"""Tower-aware LEMS rank allocation for vision-language models.

This is the novel contribution. LEMS multiplies each layer's error surrogate by
a depth bias ``1 + beta(d_i)`` -- a harmonic decay in block depth, fit
empirically -- to account for compression error compounding as it propagates
through the stack. On a plain LLM there is one stack, so "depth" is
unambiguous. On a VLM there are two, of different lengths and different
Linear-per-block counts, joined by a projector, and lems's implementation
collapses them to a single global counter (see ``vlm_lems.towers``).

We replace that single bias with a per-tower one, plus explicit cross-tower
coupling:

    m(layer) = h(d, L_vision;   a_V, g_V) * k_V     if layer is in the vision tower
               h(d, L_language; a_L, g_L)           if layer is in the language tower
               k_P                                   if layer is in the projector
               1                                     otherwise

where ``h`` is lems's own ``harmonicv2`` decay (reused, not reimplemented) and
``k_V``, ``k_P`` are scalars. The coupling terms exist because depth alone
cannot express the asymmetry: an error introduced in the vision tower passes
through the projector and perturbs the input to *every* language block, whereas
an error in language block 30 perturbs only two. Per-tower ``(a, g)`` cannot
represent that -- ``h`` is normalized within its own tower, so the deepest
vision block and the deepest language block both get the same relative weight
regardless of how much lies downstream.

Setting ``g_V = g_L = 0`` and ``k_V = k_P = 1`` makes every multiplier 1.0, so
this strictly generalizes the flat behaviour lems falls back to today
(``get_depth_multiplier(..., is_vision=True) -> 1.0``). That is what makes the
Phase-2 ablation meaningful: the tower-aware variant cannot be worse than flat
by construction, only by search noise.
"""

from __future__ import annotations

import copy
import gc
import json
import os

import torch

from compression.factorization._interface import get_valid_layers
from compression.search.lems import (
    LEMSSearch,
    get_depth_multiplier,
    optuna_inner_search,
)

from .ilp_highs import ilp_search_highs
from .vlm_eval_mixin import VLMReferenceEvalMixin
from .towers import (
    TOWER_LANGUAGE,
    TOWER_PROJECTOR,
    TOWER_VISION,
    LayerSite,
    discover_layer_sites,
    summarize_sites,
)


class TowerBias:
    """The per-tower depth bias described in the module docstring."""

    def __init__(self, a_vision=0.0, g_vision=0.0, a_language=0.0, g_language=0.0,
                 k_vision=1.0, k_projector=1.0):
        self.a_vision = a_vision
        self.g_vision = g_vision
        self.a_language = a_language
        self.g_language = g_language
        self.k_vision = k_vision
        self.k_projector = k_projector

    def __call__(self, site: LayerSite) -> float:
        if site.tower == TOWER_PROJECTOR:
            return self.k_projector
        if not site.is_block_structured:
            return 1.0
        if site.tower == TOWER_VISION:
            base = get_depth_multiplier(
                current_block=site.depth, total_blocks=site.tower_depth,
                crosslayer_term="harmonicv2", alpha=self.a_vision, gamma=self.g_vision,
            )
            return base * self.k_vision
        if site.tower == TOWER_LANGUAGE:
            return get_depth_multiplier(
                current_block=site.depth, total_blocks=site.tower_depth,
                crosslayer_term="harmonicv2", alpha=self.a_language, gamma=self.g_language,
            )
        return 1.0

    def as_dict(self) -> dict:
        return {
            "a_vision": self.a_vision, "g_vision": self.g_vision,
            "a_language": self.a_language, "g_language": self.g_language,
            "k_vision": self.k_vision, "k_projector": self.k_projector,
        }


class TOWER_LEMSSearch(VLMReferenceEvalMixin, LEMSSearch):
    """LEMS with a tower-aware depth bias. Named for lems's dynamic loader,
    which resolves ``search=tower_lems`` to ``TOWER_LEMSSearch``."""

    #: Ablation ladder. Each rung adds one mechanism, so the contribution of
    #: each can be read off directly:
    #:   flat     -- sensitivity-driven ILP, every depth multiplier 1.0. This is
    #:               what lems degenerates to on a VLM today.
    #:   tower    -- per-tower (alpha, gamma) depth decay.
    #:   coupled  -- adds the cross-tower coupling scalars k_V, k_P.
    BIAS_MODES = ("flat", "tower", "coupled")

    def __init__(self, *args, bias_mode: str = "coupled", n_trials: int = 20,
                 image_token_id: int = 32000, workspace_dir: str = "./workspace",
                 run_dir: str | None = None, solver: str = "highs", **kwargs):
        # lems's ILPSettings validates solver against {"cbc", "gurobi"} and
        # raises on anything else, so "highs" cannot be handed to it. HiGHS is
        # dispatched by our own single_search below, well before ilp_settings
        # is consulted, so lems gets a value it accepts and never acts on.
        self._ilp_backend = solver
        super().__init__(
            *args, workspace_dir=workspace_dir,
            solver=("cbc" if solver == "highs" else solver), **kwargs,
        )
        if bias_mode not in self.BIAS_MODES:
            raise ValueError(f"bias_mode must be one of {self.BIAS_MODES}, got {bias_mode!r}")
        self.bias_mode = bias_mode
        self.n_trials = n_trials
        self.image_token_id = image_token_id
        # workspace_dir is shared across the sweep (sensitivity cache); the
        # fitted bias parameters belong to this run alone.
        self.run_dir = run_dir or workspace_dir
        self.sites: dict[str, LayerSite] = {}
        self.best_bias_params: dict | None = None

    # -- structure -----------------------------------------------------

    def initialize_search(self, lrd_method, model, spec_tensor=None):
        self.sites = discover_layer_sites(model, self.name_omit)
        print(f"tower discovery: {len(self.sites)} compressible layers")
        print(summarize_sites(self.sites))
        super().initialize_search(lrd_method, model, spec_tensor)

    def prepare_data(self, size_dict, layer_sensitivity, compression_target, layers_per_block):
        """Same multiple-choice-knapsack construction as LEMS, but the depth
        multiplier comes from the layer's own tower and depth rather than from
        ``i // layers_per_block`` over a flat list."""
        from compression.factorization._interface import get_eq_rank

        bias = self._bias
        data, compression_list, active_layer_sizes = [], [], []
        layer_name_list = list(layer_sensitivity.keys())

        lower_bound = 0.1 if compression_target < 0.5 else 0.3
        upper_bound = compression_target + 1.0

        for layer_name, sensitivity_data in layer_sensitivity.items():
            if self.enforce_rank_multiples_of:
                n, m = self.shape_dict[layer_name]
                eq_rank = get_eq_rank(n, m)
                sensitivity_data = {
                    k: v for k, v in sensitivity_data.items()
                    if int(k * eq_rank) % self.enforce_rank_multiples_of == 0
                }

            site = self.sites.get(layer_name)
            multiplier = bias(site) if site is not None else 1.0

            layer_data = [(size_dict[layer_name], 0.0)] + [
                (ratio * size_dict[layer_name], sensitivity * multiplier)
                for ratio, sensitivity in sensitivity_data.items()
                if lower_bound <= ratio <= upper_bound
            ]
            layer_ratios = [1.0] + [
                r for r in sensitivity_data if lower_bound <= r <= upper_bound
            ]
            data.append(layer_data)
            compression_list.append(layer_ratios)
            active_layer_sizes.append(size_dict[layer_name])

        total_parameters = sum(active_layer_sizes)
        return data, layer_name_list, compression_list, total_parameters * compression_target

    # -- search --------------------------------------------------------

    # ``layers_per_block`` is meaningless here -- that is the very assumption
    # this class removes -- but it cannot be None: both ILP builders compute
    # ``num_blocks = num_variables // block_size`` unconditionally, before
    # checking whether any monotonicity constraint is actually enabled. LEMS
    # leaves all of those flags False, so the value is unused; 1 is the safe
    # sentinel that keeps the arithmetic valid.
    _ILP_BLOCK_SENTINEL = 1

    def single_search(self, layers_per_block, default_param_ratio):
        """Same as LEMS's, but routed to HiGHS when selected.

        lems's ``single_search`` dispatches only to Gurobi or CBC; HiGHS needs
        neither a licence nor a coarsened rank grid, so it is the default here.
        """
        if self._ilp_backend != "highs":
            return super().single_search(layers_per_block, default_param_ratio)

        data, layer_name_list, compression_list, target = self.prepare_data(
            size_dict=self.size_dict,
            layer_sensitivity=self.sensitivity_dict,
            compression_target=self.ratio_target,
            layers_per_block=layers_per_block,
        )
        chosen = ilp_search_highs(
            data=data,
            compression_list=compression_list,
            layer_name_list=layer_name_list,
            compression_param_target=target,
        )
        ratios = {name: default_param_ratio for name in self.sensitivity_dict}
        ratios.update(chosen)
        return ratios

    def search(self, model):
        self._bias = TowerBias()
        if self.bias_mode == "flat":
            # No parameters to fit: solve the ILP once with all multipliers 1.0.
            print("bias_mode=flat: single ILP solve, no depth bias, no Optuna")
            self.best_bias_params = self._bias.as_dict()
            self._persist_bias_params(best_kl=None, n_trials=0)
            return self.single_search(self._ILP_BLOCK_SENTINEL, 1.0)
        return self.grid_search(
            model, layers_per_block=self._ILP_BLOCK_SENTINEL, default_param_ratio=1.0,
        )

    def _persist_bias_params(self, best_kl, n_trials) -> None:
        """svd_core keeps the search object in a local, so the driver cannot
        read the fitted parameters off it -- persist them instead."""
        os.makedirs(self.run_dir, exist_ok=True)
        with open(os.path.join(self.run_dir, "bias_params.json"), "w") as f:
            json.dump(
                {"best_kl": best_kl, "params": self.best_bias_params,
                 "bias_mode": self.bias_mode, "n_trials": n_trials},
                f, indent=2,
            )

    def grid_search(self, model, layers_per_block, default_param_ratio,
                    n_trials=None, alpha_range=(0.0, 3.0), gamma_range=(0.0, 7.0),
                    kappa_range=(0.5, 4.0)):
        """Optuna over the per-tower bias parameters.

        Same structure as ``LEMSSearch.grid_search`` -- propose parameters,
        solve the ILP, apply the ranks temporarily, measure KL against the
        uncompressed reference -- but over 4-6 parameters instead of 2.
        """
        self.sensitivity_loss = "kl"
        self.crosslayer_term = "harmonicv2"
        n_trials = n_trials or self.n_trials

        dev = torch.device(torch.cuda.current_device())
        model_bkup = copy.deepcopy(model)
        module_bkup_dict = dict(get_valid_layers(model_bkup, self.name_omit))
        model = model.to(dev)
        module_dict = dict(get_valid_layers(model, self.name_omit))

        original_outputs = self._precompute_original_outputs(model)

        def objective(trial):
            params = {
                "a_vision": trial.suggest_float("a_vision", *alpha_range),
                "g_vision": trial.suggest_float("g_vision", *gamma_range),
                "a_language": trial.suggest_float("a_language", *alpha_range),
                "g_language": trial.suggest_float("g_language", *gamma_range),
            }
            if self.bias_mode == "coupled":
                params["k_vision"] = trial.suggest_float("k_vision", *kappa_range)
                params["k_projector"] = trial.suggest_float("k_projector", *kappa_range)
            self._bias = TowerBias(**params)

            ranks = self.single_search(layers_per_block, default_param_ratio)
            self._compress_model_ratios(module_dict, module_bkup_dict, ranks, self.lrd_method)
            metric = self._eval_llm(model, original_outputs)

            trial.set_user_attr("search_ranks", copy.deepcopy(ranks))
            trial.set_user_attr("bias_params", params)
            print(f"Trial {trial.number}: {params} -> KL={metric:.5f}", flush=True)
            return metric

        best = optuna_inner_search(objective_fn=objective, n_trials=n_trials)
        if best is None:
            raise RuntimeError("every Optuna trial failed; no rank allocation produced")

        self.best_bias_params = best.user_attrs["bias_params"]
        print(f"\nbest KL={best.value:.5f} with {self.best_bias_params}")
        self._persist_bias_params(best_kl=best.value, n_trials=n_trials)
        self._restore_model(module_dict, module_bkup_dict)
        del model_bkup
        gc.collect()
        return best.user_attrs["search_ranks"]
