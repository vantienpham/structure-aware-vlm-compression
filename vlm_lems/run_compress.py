"""Compress a VLM with lems's KFAC-SVD and evaluate it on ScienceQA-IMG.

Phase 1 driver: proves the reuse strategy end to end (lems's factorizer +
search interfaces, driven with multimodal data over a dual-tower model) before
any novel allocation code is layered on top.

    uv run --no-sync python -m vlm_lems.run_compress \
        --ratio 0.8 --search uniform --run-dir out/runs/smoke

Two lems behaviours are deliberately avoided here, both of which would silently
compress only one of the two towers:

* ``blockwise_factorization`` -- ``_get_scale_and_factorize_block_wise`` walks
  ``_find_decoder_layers(model)``, which returns the first block stack only.
  One-shot factorization hooks the whole model instead, so both towers and the
  projector are covered.
* ``KFAC_SVDFactorization(truncate=False)`` -- that mode sets
  ``post_search_calibration``, and ``svd_core`` responds by forcing
  ``blockwise_factorization=True``, re-introducing the same problem. We use
  ``truncate=True``.

Note ``vision=False`` is correct for a VLM: in lems's taxonomy that flag means
"timm-style image classifier consuming ``(data, target)`` tuples", not "has a
vision component". A VLM is fed dict batches exactly like an LLM.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from .lems_path import add_lems_to_syspath

add_lems_to_syspath()

from .data_utils import get_scienceqa_calib_data  # noqa: E402
from .evaluater import evaluate_scienceqa, evaluate_seedbench  # noqa: E402
from .model_utils import get_image_token_id, get_vlm_from_huggingface  # noqa: E402
from .towers import (  # noqa: E402
    discover_layer_sites,
    summarize_sites,
    unused_vision_layer_prefixes,
)

# lems defaults, from lems/configs/config.yaml
NAME_OMIT = ["norm", "patch_embed", "head", "downsample", "decoder.project_", "dt_proj"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--ratio", type=float, required=True,
                   help="parameter retention target, e.g. 0.8")
    p.add_argument("--search", default="uniform",
                   help="allocation rule: uniform, tower_lems, or one of the "
                        "competing rules vlm_asvd / vlm_mrcs / vlm_atp / "
                        "vlm_svdllmv2 (lems's own baselines, VLM-enabled)")
    p.add_argument("--svd", default="tower_kfac_svd",
                   help="factorization method; tower_kfac_svd is KFAC-SVD with a "
                        "blockwise schedule that covers every tower")
    p.add_argument("--calib-samples", type=int, default=256)
    p.add_argument("--eval-limit", type=int, default=None,
                   help="cap eval examples (smoke tests); default = full split")
    p.add_argument("--eval-split", default="test")
    p.add_argument("--benchmark", default="scienceqa",
                   choices=["scienceqa", "seedbench"],
                   help="evaluation benchmark; both use the same constrained "
                        "letter-scoring protocol, so methods are measured the "
                        "same way on each -- but the two datasets sample "
                        "different items, so their accuracies are not "
                        "comparable to each other")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--workspace", default="out/workspace",
                   help="shared cache root for KFAC factorizations and layer "
                        "sensitivities. Both are independent of the compression "
                        "ratio and of the search method, so every run in an "
                        "ablation sweep can reuse one copy.")
    p.add_argument("--no-cache", action="store_true",
                   help="recompute factorizations/sensitivities instead of reusing "
                        "the shared workspace")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--baseline", action="store_true",
                   help="skip compression; evaluate the model as-is")
    p.add_argument("--search-samples", type=int, default=32,
                   help="examples used for the search's KL reference metric")
    p.add_argument("--n-trials", type=int, default=20,
                   help="Optuna trials for tower_lems")
    p.add_argument("--solver", default="highs", choices=["highs", "cbc", "gurobi"],
                   help="ILP solver. Default highs: open source, no licence, and "
                        "strong enough to keep a fine rank grid. gurobi needs a "
                        "full licence (the size-limited one rejects this model); "
                        "cbc is free but weak enough that lems recommends "
                        "coarsening the grid to compensate.")
    p.add_argument("--rank-multiple", type=int, default=8,
                   help="restrict candidate ranks to multiples of this. Matches "
                        "lems's own default of 8; CBC needs a coarser 64 to stay "
                        "tractable, HiGHS does not.")
    p.add_argument("--bias-mode", default="coupled",
                   choices=["flat", "tower", "coupled"],
                   help="tower_lems ablation rung: flat = sensitivity ILP with no "
                        "depth bias (what lems degenerates to on a VLM); tower = "
                        "per-tower depth decay; coupled = adds cross-tower coupling")
    p.add_argument("--no-answer-calib", action="store_true",
                   help="build calibration sequences from the prompt only. The "
                        "calibration loss is teacher-forced next-token "
                        "prediction over the sequence, so with the assistant "
                        "turn present the gold answer tokens are targets; this "
                        "removes them, making factor estimation strictly "
                        "label-free.")
    p.add_argument("--progress", action="store_true",
                   help="show tqdm bars (off by default: they flood batch logs)")
    return p


def register_tower_search() -> None:
    """Make ``search=tower_lems`` resolvable by lems's dynamic loader.

    ``compression/svd_core.py::get_search_module`` does
    ``importlib.import_module(f"compression.search.{name}")`` and then
    ``getattr(module, f"{name.upper()}Search")``. Our search class lives in
    this repo, not in the lems clone (which we never write to), so we publish
    it under the name that loader will look up. This keeps the rest of
    svd_core's orchestration -- ordering, backup/restore, post-search
    recalibration -- reused rather than reimplemented.
    """
    import sys

    from . import factorization_tower, numerics, search_baselines, search_tower

    numerics.install_robust_whitening()

    sys.modules.setdefault("compression.search.tower_lems", search_tower)
    sys.modules.setdefault("compression.factorization.tower_kfac_svd", factorization_tower)
    # Competing allocation rules, VLM-enabled. Registered under the names
    # lems's loader derives its class names from: vlm_asvd -> VLM_ASVDSearch.
    for name in ("vlm_asvd", "vlm_mrcs", "vlm_atp", "vlm_svdllmv2"):
        sys.modules.setdefault(f"compression.search.{name}", search_baselines)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not args.progress:
        # lems's calibration loops create their own tqdm bars; without this a
        # single job writes tens of thousands of progress lines into the Slurm
        # log, which buries the numbers that matter.
        os.environ["TQDM_DISABLE"] = "1"
    os.makedirs(args.run_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    print(f"loading {args.model}")
    model, processor = get_vlm_from_huggingface(args.model)

    # Vision blocks past the feature layer are dead weight: no gradient path,
    # and compressing them would be a free ratio win. See towers.py.
    dead_prefixes = unused_vision_layer_prefixes(model)
    if dead_prefixes:
        print(f"excluding {len(dead_prefixes)} unused vision block(s) past the "
              f"feature layer: {', '.join(dead_prefixes)}")
    name_omit = NAME_OMIT + dead_prefixes

    image_token_id = get_image_token_id(model, processor)
    sites = discover_layer_sites(model, name_omit)
    print(f"discovered {len(sites)} compressible layers "
          f"(image token id {image_token_id})")
    print(summarize_sites(sites))

    result = {
        "model": args.model,
        "ratio": args.ratio if not args.baseline else 1.0,
        "search": None if args.baseline else args.search,
        "svd": None if args.baseline else args.svd,
        "bias_mode": args.bias_mode if (not args.baseline and args.search == "tower_lems") else None,
        "solver": args.solver if (not args.baseline and args.search == "tower_lems") else None,
        "rank_multiple": args.rank_multiple if (not args.baseline and args.search == "tower_lems") else None,
        "calib_samples": args.calib_samples,
        "calib_includes_answer": not args.no_answer_calib,
        "seed": args.seed,
        "towers": {
            tower: sum(1 for s in sites.values() if s.tower == tower)
            for tower in sorted({s.tower for s in sites.values()})
        },
        "excluded_unused_vision_blocks": dead_prefixes,
        "n_compressible_layers": len(sites),
    }
    params_before = sum(p.numel() for p in model.parameters())
    result["params_before"] = params_before

    if not args.baseline:
        register_tower_search()
        from compression.svd_core import ModelFactorizer

        cache_dir = os.path.join(args.run_dir, "..", "..", "cache", "calib")
        calib_data = get_scienceqa_calib_data(
            processor, nsamples=args.calib_samples, seed=args.seed,
            output_dir=cache_dir, include_answer=not args.no_answer_calib,
        )
        # The search's KL reference set is held out from the calibration set,
        # so the bias parameters are not fit on the same examples that produced
        # the KFAC statistics.
        search_data = get_scienceqa_calib_data(
            processor, nsamples=args.search_samples, seed=args.seed + 1000,
            output_dir=cache_dir, include_answer=not args.no_answer_calib,
        )
        print(f"calibration: {len(calib_data)} examples; "
              f"search reference: {len(search_data)} examples")

        use_cache = not args.no_cache
        svd_args = {
            "vision": False,          # dict-batch pipeline, not timm-style; see module docstring
            "truncate": True,         # avoids svd_core forcing progressive recalibration
            "fast": True,
            "use_cache": use_cache,
            # Blockwise keeps only one block's KFAC statistics resident; the
            # tower_kfac_svd subclass is what makes blockwise cover both towers
            # instead of whichever one _find_decoder_layers happens to return.
            "blockwise_factorization": True,
            "progressive_compression": False,
            "do_post_calibration": "default",
            # Shared, not per-run: KFAC statistics depend on the model and the
            # calibration set, not on the ratio or the search method, so one
            # copy serves the whole ablation sweep.
            "workspace_dir": args.workspace,
        }
        search_args = {
            "ratio_target": args.ratio,
            "workspace_dir": args.workspace,   # shared sensitivity cache
            "use_cache": use_cache,
        }
        if args.search.startswith("vlm_"):
            search_args["image_token_id"] = image_token_id
        if args.search == "tower_lems":
            search_args.update({
                "n_trials": args.n_trials,
                "bias_mode": args.bias_mode,
                "run_dir": args.run_dir,       # per-run: bias_params.json
                "solver": args.solver,
                "enforce_rank_multiples_of": args.rank_multiple,
                "image_token_id": image_token_id,
            })

        factorizer = ModelFactorizer(
            svd_method=args.svd, svd_method_args=svd_args,
            search_method=args.search, search_method_args=search_args,
        )
        start = time.time()
        total_time, search_time, num_params, model, rank_dict = factorizer.factorize_and_search(
            model=model, calib_data=calib_data, eval_data=search_data,
            # The cache key is built from this name, and lems does not fold the
            # sample count or seed into it -- so they go here, or a rerun with
            # different calibration would silently reuse the old statistics.
            calib_dataset_name=(f"scienceqa-n{args.calib_samples}-s{args.seed}"
                                + ("-noans" if args.no_answer_calib else "")),
            mixup_fn=None,
            name_omit=name_omit, blockwise_search=False,
        )
        result.update({
            "params_after": num_params,
            "param_ratio_actual": num_params / params_before,
            "compress_minutes": total_time,
            "search_minutes": search_time,
            "wall_minutes": (time.time() - start) / 60,
        })
        with open(os.path.join(args.run_dir, "rank_dict.json"), "w") as f:
            json.dump({k: float(v) for k, v in rank_dict.items()}, f, indent=2)
        bias_path = os.path.join(args.run_dir, "bias_params.json")
        if os.path.exists(bias_path):
            with open(bias_path) as f:
                result["bias_params"] = json.load(f)
        # Per-tower mean retained ratio: the headline diagnostic for whether
        # tower-aware allocation actually allocates differently per tower.
        per_tower: dict[str, list[float]] = {}
        for name, ratio in rank_dict.items():
            site = sites.get(name)
            if site is not None:
                per_tower.setdefault(site.tower, []).append(float(ratio))
        result["mean_ratio_by_tower"] = {
            t: sum(v) / len(v) for t, v in sorted(per_tower.items()) if v
        }

    print(f"evaluating on {args.benchmark}")
    if args.benchmark == "seedbench":
        eval_result = evaluate_seedbench(
            model, processor, limit=args.eval_limit, progress=args.progress,
        )
    else:
        eval_result = evaluate_scienceqa(
            model, processor, split=args.eval_split, limit=args.eval_limit,
            progress=args.progress,
        )
    # Keep the per-example outcomes out of result.json (2k rows would swamp it)
    # but on disk, so paired significance tests are possible after the fact.
    per_example = eval_result.pop("per_example", None)
    if per_example is not None:
        with open(os.path.join(args.run_dir, "predictions.json"), "w") as f:
            json.dump(per_example, f)
    result["eval"] = eval_result
    result["eval_split"] = args.eval_split
    result["benchmark"] = args.benchmark
    result["eval_limit"] = args.eval_limit
    print("RESULT", json.dumps(result, indent=2))

    with open(os.path.join(args.run_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
