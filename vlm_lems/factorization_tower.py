"""Tower-aware blockwise KFAC-SVD.

lems offers two factorization schedules:

* **one-shot** -- hook the whole model, run calibration once. Covers every
  layer, but holds the KFAC statistics for all of them simultaneously. Each
  layer needs an ``A`` of ``in_features^2`` and a ``G`` of ``out_features^2``;
  on LLaVA-1.5-7B the language tower alone is ~2 GB per block in fp16
  (``down_proj`` contributes ``11008^2`` by itself), so all 57 blocks at once
  does not fit on any partition in this cluster.
* **blockwise** -- hook one block at a time and free between blocks. Bounded
  memory, but ``_get_scale_and_factorize_block_wise`` finds its blocks with
  ``_find_decoder_layers(model)``, which returns a single stack. On a VLM that
  silently covers one tower and leaves the other -- and the projector --
  uncompressed.

This subclass keeps blockwise's memory behaviour and fixes its coverage: it
walks every stack found by ``vlm_lems.towers.find_block_stacks`` plus the
non-block-structured layers (the projector), so all 370 compressible layers on
LLaVA-1.5-7B are visited with only one block's statistics resident at a time.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from compression.factorization._interface import get_valid_layers
from compression.factorization.kfac_svd import KFAC_SVDFactorization

from .towers import find_block_stacks


def build_calibration_units(model: nn.Module, name_omit) -> list[tuple[nn.Module, str, str]]:
    """``(module_to_hook, name_prefix, label)`` for every unit of compression.

    ``name_prefix + local_name`` must equal the layer's full name in the model,
    because that is the key the factorization cache and the search's rank dict
    are both keyed on.
    """
    units: list[tuple[nn.Module, str, str]] = []
    stacks = find_block_stacks(model)
    stacked_prefixes = [name + "." for name, _ in stacks]
    omit = list(name_omit)

    def has_compressible_layers(module: nn.Module, prefix: str) -> bool:
        """Whether any layer in this unit survives ``name_omit``.

        ``get_valid_layers`` matches ``name_omit`` against names local to the
        module it is given, but exclusions like the unused vision block are
        expressed as full model paths -- so the check has to be redone against
        ``prefix + local_name``. Without this, an entirely-omitted unit is still
        visited, and calibrating it fails (nothing in it requires grad).
        """
        return any(
            all(pattern not in f"{prefix}{local}" for pattern in omit)
            for local, _ in get_valid_layers(module, [])
        )

    for stack_name, stack in stacks:
        for index, block in enumerate(stack):
            prefix = f"{stack_name}.{index}."
            if not has_compressible_layers(block, prefix):
                print(f"  skipping {stack_name}[{index}]: fully excluded by name_omit")
                continue
            units.append((block, prefix, f"{stack_name}[{index}]"))

    # Anything compressible that is not inside a stack -- on a VLM this is the
    # multimodal projector. Group by parent module so each is hooked once.
    modules_by_name = dict(model.named_modules())
    loose_parents: list[str] = []
    for layer_name, _ in get_valid_layers(model, list(name_omit)):
        if any(layer_name.startswith(prefix) for prefix in stacked_prefixes):
            continue
        parent = layer_name.rsplit(".", 1)[0] if "." in layer_name else ""
        if parent and parent not in loose_parents and parent in modules_by_name:
            loose_parents.append(parent)

    for parent in loose_parents:
        units.append((modules_by_name[parent], f"{parent}.", parent))

    return units


class TOWER_KFAC_SVDFactorization(KFAC_SVDFactorization):
    """KFAC-SVD whose blockwise schedule covers every tower.

    Named for lems's dynamic loader, which resolves ``svd=tower_kfac_svd`` to
    ``TOWER_KFAC_SVDFactorization``.
    """

    def _get_scale_and_factorize_block_wise(self, model, name_omit, calib_data,
                                            mixup_fn, white_list):
        units = build_calibration_units(model, name_omit)
        print(f"tower-aware blockwise factorization over {len(units)} units")

        for position, (hook_module, name_prefix, label) in enumerate(units):
            try:
                self._get_scale_and_factorize_module(
                    model=model,
                    hook_module=hook_module,
                    calib_data=calib_data,
                    name_omit=name_omit,
                    name_prefix=name_prefix,
                    white_list=white_list,
                    mixup_fn=mixup_fn,
                    tqdm_message=f"Unit {position + 1}/{len(units)} ({label}): Gathering ",
                )
            except RuntimeError as exc:
                if "does not require grad" not in str(exc):
                    raise
                # Autograd's message names no module, so translate it into the
                # actual cause: this unit's output never reaches the loss. The
                # known case (a vision block past config.vision_feature_layer)
                # is excluded up front in run_compress; anything else reaching
                # here is a real modelling surprise and should stop the run.
                raise RuntimeError(
                    f"unit {label!r} has no gradient path to the loss, so KFAC "
                    f"statistics cannot be collected for it. Its output is "
                    f"unused by the model. Exclude it via name_omit (see "
                    f"vlm_lems.towers.unused_vision_layer_prefixes) or confirm "
                    f"the model's feature-layer configuration."
                ) from exc

            if not (self.progressive_compression or self.compute_memory_efficient):
                continue

            for local_name, module_sub in get_valid_layers(
                hook_module, name_omit, white_list=white_list
            ):
                key = f"{name_prefix}{local_name}"
                if self.progressive_compression:
                    rank, ratio, skip = self._get_active_rank(
                        module_sub.weight.shape, key,
                        default_ratio=self.static_progressive_compression_ratio,
                    )
                    if skip:
                        continue
                    factorized = self.factorize_matrix(
                        module_sub.weight, name=key, rank=rank, ratio=ratio, verbose=False,
                    )
                    module_sub.weight.data.copy_(
                        factorized.mat_l.to(self.dev) @ factorized.mat_r.to(self.dev)
                    ).cpu()
                if self.compute_memory_efficient:
                    self._factorize_cleanup(key)
                    torch.cuda.empty_cache()
