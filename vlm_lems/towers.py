"""Dual-tower structure discovery for VLMs.

lems locates "the" transformer block stack with
``compression/factorization/_interface.py::_find_decoder_layers``, which walks
``named_modules()`` and returns the **first** ``nn.ModuleList`` whose child has
a ``self_attn`` attribute. ``compression/search/lems.py::search`` then derives
one global divisor from it::

    stages, _ = _find_decoder_layers(model)
    layers_per_block = sum(1 for m in stages[0].modules() if isinstance(m, nn.Linear))
    ...
    current_block = i // layers_per_block      # i indexes ALL compressible layers

Both assumptions break on a dual-tower VLM. Measured on llava-hf/llava-1.5-7b-hf:

    model.vision_tower.vision_model.encoder.layers   24 blocks x 6 Linear = 144
    model.multi_modal_projector                       -            2 Linear =   2
    model.language_model.layers                      32 blocks x 7 Linear = 224

Both stacks expose ``self_attn``, so ``_find_decoder_layers`` returns the
*vision* tower -- the smaller one -- and ``layers_per_block`` becomes 6. Applied
to the flat 370-layer list that divisor assigns language-tower layers to
fictitious block indices, so the depth-bias term LEMS multiplies its error
surrogate by is attached to the wrong layers. The failure is silent: the ILP
still solves, it just optimizes a scrambled objective.

This module replaces that single (stack, divisor) pair with an explicit
per-layer site record, so depth is always "which block of *which tower*",
never a global counter.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

# Substrings that identify a stack's role from its qualified module path.
# Checked against the lowercased path, vision first: a VLM's vision stack often
# sits under a name that also contains a language marker (e.g. Qwen2-VL's
# "visual" tower lives beside "model.layers"), but never the reverse.
_VISION_MARKERS = ("vision_tower", "vision_model", "visual", "image_encoder", "img_encoder")
_LANGUAGE_MARKERS = ("language_model", "text_model", "text_decoder")
_PROJECTOR_MARKERS = ("projector", "connector", "merger", "mm_proj", "abstractor")

TOWER_VISION = "vision"
TOWER_LANGUAGE = "language"
TOWER_PROJECTOR = "projector"
TOWER_OTHER = "other"


@dataclass(frozen=True)
class LayerSite:
    """Where one compressible Linear sits in the model's tower structure."""

    name: str
    tower: str
    depth: int        # block index within its own tower; -1 if not block-structured
    tower_depth: int  # number of blocks in that tower; 0 if not block-structured

    @property
    def is_block_structured(self) -> bool:
        return self.depth >= 0


def _is_block_like(module: nn.Module) -> bool:
    """True for a module that looks like a transformer block."""
    return any(hasattr(module, attr) for attr in ("self_attn", "mixer", "attn", "attention"))


def find_block_stacks(model: nn.Module) -> list[tuple[str, nn.ModuleList]]:
    """Every transformer block stack in the model, nested stacks removed.

    Unlike ``_find_decoder_layers`` this returns *all* stacks rather than the
    first match, which is the whole point on a dual-tower model.
    """
    candidates = [
        (name, mod)
        for name, mod in model.named_modules()
        if isinstance(mod, nn.ModuleList) and len(mod) > 0 and _is_block_like(mod[0])
    ]
    # A stack nested inside another stack's block is not a top-level tower.
    return [
        (name, mod)
        for name, mod in candidates
        if not any(name.startswith(other + ".") for other, _ in candidates if other != name)
    ]


def classify_stack(stack_name: str) -> str:
    """Assign a block stack to a tower from its module path."""
    low = stack_name.lower()
    if any(marker in low for marker in _VISION_MARKERS):
        return TOWER_VISION
    if any(marker in low for marker in _LANGUAGE_MARKERS):
        return TOWER_LANGUAGE
    # A model with one unlabelled stack is a plain LLM: that stack is the
    # language tower, which keeps this function correct for lems's own models.
    return TOWER_LANGUAGE


def _classify_unstacked(layer_name: str) -> str:
    low = layer_name.lower()
    if any(marker in low for marker in _PROJECTOR_MARKERS):
        return TOWER_PROJECTOR
    return TOWER_OTHER


def discover_layer_sites(model: nn.Module, name_omit=()) -> dict[str, LayerSite]:
    """Map every compressible Linear to its ``LayerSite``.

    The layer set is taken from lems's own ``get_valid_layers`` so it matches
    exactly what the compression pipeline will act on -- if these two disagreed,
    the search would be allocating ranks to layers the factorizer never touches.
    """
    from compression.factorization._interface import get_valid_layers

    stacks = find_block_stacks(model)
    sites: dict[str, LayerSite] = {}

    for name, _module in get_valid_layers(model, list(name_omit)):
        for stack_name, stack in stacks:
            prefix = stack_name + "."
            if not name.startswith(prefix):
                continue
            head = name[len(prefix):].split(".", 1)[0]
            if not head.isdigit():
                continue
            sites[name] = LayerSite(
                name=name,
                tower=classify_stack(stack_name),
                depth=int(head),
                tower_depth=len(stack),
            )
            break
        else:
            sites[name] = LayerSite(
                name=name, tower=_classify_unstacked(name), depth=-1, tower_depth=0,
            )

    return sites


def unused_vision_layer_prefixes(model: nn.Module) -> list[str]:
    """Name prefixes of vision-encoder blocks whose output the model discards.

    LLaVA reads its visual features from an intermediate CLIP layer, not the
    last one: ``config.vision_feature_layer = -2`` on LLaVA-1.5, so with 24
    encoder blocks, block 23 never contributes to the output. Two things follow,
    and both matter:

    * It has no gradient path, so KFAC calibration on it dies with an opaque
      ``element 0 of tensors does not require grad`` from ``loss.backward()``.
    * Compressing it would shed parameters at zero accuracy cost, inflating the
      reported compression ratio for free. A naive port of an LLM compression
      pipeline to a VLM does exactly that, silently.

    We therefore exclude these blocks from compression and leave them at full
    rank, which makes the reported ratio conservative rather than flattering.
    """
    config = getattr(model, "config", None)
    vision_config = getattr(config, "vision_config", None)
    feature_layer = getattr(config, "vision_feature_layer", None)
    num_layers = getattr(vision_config, "num_hidden_layers", None)
    if feature_layer is None or num_layers is None:
        return []
    # A list means several layers are read; the deepest one bounds what is used.
    if isinstance(feature_layer, (list, tuple)):
        if not feature_layer:
            return []
        feature_layer = max(
            (index if index >= 0 else num_layers + index) for index in feature_layer
        )
    last_used = feature_layer if feature_layer >= 0 else num_layers + feature_layer

    stacks = [
        (name, stack) for name, stack in find_block_stacks(model)
        if classify_stack(name) == TOWER_VISION
    ]
    prefixes = []
    for stack_name, stack in stacks:
        for index in range(last_used + 1, len(stack)):
            prefixes.append(f"{stack_name}.{index}.")
    return prefixes


def summarize_sites(sites: dict[str, LayerSite]) -> str:
    """One-line-per-tower summary, for logging what discovery actually found."""
    towers: dict[str, list[LayerSite]] = {}
    for site in sites.values():
        towers.setdefault(site.tower, []).append(site)

    lines = []
    for tower, members in sorted(towers.items()):
        depths = {m.depth for m in members if m.is_block_structured}
        if not depths:
            lines.append(f"  {tower:<10} {len(members):>4} layers (not block-structured)")
            continue
        # Distinct depths present, not tower_depth: they differ whenever blocks
        # are excluded (e.g. a vision block past the feature layer).
        stack_len = members[0].tower_depth
        covered = f"{len(depths)} of {stack_len} blocks" if len(depths) != stack_len \
            else f"{stack_len} blocks"
        lines.append(
            f"  {tower:<10} {len(members):>4} layers, {covered}, "
            f"{len(members) // len(depths)} Linear/block"
        )
    return "\n".join(lines)
