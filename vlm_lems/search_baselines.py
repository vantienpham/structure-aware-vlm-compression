"""Competing rank-allocation rules, made runnable on a VLM.

The reference LEMS implementation ships several allocation baselines besides
uniform --- ASVD thresholding, MRCS greedy search, ATP, and SVD-LLMv2 --- but
none of them runs on a vision-language model as shipped, for the reason given
in :mod:`vlm_lems.vlm_eval_mixin`. Each class here is the unmodified lems
search with the VLM-safe reference metric mixed in ahead of it, and, for ATP,
with tower-aware block discovery substituted for the first-match variant.

This is what allows the paper to compare allocation *rules* while holding the
decomposition (KFAC-SVD), the eligible-layer set, the calibration data, the
parameter accounting, and the evaluation protocol fixed --- so a difference
between rows is attributable to allocation and nothing else.
"""

from __future__ import annotations

import torch.nn as nn

from compression.search.asvd import ASVDSearch
from compression.search.atp import ATPSearch
from compression.search.memvit import MEMVITSearch
from compression.search.svd_llmv2 import SVD_LLMV2Search

from .towers import find_block_stacks
from .vlm_eval_mixin import VLMReferenceEvalMixin


class _VLMSearchBase(VLMReferenceEvalMixin):
    """Accept and stash the image token id without disturbing lems's kwargs."""

    def __init__(self, *args, image_token_id: int = 32000, **kwargs):
        self.image_token_id = image_token_id
        super().__init__(*args, **kwargs)


class VLM_ASVDSearch(_VLMSearchBase, ASVDSearch):
    """ASVD sensitivity-threshold allocation."""


class VLM_MRCSSearch(_VLMSearchBase, MEMVITSearch):
    """MRCS greedy rank allocation (``memvit`` in the reference code)."""


class VLM_SVDLLMV2Search(_VLMSearchBase, SVD_LLMV2Search):
    """SVD-LLMv2 allocation."""


class VLM_ATPSearch(_VLMSearchBase, ATPSearch):
    """ATP allocation.

    ATP is the one baseline that also calls ``_find_decoder_layers`` directly,
    to size its per-block schedule. On a VLM that returns the vision tower
    alone, so it is replaced here with the concatenation of every tower's
    blocks --- the same correction the proposed method applies, so the
    comparison isolates the allocation rule rather than re-testing the
    discovery bug.
    """

    def search(self, model: nn.Module):
        import compression.search.atp as atp_module

        original = atp_module._find_decoder_layers

        def all_tower_blocks(m):
            stacks = find_block_stacks(m)
            if not stacks:
                return original(m)
            merged = nn.ModuleList([block for _, stack in stacks for block in stack])
            return merged, stacks[0][0]

        atp_module._find_decoder_layers = all_tower_blocks
        try:
            return super().search(model)
        finally:
            atp_module._find_decoder_layers = original
