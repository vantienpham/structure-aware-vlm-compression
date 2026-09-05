"""VLM-safe replacements for lems's reference-output machinery.

``SensitivityBasedSearch._precompute_original_outputs`` collates the evaluation
set with ``torch.cat(..., dim=0)`` and pins the result. That requires every
batch to share a sequence length, which holds for lems's own calibration data
(fixed 2048-token windows) but not for VLM prompts: ScienceQA question lengths
vary, and the processor then inserts 576 image tokens into each. Every search
in lems that inherits from that class therefore fails on VLM data before it
reaches any of its own logic.

Iterating per batch avoids the problem without padding, which would otherwise
dilute the metric with positions carrying no signal. Mixing this in ahead of a
lems search class makes that search usable on a VLM with no other change, which
is what lets the allocation baselines in this paper run under exactly the same
protocol as the proposed method.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

#: Token id of ``<image>`` for the LLaVA family. Overridden per run.
DEFAULT_IMAGE_TOKEN_ID = 32000


class VLMReferenceEvalMixin:
    """Per-batch reference outputs and KL metric for variable-length VLM data."""

    image_token_id: int = DEFAULT_IMAGE_TOKEN_ID

    def _precompute_original_outputs(self, model):
        dev = torch.device(torch.cuda.current_device())
        model = model.to(dev).eval()
        references = []
        with torch.no_grad():
            for batch in self.eval_data:
                batch = {k: v.to(dev) for k, v in batch.items()}
                references.append(
                    model(**batch).logits.detach().to(torch.bfloat16).cpu()
                )
        torch.cuda.empty_cache()
        return references

    def _text_start(self, input_ids: torch.Tensor) -> int:
        """First position after the image block.

        Scoring text positions only. Compressing the vision tower still moves
        the metric --- its effect reaches the text stream through the projector
        --- but the image positions themselves carry no prediction signal.
        """
        image_positions = (input_ids[0] == self.image_token_id).nonzero()
        if image_positions.numel() == 0:
            return 0
        return int(image_positions[-1].item()) + 1

    def _eval_llm(self, cp_model, original_outputs):
        dev = torch.device(torch.cuda.current_device())
        cp_model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for batch, reference in zip(self.eval_data, original_outputs):
                batch = {k: v.to(dev) for k, v in batch.items()}
                logits = cp_model(**batch).logits
                reference = reference.to(dev)
                start = self._text_start(batch["input_ids"])

                target = reference[:, start:, :].float() / 0.6
                predicted = logits[:, start:, :].float() / 0.6
                if not (torch.isfinite(target).all() and torch.isfinite(predicted).all()):
                    # A numerically blown-up candidate must not score well.
                    total += 1e4
                    n += 1
                    continue
                total += float(F.kl_div(
                    F.log_softmax(predicted, dim=-1).flatten(0, 1),
                    F.softmax(target, dim=-1).flatten(0, 1),
                    reduction="batchmean",
                ).item())
                n += 1
        torch.cuda.empty_cache()
        return total / max(n, 1)
