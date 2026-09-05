"""VLM loading. Mirrors lems/all_utils/model_utils.py's shape (model, processor
in, ready-to-calibrate model out) but for image-text-to-text models, which that
module never supported -- it is hardcoded to AutoModelForCausalLM and a
text-only tokenizer.

Loading goes through ``AutoModelForImageTextToText`` rather than a named class,
so a second architecture needs no code change here. The image-token id is read
from the model config rather than assumed, because it differs across families
(LLaVA uses a single ``<image>`` placeholder that the processor expands, Qwen2-VL
uses ``<|image_pad|>`` between vision sentinels, Idefics-style models use
``<image>`` with a perceiver connector).
"""

from __future__ import annotations

import torch

#: Config fields different families use to record the image placeholder id.
_IMAGE_TOKEN_FIELDS = ("image_token_id", "image_token_index")
#: Fallback token strings, tried in order against the tokenizer.
_IMAGE_TOKEN_STRINGS = ("<image>", "<|image_pad|>", "<image_soft_token>")


def get_vlm_from_huggingface(model_id: str, fp32: bool = False,
                             cache_dir: str | None = None):
    """Load a VLM and its processor, in the precision it was released in.

    The checkpoint's own ``torch_dtype`` is respected rather than defaulting to
    float16. This matters: Qwen2-VL and SmolVLM are bfloat16 checkpoints, and
    loading them in float16 overflows during calibration. The resulting
    covariance carries non-finite entries, and the failure surfaces much later
    as an eigendecomposition that will not converge on either GPU or CPU --- a
    message that points at linear algebra rather than at precision.

    Returns ``(model, processor)``. ``processor`` bundles the image processor
    and tokenizer; ``processor.tokenizer`` is the plain tokenizer where only
    text handling is needed.
    """
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

    if fp32:
        dtype = torch.float32
    else:
        config = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir)
        dtype = getattr(config, "torch_dtype", None) or torch.float16
        if not isinstance(dtype, torch.dtype):
            dtype = getattr(torch, str(dtype).split(".")[-1], torch.float16)
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    # transformers 4.55 (the version lems pins) takes `torch_dtype`; the newer
    # `dtype` spelling is rejected here with a TypeError from __init__.
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=dtype, device_map="cpu", cache_dir=cache_dir,
    )
    return model, processor


def get_image_token_id(model, processor) -> int:
    """The token id standing for an image patch in the input sequence.

    Used to locate where the text begins, so the reference metric is scored on
    text positions only. Read from the config where the family records it, and
    otherwise recovered from the tokenizer by trying the known placeholder
    strings.
    """
    config = getattr(model, "config", None)
    for field in _IMAGE_TOKEN_FIELDS:
        value = getattr(config, field, None)
        if isinstance(value, int):
            return value
        nested = getattr(getattr(config, "text_config", None), field, None)
        if isinstance(nested, int):
            return nested

    tokenizer = processor.tokenizer
    for token in _IMAGE_TOKEN_STRINGS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id != tokenizer.unk_token_id:
            return token_id

    raise ValueError(
        f"could not determine the image token id for {model.config.model_type!r}; "
        "add its config field or placeholder string to vlm_lems.model_utils"
    )
