"""Multimodal calibration data. lems/all_utils/data_utils.py only ever builds
``{"input_ids", "attention_mask"}`` batches from raw text -- there is no image
path anywhere in that module. This builds ScienceQA-IMG batches that also
carry ``pixel_values``, in the same list-of-single-example-dicts shape lems's
own ``get_calib_train_data`` returns, so lems's ``_run_forward_calib`` /
``_run_backward_calib`` (which just do ``batch = {k: v.to(dev) for k, v in
batch.items()}; model(**batch)``) consume it unchanged.

Calibration stays label-free in the sense LEMS's own LLM calibration is: the
loss is ordinary next-token prediction over a *complete, correctly formatted*
conversation (image + question + the correct answer, via the model's own chat
template) -- not a separate classification loss against an external target,
which is what WSVD/LASER's Fisher-weighting step uses instead.
"""

from __future__ import annotations

import hashlib
import os
import random

import torch
from datasets import load_dataset


_LETTERS = "ABCDEFGH"


def _format_example(example: dict) -> tuple[str, str]:
    """Return (user_text, assistant_text) for one ScienceQA-IMG example."""
    choices = example["choices"]
    options = " ".join(f"({_LETTERS[i]}) {c}" for i, c in enumerate(choices))
    question = example["question"]
    if example.get("hint"):
        question = f"{example['hint']} {question}"
    user_text = f"{question} Options: {options}. Answer with the option letter."
    assistant_text = f"({_LETTERS[example['answer']]})"
    return user_text, assistant_text


def get_scienceqa_calib_data(
    processor,
    nsamples: int,
    seed: int = 3,
    split: str = "train",
    output_dir: str = "./.cache/",
    include_answer: bool = True,
):
    """Sample ``nsamples`` image-bearing ScienceQA-IMG examples, formatted with
    the model's own chat template, tokenized via ``processor``.

    Returns a list of ``nsamples`` dicts, each ``{"input_ids", "attention_mask",
    "pixel_values"}`` with a batch dimension of 1 -- the same shape
    ``lems/all_utils/data_utils.py::get_calib_train_data`` returns for text.
    """
    suffix = "" if include_answer else "_noans"
    cache_file = os.path.join(
        output_dir,
        f"scienceqa_{split}_{nsamples}_{seed}{suffix}_"
        f"{hashlib.sha256(processor.tokenizer.name_or_path.encode()).hexdigest()[:12]}.pt",
    )
    if os.path.exists(cache_file):
        return torch.load(cache_file, weights_only=False)

    os.makedirs(output_dir, exist_ok=True)
    ds = load_dataset("derek-thomas/ScienceQA", split=split)
    image_indices = [i for i, has_img in enumerate(ds["image"]) if has_img is not None]

    rng = random.Random(seed)
    rng.shuffle(image_indices)
    if len(image_indices) < nsamples:
        raise ValueError(
            f"requested {nsamples} image-bearing ScienceQA-{split} examples, "
            f"only {len(image_indices)} exist"
        )

    calib_data = []
    for idx in image_indices[:nsamples]:
        example = ds[idx]
        user_text, assistant_text = _format_example(example)
        conversation = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
        ]
        if include_answer:
            conversation.append(
                {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]}
            )
            text = processor.apply_chat_template(conversation, tokenize=False)
        else:
            # Prompt only. lems's calibration loss is teacher-forced next-token
            # prediction over the sequence itself, so any answer token present
            # here becomes a target -- which is precisely what "label-free"
            # must exclude. Dropping the assistant turn removes every gold
            # token from the calibration signal.
            text = processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
        batch = processor(images=example["image"].convert("RGB"), text=text, return_tensors="pt")
        calib_data.append(dict(batch))

    torch.save(calib_data, cache_file)
    return calib_data
