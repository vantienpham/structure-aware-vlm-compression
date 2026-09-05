"""ScienceQA-IMG evaluation for VLMs.

lems/all_utils/evaluater.py has no multimodal path: ``zero_shot_eval`` wraps
lm-eval-harness text tasks and ``ppl_eval`` scores plain next-token CE over
token batches. Neither can score an image+question multiple-choice benchmark.

Scoring is constrained-decoding, one forward pass per example: the prompt is
built with the model's own chat template, an opening ``(`` is appended, and the
prediction is ``argmax`` over the next-token logits restricted to the answer
letters. That is cheaper than scoring each continuation separately (one pass,
not one per choice) and it cannot be defeated by a compressed model emitting
unparseable free text -- which matters here, because the whole point is
comparing models at compression ratios aggressive enough to degrade fluency.
"""

from __future__ import annotations

import random

import torch
from tqdm import tqdm

_LETTERS = "ABCDEFGH"


def _letter_token_ids(tokenizer, n: int) -> list[int]:
    """Token id for each of the first ``n`` answer letters.

    Asserts single-token encoding: on Llama's SentencePiece vocabulary each
    bare capital letter is one token (A=319, B=350, ...), and the scoring below
    is only valid while that holds.
    """
    ids = []
    for letter in _LETTERS[:n]:
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                f"answer letter {letter!r} is not a single token ({encoded}); "
                "constrained letter scoring assumes it is"
            )
        ids.append(encoded[0])
    return ids


def build_prompt(processor, question: str, choices: list[str], hint: str = "") -> str:
    """Format one example the same way vlm_lems.data_utils formats calibration."""
    options = " ".join(f"({_LETTERS[i]}) {c}" for i, c in enumerate(choices))
    if hint:
        question = f"{hint} {question}"
    user_text = f"{question} Options: {options}. Answer with the option letter."
    conversation = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]}
    ]
    text = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    # Force the answer to open with "(" so the next token is the letter itself.
    return text + "("


@torch.no_grad()
def evaluate_scienceqa(
    model,
    processor,
    split: str = "test",
    limit: int | None = None,
    device: str = "cuda",
    progress: bool = True,
) -> dict:
    """Accuracy on the image-bearing subset of ScienceQA (i.e. ScienceQA-IMG).

    Returns ``{"accuracy", "correct", "n", "n_nonfinite"}``. ``n_nonfinite``
    counts examples whose logits contained NaN/Inf -- scored as wrong, but
    reported separately, since a compressed model that has numerically blown up
    is a different failure from one that is merely inaccurate.
    """
    from datasets import load_dataset

    model = model.to(device).eval()
    tokenizer = processor.tokenizer

    dataset = load_dataset("derek-thomas/ScienceQA", split=split)
    indices = [i for i, img in enumerate(dataset["image"]) if img is not None]
    if limit is not None:
        indices = indices[:limit]

    correct = 0
    n_nonfinite = 0
    # Per-example outcomes, so methods evaluated on the same examples can be
    # compared with a paired test (McNemar). Treating these runs as independent
    # samples wastes most of the statistical power: the accuracy differences
    # here are smaller than the independent-sample standard error, but the runs
    # differ only in rank allocation and agree on the large majority of items.
    per_example: list[dict] = []
    iterator = tqdm(indices, desc=f"ScienceQA-IMG/{split}") if progress else indices

    for idx in iterator:
        example = dataset[idx]
        choices = example["choices"]
        prompt = build_prompt(
            processor, example["question"], choices, example.get("hint", "") or ""
        )
        inputs = processor(
            images=example["image"].convert("RGB"), text=prompt, return_tensors="pt"
        ).to(device)

        logits = model(**inputs).logits[0, -1]
        if not torch.isfinite(logits).all():
            n_nonfinite += 1
            per_example.append({"i": idx, "pred": None, "gold": example["answer"],
                                "correct": False})
            continue

        letter_ids = _letter_token_ids(tokenizer, len(choices))
        prediction = int(torch.argmax(logits[letter_ids]).item())
        is_correct = prediction == example["answer"]
        correct += int(is_correct)
        per_example.append({"i": idx, "pred": prediction, "gold": example["answer"],
                            "correct": bool(is_correct)})

    n = len(indices)
    return {
        "accuracy": correct / n if n else float("nan"),
        "correct": correct,
        "n": n,
        "n_nonfinite": n_nonfinite,
        "per_example": per_example,
    }


@torch.no_grad()
def evaluate_seedbench(
    model,
    processor,
    limit: int | None = None,
    device: str = "cuda",
    progress: bool = False,
) -> dict:
    """Accuracy on the image subset of SEED-Bench.

    Same constrained-letter protocol as :func:`evaluate_scienceqa`, so the two
    benchmarks are scored identically and their numbers are comparable. Only
    ``data_type == "image"`` rows are used: SEED-Bench also contains video
    questions, which LLaVA-1.5 cannot consume.
    """
    from datasets import load_dataset

    model = model.to(device).eval()
    tokenizer = processor.tokenizer
    letter_ids = _letter_token_ids(tokenizer, 4)  # SEED-Bench is always A-D

    dataset = load_dataset("lmms-lab/SEED-Bench", split="test")
    # Read the type column straight from Arrow rather than using .filter():
    # filter materialises every row, which decodes all 18k images and takes
    # ~13 minutes, once per run.
    data_types = dataset.data.column("data_type").to_pylist()
    indices = [i for i, t in enumerate(data_types) if t == "image"]
    if limit is not None and limit < len(indices):
        # Deterministic subsample. Taking a prefix would bias the sample,
        # since the benchmark is grouped by question type.
        rng = random.Random(0)
        indices = sorted(rng.sample(indices, limit))

    correct = 0
    n_nonfinite = 0
    per_example: list[dict] = []
    iterator = tqdm(indices, desc="SEED-Bench-IMG") if progress else indices

    for position in iterator:
        row = dataset[position]
        choices = [row["choice_a"], row["choice_b"], row["choice_c"], row["choice_d"]]
        prompt = build_prompt(processor, row["question"], choices)
        # The image column is a list of one PIL image.
        image = row["image"][0] if isinstance(row["image"], list) else row["image"]
        inputs = processor(
            images=image.convert("RGB"), text=prompt, return_tensors="pt"
        ).to(device)

        gold = _LETTERS.index(row["answer"].strip().upper())
        logits = model(**inputs).logits[0, -1]
        if not torch.isfinite(logits).all():
            n_nonfinite += 1
            per_example.append({"i": position, "pred": None, "gold": gold,
                                "correct": False})
            continue

        prediction = int(torch.argmax(logits[letter_ids]).item())
        is_correct = prediction == gold
        correct += int(is_correct)
        per_example.append({"i": position, "pred": prediction, "gold": gold,
                            "correct": bool(is_correct)})

    total = len(indices)
    return {
        "accuracy": correct / total if total else float("nan"),
        "correct": correct,
        "n": total,
        "n_nonfinite": n_nonfinite,
        "per_example": per_example,
    }
