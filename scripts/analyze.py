#!/usr/bin/env python3
"""Summarise runs and compare them with paired significance tests.

    uv run --no-sync python scripts/analyze.py --runs out/runs

Every method is scored on the same evaluation examples, so comparing two runs
as independent samples throws away most of the available power: the accuracy
differences of interest here are smaller than the independent-sample standard
error, while the runs agree on the large majority of individual items. Where
per-example outcomes are available (``predictions.json``) this reports an
exact McNemar test on the discordant pairs instead.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from glob import glob


def load_runs(runs_dir: str, min_n: int, benchmark: str = "scienceqa",
              model_tag: str = "7b") -> list[dict]:
    """Runs for one (benchmark, model) pair.

    Filtering both is essential: a comparison only means anything between runs
    scored on the same examples with the same model, and the run directory
    holds several combinations at once.
    """
    runs = []
    for path in sorted(glob(os.path.join(runs_dir, "*", "result.json"))):
        record = json.load(open(path))
        if record.get("eval", {}).get("n", 0) < min_n:
            continue
        if record.get("benchmark", "scienceqa") != benchmark:
            continue
        # Match on the model id, not a loose substring: "13b" in the id is
        # false for Qwen2-VL-7B, which would otherwise be pooled with
        # LLaVA-1.5-7B and compared against runs scored on a different model.
        model_id = record.get("model", "")
        if model_tag == "13b":
            keep = "13b" in model_id
        elif model_tag == "qwen":
            keep = "Qwen" in model_id
        else:
            keep = "llava" in model_id.lower() and "13b" not in model_id
        if not keep:
            continue
        record["_name"] = os.path.basename(os.path.dirname(path))
        preds = os.path.join(os.path.dirname(path), "predictions.json")
        record["_preds"] = json.load(open(preds)) if os.path.exists(preds) else None
        runs.append(record)
    return runs


def mcnemar_exact(a: list[dict], b: list[dict]) -> tuple[int, int, float]:
    """Exact two-sided McNemar test over examples both runs scored.

    Returns ``(b01, b10, p)``: items only ``a`` got right, items only ``b`` got
    right, and the p-value under the null that each discordant pair is a coin
    flip.
    """
    by_index_a = {row["i"]: row["correct"] for row in a}
    by_index_b = {row["i"]: row["correct"] for row in b}
    shared = by_index_a.keys() & by_index_b.keys()
    b01 = sum(1 for i in shared if by_index_a[i] and not by_index_b[i])
    b10 = sum(1 for i in shared if by_index_b[i] and not by_index_a[i])

    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    lo = min(b01, b10)
    tail = sum(math.comb(n, k) for k in range(lo + 1)) / (2 ** n)
    return b01, b10, min(1.0, 2 * tail)


def paired_diff_ci(a: list[dict], b: list[dict], conf: float = 0.95):
    """Paired difference in accuracy (b minus a) with a confidence interval.

    A non-significant McNemar test is not evidence of equivalence, so the
    interval matters more than the p-value: it says how large a difference the
    data still permits. Uses the standard Wald interval for the difference of
    correlated proportions, whose variance depends only on the discordant
    counts.
    """
    import math

    by_a = {r["i"]: r["correct"] for r in a}
    by_b = {r["i"]: r["correct"] for r in b}
    shared = by_a.keys() & by_b.keys()
    n = len(shared)
    if n == 0:
        return 0.0, 0.0, 0.0
    b01 = sum(1 for i in shared if by_a[i] and not by_b[i])
    b10 = sum(1 for i in shared if by_b[i] and not by_a[i])

    diff = (b10 - b01) / n
    var = ((b01 + b10) - (b10 - b01) ** 2 / n) / (n ** 2)
    z = 1.959963984540054 if abs(conf - 0.95) < 1e-9 else 1.959963984540054
    half = z * math.sqrt(max(var, 0.0))
    return diff * 100, (diff - half) * 100, (diff + half) * 100


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="out/runs")
    parser.add_argument("--min-n", type=int, default=1000,
                        help="ignore smoke runs evaluated on fewer examples")
    parser.add_argument("--baseline", default="uniform",
                        help="substring of the run name to compare others against")
    parser.add_argument("--benchmark", default="scienceqa",
                        choices=["scienceqa", "seedbench"])
    parser.add_argument("--model", default="7b", choices=["7b", "13b", "qwen"])
    args = parser.parse_args(argv)

    runs = load_runs(args.runs, args.min_n, args.benchmark, args.model)
    if not runs:
        print(f"no {args.model}/{args.benchmark} runs with n >= {args.min_n}")
        return 1

    header = (f"{'run':30s} {'bias':9s} {'trials':>6s} {'acc%':>7s} {'+-':>5s} "
              f"{'ratio':>6s} {'vis':>6s} {'lang':>6s} {'proj':>6s} {'KL':>8s}")
    print(header)
    print("-" * len(header))
    for run in sorted(runs, key=lambda r: -r["eval"]["accuracy"]):
        acc = run["eval"]["accuracy"]
        se = math.sqrt(acc * (1 - acc) / run["eval"]["n"]) * 100
        towers = run.get("mean_ratio_by_tower") or {}
        bias = run.get("bias_params") or {}
        kl = bias.get("best_kl")
        num = lambda v, w=6, p=3: f"{v:{w}.{p}f}" if isinstance(v, float) else f"{'-':>{w}}"
        print(f"{run['_name']:30s} {str(run.get('bias_mode') or '-'):9s} "
              f"{str(bias.get('n_trials', '-')):>6s} {acc * 100:7.2f} {se:5.2f} "
              f"{run.get('param_ratio_actual', 1.0):6.4f} "
              f"{num(towers.get('vision'))} {num(towers.get('language'))} "
              f"{num(towers.get('projector'))} {num(kl, 8, 5)}")

    # Comparisons are only meaningful between runs at the same parameter
    # budget, so group by target ratio rather than testing everything against
    # one reference.
    by_ratio: dict[float, list[dict]] = {}
    for run in runs:
        if run.get("search") is not None:
            by_ratio.setdefault(run["ratio"], []).append(run)

    def compare(reference: dict, others: list[dict], label: str) -> None:
        if reference.get("_preds") is None:
            print(f"  (no per-example predictions for {reference['_name']})")
            return
        print(f"  vs {label} [{reference['_name']}]")
        for run in others:
            if run is reference or run["_preds"] is None:
                continue
            b01, b10, p = mcnemar_exact(reference["_preds"], run["_preds"])
            delta, lo, hi = paired_diff_ci(reference["_preds"], run["_preds"])
            verdict = "significant" if p < 0.05 else "n.s."
            print(f"    {run['_name']:32s} {delta:+6.2f}pp "
                  f"[{lo:+6.2f},{hi:+6.2f}]  {b01:4d}/{b10:4d}  "
                  f"p={p:.4f}  {verdict}")

    print("\npaired McNemar, within matched parameter budget")
    print("(counts are only-reference-correct / only-other-correct)")
    for ratio in sorted(by_ratio, reverse=True):
        group = by_ratio[ratio]
        print(f"\nratio {ratio}")
        uniform = next((r for r in group if r.get("search") == "uniform"), None)
        if uniform is not None:
            compare(uniform, group, "uniform")
        flat = next((r for r in group if r.get("bias_mode") == "flat"), None)
        if flat is not None:
            compare(flat, [r for r in group if r.get("bias_mode") in ("tower", "coupled")],
                    "flat (sensitivity ILP, no depth bias)")

    fp16 = next((r for r in runs if r.get("search") is None), None)
    if fp16 is not None and fp16.get("_preds") is not None:
        print("\nvs uncompressed FP16")
        for run in sorted(runs, key=lambda r: -r["eval"]["accuracy"]):
            if run is fp16 or run["_preds"] is None:
                continue
            b01, b10, p = mcnemar_exact(fp16["_preds"], run["_preds"])
            delta, lo, hi = paired_diff_ci(fp16["_preds"], run["_preds"])
            print(f"    {run['_name']:32s} {delta:+6.2f}pp "
                  f"[{lo:+6.2f},{hi:+6.2f}]  {b01:4d}/{b10:4d}  p={p:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
