#!/usr/bin/env python3
"""Full parameter accounting for a compressed configuration.

    uv run --no-sync python scripts/param_audit.py --run-dir out/runs/tower_lems-flat-0.8

Reviewers cannot audit a compression ratio from per-tower means alone, because
those means are unweighted over layers of very different sizes. This
reconstructs the budget exactly from the model and the saved rank assignment:

  total            every parameter in the deployed model
  eligible dense   the linear matrices the allocator may compress
  fixed            everything else (embeddings, norms, LM head, biases, and
                   the excluded vision block), carried at full precision
  retained         eligible parameters actually kept, after factorization

and reports the realized whole-model retention ratio, which is the number the
paper's targets refer to. A rank r replaces an m x n matrix with m*r + r*n
parameters, so it only saves anything below r* = mn/(m+n); at or above that
threshold the dense matrix is kept, and the audit reports how often that
happens.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vlm_lems.lems_path import add_lems_to_syspath  # noqa: E402

add_lems_to_syspath()


def audit(model, rank_dict: dict, name_omit) -> dict:
    import torch.nn as nn

    from compression.factorization._interface import get_valid_layers

    total = sum(p.numel() for p in model.parameters())
    eligible = dict(get_valid_layers(model, list(name_omit)))

    eligible_dense = 0
    retained = 0
    kept_dense = 0
    per_tower: dict[str, dict[str, float]] = {}

    from vlm_lems.towers import discover_layer_sites

    sites = discover_layer_sites(model, list(name_omit))

    for name, module in eligible.items():
        out_features, in_features = module.weight.shape
        dense = out_features * in_features
        eligible_dense += dense

        ratio = float(rank_dict.get(name, 1.0))
        threshold = out_features * in_features / (out_features + in_features)
        rank = int(round(ratio * threshold))
        factored = rank * (out_features + in_features)
        # A factorization is only adopted when it is actually smaller.
        if factored >= dense:
            kept = dense
            kept_dense += 1
        else:
            kept = factored
        retained += kept

        tower = sites[name].tower if name in sites else "other"
        bucket = per_tower.setdefault(tower, {"dense": 0, "retained": 0, "layers": 0})
        bucket["dense"] += dense
        bucket["retained"] += kept
        bucket["layers"] += 1

    fixed = total - eligible_dense
    final = fixed + retained
    return {
        "total_parameters": total,
        "fixed_parameters": fixed,
        "eligible_dense_parameters": eligible_dense,
        "retained_eligible_parameters": retained,
        "final_parameters": final,
        "whole_model_retention": final / total,
        "eligible_retention": retained / eligible_dense,
        "layers_kept_dense_by_threshold": kept_dense,
        "per_tower": {
            tower: {
                "layers": v["layers"],
                "dense_parameters": v["dense"],
                "retained_parameters": v["retained"],
                "parameter_weighted_retention": v["retained"] / v["dense"],
                "parameters_saved": v["dense"] - v["retained"],
            }
            for tower, v in sorted(per_tower.items())
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, nargs="+",
                        help="one or more run directories; the model is loaded once")
    parser.add_argument("--model", default=None,
                        help="defaults to the model recorded in result.json")
    parser.add_argument("--out", default=None, help="write JSON here as well")
    args = parser.parse_args(argv)

    from vlm_lems.model_utils import get_vlm_from_huggingface
    from vlm_lems.run_compress import NAME_OMIT
    from vlm_lems.towers import unused_vision_layer_prefixes

    first = json.load(open(os.path.join(args.run_dir[0], "result.json")))
    model_id = args.model or first["model"]
    model, _ = get_vlm_from_huggingface(model_id)
    name_omit = NAME_OMIT + unused_vision_layer_prefixes(model)

    reports = {}
    for run_dir in args.run_dir:
        result = json.load(open(os.path.join(run_dir, "result.json")))
        if result["model"] != model_id:
            print(f"skipping {run_dir}: different model", file=sys.stderr)
            continue
        rank_path = os.path.join(run_dir, "rank_dict.json")
        if not os.path.exists(rank_path):
            continue
        report = audit(model, json.load(open(rank_path)), name_omit)
        name = os.path.basename(os.path.abspath(run_dir))
        report["run"] = name
        report["model"] = model_id
        report["target_ratio"] = result.get("ratio")
        reports[name] = report
        print(f"{name:28s} whole={report['whole_model_retention']:.4f} "
              f"eligible={report['eligible_retention']:.4f} "
              f"dense_kept={report['layers_kept_dense_by_threshold']:3d} "
              + " ".join(f"{t[:4]}={v['parameter_weighted_retention']:.3f}"
                         for t, v in report["per_tower"].items()))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(reports, f, indent=2, sort_keys=True)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
