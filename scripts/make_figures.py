#!/usr/bin/env python3
"""Generate the manuscript's data figures from run artifacts.

    uv run --no-sync python scripts/make_figures.py --out results/figures

Two figures, both from committed run data rather than illustration:

* ``allocation_by_depth`` -- parameter-weighted retained ratio per block, per
  tower, for two architectures. This shows the allocation policy directly,
  which arithmetic tower means cannot.
* ``accuracy_vs_retention`` -- the operating curve, uniform against the
  proposed allocator, with the uncompressed model marked.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vlm_lems.lems_path import add_lems_to_syspath  # noqa: E402

add_lems_to_syspath()

# Colour-blind-safe, and distinguishable when printed in greyscale by pairing
# each colour with a distinct marker and line style.
VISION = "#0072B2"
LANGUAGE = "#D55E00"
PROJECTOR = "#009E73"
UNIFORM = "#7F7F7F"
OURS = "#0072B2"

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight",
})


def per_block_ratios(model_id: str, run_dir: str) -> dict:
    """Parameter-weighted retained ratio for each block of each tower."""
    import torch.nn as nn

    from compression.factorization._interface import get_valid_layers

    from vlm_lems.model_utils import get_vlm_from_huggingface
    from vlm_lems.run_compress import NAME_OMIT
    from vlm_lems.towers import discover_layer_sites, unused_vision_layer_prefixes

    model, _ = get_vlm_from_huggingface(model_id)
    name_omit = NAME_OMIT + unused_vision_layer_prefixes(model)
    sites = discover_layer_sites(model, name_omit)
    ranks = json.load(open(os.path.join(run_dir, "rank_dict.json")))

    buckets: dict[tuple[str, int], list[int]] = {}
    for name, module in get_valid_layers(model, list(name_omit)):
        out_features, in_features = module.weight.shape
        dense = out_features * in_features
        threshold = dense / (out_features + in_features)
        rank = int(round(float(ranks.get(name, 1.0)) * threshold))
        kept = min(rank * (out_features + in_features), dense)
        site = sites[name]
        key = (site.tower, site.depth)
        bucket = buckets.setdefault(key, [0, 0])
        bucket[0] += dense
        bucket[1] += kept

    del model
    return {f"{tower}:{depth}": kept / dense
            for (tower, depth), (dense, kept) in buckets.items() if dense}


def figure_allocation(data: dict, path: str) -> None:
    panels = list(data.items())
    fig, axes = plt.subplots(1, len(panels), figsize=(6.6, 2.5), sharey=True)
    if len(panels) == 1:
        axes = [axes]

    for ax, (title, entry) in zip(axes, panels):
        for tower, colour, marker in (("vision", VISION, "o"),
                                      ("language", LANGUAGE, "s")):
            points = sorted(
                (int(k.split(":")[1]), v) for k, v in entry["blocks"].items()
                if k.startswith(tower + ":") and int(k.split(":")[1]) >= 0
            )
            if points:
                xs, ys = zip(*points)
                ax.plot(xs, ys, marker=marker, markersize=2.5, linewidth=1.1,
                        color=colour, label=tower.capitalize() + " tower")
        projector = [v for k, v in entry["blocks"].items()
                     if k.startswith("projector:")]
        if projector:
            ax.axhline(sum(projector) / len(projector), color=PROJECTOR,
                       linestyle=":", linewidth=1.4, label="Projector")
        ax.axhline(entry["target"], color="black", linestyle="--",
                   linewidth=0.8, label="Uniform budget")
        ax.set_title(title)
        ax.set_xlabel("Block index within tower")
        # Data occupies the upper range; the floor is kept well below the
        # smallest value so the compression depth stays legible.
        ax.set_ylim(0.25, 1.05)

    axes[0].set_ylabel("Retained parameters")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.13),
               ncol=4, frameon=False)
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def figure_curve(points: dict, path: str) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for method, style in (("uniform", dict(color=UNIFORM, marker="s",
                                           linestyle="--", label="Uniform")),
                          ("ours", dict(color=OURS, marker="o",
                                        linestyle="-", label="Sensitivity ILP"))):
        series = sorted(points[method])
        if series:
            xs, ys = zip(*series)
            ax.plot(xs, ys, markersize=4, linewidth=1.3, **style)
    if points.get("fp16") is not None:
        ax.axhline(points["fp16"], color="black", linestyle=":", linewidth=1.0)
        ax.text(0.62, points["fp16"] + 0.4, "uncompressed", fontsize=7)
    ax.set_xlabel("Whole-model parameter retention")
    ax.set_ylabel("ScienceQA-IMG accuracy (%)")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="out/runs")
    parser.add_argument("--out", default="results/figures")
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    spec = [("LLaVA-1.5-7B", "llava-hf/llava-1.5-7b-hf",
             os.path.join(args.runs, "tower_lems-flat-0.8"), 0.8),
            ("Qwen2-VL-7B", "Qwen/Qwen2-VL-7B-Instruct",
             os.path.join(args.runs, "qwen-flat-0.8"), 0.8)]
    allocation = {}
    for title, model_id, run_dir, target in spec:
        if os.path.exists(os.path.join(run_dir, "rank_dict.json")):
            allocation[title] = {"blocks": per_block_ratios(model_id, run_dir),
                                 "target": target}
    if allocation:
        with open(os.path.join(args.out, "allocation_by_depth.json"), "w") as f:
            json.dump(allocation, f, indent=2)
        figure_allocation(allocation,
                          os.path.join(args.out, "allocation_by_depth.pdf"))

    curve = {"uniform": [], "ours": [], "fp16": None}
    for name in sorted(os.listdir(args.runs)):
        path = os.path.join(args.runs, name, "result.json")
        if not os.path.exists(path):
            continue
        record = json.load(open(path))
        if (record.get("benchmark", "scienceqa") != "scienceqa"
                or "llava-1.5-7b" not in record.get("model", "")
                or record["eval"]["n"] < 1000
                # The answer-free calibration ablation is a separate
                # comparison; including it would put two points per budget on
                # the operating curve.
                or record.get("calib_includes_answer") is False):
            continue
        accuracy = record["eval"]["accuracy"] * 100
        retention = record.get("param_ratio_actual", 1.0)
        if record.get("search") is None:
            curve["fp16"] = accuracy
        elif record.get("search") == "uniform":
            curve["uniform"].append((retention, accuracy))
        elif record.get("bias_mode") == "flat":
            curve["ours"].append((retention, accuracy))
    if curve["uniform"] and curve["ours"]:
        with open(os.path.join(args.out, "accuracy_vs_retention.json"), "w") as f:
            json.dump(curve, f, indent=2)
        figure_curve(curve, os.path.join(args.out, "accuracy_vs_retention.pdf"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
