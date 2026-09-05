#!/usr/bin/env python3
"""Verify the manuscript's numbers against the generated results.

    python scripts/check_manuscript.py

Every accuracy and delta quoted in the prose must reconcile with
``results/tables/all_results.json`` and with the generated LaTeX tables. Run
this after regenerating tables and before compiling; it exits non-zero on any
disagreement.

It exists because a table and its surrounding text drifted apart once: runs
from the answer-free calibration ablation and from the competing allocation
rules leaked into the main per-model table, which changed the rows and the
deltas while the prose still quoted the original figures. Automated tables are
only auditable if something actually checks them against what is claimed.
"""

from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
TEX = os.path.join(ROOT, "redaction", "manuscript.tex")
TABLES = os.path.join(ROOT, "results", "tables")

#: Accuracies the prose states, keyed by the run that must produce them.
QUOTED_ACCURACY = {
    "baseline-full": 64.80, "tower_lems-flat-0.9": 64.50,
    "uniform-0.9": 61.63, "tower_lems-flat-0.8": 62.47, "uniform-0.8": 60.59,
    "tower_lems-flat-0.7": 60.68, "uniform-0.7": 58.30,
    "tower_lems-flat-0.6": 57.66, "uniform-0.6": 51.51,
    "tower_lems-tower-0.6": 59.20,
    "seed-baseline": 61.80, "seed-tower-0.8": 61.63, "seed-uniform-0.8": 56.70,
    "seed-tower-0.6": 57.50, "seed-uniform-0.6": 43.33,
    "13b-baseline": 70.75, "13b-tower-0.8": 70.55, "13b-uniform-0.8": 66.19,
    "13b-tower-0.6": 68.47, "13b-uniform-0.6": 60.19,
    "qwen-baseline": 81.36, "qwen-uniform-0.8": 80.42, "qwen-flat-0.8": 80.07,
    "qwen-uniform-0.6": 66.24, "qwen-flat-0.6": 70.50,
    # Table 6, the prompt-only calibration ablation. Hand-written in the
    # manuscript rather than generated, and for a while absent from
    # all_results.json, so nothing checked it.
    "noans-uniform-0.8": 59.99, "noans-flat-0.8": 61.82,
    "noans-uniform-0.6": 50.42, "noans-flat-0.6": 56.42,
}

#: Deltas the prose states, as (treatment, control, expected).
QUOTED_DELTA = [
    ("tower_lems-flat-0.9", "uniform-0.9", +2.88),
    ("tower_lems-flat-0.8", "uniform-0.8", +1.88),
    ("tower_lems-flat-0.7", "uniform-0.7", +2.38),
    ("tower_lems-flat-0.6", "uniform-0.6", +6.15),
    ("tower_lems-tower-0.6", "uniform-0.6", +7.68),
    ("seed-tower-0.8", "seed-uniform-0.8", +4.93),
    ("seed-tower-0.6", "seed-uniform-0.6", +14.17),
    ("13b-tower-0.8", "13b-uniform-0.8", +4.36),
    ("13b-tower-0.6", "13b-uniform-0.6", +8.28),
    ("qwen-flat-0.6", "qwen-uniform-0.6", +4.26),
    # The quantity the label-free claim rests on: the gain over uniform
    # allocation must survive removing the answer from calibration.
    ("noans-flat-0.8", "noans-uniform-0.8", +1.83),
    ("noans-flat-0.6", "noans-uniform-0.6", +6.00),
]

#: Per-tower retention figures quoted in the prose. These must be the
#: parameter-weighted values, since that is what the tables report; quoting the
#: unweighted per-layer mean here is the drift this guards against.
QUOTED_TOWER = [
    ("tower_lems-flat-0.9", "language", 0.898),
    ("tower_lems-flat-0.8", "language", 0.794),
    ("tower_lems-flat-0.7", "language", 0.690),
    ("tower_lems-flat-0.6", "language", 0.587),
    ("tower_lems-flat-0.6", "projector", 0.644),
    ("tower_lems-coupled-t60-0.8", "projector", 0.622),
    ("qwen-flat-0.8", "vision", 0.919),
    ("qwen-flat-0.6", "vision", 0.848),
]

#: Headline improvement ranges quoted in the abstract, contributions and
#: conclusion. These must come from a single variant. They previously mixed the
#: flat allocator's minimum with the tower-biased variant's maximum, which
#: attributed to the validated method a figure that required the exploratory
#: one. Each entry is (label, [(uniform, treatment), ...], low, high).
QUOTED_RANGE = [
    ("ScienceQA-IMG", [("uniform-0.9", "tower_lems-flat-0.9"),
                       ("uniform-0.8", "tower_lems-flat-0.8"),
                       ("uniform-0.7", "tower_lems-flat-0.7"),
                       ("uniform-0.6", "tower_lems-flat-0.6")], 1.9, 6.1),
    ("SEED-Bench-IMG", [("seed-uniform-0.8", "seed-flat-0.8"),
                        ("seed-uniform-0.6", "seed-flat-0.6")], 4.4, 13.4),
    ("LLaVA-1.5-13B", [("13b-uniform-0.8", "13b-flat-0.8"),
                       ("13b-uniform-0.6", "13b-flat-0.6")], 4.1, 8.4),
]

#: Rows that must never appear in a generated table.
FORBIDDEN_ROW_PATTERNS = [
    (r"&\s*\?\s*&", "a row whose method is '?': a run with no bias mode "
                    "reached a per-model table"),
]


def unit_ambiguities(text: str) -> list:
    """Bare parameter counts sitting in a series of abbreviated ones.

    "1334M parameters out of 1358M, against 24M ... and 1024 from the
    projector" reads the last term as 1024M, which would exceed the total. The
    rule is deliberately narrow: a bare integer introduced by "and" or
    "against" inside a sentence that already carries two or more M-suffixed
    figures. Counts of things rather than parameters (for example "the 364
    compressible matrices") are not in that shape and are not flagged.
    """
    problems = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if len(re.findall(r"\b\d+(?:\.\d+)?M\b", sentence)) < 2:
            continue
        for bare in re.findall(r"\b(?:and|against)\s+(\d{3,})(?!M)(?![\d.%])",
                               sentence):
            problems.append((bare, " ".join(sentence.split())[:110]))
    return problems


#: Maps a generated table file to the label the manuscript gives it.
TABLE_LABEL = {
    "main_scienceqa.tex": "tab:scienceqa",
    "main_seedbench.tex": "tab:seedbench",
    "main_13b.tex": "tab:13b",
    "main_qwen.tex": "tab:qwen",
    "baselines.tex": "tab:baselines",
    "param_accounting.tex": "tab:accounting",
}


def tower_column_references(tex: str) -> list:
    """Check the sentence claiming per-tower figures names the right tables.

    The claim applies only to tables that actually carry the Vis./Lang./Proj.
    columns. A range reference such as "Tables 1--5" silently swept in two
    tables without them.
    """
    have_columns = {
        label for filename, label in TABLE_LABEL.items()
        if os.path.exists(os.path.join(TABLES, filename))
        and "Vis." in open(os.path.join(TABLES, filename)).read()
    }
    sentence = re.search(r"The per-tower figures in ([^.]*?)\s+are\s*\n?\s*"
                         r"parameter-weighted", tex)
    if sentence is None:
        return ["the sentence naming the parameter-weighted tables was not found"]
    named = set(re.findall(r"\\ref\{(tab:[^}]+)\}", sentence.group(1)))
    if re.search(r"\\ref\{tab:[^}]+\}\s*--\s*\\ref", sentence.group(1)):
        return ["per-tower claim uses a table range, which cannot express "
                "a non-contiguous set; name the tables individually"]
    problems = []
    for missing in sorted(have_columns - named):
        problems.append(f"per-tower claim omits {missing}, which has those columns")
    for extra in sorted(named - have_columns):
        problems.append(f"per-tower claim names {extra}, which has no tower columns")
    return problems


def accuracies() -> dict:
    data = json.load(open(os.path.join(TABLES, "all_results.json")))
    out = {}
    for key, value in data.items():
        out.setdefault(key.rsplit("/", 1)[1], []).append(value)
    return out


def main() -> int:
    failures = []
    runs = accuracies()
    # A quoted value may live in the prose or in a generated table; both are
    # part of the manuscript as compiled.
    # The manuscript source is not part of the public code release, so the
    # checks that read the prose are skipped when it is absent. Everything
    # that reconciles the generated tables against the run records still runs.
    have_tex = os.path.exists(TEX)
    tex = open(TEX).read() if have_tex else ""
    for filename in sorted(os.listdir(TABLES)):
        if filename.endswith(".tex"):
            tex += open(os.path.join(TABLES, filename)).read()

    for name, expected in QUOTED_ACCURACY.items():
        entries = runs.get(name)
        if not entries:
            failures.append(f"{name}: quoted in the paper but absent from results")
            continue
        actual = entries[0]["accuracy"] * 100
        if abs(actual - expected) > 0.006:
            failures.append(f"{name}: paper says {expected:.2f}, results give {actual:.2f}")
        # Whether the value actually appears in the paper can only be checked
        # against the manuscript source. Some quoted values (Table 6, which is
        # hand-written rather than generated) appear nowhere else, so without
        # the source this assertion is not merely weaker but wrong.
        if have_tex and f"{expected:.2f}" not in tex:
            failures.append(
                f"{name}: {expected:.2f} appears in neither the prose nor any table")

    for treatment, control, expected in QUOTED_DELTA:
        if treatment not in runs or control not in runs:
            failures.append(f"{treatment} vs {control}: missing from results")
            continue
        actual = (runs[treatment][0]["accuracy"] - runs[control][0]["accuracy"]) * 100
        if abs(actual - expected) > 0.006:
            failures.append(
                f"{treatment} vs {control}: paper says {expected:+.2f}, "
                f"results give {actual:+.2f}")

    if have_tex:
        prose = re.sub(r"%.*", "", open(TEX).read())
        for bare, sentence in unit_ambiguities(prose):
            failures.append(
                f"unit ambiguity: bare '{bare}' in a clause using M-suffixed "
                f"figures: {sentence}")

        failures.extend(tower_column_references(open(TEX).read()))

    for label, pairs, low, high in QUOTED_RANGE:
        deltas = []
        for uniform, treatment in pairs:
            if uniform not in runs or treatment not in runs:
                failures.append(f"{label} range: {uniform} or {treatment} missing")
                break
            deltas.append((runs[treatment][0]["accuracy"]
                           - runs[uniform][0]["accuracy"]) * 100)
        else:
            for stated, actual, end in ((low, min(deltas), "low"),
                                        (high, max(deltas), "high")):
                if abs(round(actual, 1) - stated) > 0.051:
                    failures.append(
                        f"{label} range {end} end: paper states {stated}, the "
                        f"single-variant value is {actual:.2f}")

    audit_path = os.path.join(TABLES, "param_audit.json")
    audit = json.load(open(audit_path)) if os.path.exists(audit_path) else {}
    for run, tower, expected in QUOTED_TOWER:
        entry = audit.get(run, {}).get("per_tower", {}).get(tower)
        if entry is None:
            failures.append(f"{run}/{tower}: quoted but not audited")
            continue
        actual = entry["parameter_weighted_retention"]
        if abs(actual - expected) > 0.0006:
            failures.append(
                f"{run}/{tower}: paper says {expected:.3f}, "
                f"parameter-weighted audit gives {actual:.3f}")

    for filename in sorted(os.listdir(TABLES)):
        if not filename.endswith(".tex"):
            continue
        body = open(os.path.join(TABLES, filename)).read()
        for pattern, message in FORBIDDEN_ROW_PATTERNS:
            if re.search(pattern, body):
                failures.append(f"{filename}: {message}")
        # A duplicated method label within one budget block means two runs of
        # the same configuration were rendered side by side. The accounting
        # table is keyed by (target, method) with no per-target rules, so a
        # repeated label there is expected rather than a fault.
        if filename == "param_accounting.tex":
            continue
        for block in body.split("\\midrule"):
            labels = re.findall(r"&\s*((?:\\;\+ )?[A-Z][A-Za-z0-9 ()+-]*?)\s*&", block)
            labels = [l.strip() for l in labels if l.strip() not in ("--",)]
            duplicated = {l for l in labels if labels.count(l) > 1}
            if duplicated:
                failures.append(f"{filename}: duplicated rows for {sorted(duplicated)}")

    if failures:
        print("MANUSCRIPT CHECK FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    scope = "" if have_tex else " (prose checks skipped: no manuscript source)"
    print(f"manuscript check passed: {len(QUOTED_ACCURACY)} accuracies, "
          f"{len(QUOTED_DELTA)} deltas, {len(QUOTED_TOWER)} tower ratios, "
          f"and every generated table consistent{scope}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
