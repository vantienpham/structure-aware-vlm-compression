# Structure-Aware Global Rank Allocation for Low-Rank Compression of Dual-Tower Vision-Language Models

Code, experiment configurations and results for the paper of the same name.

**Preprint:** [https://ssrn.com/abstract=7417408](https://ssrn.com/abstract=7417408)

<p align="center" width="100%">
    <img src="assets\pham2026structure.png" width="100%" height="100%">
</p>

Low-rank compression by truncated SVD becomes competitive when the rank budget
is allocated across layers by measured sensitivity. Existing formulations
assume a single homogeneous transformer stack, which vision-language models do
not satisfy: they comprise a vision encoder and a language decoder of unequal
geometry, joined by a projector. This repository implements a rank allocator
defined over that dual-tower structure, and reproduces every number in the
paper from the committed run records.

## What the paper reports

Relative to uniform allocation at matched realized parameter budgets:

| benchmark / model | improvement |
|---|---|
| ScienceQA-IMG, LLaVA-1.5-7B | +1.9 to +6.1 points |
| SEED-Bench-IMG, LLaVA-1.5-7B | +4.4 to +13.4 points |
| ScienceQA-IMG, LLaVA-1.5-13B | +4.1 to +8.4 points |
| ScienceQA-IMG, Qwen2-VL-7B | +4.3 points at 0.6; no detected difference at 0.8 |

Figures are for the allocator without the tower-indexed depth bias, which is
reported separately as exploratory: its measured benefit is small, inconsistent
in sign, and does not survive correction for multiplicity.

Two structural limitations of single-stack rank allocation on dual-tower models
are characterised in the paper and implemented around in `vlm_lems/towers.py`:
identifying transformer blocks by first match recovers only one tower, and
vision blocks beyond the feature-selection layer are executed without
influencing the output.

## Relationship to LEMS

This work extends **KFAC-SVD + LEMS** (Thoma et al., *Advancing SVD-based LLM
Compression via Layer-Wise Error Model Search*, ICML 2026), whose official
implementation is at [`lems-svd/lems`](https://github.com/lems-svd/lems).

**That code is not redistributed here.** It carries no LICENSE file, so this
repository depends on it the way any unlicensed research codebase is safe to
depend on: by importing it locally, unmodified. Nothing under `vlm_lems/` is a
copy of it — every extension is a subclass. Clone it as a sibling directory:

```bash
git clone https://github.com/lems-svd/lems.git lems
```

`vlm_lems/lems_path.py` puts that directory on `sys.path` and will tell you if
it is missing.

## Install

```bash
git clone https://github.com/vantienpham/structure-aware-vlm-compression.git
cd structure-aware-vlm-compression
git clone https://github.com/lems-svd/lems.git lems     # required, see above
uv sync
```

The GPU build of PyTorch is pinned deliberately; see the comments in
`pyproject.toml`. `torch.cuda.is_available()` returning `True` does not prove
the wheel runs on your device, so verify with real kernel launches:

```bash
uv run --no-sync python scripts/verify_pin.py
```

## Reproducing the results

Compute nodes on many clusters have no outbound internet, so fetch weights and
data first:

```bash
uv run --no-sync python scripts/prefetch.py --model llava-hf/llava-1.5-7b-hf
```

One compression + evaluation run:

```bash
uv run --no-sync python -m vlm_lems.run_compress \
  --model llava-hf/llava-1.5-7b-hf \
  --ratio 0.8 --calib-samples 32 \
  --search tower_lems --bias-mode flat \
  --run-dir out/runs/tower_lems-flat-0.8
```

Key arguments:

| argument | meaning |
|---|---|
| `--search` | `uniform`, `tower_lems`, or a competing rule (`vlm_asvd`, `vlm_mrcs`, `vlm_atp`, `vlm_svdllmv2`) |
| `--bias-mode` | `flat` (no depth bias), `tower`, or `coupled` |
| `--ratio` | target whole-model parameter retention |
| `--benchmark` | `scienceqa` or `seedbench` |
| `--no-answer-calib` | strictly label-free calibration (prompt only) |
| `--solver` | ILP backend; HiGHS is the default and needs no licence |

The full ablation at one budget, as submitted for the paper:

```bash
bash slurm/submit_ablation.sh 0.8
```

Then regenerate every table and figure:

```bash
uv run --no-sync python scripts/collect_results.py --runs out/runs
uv run --no-sync python scripts/param_audit.py    --runs out/runs
uv run --no-sync python scripts/make_tables.py    --out results/tables
uv run --no-sync python scripts/make_figures.py   --out results/figures
uv run --no-sync python scripts/analyze.py        --runs out/runs
```

`scripts/analyze.py` reports exact McNemar tests and paired confidence
intervals. Every method is scored on the same examples, so comparing runs as
independent samples discards most of the available power.

## Experimental configuration

- **Calibration**: 32 examples for factor estimation, a disjoint 32 for the
  search reference, one seed per configuration. Rank search and bias fitting
  are label-free (KL against the uncompressed model's own distribution). Factor
  estimation is teacher-forced by default, so the reference answer is among the
  targets; `--no-answer-calib` removes it, and the paper reports both.
- **Evaluation**: constrained multiple-choice decoding over the answer letters,
  one forward pass per example. ScienceQA-IMG uses all 2,017 image-bearing test
  examples; SEED-Bench-IMG uses a deterministic 3,000-example subsample.
- **Budgets**: retention ratios 0.9 / 0.8 / 0.7 / 0.6, matched between methods
  on *realized* whole-model retention, not on the requested target.

## Results in this repository

`results/` holds the artifacts the paper is built from.

| path | contents |
|---|---|
| `results/tables/all_results.json` | every run: accuracy, realized retention, per-tower ratios, search metadata |
| `results/tables/param_audit.json` | parameter accounting per run, including parameter-weighted per-tower retention |
| `results/tables/*.tex` | the paper's tables, generated |
| `results/figures/*.json` | the data behind each figure |
| `results/figures/*.pdf` | the figures, generated |

`scripts/check_manuscript.py` verifies that quoted numbers reconcile with these
files. It exists because a table and its prose drifted apart once, and it now
also guards the headline ranges against mixing method variants.

Per-run artifacts (`out/runs/`) are produced on the cluster and are not
committed; the two JSON files above are the summarised record they reduce to.

`all_results.json` is the complete record of every run, including the
label-free calibration ablation and the competing allocation rules; each entry
carries its `search` and `calib_includes_answer` so the canonical subset used
for the paper's main tables can be reconstructed. The tables themselves are
generated from that canonical subset only.

## Repository layout

| path | what |
|---|---|
| `vlm_lems/towers.py` | dual-tower structure discovery: block stacks, tower classification, depth indexing |
| `vlm_lems/search_tower.py` | the allocator: tower-aware bias, the global ILP |
| `vlm_lems/factorization_tower.py` | KFAC-SVD extended to traverse both towers |
| `vlm_lems/search_baselines.py` | competing allocation rules, adapted to dual-tower models |
| `vlm_lems/ilp_highs.py` | HiGHS backend, so no commercial solver is required |
| `vlm_lems/numerics.py` | robust whitening; old GPUs can return `info == 0` with a NaN factor |
| `vlm_lems/evaluater.py` | constrained multiple-choice scoring |
| `scripts/` | prefetch, environment verification, collection, auditing, tables, figures, statistics |
| `slurm/` | job submission for a Slurm cluster |

## Cluster use

`slurm/` targets a Slurm cluster. Copy `cluster.example.md` to
`cluster.local.md`, fill in your site's values, and set the three `CHANGEME`
fields at the top of `slurm/sync.sh`. `cluster.local.md` is gitignored.

Always run through `uv run --no-sync`. A plain `uv run` re-resolves against the
lockfile and can silently reinstall over the pinned CUDA build.

## Citation

If you use this work, please cite the preprint:

```bibtex
@article{pham2026structure,
  title   = {Structure-Aware Global Rank Allocation for Low-Rank Compression of Dual-Tower Vision-Language Models},
  author  = {Pham, Van Tien and Le, Thanh Trung},
  year    = {2026},
  note    = {SSRN preprint},
  url     = {https://ssrn.com/abstract=7417408}
}
```

## License

MIT, see `LICENSE`. This covers the code in this repository only. The
`lems-svd/lems` reference implementation is a separate work under its own
terms, is not redistributed here, and must be obtained from its authors. The
models and benchmarks used (LLaVA-1.5, Qwen2-VL, ScienceQA, SEED-Bench) carry
their own licenses.
