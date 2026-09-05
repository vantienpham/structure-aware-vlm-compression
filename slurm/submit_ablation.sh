#!/usr/bin/env bash
# Submit the Phase-2 ablation ladder at one compression ratio.
#
#   bash slurm/submit_ablation.sh 0.8 [extra args passed to run_compress]
#
# The four rungs isolate one mechanism each:
#
#   uniform            no sensitivity, no depth bias
#   tower_lems flat    sensitivity-driven ILP, all depth multipliers 1.0
#                      -- what lems degenerates to on a VLM today
#   tower_lems tower   + per-tower depth decay
#   tower_lems coupled + cross-tower coupling  (the full method)
#
# Ordering matters for cost, not correctness. The KFAC factorization cache is
# written by whichever run gets there first and is ratio- and search-agnostic;
# the layer-sensitivity cache is written by the first tower_lems run and reused
# by the rest. Submitting all four at once would have three jobs recompute the
# same sensitivities in parallel, so the later rungs are chained behind the
# first with --dependency=afterok. That also keeps the footprint well under the
# cluster's 12-concurrent-job cap (see cluster.local.md).

set -euo pipefail
cd "$(dirname "$0")/.."

RATIO="${1:?usage: $0 <ratio> [extra run_compress args...]}"
shift || true

PARTITIONS="gpu40G,gpu80G,prismgpup,gpul40s,gpuh100p,gpuh200p"
TIME="0-08:00:00"
# HOST RAM, not GPU. Three things sit in CPU memory at once: the model, the
# full state-dict backup svd_core takes before factorizing, and the cached
# per-layer factorizations (~30 GB for 364 layers of a 7B model, since a
# rank-eq factorization is about the size of the weight it replaces). The
# 88 GB that 8 CPUs get by default is not enough -- a run OOM-killed at unit
# 51 of 56. Nodes carry ~1.5 TB, so this is a comfortable margin.
MEM="320G"
COMMON=(--ratio "$RATIO" --calib-samples 32 "$@")
TAG="${RATIO//./}"

submit() {  # submit <name> <dependency-or-empty> <args...>
  local name="$1"; shift
  local dep="$1"; shift
  local dep_flag=()
  [[ -n "$dep" ]] && dep_flag=(--dependency="afterok:$dep")
  sbatch --parsable --job-name="$name" -p "$PARTITIONS" --time="$TIME" \
    --mem="$MEM" "${dep_flag[@]}" slurm/run.slurm -m vlm_lems.run_compress "$@" \
    | tail -1
}

uniform_id=$(submit "abl-uniform-$TAG" "" \
  "${COMMON[@]}" --search uniform --run-dir "out/runs/uniform-$RATIO")
echo "uniform            : $uniform_id"

# Chained behind uniform so the factorization cache exists, then computes and
# caches the layer sensitivities the other two rungs reuse.
flat_id=$(submit "abl-flat-$TAG" "$uniform_id" \
  "${COMMON[@]}" --search tower_lems --bias-mode flat \
  --run-dir "out/runs/tower_lems-flat-$RATIO")
echo "tower_lems flat    : $flat_id  (after $uniform_id)"

tower_id=$(submit "abl-tower-$TAG" "$flat_id" \
  "${COMMON[@]}" --search tower_lems --bias-mode tower \
  --run-dir "out/runs/tower_lems-tower-$RATIO")
echo "tower_lems tower   : $tower_id  (after $flat_id)"

coupled_id=$(submit "abl-coupled-$TAG" "$flat_id" \
  "${COMMON[@]}" --search tower_lems --bias-mode coupled \
  --run-dir "out/runs/tower_lems-coupled-$RATIO")
echo "tower_lems coupled : $coupled_id  (after $flat_id)"

echo
echo "watch:   squeue -u \$USER"
echo "collect: uv run --no-sync python scripts/collect_results.py"
