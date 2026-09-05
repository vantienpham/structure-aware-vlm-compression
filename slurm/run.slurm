#!/bin/bash
# Generic passthrough runner.
#
#   sbatch --job-name=NAME -p <PARTITIONS> --time=0-08:00:00 slurm/run.slurm \
#     scripts/<script>.py <args...> --run-dir=out/runs/NAME
#
# Everything after the script name is handed to the interpreter untouched, and
# the full command is echoed into the log so a job records what produced it.
# Resource flags are set on the sbatch command line, not here, so one file
# serves every experiment; anything passed to sbatch overrides the #SBATCH
# defaults below.
#
# PREREQUISITE: logs/ must already exist on the remote tree. Slurm opens
# --output below *before* this script runs, so a missing directory kills the job
# with no log to explain it. slurm/setup_remote.sh creates it.
#
# Pass ALL usable partitions as one comma-separated list so Slurm places the job
# wherever frees first -- but keep the list memory-feasible end to end, or the
# job lands somewhere it cannot run and dies after queueing.
#
# See cluster.local.md for the partition list, the per-partition memory, the job
# cap and the environment pin.

#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-08:00:00

set -euo pipefail

# $SLURM_SUBMIT_DIR, not a hardcoded path: the job must run from wherever it was
# submitted, or relative paths in configs resolve against the wrong root.
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs out/runs

# Compute nodes have no outbound internet; the login node does. Without this, a
# cache miss spends its timeout trying to reach huggingface.co and then reports
# a connection error, which reads like a network blip rather than a policy.
# Offline mode fails immediately and names the missing entry.
# Prefetch on the login node first: scripts/prefetch.py
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

# Python block-buffers stdout when it is a file rather than a terminal, so a
# job's log can sit empty for many minutes and look hung while it is in fact
# working. Progress bars hid this before because tqdm writes to stderr.
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

echo "job       : ${SLURM_JOB_NAME:-?} (${SLURM_JOB_ID:-?})"
echo "node      : $(hostname)"
echo "partition : ${SLURM_JOB_PARTITION:-?}"
echo "submitted : $SLURM_SUBMIT_DIR"
echo "started   : $(date -Is)"
# The remote tree is rsync'd without .git/, so `git rev-parse` finds nothing
# here. slurm/sync.sh writes .git_commit at transfer time so a run can still
# record the commit it came from; without it, cluster runs record no provenance.
if [[ -f .git_commit ]]; then
  echo "git       : $(grep '^commit=' .git_commit | cut -d= -f2- | cut -c1-12)" \
       "$(grep -q '^dirty=true' .git_commit && echo '(DIRTY at sync time)' || echo '')"
else
  echo "git       : no .git_commit stamp -- provenance unrecorded (use slurm/sync.sh push)"
fi
echo "command   : uv run --no-sync $*"
echo

# Record what the job actually got. When a run behaves differently from an
# identical one, the driver version on the assigned node is the usual reason --
# this cluster is not uniform across partitions.
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
echo

# --no-sync is not optional. A plain `uv run` re-resolves against uv.lock and
# reinstalls the wheel setup_remote.sh deliberately replaced, breaking this job
# and every later one on the same venv. See cluster.local.md, "Environment".
uv run --no-sync "$@"

echo
echo "finished  : $(date -Is)"
