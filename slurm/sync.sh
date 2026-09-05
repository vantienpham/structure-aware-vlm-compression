#!/usr/bin/env bash
# Push code to the cluster, or pull summaries back. Run from the laptop.
#
#   bash slurm/sync.sh push [--dry-run]   local tree -> cluster
#   bash slurm/sync.sh pull [--dry-run]   summaries  -> local
#
# Use this rather than hand-rolling rsync. The excludes are not optional:
# `--delete` makes the remote match the local tree exactly, so without
# `--exclude 'out/'` this command erases every result on the cluster.
#
# Values are duplicated from cluster.local.md rather than parsed out of it, so
# this script works before anything else does.

set -euo pipefail
cd "$(dirname "$0")/.."

# --- CONFIGURE ---------------------------------------------------------------
REMOTE_USER="CHANGEME"                    # <-- your cluster username
REMOTE_HOST="CHANGEME"                    # <-- your cluster login node
REMOTE_DIR="CHANGEME"                     # <-- absolute path on the cluster
# -----------------------------------------------------------------------------

REMOTE="${REMOTE_USER}@${REMOTE_HOST}"

for var in REMOTE_USER REMOTE_HOST REMOTE_DIR; do
  [[ "${!var}" == *CHANGEME* ]] && {
    echo "error: set $var in $0 first (see cluster.example.md)." >&2
    exit 2
  }
done

EXCLUDES=(
  --exclude '.git/'          # the remote tree is not a git checkout (README §2)
  --exclude 'out/'           # results live only on the cluster -- never delete
  --exclude 'logs/'
  # results/ is PRODUCED on the cluster and pulled back, so it belongs with out/
  # and logs/ rather than with the code. Without this, `push --delete` deletes it
  # on the remote because the laptop copy is empty -- precisely the "--delete
  # without --exclude erases your results" hazard.
  --exclude 'results/'
  --exclude '.venv/'
  --exclude '__pycache__/'
  --exclude '.pytest_cache/'
  --exclude '.ruff_cache/'
  --exclude '*.egg-info/'
)

usage() { echo "usage: $0 {push|pull} [--dry-run]" >&2; exit 2; }
[[ $# -ge 1 ]] || usage
MODE="$1"; shift
DRY=()
[[ "${1:-}" == "--dry-run" ]] && DRY=(-n --itemize-changes)

case "$MODE" in
  push)
    # Stamp the commit so runs on the cluster can record their provenance.
    # `.git/` does not travel, so `git rev-parse` there returns nothing; job
    # scripts and run loggers read this file instead.
    {
      echo "commit=$(git rev-parse HEAD)"
      echo "branch=$(git rev-parse --abbrev-ref HEAD)"
      echo "dirty=$([[ -n "$(git status --porcelain)" ]] && echo true || echo false)"
      echo "synced_at=$(date -Is)"
      echo "synced_from=$(hostname):$(pwd)"
    } > .git_commit

    if [[ -n "$(git status --porcelain)" ]]; then
      echo "WARNING: local tree is dirty; the cluster will run uncommitted code." >&2
      echo "         .git_commit records dirty=true so the run log will say so." >&2
    fi

    rsync -az --delete "${DRY[@]}" "${EXCLUDES[@]}" ./ "${REMOTE}:${REMOTE_DIR}/"
    [[ ${#DRY[@]} -eq 0 ]] && echo "pushed $(git rev-parse --short HEAD) -> ${REMOTE}:${REMOTE_DIR}"
    ;;

  pull)
    # Summaries only. <OUT_DIR> is typically many GB of checkpoints and dumps;
    # pull specific run directories by hand when you need the artifacts.
    #
    # Both destinations must exist first -- rsync will not create a missing local
    # root for a filtered transfer, it just moves nothing. Errors are NOT
    # suppressed here: an earlier version sent them to /dev/null and reported
    # "pulled summaries" after transferring nothing at all.
    mkdir -p results/tables out/runs

    rsync -az "${DRY[@]}" "${REMOTE}:${REMOTE_DIR}/results/tables/" results/tables/
    rsync -az "${DRY[@]}" --include '*/' \
      --include 'config.json' --include 'env.json' --include 'metrics.json' \
      --exclude '*' \
      "${REMOTE}:${REMOTE_DIR}/out/runs/" out/runs/

    n=$(find out/runs -name 'metrics.json' 2>/dev/null | wc -l)
    echo "pulled summaries for ${n} run(s); artifacts stay on the cluster"
    ;;

  *) usage ;;
esac
