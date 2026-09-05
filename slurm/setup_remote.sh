#!/usr/bin/env bash
# Build the cluster environment from scratch. Idempotent -- safe to re-run.
#
#   ssh <REMOTE_USER>@<REMOTE_HOST> 'cd <REMOTE_DIR> && bash slurm/setup_remote.sh'
#
# Run this after the first rsync, and again whenever a plain `uv run` (without
# --no-sync) has been executed on the cluster, since that undoes the pin below.
#
# TEMPLATE: the CONFIGURE block is pre-filled with the values VERIFIED on the
# cluster described in cluster.local.md. On a different cluster, re-derive them
# -- do not carry them over on faith. Record whatever you choose in
# cluster.local.md so a later session can see what this environment is supposed
# to be without reading this script.

set -euo pipefail
cd "$(dirname "$0")/.."

# --- CONFIGURE ---------------------------------------------------------------
# The build to force after the lockfile install, and where to get it. Leave both
# empty to skip step 3 entirely (correct when the cluster has one driver
# generation, or when the lockfile already resolves something that runs
# everywhere).
#
# Pick the value from the OLDEST architecture you intend to schedule on, not the
# newest -- a wheel built for newer GPUs fails on older nodes, so jobs silently
# queue only for the newest partitions, or crash after queueing.
#
# Find the right value by checking the oldest driver you intend to use:
#   sinfo -o '%P %N'
#   srun -p <oldest-partition> nvidia-smi --query-gpu=driver_version --format=csv
#
# Why THIS value on THIS cluster (measured 2026-08-11, full argument in
# cluster.local.md): the P5000 and V100 nodes run driver 560.35.05, whose CUDA
# ceiling is 12.6, while the A100 nodes run 590.44.01. An unconstrained `uv sync`
# resolves a +cu130 torch, which fails on the 560 nodes twice over -- the driver
# is below CUDA 13's minimum of 580, and CUDA 13 ships no sm_61/sm_70 cubins at
# all. It succeeds on A100 and newer, so the breakage looks like bad nodes rather
# than a bad wheel. cu124 over cu126: both clear the ceiling, but 12.6 wants
# driver >=560.28.03 against a measured 560.35.05 -- 0.07 of margin. cu124 needs
# only >=525 and still covers sm_60/61/70 through sm_90, so one build serves
# every partition.
#
# VERIFY, do not assume -- step 8 below tells you how.
# torchvision is pinned alongside torch because some VLM processors (Qwen2-VL's
# video processor, for one) import it at load time, and an unpinned install
# would drag in a torch matching *its* build and undo the CUDA pin.
PINNED_BUILD="torch==2.6.0+cu124 torchvision==0.21.0+cu124"
PINNED_INDEX="https://download.pytorch.org/whl/cu124"

# Optional extra `uv sync` group, for the dev tools the self-test needs.
# Without pytest the environment builds fine and then cannot verify itself,
# which is the worst of both worlds: a green build and no evidence it works.
SYNC_EXTRA="dev"         # set empty to skip

# Imports that scripts use but pyproject.toml does not declare. Standalone
# argparse programs are the usual source. Empty is the correct state; add here
# rather than pip-installing ad hoc, or the next rebuild loses it.
EXTRA_PACKAGES=()        # e.g. (psutil torchinfo nvidia-ml-py3 pillow)

# Directories that are gitignored and cluster-only, so they never arrive by
# rsync. `logs/` in particular MUST exist before the first sbatch.
MAKE_DIRS=(logs out/runs out/archive results/tables)
# -----------------------------------------------------------------------------

echo "=== 1. preflight ==="
command -v uv >/dev/null || {
  echo "error: uv not on PATH. Install it, or load the module that provides it." >&2
  echo "       (on the cluster in cluster.local.md: ~/.local/bin/uv)" >&2
  exit 1
}
[[ -f pyproject.toml ]] || {
  echo "error: no pyproject.toml in $(pwd). Did the rsync complete?" >&2
  exit 1
}
uv --version
# The system python is often ancient (3.6 on this cluster). uv fetches its own
# managed interpreter for `requires-python`, so this is informational only.
python3 --version 2>/dev/null || true

echo
echo "=== 2. uv sync ==="
if [[ -n "$SYNC_EXTRA" ]]; then
  uv sync --extra "$SYNC_EXTRA"
else
  uv sync
fi

echo
echo "=== 3. pinned build ==="
# NOTE: `uv pip install` writes to the environment directly and does NOT consult
# the lockfile, so it needs no --no-sync flag -- and REJECTS one. Passing it here
# aborts the script under `set -e` with "unexpected argument '--no-sync' found",
# exactly and only when PINNED_BUILD is set.
if [[ -n "$PINNED_BUILD" ]]; then
  if [[ -n "$PINNED_INDEX" ]]; then
    uv pip install $PINNED_BUILD --index-url "$PINNED_INDEX"
  else
    uv pip install $PINNED_BUILD
  fi
else
  echo "no PINNED_BUILD set -- skipping the override (see CONFIGURE block)"
fi

echo
echo "=== 4. extra packages ==="
if (( ${#EXTRA_PACKAGES[@]} )); then
  uv pip install "${EXTRA_PACKAGES[@]}"
else
  echo "none declared"
fi

echo
echo "=== 5. directories ==="
# Slurm opens --output=logs/%x-%j.out BEFORE the job script runs, so a missing
# logs/ kills the job with no log to explain it.
mkdir -p "${MAKE_DIRS[@]}"
printf '%s\n' "${MAKE_DIRS[@]}" | paste -sd' ' -

echo
echo "=== 6. environment ==="
uv run --no-sync python - <<'PY'
import sys
print(f"python   : {sys.version.split()[0]}")
try:
    import torch
    print(f"torch    : {torch.__version__}")
    print(f"cuda     : {torch.version.cuda}")
    print(f"arch list: {torch.cuda.get_arch_list()}")
    print(f"available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu      : {torch.cuda.get_device_name(0)} x{torch.cuda.device_count()}")
except ImportError:
    print("torch    : NOT INSTALLED")
PY

echo
echo "=== 7. self-test ==="
# Cheap and CPU-only; catches a broken install before a GPU job burns queue time.
if [[ -d tests ]]; then
  uv run --no-sync python -m pytest tests/ -q -m "not slow and not gpu" 2>&1 | tail -5
else
  echo "no tests/ directory -- skipping"
fi

echo
echo "=== 8. NOT DONE YET ==="
cat <<'EOF'
This ran on the LOGIN node. It has not proven the build runs on a GPU, and
`torch.cuda.is_available()` above is not that proof -- it returns True on a wheel
that carries no cubin for the device, and the failure only surfaces at kernel
launch. Verify on the OLDEST architecture you intend to schedule on:
EOF
# Emitted outside the quoted heredoc so $PWD expands into a line you can paste.
echo
echo "  srun -p <oldest-partition> --gres=gpu:1 --cpus-per-task=2 --time=0-00:05:00 \\"
echo "    bash -c 'cd $PWD && uv run --no-sync python scripts/verify_pin.py'"
cat <<'EOF'

Compute nodes have no outbound internet on this cluster. Warm the caches here,
on the login node, before submitting anything:

  uv run --no-sync python scripts/prefetch.py --model <repo> --dataset <name>

From here on, always launch with 'uv run --no-sync'.
EOF
