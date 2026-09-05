# cluster.local.md — example

The shape of `cluster.local.md`. Copy it and fill in the real values.

**Decide once whether that file is committed or ignored**, and say which at the
top of it:

- *Committed* (what this repo does) — correct when the git remote is **private**.
  A fresh clone then arrives able to reach the cluster instead of needing this
  rebuilt by hand. The cost: a login node and a username are in history, so the
  repo can never be made public without purging them first.
- *Gitignored* — correct when the remote is public, shared, or mirrored. Then
  the file never arrives with a clone, and a new session rebuilds it from this
  example plus `~/.ssh/config` and a `sinfo` on the login node.

**Either way, no secrets.** Tokens, keys and passwords never go in a tracked
file, and a file you can accidentally commit counts as tracked. Record *where*
a credential lives and *how to re-create it*, never the value.

`README.md` refers to every value here by placeholder name. Fill this in once
and every command in the README becomes executable as written.

**Record the verified state, not the intended one, and date every verification.**
Drivers, quotas and partition lists change without warning.

---

## Access

| key | value |
|---|---|
| `<REMOTE_USER>` | `jdoe` |
| `<REMOTE_HOST>` | `cluster.example.org` |
| `<REMOTE_DIR>` | `~/my-project` |
| `<LOCAL_REPO>` | `/data/my-project` |
| ssh alias | `cluster` (defined in `~/.ssh/config`, see README §1) |
| key | `~/.ssh/id_ed25519`, already authorized — no password prompt |

```bash
ssh cluster
```

## Slurm

| key | value |
|---|---|
| `<PARTITIONS>` | `gpushort,gpulong,gpua100` |
| account / QOS | *(if the site requires `--account=…`, record it here)* |
| **job limits** | *(`MaxJobs` / `MaxSubmit` from `sacctmgr show assoc user=$USER`)* |
| default wall time | `0-08:00:00` |
| log path | `<REMOTE_DIR>/logs/%x-%j.out` |

Pass the whole comma-separated list to `-p` so Slurm places the job wherever
frees first — but keep the list memory-feasible end to end (README §4).

**Record the per-user job cap.** It is the most common reason jobs pend, it
shows up as `AssocMaxJobsLimit` rather than `Resources`, and unlike contention,
waiting does not clear it — batch campaigns have to self-throttle below it.

### Partitions

One row per partition; a job that lands on 16 GB when it needs 40 dies *after*
queueing.

| partition | GPU | arch | mem/GPU | time limit |
|---|---|---|---|---|
| `<name>` | `<model> x<n>` | `sm_<xx>` | `<GB>` | `<d-hh:mm:ss>` |

Then a line per job size, ready to paste:

```bash
# small models -- the cheap silicon
-p <list>
# ~8B fp16 -- 40 GB and up
-p <list>
```

## Environment

| key | value |
|---|---|
| pinned build | `<e.g. torch==2.x.y+cuXXX>`, or *none — the lockfile is fine* |
| pin verified | *(date, and on which architectures — real kernels, not `is_available()`)* |
| runner invocation | `uv run --no-sync` |
| extra packages | *(imports not declared in `pyproject.toml`)* |
| package manager | *(path/version of `uv`/conda on the remote; the system python is often ancient)* |
| **compute-node internet** | *(yes / no — this changes the whole workflow, see below)* |

### Measured drivers

**Do not assume the cluster is uniform.** This table is the single highest-value
thing in the file.

| partition | GPU | `compute_cap` | driver | max CUDA runtime |
|---|---|---|---|---|
| `<name>` | `<model>` | `<x.y>` | `<version>` | `<x.y>` |

### Why the pin exists, and what re-breaks it

One paragraph. This is the section that saves the most time; see README §3.
State which build, chosen against which *oldest* driver, with what margin — and
name the thing that undoes it (typically a plain `uv run`, a `conda install`, or
a re-lock).

### Offline compute nodes

If compute nodes cannot reach the internet, say so here and record the three
parts: what to prefetch on the login node, which offline env vars the job sets,
and anything beyond HF that downloads at runtime (pip, `torch.hub`, NLTK).

## Data

| key | value |
|---|---|
| `<DATA_ROOT>` | `/shared/datasets/<name>` |
| layout | *(e.g. `train/` ImageFolder-style, `val/` flat + sidecar labels)* |
| model/weight cache | *(path, and how big it is)* |
| credentials | *(where the token lives and how to re-create it — never the value)* |

### Storage

| path | what | persistent? |
|---|---|---|
| home | *(filesystem, size, quota)* | yes |
| scratch | *(path, speed, **purge policy**, and whether it is shared or node-local)* | *(often not)* |

Check what `scratch` actually is before trusting it — a symlink to `/tmp` is
node-local and vanishes with the allocation, which is not what the name implies.

## Results

| key | value |
|---|---|
| `<OUT_DIR>` | `<REMOTE_DIR>/out` — gitignored, cluster-only, never rsync'd wholesale |

| directory | what |
|---|---|
| `<OUT_DIR>/runs/` | main campaign, one directory per run name |
| `<OUT_DIR>/archive/` | superseded runs; check why before reusing |
| `results/tables/` | summaries that come back to the laptop and get committed |

Naming convention for runs — `<config>-seed<N>` or similar — so results
aggregate mechanically.

## Code

| key | value |
|---|---|
| git remote | `<GIT_REMOTE>` |
| visibility | *(**private** / public — this is what licenses committing this file)* |
| default branch | `<BRANCH>` |
| commit policy | *(state it explicitly — e.g. "commit and push by default")* |
