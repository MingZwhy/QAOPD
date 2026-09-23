# Installation

## Requirements

- Linux, NVIDIA GPUs. QAD was run on 8 GPUs; OPD on 2–4.
- Python 3.12 for the training env and Python 3.10 for the evaluation env.
  `bootstrap.sh` uses `python3.10` if it is on `PATH` and otherwise asks conda
  for one; without either it builds the training env and tells you what is
  missing.
- CUDA 12.8 drivers. Both envs install their torch from the cu128 index.
- `git`, and network access for the submodules on first clone.

## Bootstrap

```bash
git clone https://github.com/MingZwhy/QAOPD.git
cd QAOPD
bash setup/bootstrap.sh
```

`bootstrap.sh` checks out the pinned submodules, applies the two third-party
patches, and creates `venvs/qaopd-train` and `venvs/qaopd-eval` from the lock
files. Re-running is safe: each step detects whether it is already done.

Let `bootstrap.sh` fetch the submodules rather than cloning with
`--recursive`. EdgeRazor records a nested gitlink it ships no `.gitmodules`
for, so recursing into it aborts the update and leaves verl off its pinned
commit.

Two environments exist on purpose. The training stack pins vLLM and the
evaluation stack pins lm-eval; their dependency closures conflict, so keep them
separate. QAD is the exception to the naming: it is a DeepSpeed job and runs in
the *evaluation* env, which is where deepspeed and bitsandbytes are pinned.

The lock files are closure freezes rather than solvable specs, so they are
replayed with `--no-deps` and torch is installed separately from the cu128
index. `bootstrap.sh` does both; installing a lock file with a plain
`pip install -r` will not reproduce a working environment.

Set `DO_VENVS=0` to skip environment creation, or `VENV_DIR` to relocate them.

## Weights

Base models come from their upstream sources (Qwen3-0.6B / 1.7B / 4B). Place or
symlink them under `models/`:

```
models/Qwen3-0.6B
models/Qwen3-1.7B
models/Qwen3-4B
```

`MODELS_DIR` overrides the location.

## Data

See [`../data/README.md`](../data/README.md) for the pools and how to build
them. `DATA_ROOT` overrides the location.

## Path conventions

Every entrypoint resolves `QAOPD_ROOT` from its own location, so scripts can be
called from anywhere. These override the defaults:

| Variable | Default |
|---|---|
| `QAOPD_ROOT` | repository root |
| `MODELS_DIR` | `$QAOPD_ROOT/models` |
| `DATA_ROOT` | `$QAOPD_ROOT/data` |
| `RUNS_DIR` | `$QAOPD_ROOT/runs` |
| `TRAIN_ENV`, `EVAL_ENV` | `$QAOPD_ROOT/venvs/qaopd-{train,eval}` |
