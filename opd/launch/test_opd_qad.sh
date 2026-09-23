#!/usr/bin/env bash
set -euo pipefail

# Path roots. Defaults derive from this file's location so a QAOPD checkout
# resolves everything inside the repo; override to point elsewhere.
QAOPD_ROOT=${QAOPD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
ROOT=${ROOT:-$QAOPD_ROOT}
VERL_ROOT=${VERL_ROOT:-$QAOPD_ROOT/third_party/verl}
EDGE_ROOT=${EDGE_ROOT:-$ROOT/third_party/edgerazor}
CONDA_SH=${CONDA_SH:-}   # optional; the release uses venvs, not conda
CONDA_ENV=${CONDA_ENV:-${CONDA_PREFIX:-}}
WANDB_API_KEY_FILE=${WANDB_API_KEY_FILE:-$ROOT/verl/.secrets/wandb_api_key}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-$QAOPD_ROOT/opd/launch/run_opd_qad_w279a8_qwen3_06b.sh}

# The project installs into venvs/ (docs/INSTALL.md), so conda is optional. This
# used to be an unconditional `source <conda>/etc/profile.d/conda.sh`, which under `set -e`
# aborted the whole run on any host without that exact path -- and the callers in
# scripts/opd/ and scripts/qad/ do not use `set -e`, so they went on to print their *_DONE marker
# and the run looked successful while nothing had trained.
# CONDA_ENV defaults to $CONDA_PREFIX, which is often just the ambient env of the
# container and says nothing about where the training stack lives. So conda is
# only used when its profile script is actually present; otherwise we assume the
# venv is already on PATH and let the import check below be the real guard.
if [[ -f "$CONDA_SH" ]]; then
    # shellcheck disable=SC1090
    source "$CONDA_SH"
    [[ -n "$CONDA_ENV" ]] && conda activate "$CONDA_ENV"
fi
# Whatever python is about to run must carry the training stack; fail here rather
# than 40 minutes in.
python -c "import verl, edgerazor, vllm" || {
    echo "python on PATH ($(command -v python)) lacks verl/edgerazor/vllm." >&2
    echo "Activate the training env: export PATH=<repo>/venvs/qaopd-train/bin:\$PATH" >&2
    exit 1; }

export PYTHONPATH="$EDGE_ROOT/src:$VERL_ROOT:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-$ROOT/.cache/huggingface}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_PROJECT=${WANDB_PROJECT:-opd_qad_gsm8k}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

if [[ "$WANDB_MODE" != "disabled" && "$WANDB_MODE" != "offline" ]]; then
    set +x
    if [[ -z "${WANDB_API_KEY:-}" && -f "$WANDB_API_KEY_FILE" ]]; then
        export WANDB_API_KEY
        WANDB_API_KEY=$(tr -d '[:space:]' < "$WANDB_API_KEY_FILE")
    fi
    set -x
fi

# The credential half of this check used to sit in the same `or` chain as the
# mode test, and Python evaluates left to right: with no ~/.netrc, netrc.netrc()
# raised FileNotFoundError before `mode != 'online'` could short-circuit it, so
# every offline run died here instead of skipping the check it does not need.
if [[ "$WANDB_MODE" == "online" ]]; then
    python -c "import edgerazor, netrc, os, transfer_queue, verl; assert os.getenv('WANDB_API_KEY') or netrc.netrc().authenticators('api.wandb.ai')"
else
    python -c "import edgerazor, transfer_queue, verl"
fi
# (removed global `ray stop --force`: it kills every other local Ray job)

bash "$TRAIN_SCRIPT" "$@"
