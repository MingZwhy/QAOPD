#!/usr/bin/env bash
# MBPP (448) and HumanEval (164) on a deployment export: a released -QAOPD model, or the
# model_* export eval_unified.sh writes for a run. Same protocol as b1_test448.sh and
# b1_humaneval.sh, which take latent checkpoints instead.
#   CKPT=<export dir> [GPU=0] [BATCH_SIZE=32] [OUT=<dir>] bash scripts/eval/code_export.sh
set -uo pipefail
QAOPD_ROOT="${QAOPD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export QAOPD_ROOT
export DATA_ROOT="${DATA_ROOT:-$QAOPD_ROOT/data}"
# Evaluation env, as in eval_unified.sh: EVAL_ENV pins a prefix, else python on PATH.
E=${EVAL_ENV:-}
PY=${PY:-${E:+$E/bin/}python}
CKPT=${CKPT:?set CKPT to a deployment export directory}
# The export's modeling_edgerazor.py imports edgerazor.
export PYTHONPATH="$QAOPD_ROOT/third_party/edgerazor/src:${PYTHONPATH:-}"
CUDA_VISIBLE_DEVICES=${GPU:-0} "$PY" "$QAOPD_ROOT/scripts/eval/code_export.py" \
    --model "$CKPT" --batch_size "${BATCH_SIZE:-32}" ${OUT:+--out "$OUT"}
