#!/usr/bin/env bash
# Scan a training run's checkpoints on GSM8K + MATH-500 to pick a phase optimum.
#
# docs/RECIPE_3PHASE.md requires selecting each phase by scanning rather than
# reusing another run's step count ("the optimum is early" has held five times).
# This walks the steps, waiting for each checkpoint to appear so it can run
# alongside the training job instead of after it.
#
# AMC is deliberately excluded: avg@16 costs ~6.7 GPU-h per checkpoint
#, which is not affordable per step. Scan on GSM8K +
# MATH-500, then run scripts/eval/amc23_avg16_parallel.sh on the finalists only.
#
# Usage:
#   RUN=<run dir> STEPS="10 20 30" TAG=w188p1 GPU=1 BITWIDTH=w1.88 \
#     bash scripts/eval/scan_math_ckpts.sh
# WAIT_MIN caps how long to wait for a checkpoint that has not been written yet.
set -uo pipefail
# Brace group so an edit mid-run cannot corrupt an in-flight job; see the note in
# scripts/eval/amc23_avg16_parallel.sh. This one runs for
# hours alongside training, so it is the most exposed of the lot.
{
QAOPD_ROOT="${QAOPD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RUN=${RUN:?}; STEPS=${STEPS:?}; TAG=${TAG:?}
GPU=${GPU:-0}; WAIT_MIN=${WAIT_MIN:-0}

# A checkpoint is one file on the small students and a shard set plus an index
# on the larger ones, so testing for model.safetensors alone silently skips
# every step of a sharded run.
have_weights() {
    [[ -f "$1/model.safetensors" || -f "$1/model.safetensors.index.json" ]]
}

for STEP in $STEPS; do
    CKPT="$RUN/checkpoints/global_step_$STEP/actor/huggingface"
    waited=0
    while ! have_weights "$CKPT" && (( waited < WAIT_MIN )); do
        sleep 60; waited=$((waited+1))
    done
    if ! have_weights "$CKPT"; then
        echo "SCAN skip step$STEP: no checkpoint after ${waited}m"
        continue
    fi
    # verl writes the safetensors before the tokenizer files; evaluating in that
    # window fails on a missing tokenizer rather than on anything meaningful.
    sleep 20
    echo "SCAN $TAG step$STEP on GPU $GPU"
    GPU=$GPU RUN_AMC=0 TAG="${TAG}_s${STEP}" RUN="$RUN" STEP="$STEP" \
        bash "$QAOPD_ROOT/scripts/eval/eval_unified.sh"
done
echo "SCAN_DONE $TAG"
}
