#!/usr/bin/env bash
set -euo pipefail

# Path roots default to this file's location so a QAOPD checkout stays self-contained.
QAOPD_ROOT=${QAOPD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
ROOT=${ROOT:-$QAOPD_ROOT}
VERL_ROOT=${VERL_ROOT:-$QAOPD_ROOT/third_party/verl}

TRAIN_DIR=${TRAIN_DIR:-${1:-}}
STEPS=${STEPS:-${2:-}}

if [[ -z "$TRAIN_DIR" || -z "$STEPS" ]]; then
    echo "Usage: TRAIN_DIR=<run-dir> STEPS='5 10 15 20' bash $0" >&2
    exit 2
fi
if [[ ! -d "$TRAIN_DIR" ]]; then
    echo "Training directory does not exist: $TRAIN_DIR" >&2
    exit 2
fi

RUN_NAME=$(basename "$TRAIN_DIR")
EVAL_ROOT=${EVAL_ROOT:-$TRAIN_DIR/evaluation_w279a8kv16}
BASELINE_ROOT=${BASELINE_ROOT:-$ROOT/experiments/opd_qad/evaluation/warmstart_w279a8kv16_baseline}
ANALYZER=$QAOPD_ROOT/opd/tools/analyze_opd_eval_results.py
EVALUATOR=$QAOPD_ROOT/opd/launch/evaluate_opd_qad_w279a8.sh
EVAL_PY=${EVAL_PY:-python}

if [[ ! -d "$BASELINE_ROOT" ]]; then
    echo "Matched baseline evaluation is missing: $BASELINE_ROOT" >&2
    exit 3
fi

mkdir -p "$EVAL_ROOT"
ray stop --force >/dev/null 2>&1 || true

run_specs=()
for step in $STEPS; do
    checkpoint=$TRAIN_DIR/checkpoints/global_step_$step/actor/huggingface
    # Single-file or sharded: everything verl saves above 0.6B is a shard set
    # plus an index, so testing for model.safetensors alone rejects every
    # checkpoint this pipeline produces.
    if [[ ! -f "$checkpoint/model.safetensors" \
       && ! -f "$checkpoint/model.safetensors.index.json" ]]; then
        echo "Checkpoint is missing: $checkpoint" >&2
        exit 3
    fi

    output=$EVAL_ROOT/step_$step
    echo "Evaluating $RUN_NAME step $step under matched W2.79A8KV16 protocol"
    env \
        CUDA_VISIBLE_DEVICES=0 \
        NUM_PROCESSES=1 \
        RUN_GSM8K=1 \
        RUN_FULL=0 \
        RUN_CHAT=0 \
        RUN_HUMANEVAL=0 \
        RUN_IFEVAL=0 \
        GSM_BATCH=16 \
        OUTPUT_ROOT=$output \
        MODEL_LABEL=w279a8kv16 \
        bash "$EVALUATOR" "$checkpoint" "${RUN_NAME}_step_${step}"
    run_specs+=(--run "step-$step=$output")
done

"$EVAL_PY" "$ANALYZER" \
    --baseline "warm-start=$BASELINE_ROOT" \
    "${run_specs[@]}" \
    --output "$EVAL_ROOT/comparison.md"

echo "Matched comparison: $EVAL_ROOT/comparison.md"
