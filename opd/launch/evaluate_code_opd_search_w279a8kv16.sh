#!/usr/bin/env bash
set -euo pipefail

# Path roots default to this file's location so a QAOPD checkout stays self-contained.
QAOPD_ROOT=${QAOPD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
ROOT=${ROOT:-$QAOPD_ROOT}
VERL_ROOT=${VERL_ROOT:-$QAOPD_ROOT/third_party/verl}

TRAIN_DIR=${TRAIN_DIR:-${1:-}}
STEPS=${STEPS:-${2:-"5 10 15 20 25 30"}}
MAX_PARALLEL_EVALS=${MAX_PARALLEL_EVALS:-4}
BASELINE_ROOT=${BASELINE_ROOT:-$ROOT/experiments/opd_qad/evaluation/warmstart_w279a8kv16_humaneval_matched}

if [[ -z "$TRAIN_DIR" || ! -d "$TRAIN_DIR" ]]; then
    echo "Usage: TRAIN_DIR=<run-dir> bash $0" >&2
    exit 2
fi
if (( MAX_PARALLEL_EVALS < 1 )); then
    echo "MAX_PARALLEL_EVALS must be positive" >&2
    exit 2
fi
if [[ ! -d "$BASELINE_ROOT" ]]; then
    echo "Matched warm-start baseline does not exist: $BASELINE_ROOT" >&2
    exit 3
fi

RUN_NAME=$(basename "$TRAIN_DIR")
EVAL_ROOT=${EVAL_ROOT:-$TRAIN_DIR/evaluation_w279a8kv16}
LOG_DIR=${LOG_DIR:-$ROOT/experiments/opd_qad/cloud_logs}
EVALUATOR=$QAOPD_ROOT/opd/launch/evaluate_opd_qad_w279a8.sh
ANALYZER=$QAOPD_ROOT/opd/tools/analyze_opd_eval_results.py
PY=${PY:-python}

mkdir -p "$EVAL_ROOT" "$LOG_DIR"
read -r -a step_list <<<"$STEPS"

eval_pids=()
eval_steps=()
eval_failed=0
slot=0

wait_for_batch() {
    local index
    for index in "${!eval_pids[@]}"; do
        if ! wait "${eval_pids[$index]}"; then
            echo "HumanEval failed at step ${eval_steps[$index]}" >&2
            eval_failed=1
        fi
    done
    eval_pids=()
    eval_steps=()
    slot=0
}

for step in "${step_list[@]}"; do
    checkpoint=$TRAIN_DIR/checkpoints/global_step_$step/actor/huggingface
    # Single-file or sharded: everything verl saves above 0.6B is a shard set
    # plus an index, so testing for model.safetensors alone rejects every
    # checkpoint this pipeline produces.
    if [[ ! -f "$checkpoint/model.safetensors" \
       && ! -f "$checkpoint/model.safetensors.index.json" ]]; then
        echo "Checkpoint is missing: $checkpoint" >&2
        exit 3
    fi

    gpu=$slot
    step_root=$EVAL_ROOT/step_$step
    step_log=$LOG_DIR/${RUN_NAME}_humaneval_step_${step}.log
    metrics_cache=$step_root/hf_metrics_cache
    mkdir -p "$metrics_cache"
    echo "Evaluating $RUN_NAME step $step on GPU $gpu"
    (
        set -o pipefail
        env \
            CUDA_VISIBLE_DEVICES=$gpu \
            HF_METRICS_CACHE="$metrics_cache" \
            NUM_PROCESSES=1 \
            HUMANEVAL_BATCH=16 \
            RUN_GSM8K=0 \
            RUN_FULL=0 \
            RUN_CHAT=0 \
            RUN_HUMANEVAL=1 \
            RUN_IFEVAL=0 \
            OUTPUT_ROOT="$step_root" \
            bash "$EVALUATOR" "$checkpoint" "${RUN_NAME}_step_${step}" \
            2>&1 | tee -a "$step_log"
    ) &
    eval_pids+=("$!")
    eval_steps+=("$step")
    ((slot += 1))

    if (( slot == MAX_PARALLEL_EVALS )); then
        wait_for_batch
    fi
done

if (( ${#eval_pids[@]} > 0 )); then
    wait_for_batch
fi
if (( eval_failed != 0 )); then
    exit 4
fi

analysis_args=(
    --baseline "warm=$BASELINE_ROOT"
    --output "$TRAIN_DIR/humaneval_comparison.md"
)
for step in "${step_list[@]}"; do
    analysis_args+=(--run "step-$step=$EVAL_ROOT/step_$step")
done
"$PY" "$ANALYZER" "${analysis_args[@]}"

echo "HumanEval comparison: $TRAIN_DIR/humaneval_comparison.md"
