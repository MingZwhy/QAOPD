#!/usr/bin/env bash
set -euo pipefail

# Path roots default to this file's location so a QAOPD checkout stays self-contained.
QAOPD_ROOT=${QAOPD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
ROOT=${ROOT:-$QAOPD_ROOT}
VERL_ROOT=${VERL_ROOT:-$QAOPD_ROOT/third_party/verl}

SOURCE_TRAIN_FILE=${SOURCE_TRAIN_FILE:-$ROOT/verl/data/gsm8k/train.parquet}
SOURCE_VAL_FILE=${SOURCE_VAL_FILE:-$ROOT/verl/data/gsm8k/test.parquet}
ALIGNED_TRAIN_FILE=${ALIGNED_TRAIN_FILE:-${DATA_ROOT:-$QAOPD_ROOT/data}/gsm8k_train_eval_aligned_5shot.parquet}
ALIGNED_VAL_FILE=${ALIGNED_VAL_FILE:-${DATA_ROOT:-$QAOPD_ROOT/data}/gsm8k_test_eval_aligned_5shot.parquet}
PREPARE_SCRIPT=$QAOPD_ROOT/opd/data_prep/prepare_gsm8k_eval_aligned.py

prepare_data() {
    local input=$1
    local output=$2
    if [[ ! -f "$output" ]]; then
        "${PY:-python}" "$PREPARE_SCRIPT" \
            --input "$input" \
            --fewshot_source "$SOURCE_TRAIN_FILE" \
            --output "$output"
    fi
}

prepare_data "$SOURCE_TRAIN_FILE" "$ALIGNED_TRAIN_FILE"
prepare_data "$SOURCE_VAL_FILE" "$ALIGNED_VAL_FILE"

export TRAIN_FILE=$ALIGNED_TRAIN_FILE
export VAL_FILE=$ALIGNED_VAL_FILE
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1664}
export ROLLOUT_N=${ROLLOUT_N:-4}
export USE_TASK_REWARDS=True
export REQUIRE_HASH_REWARD=True
export CORRECT_WITHOUT_HASH_SCORE=${CORRECT_WITHOUT_HASH_SCORE:-0.0}
export HASH_FORMAT_SCORE=${HASH_FORMAT_SCORE:-0.0}
export DISTILLATION_LOSS_COEF=${DISTILLATION_LOSS_COEF:-1.0}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-w279a8_eval_aligned_opd_n${ROLLOUT_N}_steps${TRAINING_STEPS:-100}_lr${ACTOR_LR:-3e-6}}

PLAIN_TEXT_CHAT_TEMPLATE="{% for message in messages %}{{ message.content }}{% endfor %}"

bash "$QAOPD_ROOT/opd/launch/run_opd_qad_w279a8_qwen3_06b.sh" \
    "actor_rollout_ref.model.custom_chat_template='$PLAIN_TEXT_CHAT_TEMPLATE'" \
    "+actor_rollout_ref.rollout.stop=['Question:']" \
    "$@"
