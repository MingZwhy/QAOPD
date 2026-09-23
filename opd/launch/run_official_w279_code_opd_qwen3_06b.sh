#!/usr/bin/env bash
set -euo pipefail

# Path roots default to this file's location so a QAOPD checkout stays self-contained.
QAOPD_ROOT=${QAOPD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
ROOT=${ROOT:-$QAOPD_ROOT}
VERL_ROOT=${VERL_ROOT:-$QAOPD_ROOT/third_party/verl}

# The caller (run_code_opd_search_w279a8kv16.sh) normally exports DATA_DIR. The
# fallback covers a direct invocation and names the pool that ships: the MBPP
# official train split, with the 448 test rows held out for evaluation.
DATA_VARIANT=${DATA_VARIANT:-mbpp_official}
CHAT_TEMPLATE=${CHAT_TEMPLATE:-$QAOPD_ROOT/opd/humaneval_opd_chat_template.jinja}
ROLLOUT_STOP=${ROLLOUT_STOP:-'["```"]'}
ROLLOUT_BAD_WORDS=${ROLLOUT_BAD_WORDS:-null}

if [[ -z "${DATA_DIR:-}" ]]; then
    case "$DATA_VARIANT" in
        mbpp_official)
            DATA_DIR=${DATA_ROOT:-$QAOPD_ROOT/data}/mbpp_official_protocol_v1
            ;;
        kodcode)
            # Optional extension, built by opd/data_prep/prepare_kodcode_profile_curriculum.py
            DATA_DIR=${DATA_ROOT:-$QAOPD_ROOT/data}/kodcode_v1
            ;;
        *)
            echo "Unknown DATA_VARIANT: $DATA_VARIANT (want mbpp_official or kodcode)" >&2
            exit 2
            ;;
    esac
fi

for path in "$DATA_DIR/train.parquet" "$DATA_DIR/validation.parquet" "$CHAT_TEMPLATE"; do
    if [[ ! -f "$path" ]]; then
        echo "Required file does not exist: $path" >&2
        exit 3
    fi
done

export TRAIN_FILE=$DATA_DIR/train.parquet
export VAL_FILE=$DATA_DIR/validation.parquet
if [[ -z "${STUDENT_MODEL:-}" ]]; then
    echo "STUDENT_MODEL must point to a LATENT checkpoint (docs/CHECKPOINTS.md)." >&2
    echo "A deployment export has the low-bit weights baked in and cannot be" >&2
    echo "trained from. Normally this is the phase-1 output; to start from the" >&2
    echo "QAD anchor instead, use:" >&2
    echo "  ${MODELS_DIR:-$QAOPD_ROOT/models}/w279_step10000" >&2
    exit 4
fi
export STUDENT_MODEL
export TEACHER_MODEL=${TEACHER_MODEL:-$ROOT/models/Qwen3-0.6B}
export TOTAL_GPUS_PER_NODE=${TOTAL_GPUS_PER_NODE:-4}
export STUDENT_NGPUS=${STUDENT_NGPUS:-3}
export TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-1}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-6}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-6}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}
export ROLLOUT_N=${ROLLOUT_N:-8}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.30}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-256}
export TEACHER_GPU_MEM_UTIL=${TEACHER_GPU_MEM_UTIL:-0.45}
export TRAINING_STEPS=${TRAINING_STEPS:-10}
export ACTOR_LR=${ACTOR_LR:-2e-7}
export LR_WARMUP_RATIO=${LR_WARMUP_RATIO:-0.0}
export DISTILLATION_LOSS_MODE=${DISTILLATION_LOSS_MODE:-forward_kl_topk}
export DISTILLATION_TOPK=${DISTILLATION_TOPK:-64}
export USE_POLICY_GRADIENT=${USE_POLICY_GRADIENT:-False}
export USE_TASK_REWARDS=${USE_TASK_REWARDS:-True}
export DISTILLATION_LOSS_COEF=${DISTILLATION_LOSS_COEF:-1.0}
export DISTILLATION_ENABLED=${DISTILLATION_ENABLED:-True}
export QUANTIZE_REFERENCE=${QUANTIZE_REFERENCE:-True}
export SAVE_FREQ=${SAVE_FREQ:-$TRAINING_STEPS}
export TEST_FREQ=${TEST_FREQ:--1}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
export MAX_ACTOR_CKPTS_TO_KEEP=${MAX_ACTOR_CKPTS_TO_KEEP:-2}
# Overridable so a rehearsal mix can swap in a router that also scores the math rows;
# mbpp_execution_reward silently returns 0.0 for any non-code data_source.
export REWARD_FUNCTION_PATH=${REWARD_FUNCTION_PATH:-$QAOPD_ROOT/opd/rewards/mbpp_execution_reward.py}
export REWARD_FUNCTION_NAME=${REWARD_FUNCTION_NAME:-compute_score}
export GSM_REWARD_KWARGS=False
export PROJECT_NAME=${PROJECT_NAME:-opd_qad_code_official_w279}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-official_w279_self06_${DATA_VARIANT}_${DISTILLATION_LOSS_MODE}_s${TRAINING_STEPS}_lr${ACTOR_LR}}

RANDOM_SEED=${RANDOM_SEED:-42}
DATA_SEED=${DATA_SEED:-20260717}
POSITIVE_ADVANTAGE_ONLY=${POSITIVE_ADVANTAGE_ONLY:-False}
POSITIVE_TASK_ADVANTAGE_ONLY=${POSITIVE_TASK_ADVANTAGE_ONLY:-False}
TASK_POLICY_LOSS_COEF=${TASK_POLICY_LOSS_COEF:-1.0}
CHECKPOINT_SAVE_CONTENTS=${CHECKPOINT_SAVE_CONTENTS:-"['hf_model']"}
reward_overrides=()
if [[ -n "${REQUIRE_ALL_TESTS:-}" ]]; then
    reward_overrides+=(
        "+reward.custom_reward_function.reward_kwargs.require_all_tests=$REQUIRE_ALL_TESTS"
    )
fi

HUMANEVAL_CHAT_TEMPLATE=$(<"$CHAT_TEMPLATE")
export HUMANEVAL_CHAT_TEMPLATE

# Third place this was needed: the math training script and the MBPP/HumanEval
# eval launcher each had their own copy, and the code *training* launcher still
# died on `PackageNotFoundError: flash_attn`. All four paths converge on
# run_opd_qad_w279a8_qwen3_06b.sh, which is where this really belongs.
source "$ROOT/scripts/lib/attn.sh"; attn_pick

bash "$QAOPD_ROOT/opd/launch/run_opd_qad_w279a8_qwen3_06b.sh" \
    'actor_rollout_ref.model.custom_chat_template=${oc.env:HUMANEVAL_CHAT_TEMPLATE}' \
    "+actor_rollout_ref.model.override_config.attn_implementation=$ATTN_IMPL" \
    "+actor_rollout_ref.rollout.stop=$ROLLOUT_STOP" \
    "actor_rollout_ref.rollout.bad_words=$ROLLOUT_BAD_WORDS" \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    data.shuffle=True \
    data.seed="$DATA_SEED" \
    actor_rollout_ref.actor.data_loader_seed="$RANDOM_SEED" \
    actor_rollout_ref.actor.fsdp_config.seed="$RANDOM_SEED" \
    actor_rollout_ref.ref.fsdp_config.seed="$RANDOM_SEED" \
    actor_rollout_ref.rollout.seed="$RANDOM_SEED" \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.optim.weight_decay=0.0 \
    +distillation.distillation_loss.positive_advantage_only="$POSITIVE_ADVANTAGE_ONLY" \
    +distillation.distillation_loss.positive_task_advantage_only="$POSITIVE_TASK_ADVANTAGE_ONLY" \
    +distillation.distillation_loss.task_policy_loss_coef="$TASK_POLICY_LOSS_COEF" \
    distillation.enabled="$DISTILLATION_ENABLED" \
    actor_rollout_ref.actor.checkpoint.save_contents="$CHECKPOINT_SAVE_CONTENTS" \
    "${reward_overrides[@]}" \
    "$@"
