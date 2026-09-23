#!/usr/bin/env bash
set -euo pipefail

# Path roots default to this file's location so a QAOPD checkout stays self-contained.
QAOPD_ROOT=${QAOPD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
ROOT=${ROOT:-$QAOPD_ROOT}
VERL_ROOT=${VERL_ROOT:-$QAOPD_ROOT/third_party/verl}

VARIANT=${MATH_VARIANT:-${1:-k1_teacher17_steps30}}
RUN_STAMP=${RUN_STAMP:-$(date -u +%Y%m%d_%H%M%S)}
if [[ -z "${MATH_VARIANT:-}" && -n "${1:-}" ]]; then
    shift
fi

export STUDENT_MODEL=${STUDENT_MODEL:-${MODELS_DIR:-$QAOPD_ROOT/models}/w188_step9000}
export QAT_CONFIG=${QAT_CONFIG:-$QAOPD_ROOT/configs/opd/edgerazor_w1_88a8_qwen3.yaml}
export QAT_MODE=${QAT_MODE:-w1.88a8}
export EDGERAZOR_QUANT_MODE=${EDGERAZOR_QUANT_MODE:-w1_88a8kv8_embint4_qwen3}

export TOTAL_GPUS_PER_NODE=2
export STUDENT_NGPUS=1
export TEACHER_WORLD_SIZE=1
export TRAIN_BATCH_SIZE=4
export PPO_MINI_BATCH_SIZE=4
export MAX_PROMPT_LENGTH=1664
export MAX_RESPONSE_LENGTH=512
export PPO_MAX_TOKEN_LEN_PER_GPU=8192
export ROLLOUT_N=4
export ROLLOUT_GPU_MEM_UTIL=0.30
export ROLLOUT_MAX_NUM_SEQS=256
export TEACHER_GPU_MEM_UTIL=0.50
export LR_WARMUP_RATIO=0.0
export VAL_BEFORE_TRAIN=False
export TEST_FREQ=100000
export LOG_VAL_GENERATIONS=0
export MAX_ACTOR_CKPTS_TO_KEEP=20
export USE_TASK_REWARDS=True
export REQUIRE_HASH_REWARD=True
export CORRECT_WITHOUT_HASH_SCORE=0.0
export HASH_FORMAT_SCORE=0.0
export PROJECT_NAME=opd_qad_w188_math_search

case "$VARIANT" in
    k1_teacher17_steps30)
        export TEACHER_MODEL=$ROOT/models/Qwen3-1.7B
        export DISTILLATION_LOSS_MODE=k1
        export USE_POLICY_GRADIENT=True
        export DISTILLATION_TOPK=32
        export DISTILLATION_LOSS_COEF=1.0
        export ACTOR_LR=3e-6
        export TRAINING_STEPS=30
        export SAVE_FREQ=5
        ;;
    k1_teacher17_lr2e6_steps40)
        export TEACHER_MODEL=$ROOT/models/Qwen3-1.7B
        export DISTILLATION_LOSS_MODE=k1
        export USE_POLICY_GRADIENT=True
        export DISTILLATION_TOPK=32
        export DISTILLATION_LOSS_COEF=1.0
        export ACTOR_LR=2e-6
        export TRAINING_STEPS=40
        export SAVE_FREQ=5
        ;;
    fkl64_teacher17_n8_steps30)
        export TEACHER_MODEL=$ROOT/models/Qwen3-1.7B
        export DISTILLATION_LOSS_MODE=forward_kl_topk
        export USE_POLICY_GRADIENT=False
        export DISTILLATION_TOPK=64
        export DISTILLATION_LOSS_COEF=1.0
        export ACTOR_LR=3e-6
        export TRAINING_STEPS=30
        export SAVE_FREQ=5
        export TRAIN_BATCH_SIZE=2
        export PPO_MINI_BATCH_SIZE=2
        export ROLLOUT_N=8
        ;;
    k1_teacher8_steps20)
        export TEACHER_MODEL=$ROOT/models/Qwen3-8B
        export DISTILLATION_LOSS_MODE=k1
        export USE_POLICY_GRADIENT=True
        export DISTILLATION_TOPK=32
        export DISTILLATION_LOSS_COEF=1.0
        export ACTOR_LR=3e-6
        export TRAINING_STEPS=20
        export SAVE_FREQ=5
        ;;
    *)
        echo "Unknown MATH_VARIANT: $VARIANT" >&2
        echo "Supported variants:" >&2
        echo "  k1_teacher17_steps30" >&2
        echo "  k1_teacher17_lr2e6_steps40" >&2
        echo "  fkl64_teacher17_n8_steps30" >&2
        echo "  k1_teacher8_steps20" >&2
        exit 2
        ;;
esac

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-w188a8kv16_step9000_math_${VARIANT}_${RUN_STAMP}}
export OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/experiments/opd_qad/math_search_w188/$EXPERIMENT_NAME}

echo "W1.88A8KV16 math OPD search configuration"
echo "  variant=$VARIANT"
echo "  experiment_name=$EXPERIMENT_NAME"
echo "  output_dir=$OUTPUT_DIR"
echo "  student_model=$STUDENT_MODEL"
echo "  teacher_model=$TEACHER_MODEL"
echo "  qat_config=$QAT_CONFIG"
echo "  loss_mode=$DISTILLATION_LOSS_MODE"
echo "  use_policy_gradient=$USE_POLICY_GRADIENT"
echo "  topk=$DISTILLATION_TOPK"
echo "  actor_lr=$ACTOR_LR"
echo "  training_steps=$TRAINING_STEPS"
echo "  train_batch_size=$TRAIN_BATCH_SIZE"
echo "  rollout_n=$ROLLOUT_N"

mkdir -p "$OUTPUT_DIR"
cat > "$OUTPUT_DIR/search_manifest.json" <<EOF
{
  "variant": "$VARIANT",
  "experiment_name": "$EXPERIMENT_NAME",
  "student_model": "$STUDENT_MODEL",
  "student_model_sha256": "624ad6bc483e13029e5ed974ede77ac80e3709c11e7e579bc5d05a7de598f88d",
  "teacher_model": "$TEACHER_MODEL",
  "qat_config": "$QAT_CONFIG",
  "training_kv_cache": "bf16",
  "distillation_loss_mode": "$DISTILLATION_LOSS_MODE",
  "use_policy_gradient": $([[ "$USE_POLICY_GRADIENT" == "True" ]] && echo true || echo false),
  "use_task_rewards": true,
  "distillation_topk": $DISTILLATION_TOPK,
  "distillation_loss_coef": $DISTILLATION_LOSS_COEF,
  "actor_lr": $ACTOR_LR,
  "training_steps": $TRAINING_STEPS,
  "save_freq": $SAVE_FREQ,
  "train_batch_size": $TRAIN_BATCH_SIZE,
  "ppo_mini_batch_size": $PPO_MINI_BATCH_SIZE,
  "rollout_n": $ROLLOUT_N,
  "max_prompt_length": $MAX_PROMPT_LENGTH,
  "max_response_length": $MAX_RESPONSE_LENGTH,
  "student_gpus": $STUDENT_NGPUS,
  "teacher_gpus": $TEACHER_WORLD_SIZE,
  "reward_function_path": "$QAOPD_ROOT/opd/rewards/gsm8k_boxed_reward.py",
  "reward_function_name": "compute_score",
  "evaluation_target": "W1.88A8KV16 GSM8K 5-shot batch16 one-process"
}
EOF

bash "$QAOPD_ROOT/opd/launch/run_eval_aligned_grouped_opd_qad_w279a8_qwen3_06b.sh" \
    "$@"
