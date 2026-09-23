#!/usr/bin/env bash
set -euo pipefail

# Path roots. Defaults keep the original workstation layout working, but a
# checkout of QAOPD sets these so nothing resolves outside the repo.
QAOPD_ROOT=${QAOPD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
ROOT=${ROOT:-$QAOPD_ROOT}
VERL_ROOT=${VERL_ROOT:-$QAOPD_ROOT/third_party/verl}
DATA_ROOT=${DATA_ROOT:-$QAOPD_ROOT/data}
MODELS_DIR=${MODELS_DIR:-$ROOT/models}
RUNS_DIR=${RUNS_DIR:-$ROOT/experiments/opd_qad/code_search}
VARIANT=${CODE_VARIANT:-${1:-mbpp_k1_t17_task_lr3e6_s30}}
# Consume the positional variant regardless of whether CODE_VARIANT is also set.
# run_unified_code.sh always passes the variant positionally *and* exports
# CODE_VARIANT, so keying the shift on CODE_VARIANT being empty left the variant
# name in "$@", which is forwarded verbatim to hydra at the bottom of this script
# and dies as "Error parsing override ... missing EQUAL at '<EOF>'". Everything
# after the variant is a hydra override and those always contain '=', so that is
# the safe discriminator for what to eat.
if [[ $# -gt 0 && "$1" != *=* ]]; then
    shift
fi
RUN_STAMP=${RUN_STAMP:-$(date -u +%Y%m%d_%H%M%S)}

export STUDENT_MODEL=${STUDENT_MODEL:-${MODELS_DIR:-$QAOPD_ROOT/models}/w279_step10000}
export TOTAL_GPUS_PER_NODE=${TOTAL_GPUS_PER_NODE:-4}
export STUDENT_NGPUS=${STUDENT_NGPUS:-2}
export TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-1}
export TEACHER_TP_SIZE=${TEACHER_TP_SIZE:-1}
# Overridable. These were plain assignments sitting above the variant block, so a variant
# that carefully wrote ${TRAIN_BATCH_SIZE:-8} still could not be overridden from outside:
# by the time the branch ran, the value was already 4 and the :- default never applied. The
# delivery arm was launched asking for 8 and silently trained at 4, which is 0.49 epochs
# and its training reward stayed flat at ~0.45 for 135 steps as a result.
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-4}
# Overridable, defaults unchanged so a pure-code run reproduces byte-for-byte. Code
# prompts are short (p99 = 310 tokens) and 512 clips only 6-12% of rollouts here, but
# any variant that mixes math rows into this runner is a different story: the 5-shot
# gsm8k prompt has p50 = 981 and 38.6% of rows exceed 1024, and phase 1 gave math a
# 1536-token response budget rather than 512. Left at the defaults, such an arm trains
# math on truncated prompts and truncated answers without saying so anywhere.
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}
export ROLLOUT_N=${ROLLOUT_N:-4}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.30}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-256}
export TEACHER_GPU_MEM_UTIL=${TEACHER_GPU_MEM_UTIL:-0.45}
export LR_WARMUP_RATIO=0.0
export DISTILLATION_LOSS_MODE=k1
export DISTILLATION_TOPK=32
export DISTILLATION_LOSS_COEF=1.0
export USE_POLICY_GRADIENT=True
export TRAINING_STEPS=30
export SAVE_FREQ=5
export TEST_FREQ=5
export VAL_BEFORE_TRAIN=True
export LOG_VAL_GENERATIONS=8
export MAX_ACTOR_CKPTS_TO_KEEP=6
export CHECKPOINT_SAVE_CONTENTS="['hf_model']"
export RANDOM_SEED=${RANDOM_SEED:-42}
export DATA_SEED=${DATA_SEED:-20260718}
export PROJECT_NAME=opd_qad_code_search_w279
export EVALUATION_TARGET="W2.79A8KV16 HumanEval instruct, batch 16, one process"

effective_mini_batch=$((PPO_MINI_BATCH_SIZE * ROLLOUT_N))
if (( effective_mini_batch % STUDENT_NGPUS != 0 )); then
    echo "PPO_MINI_BATCH_SIZE * ROLLOUT_N must divide evenly across student GPUs" >&2
    echo "Got $PPO_MINI_BATCH_SIZE * $ROLLOUT_N = $effective_mini_batch over $STUDENT_NGPUS ranks" >&2
    exit 2
fi

case "$VARIANT" in
    mbpp_official_k1_t17_strict_lr3e6_s30)
        export DATA_VARIANT=mbpp_official
        # DATA_DIR_OVERRIDE swaps in a different code pool while keeping this
        # variant's reward and PG switches. A substituted pool must keep the 448
        # test rows out of training.
        export DATA_DIR=${DATA_DIR_OVERRIDE:-$DATA_ROOT/mbpp_official_protocol_v1}
        export TEACHER_MODEL=$MODELS_DIR/Qwen3-1.7B
        export USE_TASK_REWARDS=True
        export REQUIRE_ALL_TESTS=True
        export ACTOR_LR=3e-6
        export EVALUATION_TARGET="MBPP official validation selection and held-out official test"
        ;;
    kodcode_k1_t17_strict_lr3e6_s30)
        # The wider code pool: KodCode on top of the MBPP official train split,
        # built by opd/data_prep/prepare_kodcode_profile_curriculum.py. Same
        # rewards and learning rate as above, screened against HumanEval and
        # the 448 MBPP test rows.
        export DATA_VARIANT=kodcode
        export DATA_DIR=${DATA_DIR_OVERRIDE:-$DATA_ROOT/kodcode_v1}
        export TEACHER_MODEL=$MODELS_DIR/Qwen3-1.7B
        export USE_TASK_REWARDS=True
        export REQUIRE_ALL_TESTS=True
        export ACTOR_LR=3e-6
        export EVALUATION_TARGET="MBPP official validation selection and held-out official test"
        ;;
    *)
        echo "unknown variant: $VARIANT" >&2
        echo "  mbpp_official_k1_t17_strict_lr3e6_s30 -- MBPP official train" >&2
        echo "  kodcode_k1_t17_strict_lr3e6_s30       -- KodCode, the wider pool" >&2
        exit 1
        ;;
esac
if [[ ! -d "$DATA_DIR" ]]; then
    echo "no code pool at $DATA_DIR -- build it first, see data/README.md" >&2
    exit 1
fi

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-w279a8kv16_code_${VARIANT}_${RUN_STAMP}}
export OUTPUT_DIR=${OUTPUT_DIR:-$RUNS_DIR/$EXPERIMENT_NAME}

echo "W2.79A8KV16 code OPD search configuration"
echo "  variant=$VARIANT"
echo "  experiment_name=$EXPERIMENT_NAME"
echo "  output_dir=$OUTPUT_DIR"
echo "  student_model=$STUDENT_MODEL"
echo "  data_variant=$DATA_VARIANT"
echo "  data_dir=${DATA_DIR:-resolved_by_data_variant}"
echo "  teacher_model=$TEACHER_MODEL"
echo "  chat_template=${CHAT_TEMPLATE:-$QAOPD_ROOT/opd/humaneval_opd_chat_template.jinja}"
echo "  loss_mode=$DISTILLATION_LOSS_MODE"
echo "  distillation_topk=$DISTILLATION_TOPK"
echo "  use_task_rewards=$USE_TASK_REWARDS"
echo "  require_all_tests=${REQUIRE_ALL_TESTS:-default}"
echo "  distillation_loss_coef=$DISTILLATION_LOSS_COEF"
echo "  task_policy_loss_coef=${TASK_POLICY_LOSS_COEF:-1.0}"
echo "  actor_lr=$ACTOR_LR"
echo "  training_steps=$TRAINING_STEPS"
echo "  random_seed=$RANDOM_SEED"
echo "  data_seed=$DATA_SEED"
echo "  train_batch_size=$TRAIN_BATCH_SIZE"
echo "  rollout_n=$ROLLOUT_N"
echo "  student_gpus=$STUDENT_NGPUS"
echo "  teacher_gpus=$TEACHER_WORLD_SIZE"
echo "  teacher_tensor_parallel_size=$TEACHER_TP_SIZE"

mkdir -p "$OUTPUT_DIR"
cat > "$OUTPUT_DIR/search_manifest.json" <<EOF
{
  "variant": "$VARIANT",
  "experiment_name": "$EXPERIMENT_NAME",
  "student_model": "$STUDENT_MODEL",
  "data_variant": "$DATA_VARIANT",
  "data_dir": "${DATA_DIR:-resolved_by_data_variant}",
  "teacher_model": "$TEACHER_MODEL",
  "chat_template": "${CHAT_TEMPLATE:-$QAOPD_ROOT/opd/humaneval_opd_chat_template.jinja}",
  "distillation_loss_mode": "$DISTILLATION_LOSS_MODE",
  "distillation_topk": $DISTILLATION_TOPK,
  "distillation_enabled": $([[ "${DISTILLATION_ENABLED:-True}" == "True" ]] && echo true || echo false),
  "use_policy_gradient": $([[ "$USE_POLICY_GRADIENT" == "True" ]] && echo true || echo false),
  "use_task_rewards": $([[ "$USE_TASK_REWARDS" == "True" ]] && echo true || echo false),
  "require_all_tests": "${REQUIRE_ALL_TESTS:-default}",
  "distillation_loss_coef": $DISTILLATION_LOSS_COEF,
  "task_policy_loss_coef": ${TASK_POLICY_LOSS_COEF:-1.0},
  "positive_advantage_only": $([[ "${POSITIVE_ADVANTAGE_ONLY:-False}" == "True" ]] && echo true || echo false),
  "positive_task_advantage_only": $([[ "${POSITIVE_TASK_ADVANTAGE_ONLY:-False}" == "True" ]] && echo true || echo false),
  "actor_lr": "$ACTOR_LR",
  "training_steps": $TRAINING_STEPS,
  "random_seed": $RANDOM_SEED,
  "data_seed": $DATA_SEED,
  "save_freq": $SAVE_FREQ,
  "validation_freq": $TEST_FREQ,
  "train_batch_size": $TRAIN_BATCH_SIZE,
  "ppo_mini_batch_size": $PPO_MINI_BATCH_SIZE,
  "rollout_n": $ROLLOUT_N,
  "max_prompt_length": $MAX_PROMPT_LENGTH,
  "max_response_length": $MAX_RESPONSE_LENGTH,
  "student_gpus": $STUDENT_NGPUS,
  "teacher_gpus": $TEACHER_WORLD_SIZE,
  "teacher_tensor_parallel_size": $TEACHER_TP_SIZE,
  "evaluation_target": "$EVALUATION_TARGET"
}
EOF

bash "$QAOPD_ROOT/opd/launch/run_official_w279_code_opd_qwen3_06b.sh" \
    "$@"
