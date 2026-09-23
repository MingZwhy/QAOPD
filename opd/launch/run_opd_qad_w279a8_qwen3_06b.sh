#!/usr/bin/env bash
set -euo pipefail

# Path roots. Defaults derive from this file's location so a QAOPD checkout
# resolves everything inside the repo; override to point elsewhere.
QAOPD_ROOT=${QAOPD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
ROOT=${ROOT:-$QAOPD_ROOT}
VERL_ROOT=${VERL_ROOT:-$QAOPD_ROOT/third_party/verl}
EDGE_ROOT=${EDGE_ROOT:-$ROOT/third_party/edgerazor}
CHECKPOINT_VALIDATOR=$QAOPD_ROOT/opd/tools/validate_qat_latent_checkpoint.py

STUDENT_MODEL=${STUDENT_MODEL:-${MODELS_DIR:-$QAOPD_ROOT/models}/w279_step10000}
TEACHER_MODEL=${TEACHER_MODEL:-${MODELS_DIR:-$ROOT/models}/Qwen3-0.6B}
QAT_CONFIG=${QAT_CONFIG:-$QAOPD_ROOT/configs/opd/edgerazor_w2_79a8_qwen3.yaml}
QAT_MODE=${QAT_MODE:-w2.79a8}
EDGERAZOR_QUANT_MODE=${EDGERAZOR_QUANT_MODE:-w2_79a8kv8_embint4_qwen3}
QUANTIZE_REFERENCE=${QUANTIZE_REFERENCE:-True}
TRAIN_FILE=${TRAIN_FILE:-$ROOT/verl/data/gsm8k/train.parquet}
VAL_FILE=${VAL_FILE:-$ROOT/verl/data/gsm8k/test.parquet}

NNODES=${NNODES:-1}
TOTAL_GPUS_PER_NODE=${TOTAL_GPUS_PER_NODE:-4}
STUDENT_NGPUS=${STUDENT_NGPUS:-3}
TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-1}
TEACHER_TP_SIZE=${TEACHER_TP_SIZE:-1}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-24}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-24}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}
TRAINING_STEPS=${TRAINING_STEPS:-100}
ACTOR_LR=${ACTOR_LR:-3e-6}
LR_WARMUP_RATIO=${LR_WARMUP_RATIO:-0.05}

ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.30}
ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-True}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-256}
TEACHER_GPU_MEM_UTIL=${TEACHER_GPU_MEM_UTIL:-0.50}
DISTILLATION_LOSS_MODE=${DISTILLATION_LOSS_MODE:-k1}
USE_POLICY_GRADIENT=${USE_POLICY_GRADIENT:-True}
DISTILLATION_TOPK=${DISTILLATION_TOPK:-32}
ROLLOUT_N=${ROLLOUT_N:-1}
USE_TASK_REWARDS=${USE_TASK_REWARDS:-False}
REQUIRE_HASH_REWARD=${REQUIRE_HASH_REWARD:-False}
CORRECT_WITHOUT_HASH_SCORE=${CORRECT_WITHOUT_HASH_SCORE:-0.0}
HASH_FORMAT_SCORE=${HASH_FORMAT_SCORE:-0.0}
DISTILLATION_LOSS_COEF=${DISTILLATION_LOSS_COEF:-1.0}

REWARD_FUNCTION_PATH=${REWARD_FUNCTION_PATH-$QAOPD_ROOT/opd/rewards/gsm8k_boxed_reward.py}
REWARD_FUNCTION_NAME=${REWARD_FUNCTION_NAME:-compute_score}
REWARD_MANAGER_NAME=${REWARD_MANAGER_NAME:-naive}
GSM_REWARD_KWARGS=${GSM_REWARD_KWARGS:-True}
SANDBOX_FUSION_URL=${SANDBOX_FUSION_URL:-}
SANDBOX_MAX_CONCURRENT=${SANDBOX_MAX_CONCURRENT:-64}
SANDBOX_MEMORY_LIMIT_MB=${SANDBOX_MEMORY_LIMIT_MB:-1024}

PROJECT_NAME=${PROJECT_NAME:-opd_qad_gsm8k}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-w279a8_self_teacher_steps${TRAINING_STEPS}_lr${ACTOR_LR}}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/experiments/opd_qad/$EXPERIMENT_NAME}
SAVE_FREQ=${SAVE_FREQ:-$TRAINING_STEPS}
TEST_FREQ=${TEST_FREQ:-25}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-8}
MAX_ACTOR_CKPTS_TO_KEEP=${MAX_ACTOR_CKPTS_TO_KEEP:-2}
WANDB_LOGGER=${WANDB_LOGGER:-wandb}

if (( STUDENT_NGPUS + TEACHER_WORLD_SIZE > TOTAL_GPUS_PER_NODE )); then
    echo "student and teacher GPU counts exceed TOTAL_GPUS_PER_NODE" >&2
    exit 1
fi
if (( TEACHER_TP_SIZE < 1 || TEACHER_WORLD_SIZE % TEACHER_TP_SIZE != 0 )); then
    echo "TEACHER_TP_SIZE must be positive and divide TEACHER_WORLD_SIZE" >&2
    exit 1
fi
if (( PPO_MINI_BATCH_SIZE > TRAIN_BATCH_SIZE || TRAIN_BATCH_SIZE % PPO_MINI_BATCH_SIZE != 0 )); then
    echo "PPO_MINI_BATCH_SIZE must divide TRAIN_BATCH_SIZE; both are prompt counts before ROLLOUT_N" >&2
    exit 1
fi
for path in "$STUDENT_MODEL" "$TEACHER_MODEL" "$QAT_CONFIG" "$TRAIN_FILE" "$VAL_FILE" "$CHECKPOINT_VALIDATOR"; do
    if [[ ! -e "$path" ]]; then
        echo "Required path does not exist: $path" >&2
        exit 1
    fi
done
if [[ -n "$REWARD_FUNCTION_PATH" && ! -f "$REWARD_FUNCTION_PATH" ]]; then
    echo "Reward function does not exist: $REWARD_FUNCTION_PATH" >&2
    exit 1
fi

python "$CHECKPOINT_VALIDATOR" --model "$STUDENT_MODEL" --purpose "OPD/QAT training"

cd "$VERL_ROOT"
export PYTHONPATH="$EDGE_ROOT/src:$VERL_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=1

max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1))
rollout_max_num_seqs=$ROLLOUT_MAX_NUM_SEQS
if (( rollout_max_num_seqs > max_model_len )); then
    rollout_max_num_seqs=$max_model_len
fi

reward_overrides=(
    "reward.reward_manager.name=$REWARD_MANAGER_NAME"
)
if [[ -n "$REWARD_FUNCTION_PATH" ]]; then
    reward_overrides+=(
        "reward.custom_reward_function.path=$REWARD_FUNCTION_PATH"
        "reward.custom_reward_function.name=$REWARD_FUNCTION_NAME"
    )
else
    reward_overrides+=("reward.custom_reward_function.path=null")
fi
if [[ "${GSM_REWARD_KWARGS,,}" == "true" || "$GSM_REWARD_KWARGS" == "1" ]]; then
    reward_overrides+=(
        "+reward.custom_reward_function.reward_kwargs.require_hash=$REQUIRE_HASH_REWARD"
        "+reward.custom_reward_function.reward_kwargs.correct_without_hash_score=$CORRECT_WITHOUT_HASH_SCORE"
        "+reward.custom_reward_function.reward_kwargs.hash_format_score=$HASH_FORMAT_SCORE"
    )
fi
if [[ -n "$SANDBOX_FUSION_URL" ]]; then
    reward_overrides+=(
        "reward.sandbox_fusion.url=$SANDBOX_FUSION_URL"
        "reward.sandbox_fusion.max_concurrent=$SANDBOX_MAX_CONCURRENT"
        "reward.sandbox_fusion.memory_limit_mb=$SANDBOX_MEMORY_LIMIT_MB"
    )
fi

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="['$TRAIN_FILE']" \
    data.val_files="['$VAL_FILE']" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.shuffle=False \
    actor_rollout_ref.model.path="$STUDENT_MODEL" \
    actor_rollout_ref.model.trust_remote_code=False \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.optim.lr="$ACTOR_LR" \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio="$LR_WARMUP_RATIO" \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU" \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.qat.enable=True \
    actor_rollout_ref.actor.fsdp_config.qat.backend=edgerazor \
    actor_rollout_ref.actor.fsdp_config.qat.mode="$QAT_MODE" \
    actor_rollout_ref.actor.fsdp_config.qat.edgerazor_config_path="$QAT_CONFIG" \
    actor_rollout_ref.actor.fsdp_config.qat.edgerazor_quant_mode="$EDGERAZOR_QUANT_MODE" \
    actor_rollout_ref.ref.fsdp_config.qat.enable="$QUANTIZE_REFERENCE" \
    actor_rollout_ref.ref.fsdp_config.qat.backend=edgerazor \
    actor_rollout_ref.ref.fsdp_config.qat.mode="$QAT_MODE" \
    actor_rollout_ref.ref.fsdp_config.qat.edgerazor_config_path="$QAT_CONFIG" \
    actor_rollout_ref.ref.fsdp_config.qat.edgerazor_quant_mode="$EDGERAZOR_QUANT_MODE" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEM_UTIL" \
    actor_rollout_ref.rollout.enforce_eager="$ROLLOUT_ENFORCE_EAGER" \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.max_model_len="$max_model_len" \
    actor_rollout_ref.rollout.max_num_seqs="$rollout_max_num_seqs" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU" \
    "${reward_overrides[@]}" \
    trainer.balance_batch=True \
    trainer.logger="['console','$WANDB_LOGGER']" \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.default_local_dir="$OUTPUT_DIR/checkpoints" \
    trainer.rollout_data_dir="$OUTPUT_DIR/rollouts" \
    trainer.validation_data_dir="$OUTPUT_DIR/validation" \
    trainer.n_gpus_per_node="$STUDENT_NGPUS" \
    trainer.nnodes="$NNODES" \
    trainer.val_before_train="$VAL_BEFORE_TRAIN" \
    trainer.log_val_generations="$LOG_VAL_GENERATIONS" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.total_epochs=1000 \
    trainer.total_training_steps="$TRAINING_STEPS" \
    trainer.resume_mode=disable \
    trainer.max_actor_ckpt_to_keep="$MAX_ACTOR_CKPTS_TO_KEEP" \
    actor_rollout_ref.actor.checkpoint.save_contents="['model','optimizer','extra','hf_model']" \
    distillation.enabled=True \
    distillation.n_gpus_per_node="$TEACHER_WORLD_SIZE" \
    distillation.nnodes="$NNODES" \
    distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL" \
    distillation.teacher_models.teacher_model.inference.name=vllm \
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size="$TEACHER_TP_SIZE" \
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization="$TEACHER_GPU_MEM_UTIL" \
    distillation.teacher_models.teacher_model.inference.max_model_len="$max_model_len" \
    distillation.distillation_loss.loss_mode="$DISTILLATION_LOSS_MODE" \
    distillation.distillation_loss.topk="$DISTILLATION_TOPK" \
    distillation.distillation_loss.use_task_rewards="$USE_TASK_REWARDS" \
    distillation.distillation_loss.distillation_loss_coef="$DISTILLATION_LOSS_COEF" \
    distillation.distillation_loss.use_policy_gradient="$USE_POLICY_GRADIENT" \
    distillation.distillation_loss.loss_max_clamp=10.0 \
    distillation.distillation_loss.log_prob_min_clamp=-10.0 \
    "$@"
