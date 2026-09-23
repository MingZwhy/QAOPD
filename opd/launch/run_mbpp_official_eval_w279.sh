#!/usr/bin/env bash
set -euo pipefail

# Path roots default to this file's location so a QAOPD checkout stays self-contained.
QAOPD_ROOT=${QAOPD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
ROOT=${ROOT:-$QAOPD_ROOT}
VERL_ROOT=${VERL_ROOT:-$QAOPD_ROOT/third_party/verl}

CONDA_BIN=${CONDA_BIN:-$(dirname "$(command -v python)")}
DATA_DIR=${DATA_DIR:-${DATA_ROOT:-$QAOPD_ROOT/data}/mbpp_official_protocol_v1}
EVAL_SPLIT=${EVAL_SPLIT:-test}
STUDENT_MODEL=${STUDENT_MODEL:-${MODELS_DIR:-$QAOPD_ROOT/models}/w279_step10000}
MODEL_LABEL=${MODEL_LABEL:-qad_warm}
QAT_ENABLE=${QAT_ENABLE:-True}
EXPECTED_ROWS=${EXPECTED_ROWS:-448}
RUN_STAMP=${RUN_STAMP:-$(date -u +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/experiments/opd_qad/mbpp_official_eval/${MODEL_LABEL}_${EVAL_SPLIT}_${RUN_STAMP}}
CHAT_TEMPLATE=$QAOPD_ROOT/opd/humaneval_opd_chat_template.jinja

case "$EVAL_SPLIT" in
    validation|test) ;;
    *)
        echo "EVAL_SPLIT must be validation or test" >&2
        exit 2
        ;;
esac

for path in "$DATA_DIR/train.parquet" "$DATA_DIR/$EVAL_SPLIT.parquet" "$STUDENT_MODEL" "$CHAT_TEMPLATE"; do
    if [[ ! -e "$path" ]]; then
        echo "Required path does not exist: $path" >&2
        exit 1
    fi
done
if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Output already exists: $OUTPUT_DIR" >&2
    exit 1
fi

export PATH="$CONDA_BIN:$PATH"
export VIRTUAL_ENV=${VIRTUAL_ENV:-${CONDA_PREFIX:-}}
export TRAIN_FILE=$DATA_DIR/train.parquet
export VAL_FILE=$DATA_DIR/$EVAL_SPLIT.parquet
export STUDENT_MODEL
# TEACHER_WORLD_SIZE=0, so this teacher never runs inference; it is only a
# required placeholder in the config. The existence guard above still checks it,
# so any machine without Qwen3-0.6B died here -- and the error ("Required path
# does not exist") named a model with nothing to do with this evaluation, which
# is very hard to connect. Fall back to the student itself when the placeholder
# is absent: it always exists and its vocabulary matches by construction.
if [[ -z "${TEACHER_MODEL:-}" ]]; then
    if [[ -e "$ROOT/models/Qwen3-0.6B" ]]; then
        TEACHER_MODEL=$ROOT/models/Qwen3-0.6B
    else
        TEACHER_MODEL=$STUDENT_MODEL
    fi
fi
export TEACHER_MODEL
export TOTAL_GPUS_PER_NODE=1
export STUDENT_NGPUS=1
export TEACHER_WORLD_SIZE=0
export TEACHER_TP_SIZE=1
export TRAIN_BATCH_SIZE=1
export PPO_MINI_BATCH_SIZE=1
export MAX_PROMPT_LENGTH=1024
export MAX_RESPONSE_LENGTH=512
export PPO_MAX_TOKEN_LEN_PER_GPU=32768
export ROLLOUT_N=1
export TRAINING_STEPS=1
export ACTOR_LR=0
export LR_WARMUP_RATIO=0
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.70}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-256}
export USE_POLICY_GRADIENT=False
export USE_TASK_REWARDS=False
export DISTILLATION_LOSS_COEF=0
export REWARD_FUNCTION_PATH=$QAOPD_ROOT/opd/rewards/mbpp_execution_reward.py
export REWARD_FUNCTION_NAME=compute_score
export GSM_REWARD_KWARGS=False
export VAL_BEFORE_TRAIN=True
export SAVE_FREQ=-1
export TEST_FREQ=-1
export LOG_VAL_GENERATIONS=0
export MAX_ACTOR_CKPTS_TO_KEEP=1
export WANDB_LOGGER=wandb
export WANDB_MODE=${WANDB_MODE:-offline}
export PROJECT_NAME=opd_qad_mbpp_official
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-${MODEL_LABEL}_${EVAL_SPLIT}_${RUN_STAMP}}
export OUTPUT_DIR

HUMANEVAL_CHAT_TEMPLATE=$(<"$CHAT_TEMPLATE")
export HUMANEVAL_CHAT_TEMPLATE

# NOTE: this used to run a GLOBAL `ray stop --force` here and in an EXIT
# trap. That kills every other Ray job on the machine, so two local
# evaluations could never run concurrently (and each new checkpoint in a
# selection loop would kill any parallel job). Ray is now isolated per
# invocation via RAY_TMPDIR instead.
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray_eval_$$}
mkdir -p "$RAY_TMPDIR"

mkdir -p "$OUTPUT_DIR"
{
    echo "MBPP official-split deterministic evaluation"
    echo "model=$STUDENT_MODEL"
    echo "model_label=$MODEL_LABEL"
    echo "qat_enable=$QAT_ENABLE"
    echo "eval_split=$EVAL_SPLIT"
    echo "eval_file=$VAL_FILE"
} | tee "$OUTPUT_DIR/eval_protocol.txt"

# Without this the eval dies on `PackageNotFoundError: flash_attn` before loading
# the model. The training scripts already do this; the eval path did not.
source "$ROOT/scripts/lib/attn.sh"; attn_pick

bash "$QAOPD_ROOT/opd/launch/run_opd_qad_w279a8_qwen3_06b.sh" \
    'actor_rollout_ref.model.custom_chat_template=${oc.env:HUMANEVAL_CHAT_TEMPLATE}' \
    "+actor_rollout_ref.model.override_config.attn_implementation=$ATTN_IMPL" \
    '+actor_rollout_ref.rollout.stop=["```"]' \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    data.shuffle=False \
    `# VAL_N / VAL_TEMP / VAL_DO_SAMPLE stay at 1 / 0 / False, the greedy` \
    `# single-sample protocol. They are overridable for learnability screening:` \
    `# n=8 at temperature 1.0 reproduces the rollout conditions of training, and` \
    `# the per-problem pass rate then isolates the problems that are neither` \
    `# always right nor always wrong -- the only ones that give GRPO a non-zero` \
    `# advantage.` \
    actor_rollout_ref.actor.fsdp_config.qat.enable="$QAT_ENABLE" \
    actor_rollout_ref.ref.fsdp_config.qat.enable="$QAT_ENABLE" \
    `# These three used to be omitted -- only enable was passed. vLLM then saw an` \
    `# empty quant_mode and a None _quant_mode_cfg, so embed_tokens / lm_head` \
    `# never picked up the embint4 override and fell back to the global width` \
    `# (2.79 / 1.88). The embedding quantizer only implements int4 and int1.58,` \
    `# so 2.79 fails outright with "Unsupported embedding weight_bits=2.79".` \
    `# Qwen3 hid this not because it lacks tied embeddings (4B is tie=True too),` \
    `# but because vLLM qwen2.py passes a prefix when building the embedding and` \
    `# llama.py does not: the former matches the ".*embed_tokens" override, the` \
    `# latter gets an empty prefix and falls back to the global width.` \
    `# See dynamic_reload.py.` \
    actor_rollout_ref.actor.fsdp_config.qat.backend="${QAT_BACKEND:-edgerazor}" \
    actor_rollout_ref.actor.fsdp_config.qat.edgerazor_config_path="${QAT_CONFIG:-}" \
    actor_rollout_ref.actor.fsdp_config.qat.edgerazor_quant_mode="${EDGERAZOR_QUANT_MODE:-}" \
    actor_rollout_ref.ref.fsdp_config.qat.backend="${QAT_BACKEND:-edgerazor}" \
    actor_rollout_ref.ref.fsdp_config.qat.edgerazor_config_path="${QAT_CONFIG:-}" \
    actor_rollout_ref.ref.fsdp_config.qat.edgerazor_quant_mode="${EDGERAZOR_QUANT_MODE:-}" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=${VAL_DO_SAMPLE:-False} \
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_N:-1} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMP:-0} \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    +reward.custom_reward_function.reward_kwargs.require_all_tests=True \
    distillation.enabled=False \
    trainer.val_only=True \
    2>&1 | tee "$OUTPUT_DIR/eval.log"

test -s "$OUTPUT_DIR/validation/0.jsonl"
"$CONDA_BIN/python" \
    "$QAOPD_ROOT/opd/tools/analyze_mbpp_official_eval.py" \
    --result "$OUTPUT_DIR/validation/0.jsonl" \
    --expected_rows "$EXPECTED_ROWS" \
    --output "$OUTPUT_DIR/summary.json"
