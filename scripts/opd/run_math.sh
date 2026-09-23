#!/usr/bin/env bash
# OPD phase 1: mathematics, starting from a QAD checkpoint.
#
# One phase over a balanced GSM8K / MATH / DAPO mix (2,500 prompts each), which
# lifts GSM8K, MATH-500 and AMC23 together. Splitting the corpus into
# per-benchmark phases buys more on whichever one runs last and less on the
# others, so the balanced mix is the default.
set -uo pipefail
QAOPD_ROOT="${QAOPD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ROOT="${ROOT:-$QAOPD_ROOT}"
VERL="${VERL:-$QAOPD_ROOT/third_party/verl}"
export VERL_ROOT="$VERL"
export MODELS_DIR="${MODELS_DIR:-$QAOPD_ROOT/models}"
export DATA_ROOT="${DATA_ROOT:-$QAOPD_ROOT/data}"
STAMP=$(date -u +%Y%m%d_%H%M%S)

# Per-arm budget and checkpoint density, from the runs behind the published
# results: 80 steps on Qwen3-1.7B, 120 on Qwen3-4B, saving every 10 either way.
# The picked checkpoint on those runs was well inside the budget -- s70 of 80
# and s60 of 120 -- so this is room to scan, not a target to reach.
# BITWIDTH selects the width; MODEL_SIZE selects the budget. Read it out of the
# student's config rather than guessing from the path: checkpoint directories
# are named by arm and stage, not by model, so `qad17_...` and `qad4b_...` are
# a convention a user has no reason to follow. hidden_size separates the three
# cleanly (1024 / 2048 / 2560).
if [[ -z "${MODEL_SIZE:-}" && -f "${STUDENT_MODEL:-}/config.json" ]]; then
    MODEL_SIZE=$(python3 - "$STUDENT_MODEL/config.json" <<'PY'
import json, sys
h = json.load(open(sys.argv[1])).get("hidden_size")
print({1024: "06b", 2048: "1_7b", 2560: "4b"}.get(h, "unknown"))
PY
)
fi
MODEL_SIZE=${MODEL_SIZE:-unknown}
if [[ "$MODEL_SIZE" == unknown ]]; then
    echo "cannot tell the model size from $STUDENT_MODEL; set MODEL_SIZE=06b|1_7b|4b" >&2
    exit 2
fi
case "$MODEL_SIZE" in
    4b) DEF_STEPS=120 ;;
    *)  DEF_STEPS=80  ;;
esac
STEPS=${STEPS:-$DEF_STEPS}

# QAT starting point. Fetch it with tools/fetch_weights.sh (it lives on the
# Hub, not in this repo -- see docs/CHECKPOINTS.md).
export STUDENT_MODEL=${STUDENT_MODEL:-$QAOPD_ROOT/models/w279_step10000}
# Single-file or sharded. The fetched QAD checkpoints are single-file, but
# anything verl saves is sharded (model-000N-of-000M + index.json), so
# requiring model.safetensors alone rejects every checkpoint this pipeline
# produces -- and the message then sends you off to re-download weights that
# are already there. The b1 scripts and eval_unified.sh accept both forms too.
if [[ ! -f "$STUDENT_MODEL/model.safetensors" \
   && ! -f "$STUDENT_MODEL/model.safetensors.index.json" ]]; then
    echo "no weights at $STUDENT_MODEL -- run: bash tools/fetch_weights.sh" >&2
    exit 1
fi
export TEACHER_MODEL=${TEACHER_MODEL_PATH:-$QAOPD_ROOT/models/Qwen3-1.7B}
DATA=${DATA:-$QAOPD_ROOT/data/unified_math_v1}
export TRAIN_FILE=$DATA/train.parquet
export VAL_FILE=$DATA/validation.parquet
export REWARD_FUNCTION_PATH=${REWARD_FUNCTION_PATH:-$QAOPD_ROOT/opd/rewards/math_mixed_reward.py}
export REWARD_FUNCTION_NAME=compute_score
export GSM_REWARD_KWARGS=True
export REQUIRE_HASH_REWARD=True
export CORRECT_WITHOUT_HASH_SCORE=0.0
export HASH_FORMAT_SCORE=0.0

export DISTILLATION_LOSS_MODE=k1
export USE_POLICY_GRADIENT=True
export USE_TASK_REWARDS=True
# ⚠ These two must stay overridable. They were plain assignments, which silently
# clobbered whatever the caller exported: the s4b_c arm was launched with
# DISTILLATION_LOSS_COEF=0.7 and ran at 1.0, i.e. as an exact duplicate of s4b_a,
# and nothing in the logs said so. (The accident did pay for itself -- the two arms
# became a same-config replicate and measured the noise floor at +-3.3 gsm8k
# points, see docs/OPD_1_7B.md. But that was luck, not design.)
export DISTILLATION_TOPK=${DISTILLATION_TOPK:-32}
export DISTILLATION_LOSS_COEF=${DISTILLATION_LOSS_COEF:-1.0}
export ACTOR_LR=${ACTOR_LR:-3e-6}
export TRAINING_STEPS=$STEPS
export SAVE_FREQ=${SAVE_FREQ:-10}
export TEST_FREQ=100000
export VAL_BEFORE_TRAIN=False
export LOG_VAL_GENERATIONS=0
export MAX_ACTOR_CKPTS_TO_KEEP=20

# Bit width. Everything below funnels into run_opd_qad_w279a8_qwen3_06b.sh, which
# reads QAT_CONFIG + EDGERAZOR_QUANT_MODE, so one switch covers math/code/refresh.
source "$QAOPD_ROOT/scripts/lib/bitwidth.sh"; bitwidth_setup || exit 1

# GPU split. These used to be hard assignments, so the recipe could not scale off
# the 2-GPU box it was written on. student+teacher must be <= total, and
# PPO_MINI_BATCH_SIZE*ROLLOUT_N must divide by STUDENT_NGPUS (the trainer checks).
# Student and teacher each get their own GPUs. TRAIN_BATCH_SIZE x ROLLOUT_N
# fixes the optimizer step regardless, so the split is a throughput choice --
# but it also sets how the batch is sharded, and the published runs used the
# splits below. Widening one shortens every rank's share of the batch, which
# is visible in global_seqlen/mean, so keep these to reproduce a published arm
# and override them only when the hardware forces it.
case "${MODEL_SIZE:-06b}" in
    4b) DEF_STU=4; DEF_TEA=4 ;;
    *)  DEF_STU=2; DEF_TEA=2 ;;
esac
export STUDENT_NGPUS=${STUDENT_NGPUS:-$DEF_STU}
export TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-$DEF_TEA}
export TOTAL_GPUS_PER_NODE=${TOTAL_GPUS_PER_NODE:-$((STUDENT_NGPUS + TEACHER_WORLD_SIZE))}
# Were hard assignments. ROLLOUT_N is the GRPO group size, so N=4 estimates each
# advantage from four samples -- noisy enough that the 1.7B phase-1 gsm8k curve
# oscillates ~1.3 points between adjacent checkpoints with no trend. Making these
# overridable is what lets an arm trade steps for a cleaner gradient.
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-$TRAIN_BATCH_SIZE}
export ROLLOUT_N=${ROLLOUT_N:-4}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1664} MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1536}
# The token budget per GPU for the actor update, and how much of each card vLLM
# may hold. These three are the values the published runs used; they are memory
# knobs rather than algorithm ones, but the 4B student only just fits, so a
# larger budget or a greedier rollout engine goes out of memory in the opening
# steps -- a badly damaged student does not terminate, so most of its early
# rollouts run to MAX_RESPONSE_LENGTH and the update sees far longer sequences
# than it will once recovery starts.
case "$MODEL_SIZE" in
    4b) DEF_TOK=4096;  DEF_MEM=0.15 ;;
    *)  DEF_TOK=16384; DEF_MEM=0.25 ;;
esac
export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-$DEF_TOK}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-$DEF_MEM} TEACHER_GPU_MEM_UTIL=${TEACHER_GPU_MEM_UTIL:-0.55}
export LR_WARMUP_RATIO=0.0
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export DATA_SEED=${DATA_SEED:-20260727} RANDOM_SEED=${RANDOM_SEED:-42}

# Everything above takes an override, and an `export` in a calling script beats
# a `${VAR:-default}` silently. That is not hypothetical: a wrapper exporting
# STUDENT_NGPUS=8 ran a published 4B arm on twice the student shards, and it
# trained happily for eight steps before global_seqlen/mean -- half what the
# original logged -- gave it away. A wrong split does not fail, it trains a
# different thing. So print what actually resolved and name whatever is no
# longer the published recipe.
_off_recipe=()
_show() {
    local name=$1 want=$2 got=${!1}
    if [[ "$got" == "$want" ]]; then
        printf '    %-26s %s\n' "$name" "$got"
    else
        printf '    %-26s %s   <- recipe says %s\n' "$name" "$got" "$want"
        _off_recipe+=("$name=$got (recipe $want)")
    fi
}
echo "  resolved recipe -- MODEL_SIZE=$MODEL_SIZE  BITWIDTH=$BITWIDTH"
_show STEPS                     "$DEF_STEPS"
_show SAVE_FREQ                 10
_show ACTOR_LR                  3e-6
_show STUDENT_NGPUS             "$DEF_STU"
_show TEACHER_WORLD_SIZE        "$DEF_TEA"
_show TRAIN_BATCH_SIZE          8
_show PPO_MINI_BATCH_SIZE       8
_show ROLLOUT_N                 4
_show MAX_PROMPT_LENGTH         1664
_show MAX_RESPONSE_LENGTH       1536
_show PPO_MAX_TOKEN_LEN_PER_GPU "$DEF_TOK"
_show ROLLOUT_GPU_MEM_UTIL      "$DEF_MEM"
_show TEACHER_GPU_MEM_UTIL      0.55
if (( ${#_off_recipe[@]} )); then
    echo "  !! ${#_off_recipe[@]} value(s) are not the published recipe:" >&2
    printf '       %s\n' "${_off_recipe[@]}" >&2
    if [[ "${STRICT_RECIPE:-0}" == 1 ]]; then
        echo "  STRICT_RECIPE=1, refusing to start." >&2
        exit 2
    fi
    echo "     Continuing. Set STRICT_RECIPE=1 to make this an error." >&2
fi

export PROJECT_NAME=opd_qad_unified
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-UNIFIED_math_k1_t17_lr3e6_s${STEPS}_${STAMP}}
export OUTPUT_DIR=${OUTPUT_DIR:-${RUNS_DIR:-$QAOPD_ROOT/runs}/$EXPERIMENT_NAME}
export WANDB_MODE=${WANDB_MODE:-offline}

# verl defaults to flash_attention_2, and transformers then hard-fails with
# `PackageNotFoundError: flash_attn` when the package is absent -- it does not
# fall back. docs/INSTALL.md already treats flash_attn as optional for this
# project, so pick sdpa when it is missing instead of dying 3 minutes in.
# Probe with importlib.metadata, the same call transformers makes. `import
# flash_attn` is NOT a valid test: flashinfer-python ships a `flash_attn.cute`
# subpackage, so a bare import succeeds against an empty namespace package
# (__file__ is None) on an env with no flash-attn at all.
if [[ -z "${ATTN_IMPL:-}" ]]; then
    if python -c "import importlib.metadata as m; m.version('flash_attn')" 2>/dev/null; then
        ATTN_IMPL=flash_attention_2
    else
        ATTN_IMPL=sdpa
        echo "flash_attn not installed -> attn_implementation=sdpa"
    fi
fi
export ATTN_IMPL

# EXTRA_OVERRIDES lets a caller append hydra overrides; it expands to nothing by
# default. It exists because the 4B student OOMs on a 72 GB card and needs
# actor_rollout_ref.actor.fsdp_config.optimizer_offload=True,
# and that override string used to be hardcoded with no injection point.
# Deliberately unquoted: word splitting is what turns it into separate args.
echo "UNIFIED math OPD: data=$(basename $DATA) steps=$STEPS resp=$MAX_RESPONSE_LENGTH attn=$ATTN_IMPL${EXTRA_OVERRIDES:+ extra=$EXTRA_OVERRIDES}"
export TRAIN_SCRIPT=$QAOPD_ROOT/opd/launch/run_opd_qad_w279a8_qwen3_06b.sh
PLAIN_TEXT_CHAT_TEMPLATE="{% for message in messages %}{{ message.content }}{% endfor %}"

# CHAT_TEMPLATE_MODE=plain is the continuation-style prompt the Qwen line has
# always used: raw text in, truncated by stop=['Question:','Problem:'] once the
# model starts inventing the next problem. Qwen3-QAD works well with it.
# Instruction-tuned models of other families may not: without their header
# structure the model never enters an assistant turn, never emits its end token,
# and nearly every rollout runs to max_response_length, pushing entropy towards
# ln(V) so nothing is learned.
# mode=native uses the tokenizer's own template so generation terminates
# normally; enable_thinking is a Qwen3-specific kwarg and is dropped there.
case "${CHAT_TEMPLATE_MODE:-plain}" in
    plain)
        TEMPLATE_ARGS=(
            "actor_rollout_ref.model.custom_chat_template='$PLAIN_TEXT_CHAT_TEMPLATE'"
            "+actor_rollout_ref.rollout.stop=['Question:','Problem:']"
            +data.apply_chat_template_kwargs.enable_thinking=False
        ) ;;
    native)
        TEMPLATE_ARGS=() ;;
    *)
        echo "CHAT_TEMPLATE_MODE must be plain or native, got '${CHAT_TEMPLATE_MODE}'" >&2
        exit 2 ;;
esac
echo "  chat template: ${CHAT_TEMPLATE_MODE:-plain}"

bash "$QAOPD_ROOT/opd/launch/test_opd_qad.sh" \
    "${TEMPLATE_ARGS[@]}" \
    data.shuffle=True \
    data.seed="$DATA_SEED" \
    +reward.custom_reward_function.reward_kwargs.overlong_penalty="${OVERLONG_PENALTY:-0.5}" \
    +reward.custom_reward_function.reward_kwargs.overlong_chars="${OVERLONG_CHARS:-3000}" \
    +actor_rollout_ref.model.override_config.attn_implementation="$ATTN_IMPL" \
    ${EXTRA_OVERRIDES:-}
rc=$?
# The marker used to print unconditionally, so a trainer that died on line 1 still
# ended with "UNIFIED_DONE" and read as a clean run.
if [[ $rc -ne 0 ]]; then
    echo "UNIFIED_FAILED rc=$rc out=$OUTPUT_DIR steps=$STEPS" >&2
    exit $rc
fi
echo "UNIFIED_DONE out=$OUTPUT_DIR steps=$STEPS"
