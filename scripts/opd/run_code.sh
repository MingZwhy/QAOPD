#!/usr/bin/env bash
# OPD phase 2: code, starting from a phase-1 checkpoint.
#
# The student samples completions on a pool of Python problems. The BF16 teacher's top-64
# log-probabilities on those completions give a forward-KL target, applied as a direct gradient,
# and a GRPO term scores each completion 1 when it passes every test of its problem and 0
# otherwise. One optimizer update per step, clip_ratio_high 0.28, constant learning rate, a
# checkpoint every 50 steps.
#
# Pool, stages, steps, learning rate, batch, GPU split and seed resolve per arm from the
# student's config and BITWIDTH, and the script prints what it picked. An arm with two stages
# runs them in order, the second from the last checkpoint of the first; the last stage writes
# to runs/$EXPERIMENT_NAME.
#
#   BITWIDTH=w2.79 STUDENT_MODEL=<phase-1 checkpoint> EXPERIMENT_NAME=<name> bash scripts/opd/run_code.sh
set -uo pipefail
QAOPD_ROOT="${QAOPD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export QAOPD_ROOT
export ROOT="${ROOT:-$QAOPD_ROOT}"
VERL="${VERL:-$QAOPD_ROOT/third_party/verl}"
export VERL_ROOT="$VERL"
export PYTHONPATH="$QAOPD_ROOT/third_party/edgerazor/src:$VERL:${PYTHONPATH:-}"
export DATA_ROOT="${DATA_ROOT:-$QAOPD_ROOT/data}"
export MODELS_DIR="${MODELS_DIR:-$QAOPD_ROOT/models}"
export RUNS_DIR="${RUNS_DIR:-$QAOPD_ROOT/runs}"
export HF_HOME=${HF_HOME:-$ROOT/.cache/huggingface}
export WANDB_MODE=${WANDB_MODE:-offline}
# The student and teacher vLLM engines start together, and the flashinfer sampler's JIT
# build can race on a cold cache.
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
# Training is single-node. When the student's GPUs sit under different PCIe roots NCCL may
# route between them over InfiniBand, which fails on hosts whose IB devices are not usable
# from the container.
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}

python -c "import verl, edgerazor, vllm" 2>/dev/null || {
    echo "python on PATH ($(command -v python)) lacks verl/edgerazor/vllm." >&2
    echo "Activate the training env: export PATH=<repo>/venvs/qaopd-train/bin:\$PATH" >&2
    exit 1; }

# The phase-1 optimum is often well before the end of that run, so the starting point is
# always named rather than guessed.
if [[ -z "${STUDENT_MODEL:-}" ]]; then
    echo "set STUDENT_MODEL to the phase-1 checkpoint you picked, e.g." >&2
    echo "  STUDENT_MODEL=runs/p1_w279/checkpoints/global_step_80/actor/huggingface" >&2
    exit 2
fi
if [[ ! -f "$STUDENT_MODEL/model.safetensors" && ! -f "$STUDENT_MODEL/model.safetensors.index.json" ]]; then
    echo "no weights at $STUDENT_MODEL" >&2
    exit 1
fi
STUDENT_MODEL=$(cd "$STUDENT_MODEL" && pwd)

source "$QAOPD_ROOT/scripts/lib/bitwidth.sh"; bitwidth_setup || exit 1

# Model size from the student's config, as in run_math.sh: hidden_size separates the three
# (1024 / 2048 / 2560), and checkpoint paths carry no reliable hint.
if [[ -z "${MODEL_SIZE:-}" ]]; then
    MODEL_SIZE=$(python3 - "$STUDENT_MODEL/config.json" <<'PY'
import json, sys
h = json.load(open(sys.argv[1])).get("hidden_size")
print({1024: "06b", 2048: "1_7b", 2560: "4b"}.get(h, "unknown"))
PY
)
fi
case "$MODEL_SIZE" in
    06b|1_7b|4b) ;;
    *) echo "cannot tell the model size from $STUDENT_MODEL; set MODEL_SIZE=06b|1_7b|4b" >&2; exit 2 ;;
esac

# Per-arm recipe. A stage is pool:steps:lr; the pools are built by data/build_code_pools.py.
case "$MODEL_SIZE/$BITWIDTH" in
    06b/w2.79)  DEF_STAGES=code_opd_12k:300:3e-6                    DEF_BATCH=32 DEF_STU=1 DEF_SEED=45 ;;
    06b/w1.88)  DEF_STAGES=code_opd_24k:600:3e-6                    DEF_BATCH=32 DEF_STU=2 DEF_SEED=42 ;;
    1_7b/*)     DEF_STAGES=code_opd_16k:300:3e-6,code_opd_18k:300:1e-6 DEF_BATCH=32 DEF_STU=4 DEF_SEED=44 ;;
    4b/w2.79)   DEF_STAGES=code_opd_24k:300:3e-6                    DEF_BATCH=32 DEF_STU=4 DEF_SEED=43 ;;
    4b/w1.88)   DEF_STAGES=code_opd_24k:300:3e-6                    DEF_BATCH=64 DEF_STU=4 DEF_SEED=42 ;;
esac
case "$MODEL_SIZE" in
    4b) DEF_TEACHER=Qwen3-4B ;;
    *)  DEF_TEACHER=Qwen3-1.7B ;;
esac

CODE_STAGES=${CODE_STAGES:-$DEF_STAGES}
# STEPS sets the length of every stage, e.g. for a short trial run.
if [[ -n "${STEPS:-}" ]]; then
    CODE_STAGES=$(awk -F, -v OFS=, -v s="$STEPS" \
        '{ for (i = 1; i <= NF; i++) { split($i, a, ":"); $i = a[1] ":" s ":" a[3] } print }' <<<"$CODE_STAGES")
fi
IFS=, read -ra STAGE_LIST <<<"$CODE_STAGES"
TEACHER_MODEL=${TEACHER_MODEL_PATH:-$MODELS_DIR/$DEF_TEACHER}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-$DEF_BATCH}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-$TRAIN_BATCH_SIZE}
ROLLOUT_N=${ROLLOUT_N:-4}
STUDENT_NGPUS=${STUDENT_NGPUS:-$DEF_STU}
TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-1}
TOTAL_GPUS=$((STUDENT_NGPUS + TEACHER_WORLD_SIZE))
GPUS=${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((TOTAL_GPUS - 1)))}
SAVE_FREQ=${SAVE_FREQ:-50}
TEST_FREQ=${TEST_FREQ:-$SAVE_FREQ}
SEED=${SEED:-$DEF_SEED}
RANDOM_SEED=${RANDOM_SEED:-$SEED}
# Data-order seed. Seed 42 maps to 20260718, the launcher's own default.
DATA_SEED=${DATA_SEED:-$((20260676 + SEED))}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-code_${MODEL_SIZE}_${BW_LABEL}_$(date -u +%Y%m%d_%H%M%S)}

[[ -f "$TEACHER_MODEL/config.json" ]] || { echo "no teacher at $TEACHER_MODEL (set TEACHER_MODEL_PATH)" >&2; exit 1; }
# Absolute paths only from here on: the trainer runs in Ray workers, which do not start in
# this directory.
TEACHER_MODEL=$(cd "$TEACHER_MODEL" && pwd)
[[ -d "$DATA_ROOT" ]] && DATA_ROOT=$(cd "$DATA_ROOT" && pwd)
mkdir -p "$RUNS_DIR" && RUNS_DIR=$(cd "$RUNS_DIR" && pwd)
for stage in "${STAGE_LIST[@]}"; do
    IFS=: read -r pool steps lr <<<"$stage"
    if [[ -z "$lr" || ! "$steps" =~ ^[0-9]+$ ]]; then
        echo "bad stage '$stage' in CODE_STAGES (want pool:steps:lr)" >&2; exit 2
    fi
    if [[ ! -f "$DATA_ROOT/$pool/train.parquet" ]]; then
        echo "no code pool at $DATA_ROOT/$pool -- run: python data/build_code_pools.py" >&2; exit 1
    fi
done

# An override is easy to miss -- an `export` in a calling script beats every default here --
# and a different batch split or GPU count trains a different model without failing. Print
# what resolved and name whatever differs from the recipe.
_off_recipe=()
_show() {
    local name=$1 got=$2 want=$3
    if [[ "$got" == "$want" ]]; then
        printf '    %-22s %s\n' "$name" "$got"
    else
        printf '    %-22s %s   <- recipe says %s\n' "$name" "$got" "$want"
        _off_recipe+=("$name=$got (recipe $want)")
    fi
}
echo "  resolved recipe -- MODEL_SIZE=$MODEL_SIZE  BITWIDTH=$BITWIDTH"
_show stages              "$CODE_STAGES"        "$DEF_STAGES"
_show teacher             "$(basename "$TEACHER_MODEL")" "$DEF_TEACHER"
_show TRAIN_BATCH_SIZE    "$TRAIN_BATCH_SIZE"    "$DEF_BATCH"
_show PPO_MINI_BATCH_SIZE "$PPO_MINI_BATCH_SIZE" "$DEF_BATCH"
_show ROLLOUT_N           "$ROLLOUT_N"           4
_show STUDENT_NGPUS       "$STUDENT_NGPUS"       "$DEF_STU"
_show TEACHER_WORLD_SIZE  "$TEACHER_WORLD_SIZE"  1
_show SAVE_FREQ           "$SAVE_FREQ"           50
_show SEED                "$SEED"                "$DEF_SEED"
echo "    GPUs                   $GPUS"
if (( ${#_off_recipe[@]} )); then
    echo "  !! ${#_off_recipe[@]} value(s) differ from the recipe:" >&2
    printf '       %s\n' "${_off_recipe[@]}" >&2
    if [[ "${STRICT_RECIPE:-0}" == 1 ]]; then
        echo "  STRICT_RECIPE=1, refusing to start." >&2
        exit 2
    fi
    echo "     Continuing. Set STRICT_RECIPE=1 to make this an error." >&2
fi

# The 4B student and its optimizer state do not fit on a 72 GB card without offloading the
# optimizer.
ARM_ARGS=()
[[ "$MODEL_SIZE" == 4b ]] && ARM_ARGS+=(actor_rollout_ref.actor.fsdp_config.optimizer_offload=True)

student=$STUDENT_MODEL
for i in "${!STAGE_LIST[@]}"; do
    IFS=: read -r pool steps lr <<<"${STAGE_LIST[i]}"
    k=$((i + 1))
    name=$EXPERIMENT_NAME
    (( k < ${#STAGE_LIST[@]} )) && name=${EXPERIMENT_NAME}_stage$k
    echo "== code stage $k/${#STAGE_LIST[@]}: $pool, $steps steps at lr $lr -> $RUNS_DIR/$name"
    echo "   from $student"
    env STUDENT_MODEL="$student" TEACHER_MODEL="$TEACHER_MODEL" \
        EXPERIMENT_NAME="$name" OUTPUT_DIR="$RUNS_DIR/$name" DATA_DIR_OVERRIDE="$DATA_ROOT/$pool" \
        CUDA_VISIBLE_DEVICES="$GPUS" TOTAL_GPUS_PER_NODE="$TOTAL_GPUS" STUDENT_NGPUS="$STUDENT_NGPUS" \
        TEACHER_WORLD_SIZE="$TEACHER_WORLD_SIZE" TEACHER_TP_SIZE="${TEACHER_TP_SIZE:-1}" \
        TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" PPO_MINI_BATCH_SIZE="$PPO_MINI_BATCH_SIZE" ROLLOUT_N="$ROLLOUT_N" \
        RANDOM_SEED="$RANDOM_SEED" DATA_SEED="$DATA_SEED" \
        DISTILLATION_LOSS_MODE=forward_kl_topk DISTILLATION_TOPK=64 USE_POLICY_GRADIENT=False ACTOR_LR="$lr" \
        TRAINING_STEPS="$steps" SAVE_FREQ="$SAVE_FREQ" TEST_FREQ="$TEST_FREQ" \
        MAX_ACTOR_CKPTS_TO_KEEP=$((steps / SAVE_FREQ + 1)) \
        bash "$QAOPD_ROOT/opd/launch/run_code_opd_search_w279a8kv16.sh" mbpp_official_k1_t17_strict_lr3e6_s30 \
        actor_rollout_ref.actor.clip_ratio_high=0.28 \
        "${ARM_ARGS[@]}" ${EXTRA_OVERRIDES:-}
    rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "CODE_FAILED stage=$k rc=$rc out=$RUNS_DIR/$name" >&2
        exit $rc
    fi
    student=$RUNS_DIR/$name/checkpoints/global_step_$steps/actor/huggingface
    if [[ ! -f "$student/config.json" ]]; then
        echo "CODE_FAILED stage=$k left no checkpoint at $student" >&2
        exit 1
    fi
done
echo "CODE_DONE final=$student"
