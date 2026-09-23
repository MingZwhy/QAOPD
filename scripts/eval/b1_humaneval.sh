#!/usr/bin/env bash
# HumanEval-164 deterministic (b1) evaluation.
set -uo pipefail
# Brace group so an edit mid-run cannot corrupt an in-flight job; see the note in
# scripts/eval/amc23_avg16_parallel.sh.
{
QAOPD_ROOT="${QAOPD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ROOT="${ROOT:-$QAOPD_ROOT}"
VERL="${VERL:-$QAOPD_ROOT/third_party/verl}"
PY=${PY:-python}   # activate the eval env (docs/INSTALL.md) or set PY
# The vendored eval scripts resolve datasets under DATA_ROOT; the repo keeps
# them in data/, resolved through DATA_ROOT.
export DATA_ROOT="${DATA_ROOT:-$QAOPD_ROOT/data}"
export MODELS_DIR="${MODELS_DIR:-$QAOPD_ROOT/models}"
export VERL_ROOT="$VERL"
# Same as b1_test448.sh: quantization is re-applied here, so pass a latent
# checkpoint and select the bit width explicitly.
source "$QAOPD_ROOT/scripts/lib/bitwidth.sh"; bitwidth_setup || exit 1
EXP_DIR=${EXP_DIR:?}; STEPS=${STEPS:?}; GPU=${GPU:-0}
SEL="$EXP_DIR/b1_humaneval"; mkdir -p "$SEL"

for STEP in $STEPS; do
    [[ -s "$SEL/$STEP.jsonl" ]] && { echo "step $STEP done"; continue; }
    if [[ "$STEP" == warm ]]; then
        MODEL=${WARM_MODEL:-$QAOPD_ROOT/models/$BW_WARM}
    else
        MODEL="$EXP_DIR/checkpoints/global_step_$STEP/actor/huggingface"
    fi
    # A checkpoint is either one model.safetensors or a shard set described by
    # model.safetensors.index.json. verl's OPD arms save sharded, so requiring the
    # single-file form silently skipped every one of them -- the scan printed
    # "no ckpt" per step and reported DONE in seconds. b1_test448.sh took the
    # opposite bet (`-e "$MODEL"`, which an empty directory also passes), so the
    # same checkpoints scored fine there and only HumanEval went missing.
    [[ -f "$MODEL/model.safetensors" || -f "$MODEL/model.safetensors.index.json" ]] \
        || { echo "HE step=$STEP no ckpt"; continue; }
    OUT="$EXP_DIR/evaluation_humaneval/b1g${GPU}_step_$STEP"
    mkdir -p "$EXP_DIR/evaluation_humaneval"; rm -rf "$OUT"
    CUDA_VISIBLE_DEVICES=$GPU RAY_TMPDIR=/tmp/ray_heg${GPU}_$STEP \
    STUDENT_MODEL="$MODEL" MODEL_LABEL="b1he_step_$STEP" OUTPUT_DIR="$OUT" \
    DATA_DIR=${HUMANEVAL_EVAL_DIR:-$QAOPD_ROOT/data/humaneval_eval} \
    EVAL_SPLIT=validation EXPECTED_ROWS=164 ROLLOUT_MAX_NUM_SEQS=1 WANDB_MODE=offline \
    bash "$QAOPD_ROOT/opd/launch/run_mbpp_official_eval_w279.sh" \
      > "$OUT.log" 2>&1
    ROWS=$(wc -l < "$OUT/validation/0.jsonl" 2>/dev/null || echo 0)
    if [[ "$ROWS" -eq 164 ]]; then
        cp "$OUT/validation/0.jsonl" "$SEL/$STEP.jsonl"
        echo "HE step=$STEP passed=$($PY -c "import json;print(sum(1 for l in open('$SEL/$STEP.jsonl') if float(json.loads(l)['score'])>=1.0))")/164"
    else
        echo "HE step=$STEP INCOMPLETE rows=$ROWS (expected 164)"
        # Same reasoning as b1_test448.sh: surface the cause, and note that ray's
        # ANSI makes the log look binary to grep.
        echo "--- root cause from $OUT.log:"
        grep -aoE "(ValueError|RuntimeError|ModuleNotFoundError|AssertionError|OSError|torch\.OutOfMemoryError): .*" \
            "$OUT.log" 2>/dev/null | sort -u | head -5
    fi
done
echo "HUMANEVAL_DONE"
}
