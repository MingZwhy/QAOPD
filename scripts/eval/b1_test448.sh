#!/usr/bin/env bash
# b1 (deterministic, ROLLOUT_MAX_NUM_SEQS=1) evaluation on the MBPP official
# test-448 split, which is the split the reported MBPP number is measured on.
# (b1_batch.sh evaluates the 82-row validation split instead, which is what
# checkpoint selection uses.)
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
# This eval re-applies EdgeRazor quantization itself and so takes a *latent*
# checkpoint, not a deployment export. The bit width has to be selected here too.
source "$QAOPD_ROOT/scripts/lib/bitwidth.sh"; bitwidth_setup || exit 1
EXP_DIR=${EXP_DIR:?}; STEPS=${STEPS:?}; GPU=${GPU:-0}
# The checkpoint is loaded inside Ray workers, which do not run in this directory,
# and a relative path there is taken for a Hub repo id.
EXP_DIR=$(cd "$EXP_DIR" && pwd) || exit 1
SEL="$EXP_DIR/b1_test448"; mkdir -p "$SEL"

for STEP in $STEPS; do
    [[ -s "$SEL/$STEP.jsonl" ]] && { echo "step $STEP done"; continue; }
    # STEP=warm evaluates the QAT starting point, which has no global_step_* dir.
    if [[ "$STEP" == warm ]]; then
        MODEL=${WARM_MODEL:-$QAOPD_ROOT/models/$BW_WARM}
        [[ -d "$MODEL" ]] && MODEL=$(cd "$MODEL" && pwd)
    else
        MODEL="$EXP_DIR/checkpoints/global_step_$STEP/actor/huggingface"
    fi
    # Same guard as b1_humaneval.sh: accept either the single-file or the sharded
    # form. `-e "$MODEL"` passed any directory that merely existed, including an
    # empty one, and then failed much later inside the rollout.
    [[ -f "$MODEL/model.safetensors" || -f "$MODEL/model.safetensors.index.json" ]] \
        || { echo "missing ckpt $STEP"; continue; }
    # A deployment export would be quantized a second time here, which changes it.
    if "$PY" - "$QAOPD_ROOT/opd/tools" "$MODEL" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from validate_qat_latent_checkpoint import find_prequantized_evidence
sys.exit(0 if find_prequantized_evidence(Path(sys.argv[2])) else 1)
PY
    then
        echo "B1T448 step=$STEP $MODEL is a deployment export; score it with scripts/eval/code_export.sh"
        continue
    fi
    # At the launcher's 0.70 a Qwen3-4B engine (hidden_size 2560) does not fit on a
    # 72 GB card beside its fp32 actor.
    MEM=${ROLLOUT_GPU_MEM_UTIL:-0.70}
    if [[ -z "${ROLLOUT_GPU_MEM_UTIL:-}" ]] && grep -qE '"hidden_size": *2560' "$MODEL/config.json"; then
        MEM=0.50
    fi
    OUT="$EXP_DIR/evaluation_mbpp_test448/b1g${GPU}_step_$STEP"
    mkdir -p "$EXP_DIR/evaluation_mbpp_test448"; rm -rf "$OUT"
    CUDA_VISIBLE_DEVICES=$GPU RAY_TMPDIR=/tmp/ray_t448g${GPU}_$STEP ROLLOUT_GPU_MEM_UTIL=$MEM \
    STUDENT_MODEL="$MODEL" MODEL_LABEL="b1t448_step_$STEP" OUTPUT_DIR="$OUT" \
    EVAL_SPLIT=test EXPECTED_ROWS=448 ROLLOUT_MAX_NUM_SEQS=1 WANDB_MODE=offline \
    bash "$QAOPD_ROOT/opd/launch/run_mbpp_official_eval_w279.sh" \
      > "$OUT.log" 2>&1
    ROWS=$(wc -l < "$OUT/validation/0.jsonl" 2>/dev/null || echo 0)
    if [[ "$ROWS" -eq 448 ]]; then
        cp "$OUT/validation/0.jsonl" "$SEL/$STEP.jsonl"
        echo "B1T448 step=$STEP passed=$($PY -c "import json;print(sum(1 for l in open('$SEL/$STEP.jsonl') if float(json.loads(l)['score'])>=1.0))")/448"
    else
        echo "B1T448 step=$STEP INCOMPLETE rows=$ROWS (expected 448)"
        # The row count alone says nothing about why. The vendored eval buries the
        # cause a thousand lines into $OUT.log, and hydra writes that log as binary
        # (ANSI from ray), so grep needs -a or it prints "Binary file matches".
        echo "--- root cause from $OUT.log:"
        grep -aoE "(ValueError|RuntimeError|ModuleNotFoundError|AssertionError|OSError|torch\.OutOfMemoryError): .*" \
            "$OUT.log" 2>/dev/null | sort -u | head -5
        echo "--- (a vLLM 'Free memory on device' ValueError usually means a previous"
        echo "    failed run left ray workers holding the GPU)"
    fi
done
echo "B1T448_DONE"
}
