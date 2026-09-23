#!/usr/bin/env bash
# Measure FP-1.7B on the three benchmarks it was missing, so the ceiling exists on
# all five and code/AMC can be reported as recovery rates instead of only as a
# delta against the release.
#
# Runs on ONE gpu, sequentially, cheapest first: MBPP-448 and HumanEval land within
# the hour, AMC avg@16 is ~6.4 h single-card (the 4-card runs took 5750 s).
#
# Two things make this different from scoring a checkpoint:
#   QAT_ENABLE=False  - b1 normally re-applies EdgeRazor quantization to a latent
#                       checkpoint. Leaving it on would score a round-to-nearest
#                       quantization of FP, which is a different reference entirely.
#   EXPECT=fp         - the lm_eval wrappers hard-stop on a model with >32 distinct
#                       values per 128-block. Here that is the point.
# The card is one of the Lustre-wedged ones with ~37 GB of
# its 95 GB free, hence the low ROLLOUT_GPU_MEM_UTIL; b1 only needs 1024+512 tokens
# of KV so this costs nothing.
set -uo pipefail
Q=${Q:-$QAOPD_ROOT}
S=${S:-$QAOPD_ROOT/shared}
GPU=${GPU:-5}
FP=$Q/models/Qwen3-1.7B
# The merged copy lives on LOCAL disk. Writing it to Lustre wedges the writer in
# osc_extent_wait forever and leaves an unkillable process -- safetensors'
# save_file hits the same wall torch.save does, which is
# why opd17_mirror.sh merges locally and only then copies. Nothing needs this
# copy off-node, so there is no reason to publish it at all.
SINGLE=$QAOPD_ROOT/models/Qwen3-1.7B-single
TRAIN_PY=$QAOPD_ROOT/venvs/qaopd-train/bin/python

cd "$Q" || exit 1
export QAOPD_ROOT=$Q MODELS_DIR=$Q/models DATA_ROOT=$Q/data
export HF_HOME=$S/hf_cache HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_EVALUATE_OFFLINE=1
export HF_ALLOW_CODE_EVAL=1 BITWIDTH=w2.79
export EVAL_ENV=$QAOPD_ROOT/venvs/qaopd-eval

# b1_humaneval.sh requires a single model.safetensors; the FP release ships two
# shards. Merge once, keeping bf16 -- that is the precision being used as the
# ceiling, so casting it would misstate the reference.
if [[ ! -f "$SINGLE/model.safetensors" ]]; then
    echo "=== merging FP shards into $SINGLE"
    mkdir -p "$SINGLE"
    find "$FP" -maxdepth 1 -type f ! -name 'model-*.safetensors' \
        ! -name 'model.safetensors.index.json' -exec cp {} "$SINGLE/" \;
    "$TRAIN_PY" - "$FP" "$SINGLE" <<'PY' || { echo "merge failed"; exit 1; }
import sys, glob, os
from safetensors.torch import load_file, save_file
src, dst = sys.argv[1], sys.argv[2]
merged = {}
for s in sorted(glob.glob(os.path.join(src, "model-*.safetensors"))):
    merged.update(load_file(s))
tmp = os.path.join(dst, "model.safetensors.partial")
save_file(merged, tmp, metadata={"format": "pt"})
os.replace(tmp, os.path.join(dst, "model.safetensors"))
print(f"merged {len(merged)} tensors")
PY
fi

# ---- 1. MBPP-448 (b1) and HumanEval (b1) -------------------------------------
# The train venv goes on PATH because b1_* shell out to run_mbpp_official_eval_w279.sh,
# which picks python off PATH and needs verl's tensordict.
export PATH=$QAOPD_ROOT/venvs/qaopd-train/bin:$PATH
export PY=$TRAIN_PY
export QAT_ENABLE=False ROLLOUT_GPU_MEM_UTIL=0.20
EXP=$QAOPD_ROOT/runs/fpeval_17b
echo "=== MBPP-448 b1 on FP-1.7B (gpu $GPU)"
EXP_DIR=$EXP STEPS=warm WARM_MODEL=$SINGLE GPU=$GPU bash "$Q/scripts/eval/b1_test448.sh"
echo "=== HumanEval b1 on FP-1.7B (gpu $GPU)"
EXP_DIR=$EXP STEPS=warm WARM_MODEL=$SINGLE GPU=$GPU bash "$Q/scripts/eval/b1_humaneval.sh"

# ---- 2. HumanEval instruct ---------------------------------------------------
# A different protocol from b1 and not interchangeable with it, but it is the one
# the release's 51.83 and the QAD start's 39.63 live in, so the ceiling is needed
# there too.
echo "=== HumanEval instruct on FP-1.7B (gpu $GPU)"
MODEL=$SINGLE TAG=fp17b GPUS=$GPU EXPECT=fp BATCH=8 \
    bash "$Q/scripts/eval/chat_eval.sh"

# ---- 3. AMC23 avg@16 ---------------------------------------------------------
echo "=== AMC23 avg@16 on FP-1.7B (gpu $GPU) -- expect ~6.4 h"
MODEL=$SINGLE TAG=fp17b GPUS=$GPU EXPECT=fp BATCH=16 \
    bash "$Q/scripts/eval/amc23_avg16_parallel.sh"

echo "FP_CEILING_DONE"
