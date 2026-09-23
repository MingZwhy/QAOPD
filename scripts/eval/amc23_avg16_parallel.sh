#!/usr/bin/env bash
# AMC23 avg@16 (640 samples) on a deployment-export model, data-parallel over N GPUs.
#
# Why this exists: AMC23 avg@16 through the HF + EdgeRazor path costs ~37.7 s per
# sample on one GPU (measured 2026-07-29 on RTX PRO 5000, 32 samples in 20m04s),
# i.e. ~6.7 h for one checkpoint. Sweeping phase3 step10..50 that way is ~34 GPU-h
# serial. accelerate shards the 640 requests across processes and lm_eval gathers
# them, so wall time drops by roughly the process count.
#
# The protocol is unchanged from scripts/eval/eval_unified.sh: same task
# (amc23_avg16 = 40 problems x 16 samples, T0.7, top_p 0.95, 4-shot, 8192 cap),
# same lm_eval, same model args. Only the process count differs.
#
# ⚠ MODEL must be a DEPLOYMENT EXPORT, not a latent QAT checkpoint. A latent
# checkpoint loads without error and scores an unquantized model (see
# docs/CHECKPOINTS.md); this script refuses it rather than reporting a wrong number.
#
# Usage:
#   MODEL=<exported model dir> TAG=p3_s30 GPUS=8,9,10,11,12,13,14,15 \
#     bash scripts/eval/amc23_avg16_parallel.sh
set -uo pipefail
# Everything below runs inside one brace group on purpose. bash reads a script
# incrementally by byte offset, so editing this file while a run is in flight makes
# the running copy resume at a shifted offset and execute a fragment of a line
# (it has cost three runs here). A brace group is a single
# compound command, so bash must parse to the closing brace before executing any of
# it, and an edit mid-run can no longer corrupt the job. Body is left unindented to
# keep the diff readable.
{
QAOPD_ROOT="${QAOPD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# eval_tasks/*.yaml hold repo-root-relative data_files (tools/fix_eval_task_paths.sh)
cd "$QAOPD_ROOT" || exit 1
bash "$QAOPD_ROOT/tools/fix_eval_task_paths.sh" --check >/dev/null || {
    echo "eval task data_files are stale; run: bash tools/fix_eval_task_paths.sh" >&2; exit 1; }

MODEL=${MODEL:?set MODEL to a deployment-export model dir}
TAG=${TAG:?set TAG}
GPUS=${GPUS:-0}
BATCH=${BATCH:-16}
# TASK=amc23 runs the greedy single-pass variant instead. avg@16 is the number we report,
# but at ~6.4 h on one card it is useless as a signal for deciding whether an arm is worth
# continuing -- amcL and amcH would both finish training before their first AMC measurement
# came back. The greedy pass is 40 requests instead of 640 and answers the screening
# question ("did this move AMC at all") at a sixteenth of the cost. Not comparable to
# reported avg@16 figures: different protocol, and 40 problems greedy is a coarse ruler.
TASK=${TASK:-amc23_avg16}
E=${EVAL_ENV:-$QAOPD_ROOT/venvs/qaopd-eval}
ACCELERATE=${ACCELERATE:-$E/bin/accelerate}
# NOTE: without disabling IB the whole group dies on the first all_gather, and
#   every rank reports only "remote process exited early" -- no rank gives the
#   real cause, so it looks like a model-loading failure. Measured: it fails
#   after 44 seconds without this setting (even on idle GPUs) and runs fine with
#   it. qa_suite.sh always set it explicitly; this script did not, so on the same
#   machine QA ran and AMC did not.
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
PY=${PY:-$E/bin/python}
OUT=${OUT:-$QAOPD_ROOT/experiments/opd_qad/$TASK/$TAG}
# A constant here meant two AMC runs on one host collided with each other, and
# either of them collided with a Ray worker when training shared the box.
source "$QAOPD_ROOT/scripts/lib/port.sh"
PORT=${PORT:-$(free_port)} || { echo "could not find a free port" >&2; exit 1; }
# lm_eval's default seed is fixed, so re-running a model reproduces bit for bit.
# Set SEED to a different value to draw a different set of 16 samples per problem;
# that is the only way to measure how much of an AMC gap between two checkpoints is
# sampling noise rather than the models.
SEED=${SEED:-0,1234,1234,1234}

[[ -f "$MODEL/config.json" ]] || { echo "no config.json at $MODEL" >&2; exit 1; }
command -v "$ACCELERATE" >/dev/null || { echo "accelerate not found at $ACCELERATE" >&2; exit 1; }

# Refuse a latent checkpoint: the deployment export bakes quantization into the
# tensors (~12 unique values per 128-element block), latent keeps bf16 masters
# (~125). Getting this backwards is silent, so it is a hard stop.
#
# EXPECT=fp opts out, same spelling as chat_eval.sh, for the one case where an
# unquantized model is the point: scoring FP-1.7B to establish the ceiling that the
# recovery rates are measured against. Do not set it for a checkpoint that is
# supposed to be quantized -- that is the mistake this check exists to catch.
if [[ "${EXPECT:-export}" == fp ]]; then
    echo "   EXPECT=fp: skipping the deployment-export check on purpose"
else
"$PY" - "$MODEL" <<'PY' || exit 1
import sys, os, json, torch
from safetensors import safe_open
ck = sys.argv[1]
# 4B and larger are sharded; only index.json says which shard holds a weight
idx = os.path.join(ck, "model.safetensors.index.json")
if os.path.exists(idx):
    wm = json.load(open(idx))["weight_map"]
    k = next(x for x in wm if "mlp.down_proj.weight" in x)
    shard = os.path.join(ck, wm[k])
else:
    k, shard = None, os.path.join(ck, "model.safetensors")
with safe_open(shard, "pt") as f:
    if k is None:
        k = next(x for x in f.keys() if "mlp.down_proj.weight" in x)
    n = len(torch.unique(f.get_tensor(k).float().flatten()[:128]))
if n > 32:
    sys.exit(f"REFUSING: {ck} looks like a LATENT QAT checkpoint "
             f"({n} unique values in a 128-element block). Export it first "
             f"(docs/CHECKPOINTS.md), otherwise you are scoring an unquantized model.")
print(f"   deployment export confirmed ({n} unique values per 128-block)")
PY
fi

NPROC=$(awk -F, '{print NF}' <<<"$GPUS")
mkdir -p "$(dirname "$OUT")"
echo "AMC23 $TASK: model=$MODEL tag=$TAG gpus=$GPUS nproc=$NPROC batch=$BATCH"
START=$(date +%s)

# The task yaml hardcodes max_gen_toks=8192. Qwen has a 32k context so this is
# fine, but a model whose context is also exactly 8192 trips lm_eval's
# "generation length < context length" assertion and the whole avg@16 produces
# nothing. This hook allows a per-model override.
# NOTE: pass it as an array. Written as ${GEN_KWARGS:+--gen_kwargs "$GEN_KWARGS"}
# it is re-split after expansion and any value containing a space falls apart.
GEN_KWARGS=${GEN_KWARGS:-}
gen_args=()
[[ -n "$GEN_KWARGS" ]] && gen_args=(--gen_kwargs "$GEN_KWARGS")

# MODEL_ARGS_EXTRA appends key=value pairs to lm_eval's --model_args, for
# overrides such as max_length. amc23_avg16 uses max_gen_toks=8192; a model whose
# max_position_embeddings is also 8192 makes lm_eval compute
# max_ctx_len = max_length - max_gen_toks = 0 and abort on an assertion.
# NOTE: do not put comments inside the continuation block below. The trailing
# `\` joins the comment onto the command line, everything after `#` -- including
# --model_args -- is swallowed, the command is truncated to `-m lm_eval`, and
# `bash -n` does not catch it.
CUDA_VISIBLE_DEVICES=$GPUS \
PYTHONPATH="$QAOPD_ROOT/third_party/edgerazor/src:${PYTHONPATH:-}" \
HF_ALLOW_CODE_EVAL=1 TOKENIZERS_PARALLELISM=false \
"$ACCELERATE" launch --num_processes="$NPROC" --main_process_port "$PORT" -m lm_eval \
  --model hf --model_args "pretrained=$MODEL,dtype=bfloat16,trust_remote_code=True${MODEL_ARGS_EXTRA:+,$MODEL_ARGS_EXTRA}" \
  --include_path "$QAOPD_ROOT/eval_tasks" \
  --tasks "$TASK" --batch_size "$BATCH" --log_samples --trust_remote_code \
  --seed "$SEED" "${gen_args[@]}" \
  --output_path "$OUT" > "$OUT.log" 2>&1
RC=$?
ELAPSED=$(( $(date +%s) - START ))

R=$(find "$OUT" -name 'results_*.json' 2>/dev/null | sort | tail -1)
if [[ $RC -ne 0 || -z "$R" ]]; then
    echo "AMC_FAILED tag=$TAG rc=$RC after ${ELAPSED}s -- see $OUT.log" >&2
    tail -20 "$OUT.log" >&2
    exit 1
fi
"$PY" - "$R" "$TAG" "$ELAPSED" "$TASK" <<'PY'
import json, sys
task = sys.argv[4]
res = json.load(open(sys.argv[1]))["results"][task]
acc = res["exact_match,none"] * 100
se = res.get("exact_match_stderr,none", float("nan")) * 100
n = 640 if task.endswith("avg16") else 40
print(f"AMC_{task.upper()} tag={sys.argv[2]} acc={acc:.2f} stderr={se:.2f} "
      f"n={n} elapsed={sys.argv[3]}s")
PY
}
