#!/usr/bin/env bash
# Short-form QA suite on a deployment-export model, data-parallel over N GPUs.
#
# Why this exists: the other evaluations here measure mathematics or code, and
# "how much general ability did OPD cost" cannot be answered from GSM8K alone.
#
# Two task groups ship. The default, eval_tasks/qa9/, is the nine-benchmark
# average the paper reports as QA9. eval_tasks/edgerazor_qa/ is the twelve-task
# group EdgeRazor reports its Avg. over, for a like-for-like comparison against
# it; upstream's own copy cannot be used because two of its tasks resolve to hub
# datasets that still ship loading scripts, which datasets>=4 refuses, and one
# failure aborts the whole group.
#
# ⚠ MODEL must be a DEPLOYMENT EXPORT, not a latent QAT checkpoint -- a latent
# checkpoint loads fine and scores an unquantized model (docs/CHECKPOINTS.md).
#
# Usage:
#   MODEL=<exported model dir> TAG=w188_warm GPUS=0,1,2,3 \
#     bash scripts/eval/qa_suite.sh
# QA_TASKS overrides the task list (default: qa9).
set -uo pipefail
# Brace group so an edit mid-run cannot corrupt an in-flight job; see the note in
# scripts/eval/amc23_avg16_parallel.sh.
{
QAOPD_ROOT="${QAOPD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$QAOPD_ROOT" || exit 1

MODEL=${MODEL:?set MODEL to a deployment-export model dir}
TAG=${TAG:?set TAG}
GPUS=${GPUS:-0}
BATCH=${BATCH:-16}
QA_TASKS=${QA_TASKS:-qa9}
E=${EVAL_ENV:-$QAOPD_ROOT/venvs/qaopd-eval}
ACCELERATE=${ACCELERATE:-$E/bin/accelerate}
PY=${PY:-$E/bin/python}
OUT=${OUT:-$QAOPD_ROOT/experiments/opd_qad/qa_suite/$TAG}
# A constant here meant two QA runs on one host collided with each other, and
# either of them collided with a Ray worker when training shared the box.
source "$QAOPD_ROOT/scripts/lib/port.sh"
PORT=${PORT:-$(free_port)} || { echo "could not find a free port" >&2; exit 1; }
INCLUDE=${QA_INCLUDE_PATH:-$QAOPD_ROOT/eval_tasks}

[[ -f "$MODEL/config.json" ]] || { echo "no config.json at $MODEL" >&2; exit 1; }
[[ -d "$INCLUDE" ]] || { echo "no task dir at $INCLUDE" >&2; exit 1; }
command -v "$ACCELERATE" >/dev/null || { echo "accelerate not found at $ACCELERATE" >&2; exit 1; }

# Same hard stop as the AMC script: latent checkpoints keep bf16 masters
# (~125 unique values per 128-element block), exports bake the quantization in.
# EXPECT=fp is for deliberately scoring an unquantized reference (the FP base
# model as a ceiling); it has to be asked for by name so it cannot happen by
# accident, which is the whole failure mode docs/CHECKPOINTS.md warns about.
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
echo "QA suite: model=$MODEL tag=$TAG gpus=$GPUS nproc=$NPROC tasks=$QA_TASKS"

# Warm the dataset cache in ONE process before fanning out. mmlu alone is 57
# separate hub configs, so N ranks x M models resolving them at once gets the
# whole job 429'd by huggingface.co -- which is how this first failed. Cheap
# once cached; the parallel run below then goes fully offline so no rank can
# touch the network at all.
cat > /tmp/qa_prefetch_$$.py <<'PY'
import sys
from lm_eval.tasks import TaskManager, get_task_dict
tm = TaskManager(include_path=sys.argv[1])
names = sys.argv[2].split(",")
d = get_task_dict(names, tm)
def count(x):
    return sum(count(v) for v in x.values()) if isinstance(x, dict) else 1
print(f"   dataset cache warm: {count(d)} tasks under {names}")
PY
prefetch() {
    env "$@" PYTHONPATH="$QAOPD_ROOT/third_party/edgerazor/src:${PYTHONPATH:-}" \
        "$PY" /tmp/qa_prefetch_$$.py "$INCLUDE" "$QA_TASKS" 2>/tmp/qa_prefetch_$$.err
}
# Offline first: when several tags launch together the cache is usually already
# warm, and going offline keeps them from stampeding the hub API for nothing.
prefetch HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 || prefetch HF_DATASETS_OFFLINE=0 || {
    echo "dataset prefetch failed" >&2; tail -5 /tmp/qa_prefetch_$$.err >&2
    rm -f /tmp/qa_prefetch_$$.py /tmp/qa_prefetch_$$.err; exit 1; }
rm -f /tmp/qa_prefetch_$$.py /tmp/qa_prefetch_$$.err

START=$(date +%s)

CUDA_VISIBLE_DEVICES=$GPUS \
PYTHONPATH="$QAOPD_ROOT/third_party/edgerazor/src:${PYTHONPATH:-}" \
HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_EVALUATE_OFFLINE=1 \
HF_ALLOW_CODE_EVAL=1 TOKENIZERS_PARALLELISM=false \
"$ACCELERATE" launch --num_processes="$NPROC" --main_process_port "$PORT" -m lm_eval \
  --model hf --model_args "pretrained=$MODEL,dtype=bfloat16,trust_remote_code=True" \
  --include_path "$INCLUDE" \
  --tasks "$QA_TASKS" --batch_size "$BATCH" --trust_remote_code \
  --confirm_run_unsafe_code \
  --output_path "$OUT" > "$OUT.log" 2>&1
RC=$?
ELAPSED=$(( $(date +%s) - START ))

R=$(find "$OUT" -name 'results_*.json' 2>/dev/null | sort | tail -1)
if [[ $RC -ne 0 || -z "$R" ]]; then
    echo "QA_FAILED tag=$TAG rc=$RC after ${ELAPSED}s -- see $OUT.log" >&2
    grep -aoE "(ValueError|RuntimeError|OSError|ConnectionError|AssertionError): .{0,160}" \
        "$OUT.log" 2>/dev/null | sort -u | head -5 >&2
    tail -20 "$OUT.log" >&2
    exit 1
fi
"$PY" - "$R" "$TAG" "$ELAPSED" "$INCLUDE" "$QA_TASKS" <<'PY'
import json, pathlib, re, sys
res = json.load(open(sys.argv[1]))["results"]
tag, elapsed, include, spec = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]

# The member list of the requested group, so the average counts each benchmark
# once. mmlu expands into 57 subject rows, so a mean over everything lm_eval
# printed weights it 57/65 instead of 1/9 and reports several points low.
def members(spec):
    out = []
    for name in spec.split(","):
        name = name.strip()
        y = pathlib.Path(include) / name / f"{name}.yaml"
        if not y.is_file():
            out.append(name)
            continue
        body, seen_task = [], False
        for line in y.read_text().splitlines():
            if re.match(r"^task:\s*$", line):
                seen_task = True
            elif seen_task:
                m = re.match(r"^\s+-\s+(\S+)", line)
                if m:
                    body.append(m.group(1))
                elif line.strip() and not line.lstrip().startswith("#"):
                    break
        out += body or [name]
    return out

def pick(m):
    for k in ("acc_norm,none", "acc,none", "exact_match,strict-match",
              "exact_match,flexible-extract", "exact_match,none"):
        if k in m:
            return m[k] * 100, k.split(",")[0]
    return None, None

for task in sorted(res):
    v, metric = pick(res[task])
    if v is not None:
        print(f"QA {tag} {task}: {v:.2f} ({metric})")

wanted = members(spec)
vals, missing = [], []
for t in wanted:
    v, _ = pick(res.get(t, {}))
    (vals.append(v) if v is not None else missing.append(t))
if missing:
    print(f"QA_AVG tag={tag} INCOMPLETE missing={','.join(missing)}")
elif vals:
    print(f"QA_AVG tag={tag} avg={sum(vals)/len(vals):.2f} "
          f"over={len(vals)} elapsed={elapsed}s")
PY
}
