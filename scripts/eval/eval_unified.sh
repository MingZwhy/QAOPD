#!/usr/bin/env bash
# Evaluate a unified-math checkpoint on the three math benchmarks that the
# consolidation targets, under the delivery protocol:
#   GSM8K test 1319 (5-shot) | MATH-500 (4-shot) | AMC23 avg@16 (T0.7, 8k)
# AMC uses avg@16 -- greedy n=40 has no power below ~10% (see amc23_baseline.md).
set -uo pipefail
# Brace group so an edit mid-run cannot corrupt an in-flight job; see the note in
# scripts/eval/amc23_avg16_parallel.sh.
{
QAOPD_ROOT="${QAOPD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ROOT="${ROOT:-$QAOPD_ROOT}"
VERL="${VERL:-$QAOPD_ROOT/third_party/verl}"
# eval_tasks/*.yaml carry repo-root-relative data_files, and lm_eval resolves
# them against the CWD (see tools/fix_eval_task_paths.sh), so anchor it here.
cd "$QAOPD_ROOT" || exit 1
bash "$QAOPD_ROOT/tools/fix_eval_task_paths.sh" --check >/dev/null || {
    echo "eval task data_files are stale; run: bash tools/fix_eval_task_paths.sh" >&2; exit 1; }
# Evaluation env (lm_eval + EdgeRazor, see docs/INSTALL.md section 2). Falls back
# to whatever python/accelerate is on PATH so a fresh clone works with an
# activated conda env; EVAL_ENV pins a specific prefix.
E=${EVAL_ENV:-}
ACCELERATE=${ACCELERATE:-${E:+$E/bin/}accelerate}
PY=${PY:-${E:+$E/bin/}python}
command -v "$ACCELERATE" >/dev/null || { echo "accelerate not found; activate the eval env or set EVAL_ENV" >&2; exit 1; }
GPU=${GPU:-0}
# Bit width. Same switch as scripts/opd/run_math.sh. The label, the
# quant config and the tokenizer source all have to move together: exporting a
# W1.88 checkpoint through the W2.79 config produces a valid-looking model at the
# wrong bit width and no error (docs/CHECKPOINTS.md).
source "$QAOPD_ROOT/scripts/lib/bitwidth.sh"; bitwidth_setup || exit 1
MODEL_LABEL=${MODEL_LABEL:-${BW_LABEL}a8kv16}
# The exporter copies tokenizer assets (incl. chat_template.jinja) from the QAT
# source; the shipped warm checkpoints carry them.
export LM_EVAL_INCLUDE_PATH="${LM_EVAL_INCLUDE_PATH:-$QAOPD_ROOT/eval_tasks}"
export EVAL_TOKENIZER_SOURCE="${EVAL_TOKENIZER_SOURCE:-$QAOPD_ROOT/models/$BW_WARM}"
[[ -d "$EVAL_TOKENIZER_SOURCE" ]] || { echo "no tokenizer source at $EVAL_TOKENIZER_SOURCE" >&2; exit 1; }
# Usual call is RUN=<run-dir> STEP=<n>. CKPT= names a checkpoint directly, which
# is how the warm QAT baseline (no run dir, no global_step_*) gets evaluated.
if [[ -n "${CKPT:-}" ]]; then
    STEP=${STEP:-warm}
else
    RUN=${RUN:?}; STEP=${STEP:?}
    CKPT="$RUN/checkpoints/global_step_$STEP/actor/huggingface"
fi
# Single-file or sharded (model.safetensors.index.json + model-000N-of-000M).
# verl's OPD arms and the archived phase-1 checkpoints are sharded, so the
# single-file form alone rejected them with "no ckpt" before anything loaded.
# This is the third script with the same assumption -- see
# the two b1 ones, where it showed up as HumanEval silently skipping every point.
[[ -f "$CKPT/model.safetensors" || -f "$CKPT/model.safetensors.index.json" ]] \
    || { echo "no ckpt at $CKPT"; exit 1; }
# EXPORT_DIR reuses an export somebody else already produced (scan_qad_ckpts.sh
# exports once and then runs the QA suite and this script against the same
# directory). Without it every caller re-quantizes the same checkpoint.
EXPORT_DIR=${EXPORT_DIR:-}
if [[ -n "$EXPORT_DIR" ]]; then
    [[ -f "$EXPORT_DIR/model_$MODEL_LABEL/config.json" ]] || {
        echo "EXPORT_DIR has no model_$MODEL_LABEL/config.json: $EXPORT_DIR" >&2; exit 1; }
    OUT=$EXPORT_DIR
    CONVERT=0
    # Stale result files from an earlier eval of this same export would be picked
    # up by the find below and reported as this run's numbers.
    find "$OUT" -name 'results*.json' -delete 2>/dev/null
else
    OUT=$ROOT/experiments/opd_qad/unieval_${TAG:-x}_s$STEP
    CONVERT=1
    rm -rf "$OUT"
fi
# experiments/ is gitignored, so on a fresh clone this directory does not exist
# and the `> "$OUT.log"` redirect below fails before the eval ever starts -- the
# script then falls through to UNIEVAL_DONE having measured nothing.
mkdir -p "$(dirname "$OUT")" || exit 1
# GPU may be a comma-separated list; the two math benchmarks then run
# data-parallel. Single-GPU math is ~45 min per checkpoint on 1.7B, which is
# slower than QAD produces checkpoints (scripts/eval/scan_qad_ckpts.sh).
NPROC=$(awk -F, '{print NF}' <<<"$GPU")
# No port is chosen here on purpose. The export below runs for minutes before
# anything binds, so a port picked now would be stale by the time accelerate
# needs it; the inner script picks one immediately before its launch instead.
# PORT_BASE still forces a fixed port for callers that need a known one.
CUDA_VISIBLE_DEVICES=$GPU NUM_PROCESSES=$NPROC ACCELERATE_MAIN_PROCESS_PORT=${PORT_BASE:-} \
RUN_GSM8K=${RUN_GSM8K:-1} RUN_MATH500=${RUN_MATH500:-1} RUN_AMC23=0 RUN_FULL=0 RUN_CHAT=0 RUN_HUMANEVAL=0 RUN_IFEVAL=0 \
RUN_CONVERT=$CONVERT GSM_BATCH=16 MATH500_BATCH=16 \
OUTPUT_ROOT="$OUT" MODEL_LABEL="$MODEL_LABEL" QAT_CONFIG="$QAT_CONFIG" \
bash "$QAOPD_ROOT/opd/launch/evaluate_opd_qad_w279a8.sh" "$CKPT" "uni_${TAG:-x}_s$STEP" > "$OUT.log" 2>&1
# AMC avg@16 on the freshly exported quantized model. Single-GPU avg@16 takes
# ~6.7h; RUN_AMC=0 skips it so scripts/eval/amc23_avg16_parallel.sh
# can do it data-parallel instead.
M=$OUT/model_$MODEL_LABEL
if [[ "${RUN_AMC:-1}" == 1 && -f "$M/config.json" ]]; then
  source "$QAOPD_ROOT/scripts/lib/port.sh"
  AMC_PORT=${PORT_BASE:+$((PORT_BASE+10))}
  AMC_PORT=${AMC_PORT:-$(free_port)} || { echo "no free port for AMC" >&2; exit 3; }
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$QAOPD_ROOT/third_party/edgerazor/src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$ACCELERATE" launch --num_processes="$NPROC" --main_process_port "$AMC_PORT" -m lm_eval \
    --model hf --model_args "pretrained=$M,dtype=bfloat16,trust_remote_code=True" \
    --include_path "$QAOPD_ROOT/eval_tasks" \
    --tasks amc23_avg16 --batch_size 16 --log_samples --trust_remote_code \
    --output_path "$ROOT/experiments/opd_qad/amc23_avg16/uni_${TAG:-x}_s$STEP" >> "$OUT.log" 2>&1
fi
FOUND=0
WANT=0
[[ "${RUN_GSM8K:-1}" == 1 ]] && WANT=$((WANT+1))
[[ "${RUN_MATH500:-1}" == 1 ]] && WANT=$((WANT+1))
for b in gsm8k math500; do
  R=$(find "$OUT" -path "*$b*" -name 'results*.json' 2>/dev/null | head -1)
  [[ -n "$R" ]] && FOUND=$((FOUND+1)) && "$PY" -c "
import json
d=json.load(open('$R'))['results']; k=[x for x in d if '$b' in x][0]; m=d[k]

# NOTE: this used to read v = m.get(A) or m.get(B). When the score is exactly
#   0, \`0 or ...\` takes the falsy branch; if the second key is absent the result
#   is None and the following %.4f raises TypeError. A legitimate score of 0 was
#   therefore reported as NA and looked like a failed evaluation. Test with
#   \`is not None\`, not \`or\`.
v = m.get('exact_match,strict-match')
if v is None:
    v = m.get('exact_match,none')
if v is None:
    raise SystemExit('no exact_match key in %s' % '$R')
print('UNIEVAL ${TAG:-x} step$STEP $b: %.4f' % v)"
done
R=$(find "$ROOT/experiments/opd_qad/amc23_avg16/uni_${TAG:-x}_s$STEP" -name 'results_*.json' 2>/dev/null | head -1)
[[ -n "$R" ]] && "$PY" -c "
import json
m=json.load(open('$R'))['results']['amc23_avg16']
print('UNIEVAL ${TAG:-x} step$STEP amc23_avg16: %.4f' % m['exact_match,none'])"
# Print DONE only when both math benchmarks actually produced a results file.
# Otherwise a caller scanning for UNIEVAL_DONE (scripts/eval/scan_math_ckpts.sh)
# records a step as evaluated when nothing ran.
if (( FOUND < WANT )); then
    echo "UNIEVAL_FAILED step$STEP: $FOUND/$WANT result files -- see $OUT.log" >&2
    tail -20 "$OUT.log" 2>/dev/null >&2
    exit 1
fi
echo "UNIEVAL_DONE step$STEP"
}
