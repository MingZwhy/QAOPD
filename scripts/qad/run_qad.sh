#!/usr/bin/env bash
# EdgeRazor QAD (quantization-aware distillation) -- produces the latent QAT
# checkpoint that OPD starts from.
#
# This is the *other* trainer: QAD runs in EdgeRazor's own DeepSpeed script
# (third_party/edgerazor/example/edgerazor-llm/src/main.py), not in verl. verl only
# re-applies the same quantization later, during OPD.
#
# Deviations from the paper recipe, all deliberate and all logged by the script:
#   - sdpa instead of flash_attention_2 (no sm_120 wheel; scripts/lib/attn.sh)
#   - a data subset rather than the full ~11M corpus, and no offline teacher
#     regeneration pass -- see docs/QAD.md
#   - dense save_steps, because the plan is to take a good intermediate
#     checkpoint rather than finish two epochs
#
# Usage:
#   COMBO=qwen3_1_7b_w2.79 MODEL=models/Qwen3-1.7B NGPU=8 \
#     bash scripts/qad/run_qad.sh
set -uo pipefail
{
QAOPD_ROOT="${QAOPD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$QAOPD_ROOT" || exit 1

COMBO=${COMBO:?set COMBO, e.g. qwen3_1_7b_w2.79 (a stem under configs/qad/)}
MODEL=${MODEL:?set MODEL to the FP base model dir (teacher and student both)}
WORKSPACE=${WORKSPACE:-$QAOPD_ROOT/runs/qad_workspace}
NGPU=${NGPU:-8}
GPUS=${GPUS:-$(seq -s, 0 $((NGPU - 1)))}
# QAD runs in the EVALUATION env, not the training one. It is a DeepSpeed job,
# and deepspeed and bitsandbytes are pinned in requirements-eval.lock.txt; the
# training env carries vLLM instead and has neither. QAD_ENV overrides it.
E=${QAD_ENV:-$QAOPD_ROOT/venvs/qaopd-eval}
PY=$E/bin/python
ER_SRC=$QAOPD_ROOT/third_party/edgerazor/example/edgerazor-llm/src

export ER_TEACHER=${ER_TEACHER:-$(cd "$MODEL" && pwd)}
export ER_STUDENT=${ER_STUDENT:-$ER_TEACHER}
# The corpus: the downstream QA set six times over, AM twice, and tulu once.
# 4,868,610 rows, so one epoch is 6,339 steps at 768 per step -- the round
# 6,400-step budgets below are one epoch each.
#
# AM carries by far the most code and mathematics of the available sources
# (30.5% / 38.7%, against 9.5% / 14.2% for Infinity-Instruct-Gen), so doubling
# it and dropping Gen lifts the mixture to 20.8% code / 25.1% mathematics from
# 14.7% / 18.0%, at a smaller corpus. Build it with
# scripts/qad/prepare_qad_data.sh, which creates the repeat names as symlinks to
# one file so the exposure repeats without the disk cost.
if [[ -z "${ER_DATASETS:-}" ]]; then
    ER_DATASETS=task_0.2M_instruct.jsonl
    for i in 2 3 4 5 6; do ER_DATASETS=$ER_DATASETS,task_0.2M_instruct_x$i.jsonl; done
    ER_DATASETS=$ER_DATASETS,am_1.4M_instruct.jsonl,am_1.4M_instruct_x2.jsonl
    ER_DATASETS=$ER_DATASETS,tulu_0.6M_instruct.jsonl
fi
export ER_DATASETS
export ER_DS_PATH=${ER_DS_PATH:-$ER_SRC/ds_z3_config_qwen3.json}

# Per-arm budgets, in whole epochs of the corpus above. Each was chosen so the
# checkpoint this project delivered for that arm falls inside the range with
# room to scan around it; see docs/QAD.md.
case "$COMBO" in
    qwen3_06b_*)   DEF_STEPS=12800 ;;   # 2 epochs
    qwen3_1_7b_w2.79) DEF_STEPS=6400 ;; # 1 epoch
    qwen3_1_7b_w1.88) DEF_STEPS=12800 ;;# 2 epochs
    qwen3_4b_w2.79)   DEF_STEPS=9600 ;; # 1.5 epochs
    qwen3_4b_w1.88)   DEF_STEPS=6400 ;; # 1 epoch
    *)             DEF_STEPS=6400 ;;
esac
export ER_STEPS=${ER_STEPS:-$DEF_STEPS}

# A 4B checkpoint is ~31 GB against ~13 GB for 1.7B, so the dense every-100
# policy needs a couple of TB on the larger model. Halve the frequency there and
# cap what stays on local disk.
case "$COMBO" in
    qwen3_4b_*) DEF_SAVE=200; DEF_KEEP=10 ;;
    *)          DEF_SAVE=100; DEF_KEEP=0  ;;   # 0 = keep every checkpoint
esac
export ER_SAVE_STEPS=${ER_SAVE_STEPS:-$DEF_SAVE}
export ER_SAVE_TOTAL_LIMIT=${ER_SAVE_TOTAL_LIMIT:-$DEF_KEEP}

# 768 sequences per step on every arm. Only the split between per-device batch
# and accumulation differs, and that is mathematically the same optimizer step,
# so the arms stay comparable.
#
# Fewer, larger micro-steps are faster whenever they fit: on Qwen3-4B, 6 x 16
# beat 3 x 32 by 19%, and on Qwen3-0.6B, 12 x 8 beats 6 x 16 by 17% (15.5 s
# against 18.6 s per step, measured on 72 GB cards). The ceiling is the KD
# logits tensor, bs x seq x vocab x 4 B: 0.6B at bs 24 asks for another 7 GiB
# and dies, and 1.7B and 4B already peak at 42.6 and 52.6 of 71.1 GiB at bs 6.
case "$COMBO" in
    qwen3_06b_*) DEF_BS=12; DEF_ACC=8  ;;
    *)           DEF_BS=6;  DEF_ACC=16 ;;
esac
export ER_PER_DEVICE_BS=${ER_PER_DEVICE_BS:-$DEF_BS}
export ER_GRAD_ACC=${ER_GRAD_ACC:-$DEF_ACC}
export ER_MAX_SEQ_LEN=${ER_MAX_SEQ_LEN:-1024}
# ER_STEPS is the real stopping condition; the epoch cap only has to be large
# enough not to cut a run short.
export ER_EPOCH=${ER_EPOCH:-4}
export ER_LR=${ER_LR:-2e-5}
# Keep this constant. Extending a constant_with_warmup run is free -- warmup is
# long past and the rate never changes -- whereas any cosine variant derives its
# whole curve from the total step count, so a later extension would not be a
# continuation of the same schedule. The selection method here is dense
# checkpoints plus a scan, not a decay to the endpoint.
export ER_SCHED=${ER_SCHED:-constant_with_warmup}
# transformers' get_scheduler returns early for constant_with_warmup and never
# looks at lr_scheduler_kwargs; every other type goes down the generic path and
# chokes on the min_lr this trainer always passes. cosine_with_min_lr accepts it.
case "$ER_SCHED" in
    constant_with_warmup|cosine_with_min_lr) ;;
    *) echo "ER_SCHED=$ER_SCHED will fail: the trainer always passes min_lr," >&2
       echo "which only constant_with_warmup and cosine_with_min_lr tolerate." >&2
       exit 2 ;;
esac
export ER_TAG=${ER_TAG:-$COMBO}
# Read at runtime by main.py (the wired config.py covers everything else).
# ER_RESUME=1 continues from the newest checkpoint, or give a checkpoint dir.
# Extending a run this way is safe: the schedule is constant, so the rate after
# the resume is the rate before it. Keep ER_SAVE_TOTAL_LIMIT=0 when you do, or
# upstream's cap silently deletes the early checkpoints -- which are often the
# ones a scan picks.
export ER_RESUME=${ER_RESUME:-}

# EdgeRazor's config.py hardcodes flash_attention_2; transformers raises rather
# than falling back when it is missing, several minutes into model load.
source "$QAOPD_ROOT/scripts/lib/attn.sh"
PATH="$E/bin:$PATH" attn_pick
export ER_ATTN=${ER_ATTN:-$ATTN_IMPL}

for m in deepspeed bitsandbytes edgerazor; do
    "$PY" -c "import $m" 2>/dev/null || { echo "$E is missing $m" >&2; exit 1; }
done

DATA_DIR=$WORKSPACE/EdgeRazor-QLLM/data
IFS=, read -ra WANT <<< "$ER_DATASETS"
for f in "${WANT[@]}"; do
    [[ -e "$DATA_DIR/$f" ]] || {
        echo "missing $DATA_DIR/$f" >&2
        echo "build the corpus first: WORKSPACE=$WORKSPACE bash scripts/qad/prepare_qad_data.sh" >&2
        exit 1; }
done

echo "=== wiring $COMBO"
PY_BIN=$PY bash "$QAOPD_ROOT/tools/wire_edgerazor_qat.sh" "$COMBO" "$WORKSPACE" || exit 1

# Post-condition on the wiring. config.py is a working-tree file that keeps
# whatever the previous run left in it and clears nothing, so any field the
# wiring does not write survives into the next run -- a stale lr_scheduler
# decays a fresh arm towards zero LR and makes its checkpoints incomparable
# with nothing in the log to say so. The defaults above always set the schedule
# for exactly that reason; this re-reads the file to confirm it landed.
CFG=$QAOPD_ROOT/third_party/edgerazor/example/edgerazor-llm/src/config.py
CLS=$(grep -oE 'EdgeRazorTrainConfigFor[A-Za-z0-9_]+' "$ER_SRC/main.py" | tail -1)
BODY=$(awk -v c="class $CLS:" '$0 ~ c {f=1; next} /^class /{f=0} f' "$CFG")
# config_path is the one that matters most: the widths differ by a single field
# (w_mixed_precision_prop, 0.50 against 0.125) and training the wrong one
# produces a checkpoint that looks healthy and is the wrong model.
for want in "lr_scheduler.*$ER_SCHED" "lr *= *$ER_LR" \
            "per_device_bs *= *$ER_PER_DEVICE_BS" "grad_acc_steps *= *$ER_GRAD_ACC" \
            "config_path.*/$COMBO\.yaml"; do
    grep -qE "$want" <<<"$BODY" || {
        echo "wiring assertion failed: expected $want in $CLS" >&2
        grep -E "lr|sched|per_device|grad_acc|config_path" <<<"$BODY" >&2
        exit 1; }
done
echo "    wiring asserted: $COMBO.yaml / $ER_SCHED / lr $ER_LR" \
     "/ bs $ER_PER_DEVICE_BS x acc $ER_GRAD_ACC"

OUT=${OUT:-$WORKSPACE/EdgeRazor-QLLM/train}
LOG=${LOG:-/tmp/qaopd_logs/qad_$ER_TAG.log}
mkdir -p "$(dirname "$LOG")" "$OUT"

echo "=== launching on GPUs $GPUS -> $OUT"
echo "    log: $LOG"

# Heartbeat, so a scanner on another node can tell whether training is still
# alive. scan_qad_ckpts.sh used to answer that with pgrep, which only ever works
# when the scanner shares a PID namespace with the trainer; once the two are
# split across nodes pgrep finds nothing and the scan quits on its first pass.
# A file whose mtime is refreshed beats a plain marker because a killed trainer
# leaves the marker behind forever.
HEARTBEAT=${HEARTBEAT:-$OUT/.training_alive}
touch "$HEARTBEAT"
( while :; do touch "$HEARTBEAT"; sleep 60; done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null; rm -f "$HEARTBEAT"' EXIT

cd "$ER_SRC" || exit 1
# --include rather than --num_gpus, and no outer CUDA_VISIBLE_DEVICES.
#
# DeepSpeed's launcher *discards* an inherited CUDA_VISIBLE_DEVICES as soon as any
# of --include/--exclude/--num_gpus/--num_nodes is passed -- it says so and then
# prints "Setting CUDA_VISIBLE_DEVICES=0,..,N-1". So the old
# `CUDA_VISIBLE_DEVICES=$GPUS deepspeed --num_gpus=$NGPU` put every rank on
# physical GPUs 0..N-1 no matter what GPUS said. That stays invisible while all
# eight cards are free and GPUS is 0-7 anyway; it surfaces the moment the run
# has to avoid some cards, e.g. when another container on the same host holds
# 0-7 and only 8-15 are yours. Verified by probing torch.cuda uuid per rank.
PYTHONPATH="$QAOPD_ROOT/third_party/edgerazor/src:${PYTHONPATH:-}" \
TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 \
"$E/bin/deepspeed" --include "localhost:$GPUS" main.py > "$LOG" 2>&1
RC=$?
kill $HB_PID 2>/dev/null; rm -f "$HEARTBEAT"
if [[ $RC -ne 0 ]]; then
    echo "QAD_FAILED rc=$RC -- see $LOG" >&2
    grep -aoE "[A-Za-z]*(Error|Exception): .{0,160}" "$LOG" | sort -u | tail -5 >&2
    exit 1
fi
echo "QAD_DONE tag=$ER_TAG out=$OUT"
}
