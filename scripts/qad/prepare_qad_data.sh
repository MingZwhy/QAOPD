#!/usr/bin/env bash
# Build the QAD distillation corpus, then lay it out the way the trainer wants.
#
# The corpus is EdgeRazor's, not ours: the four jsonl files below are produced
# by upstream's own builder in
# third_party/edgerazor/example/data-for-qllm/src/data_prepare.sh, from public
# HuggingFace datasets. Nothing is redistributed here.
#
#   task_0.2M_instruct.jsonl     mixed downstream QA  ~0.24M rows
#   am_1.4M_instruct.jsonl       AM-DeepSeek-R1       ~1.40M rows
#   tulu_0.6M_instruct.jsonl     tulu-v3.1 mix        ~0.61M rows
#
# task_0.2M is exposed six times and am_1.4M twice, as
# task_0.2M_instruct_x{2..6}.jsonl and am_1.4M_instruct_x2.jsonl. These are
# symlinks to the one physical file: the trainer opens each name separately, so
# repeating the link repeats the exposure without costing the disk.
#
# ii_gen_1.4M is deliberately not in the default set. It is the weakest source
# for code and mathematics of the four (9.5% / 14.2%, against AM's 30.5% /
# 38.7%) and it was the largest block in the original mixture. DATASETS can
# still name it, and ii_7M, if you want to reproduce that older mix.
#
# The build runs in a THROWAWAY venv on purpose: it installs its own datasets
# release, and doing that inside venvs/qaopd-{train,eval} would rewrite a
# pinned environment.
#
# Expect tens of GB and several hours for the full mix. DATASETS=task_0.2M
# builds only the small one, which is enough to smoke-test the QAD launcher.
#
# Usage:
#   WORKSPACE=/path/to/qad_workspace bash scripts/qad/prepare_qad_data.sh
#   DATASETS=task_0.2M WORKSPACE=... bash scripts/qad/prepare_qad_data.sh
set -uo pipefail
QAOPD_ROOT="${QAOPD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

WORKSPACE=${WORKSPACE:-$QAOPD_ROOT/runs/qad_workspace}
DATA_DIR=${DATA_DIR:-$WORKSPACE/EdgeRazor-QLLM/data}
TASK_UPSAMPLE=${TASK_UPSAMPLE:-${UPSAMPLE:-6}}
AM_UPSAMPLE=${AM_UPSAMPLE:-2}
PREP_VENV=${PREP_VENV:-$WORKSPACE/.dataprep-venv}
PYTHON=${PYTHON:-python3}

ER_DATA_SRC="$QAOPD_ROOT/third_party/edgerazor/example/data-for-qllm/src"
[[ -f "$ER_DATA_SRC/data_prepare.sh" ]] || {
    echo "missing $ER_DATA_SRC/data_prepare.sh -- run setup/bootstrap.sh first" >&2
    exit 1; }

DEFAULT="task_0.2M am_1.4M tulu_0.6M"
DATASETS=${DATASETS:-$DEFAULT}

declare -A FILE=(
    [task_0.2M]=task_0.2M_instruct.jsonl
    [tulu_0.6M]=tulu_0.6M_instruct.jsonl
    [am_1.4M]=am_1.4M_instruct.jsonl
    [ii_gen_1.4M]=ii_gen_1.4M_instruct.jsonl
    [ii_7M]=ii_7M_instruct.jsonl
)
for d in $DATASETS; do
    [[ -n "${FILE[$d]:-}" ]] || {
        echo "unknown dataset '$d' (known: ${!FILE[*]})" >&2; exit 2; }
done

mkdir -p "$DATA_DIR"
echo "=== QAD corpus"
echo "    datasets:  $DATASETS"
echo "    data dir:  $DATA_DIR"
echo "    prep venv: $PREP_VENV"

if [[ ! -x "$PREP_VENV/bin/python" ]]; then
    echo "=== creating the throwaway build venv"
    "$PYTHON" -m venv "$PREP_VENV" || exit 1
    "$PREP_VENV/bin/pip" install --quiet --upgrade pip
fi
export PATH="$PREP_VENV/bin:$PATH"

# Several of the sources task_0.2M draws on (hendrycks_ethics, super_glue,
# winogrande, social_i_qa) are still script-based datasets that datasets>=4
# refuses to run. The patch in third_party/patches/edgerazor-qaopd.patch gives
# data_prepare.sh a fallback onto each dataset's auto-converted parquet branch,
# so the builder works on a current datasets release. Fail early rather than
# part-way through a multi-hour build if that patch is not in.
grep -q '_load_with_parquet_fallback' "$ER_DATA_SRC/data_prepare.sh" || {
    echo "third_party/edgerazor is not patched -- run setup/bootstrap.sh first." >&2
    echo "Without the patch data_prepare.sh dies on the first script-based dataset." >&2
    exit 1; }

for d in $DATASETS; do
    f="${FILE[$d]}"
    if [[ -s "$DATA_DIR/$f" ]]; then
        echo "=== $f: already present ($(wc -l < "$DATA_DIR/$f") rows), skipping"
        continue
    fi
    echo "=== building $f"
    bash "$ER_DATA_SRC/data_prepare.sh" --data "$f" --data-dir "$DATA_DIR" || {
        echo "data_prepare.sh failed for $f" >&2; exit 1; }
    [[ -s "$DATA_DIR/$f" ]] || { echo "$f is empty after the build" >&2; exit 1; }
done

# Upsample by exposing a file under extra names. Symlinks, not copies.
upsample() {
    local base="$1" n="$2" stem="${1%.jsonl}"
    [[ -s "$DATA_DIR/$base" ]] || return 0
    (( n > 1 )) || return 0
    echo "=== upsampling $base x$n (symlinks, not copies)"
    for i in $(seq 2 "$n"); do
        ln -sfn "$base" "$DATA_DIR/${stem}_x$i.jsonl"
    done
}
upsample task_0.2M_instruct.jsonl "$TASK_UPSAMPLE"
upsample am_1.4M_instruct.jsonl   "$AM_UPSAMPLE"

echo
echo "=== contents of $DATA_DIR"
for f in "$DATA_DIR"/*.jsonl; do
    [[ -e "$f" ]] || continue
    if [[ -L "$f" ]]; then
        printf '  %-34s -> %s\n' "$(basename "$f")" "$(readlink "$f")"
    else
        printf '  %-34s %9s rows  %6s\n' "$(basename "$f")" \
            "$(wc -l < "$f")" "$(du -h "$f" | cut -f1)"
    fi
done
echo
echo "QAD_DATA_READY $DATA_DIR"
echo "next: COMBO=<combo> MODEL=models/Qwen3-1.7B WORKSPACE=$WORKSPACE \\"
echo "        bash scripts/qad/run_qad.sh"
