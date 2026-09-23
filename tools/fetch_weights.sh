#!/usr/bin/env bash
# Download a released checkpoint from the Hub into models/.
#
# Two kinds, and they are not interchangeable (docs/CHECKPOINTS.md):
#
#   *-QAOPD   deployment export, quantization baked in. Load it and evaluate.
#   *-QAD     latent QAT checkpoint, bf16 masters. Only a training start point.
#
# Usage:
#   bash tools/fetch_weights.sh                    # list what is available
#   bash tools/fetch_weights.sh qwen3_1_7b_w2.79   # the recovered checkpoint
#   bash tools/fetch_weights.sh qwen3_06b_w2.79 --qad
#   DEST=/somewhere bash tools/fetch_weights.sh <arm>
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OWNER=MingZwhy

declare -A MODEL=(
    [qwen3_06b_w2.79]=Qwen3-0.6B
    [qwen3_06b_w1.88]=Qwen3-0.6B
    [qwen3_1_7b_w2.79]=Qwen3-1.7B
    [qwen3_1_7b_w1.88]=Qwen3-1.7B
    [qwen3_4b_w2.79]=Qwen3-4B
    [qwen3_4b_w1.88]=Qwen3-4B
)
# Every arm publishes the QAD checkpoint its own OPD run started from, so
# stage 1 can be skipped entirely.
QAD_AVAILABLE="qwen3_06b_w2.79 qwen3_06b_w1.88 qwen3_1_7b_w2.79 qwen3_1_7b_w1.88 qwen3_4b_w2.79 qwen3_4b_w1.88"

usage() {
    echo "usage: bash tools/fetch_weights.sh <arm> [--qad]" >&2
    echo >&2
    echo "  arm                recovered            QAD start" >&2
    for a in qwen3_06b_w2.79 qwen3_06b_w1.88 qwen3_1_7b_w2.79 \
             qwen3_1_7b_w1.88 qwen3_4b_w2.79 qwen3_4b_w1.88; do
        w=${a##*_}
        q="--"
        [[ " $QAD_AVAILABLE " == *" $a "* ]] && q="${MODEL[$a]}-${w^^}-QAD"
        printf "  %-18s %-22s %s\n" "$a" "${MODEL[$a]}-${w^^}-QAOPD" "$q" >&2
    done
    exit 2
}

ARM=${1:-}; [[ -n "$ARM" ]] || usage
[[ -n "${MODEL[$ARM]:-}" ]] || { echo "unknown arm: $ARM" >&2; usage; }
KIND=QAOPD
if [[ "${2:-}" == "--qad" ]]; then
    [[ " $QAD_AVAILABLE " == *" $ARM "* ]] || {
        echo "no published QAD start for $ARM -- run stage 1 (docs/QAD.md)" >&2; exit 1; }
    KIND=QAD
fi

W=${ARM##*_}
REPO="$OWNER/${MODEL[$ARM]}-${W^^}-$KIND"
DEST="${DEST:-$HERE/models/${ARM}_${KIND,,}}"

command -v hf >/dev/null || { echo "hf CLI not found; pip install huggingface_hub" >&2; exit 1; }
[[ -n "${HF_ENDPOINT:-}" ]] &&
    echo "note: HF_ENDPOINT=$HF_ENDPOINT (a mirror); unset it if downloads fail" >&2
# The hub rate-limits anonymous IPs partway through a multi-GB download and the
# failure reads like a network error. A token makes it go away.
[[ -n "${HF_TOKEN:-}" ]] ||
    echo "note: HF_TOKEN is not set; the hub rate-limits anonymous downloads" >&2
[[ "${HF_HUB_OFFLINE:-0}" == "1" ]] &&
    { echo "HF_HUB_OFFLINE=1 is set; unset it to download" >&2; exit 1; }

mkdir -p "$(dirname "$DEST")"
echo "downloading $REPO -> $DEST"
if ! hf download "$REPO" --local-dir "$DEST"; then
    echo >&2
    echo "download failed. If that was a 429, set HF_TOKEN and re-run --" >&2
    echo "hf download resumes, so nothing already fetched is lost." >&2
    exit 1
fi

[[ -f "$DEST/model.safetensors" || -f "$DEST/model.safetensors.index.json" ]] || {
    echo "download incomplete: no weights at $DEST" >&2; exit 1; }

echo
if [[ "$KIND" == QAOPD ]]; then
    echo "done. evaluate it:"
    echo "  BITWIDTH=${W/w/w} CKPT=$DEST bash scripts/eval/eval_unified.sh"
else
    echo "done. use it as the OPD starting point:"
    echo "  BITWIDTH=$W STUDENT_MODEL=$DEST bash scripts/opd/run_math.sh"
fi
