#!/usr/bin/env bash
# Single place that maps a bit width to the two EdgeRazor knobs every train/eval
# script in this repo ends up passing to run_opd_qad_w279a8_qwen3_06b.sh.
#
# Why this exists: docs/CHECKPOINTS.md warns that the quant *config* and the quant
# *mode* are separate settings, and that changing only one of them trains or
# evaluates a different model with no error anywhere. Deriving both from one
# BITWIDTH variable, then re-checking them, makes that failure ring.
#
# Usage:  source "$QAOPD_ROOT/scripts/lib/bitwidth.sh"; bitwidth_setup
# Reads:  BITWIDTH (w2.79 | w1.88, default w2.79), QAOPD_ROOT
# Sets:   QAT_CONFIG, EDGERAZOR_QUANT_MODE (exported, honouring pre-set values),
#         BW_TAG (w2_79 | w1_88), BW_LABEL (w279 | w188), BW_WARM (warm ckpt dir name)

bitwidth_setup() {
    BITWIDTH=${BITWIDTH:-w2.79}
    case "$BITWIDTH" in
        w2.79) BW_TAG=w2_79; BW_LABEL=w279; BW_WARM=w279_step10000 ;;
        w1.88) BW_TAG=w1_88; BW_LABEL=w188; BW_WARM=w188_step9000 ;;
        *) echo "unknown BITWIDTH=$BITWIDTH (want w2.79 or w1.88)" >&2; return 1 ;;
    esac
    local cfg_dir="${QAOPD_ROOT:?QAOPD_ROOT must be set before bitwidth_setup}/configs/opd"
    export QAT_CONFIG=${QAT_CONFIG:-$cfg_dir/edgerazor_${BW_TAG}a8_qwen3.yaml}
    export EDGERAZOR_QUANT_MODE=${EDGERAZOR_QUANT_MODE:-${BW_TAG}a8kv8_embint4_qwen3}
    if [[ ! -f "$QAT_CONFIG" ]]; then
        echo "no quant config at $QAT_CONFIG" >&2; return 1
    fi
    # Catches a half-applied override, e.g. exporting EDGERAZOR_QUANT_MODE for
    # W1.88 while QAT_CONFIG still points at the W2.79 yaml.
    if [[ "$QAT_CONFIG" != *"$BW_TAG"* || "$EDGERAZOR_QUANT_MODE" != "$BW_TAG"* ]]; then
        echo "bit-width mismatch: BITWIDTH=$BITWIDTH but" >&2
        echo "  QAT_CONFIG=$QAT_CONFIG" >&2
        echo "  EDGERAZOR_QUANT_MODE=$EDGERAZOR_QUANT_MODE" >&2
        return 1
    fi
    return 0
}
