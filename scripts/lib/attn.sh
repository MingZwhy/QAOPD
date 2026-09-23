#!/usr/bin/env bash
# verl defaults to attn_implementation=flash_attention_2, and transformers then
# raises PackageNotFoundError instead of falling back when flash-attn is absent.
# docs/INSTALL.md treats flash-attn as optional (it has no sm_120 wheel), so pick
# sdpa rather than dying several minutes into a run.
#
# Usage: source "$QAOPD_ROOT/scripts/lib/attn.sh"; attn_pick   # sets ATTN_IMPL
#
# The probe must be importlib.metadata, the same call transformers makes.
# `import flash_attn` is NOT a valid test: flashinfer-python ships a
# `flash_attn.cute` subpackage, so a bare import succeeds against an empty
# namespace package on an env with no flash-attn at all.

attn_pick() {
    if [[ -n "${ATTN_IMPL:-}" ]]; then
        export ATTN_IMPL
        return 0
    fi
    if python -c "import importlib.metadata as m; m.version('flash_attn')" 2>/dev/null; then
        ATTN_IMPL=flash_attention_2
    else
        ATTN_IMPL=sdpa
        echo "flash_attn not installed -> attn_implementation=sdpa"
    fi
    export ATTN_IMPL
}
