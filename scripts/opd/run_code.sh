#!/usr/bin/env bash
# Phase 2: code OPD on top of the phase-1 mathematics checkpoint.
# phase (1) winner (UNI_balanced80 step80).
# Merges what used to be phase2(100) + phase3(20) + phase5(5) into one run;
# ckpt every 10 so the early-optimum rule can be applied by scanning.
set -uo pipefail
QAOPD_ROOT="${QAOPD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ROOT="${ROOT:-$QAOPD_ROOT}"
# conda is optional: the project installs into venvs/ (docs/INSTALL.md). This was
# an unconditional source+activate of paths that exist only on the original host.
CONDA_SH=${CONDA_SH:-}   # optional; the release uses venvs, not conda
if [[ -n "${VERL_CONDA_ENV:-}" ]]; then
    # shellcheck disable=SC1090
    source "$CONDA_SH" && conda activate "$VERL_CONDA_ENV"
fi
python -c "import verl, edgerazor, vllm" || {
    echo "python on PATH ($(command -v python)) lacks verl/edgerazor/vllm." >&2
    echo "Activate the training env: export PATH=<repo>/venvs/qaopd-train/bin:\$PATH" >&2
    exit 1; }
VERL="${VERL:-$QAOPD_ROOT/third_party/verl}"
export PYTHONPATH="$QAOPD_ROOT/third_party/edgerazor/src:$VERL:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-$ROOT/.cache/huggingface}
# The vendored trainer resolves data/models/output under these; point them at
# the repo layout (data/, models/, runs/) rather than its own default tree.
export VERL_ROOT="$VERL"
export DATA_ROOT="${DATA_ROOT:-$QAOPD_ROOT/data}"
export MODELS_DIR="${MODELS_DIR:-$QAOPD_ROOT/models}"
export RUNS_DIR="${RUNS_DIR:-$QAOPD_ROOT/runs}"
# Defaults to the phase (1) output produced by scripts/opd/run_math.sh in this repo.
if [[ -z "${STUDENT_MODEL:-}" ]]; then
    # Newest run directory, then its highest-numbered checkpoint. This is only a
    # convenience for a single-run tree: the phase-1 optimum is frequently well
    # before the end (docs/OPD.md), so a real selection should pass the chosen
    # checkpoint in STUDENT_MODEL rather than rely on this.
    P1_RUN=$(ls -dt "$RUNS_DIR"/*/checkpoints 2>/dev/null | head -1)
    # Sort on the step number itself. Splitting the path on "_" breaks as soon
    # as the run directory has an underscore in its name.
    P1=$(ls -d "$P1_RUN"/global_step_*/actor/huggingface 2>/dev/null \
         | awk -F'global_step_' '{split($2, a, "/"); print a[1], $0}' \
         | sort -k1,1n | tail -1 | cut -d' ' -f2-)
    [[ -n "$P1" ]] || { echo "no phase-1 checkpoint under $RUNS_DIR; run scripts/opd/run_math.sh first or set STUDENT_MODEL" >&2; exit 1; }
    echo "no STUDENT_MODEL given; using newest phase-1 checkpoint $P1"
    export STUDENT_MODEL="$P1"
fi
# Same bit-width switch as scripts/opd/run_math.sh: this phase funnels into the same
# run_opd_qad_w279a8_qwen3_06b.sh, so QAT_CONFIG + EDGERAZOR_QUANT_MODE control it.
source "$QAOPD_ROOT/scripts/lib/bitwidth.sh"; bitwidth_setup || exit 1

# Was a hard 2-GPU assignment; kept as the default so the recipe reproduces, but
# overridable to scale out. run_code_opd_search checks PPO_MINI_BATCH_SIZE*ROLLOUT_N
# divides by STUDENT_NGPUS and refuses otherwise.
export TOTAL_GPUS_PER_NODE=${TOTAL_GPUS_PER_NODE:-2}
export STUDENT_NGPUS=${STUDENT_NGPUS:-1}
export TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-1}
export TEACHER_TP_SIZE=${TEACHER_TP_SIZE:-1}
export WANDB_MODE=offline
# Was a hard assignment, so a caller could not name the run (and could not know
# where the checkpoints would land without parsing the log).
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-UNI2_code_t17_$(date -u +%Y%m%d_%H%M%S)}
export OUTPUT_DIR=${OUTPUT_DIR:-$RUNS_DIR/$EXPERIMENT_NAME}
echo "phase2 code from: $STUDENT_MODEL"
# Keep every save: the optimum on this phase is often early (step 20-40), and a
# retention cap can delete it before it is scanned. The cap is overridden after
# the variant block via a trainer.* argument.
KEEP=${MAX_ACTOR_CKPTS_TO_KEEP:-$(( ${STEPS:-140} / ${SAVE_FREQ:-20} + 1 ))}
# Two code pools ship. The default is the MBPP official train split;
# CODE_VARIANT=kodcode_k1_t17_strict_lr3e6_s30 takes the wider KodCode pool
# instead (data/README.md builds it). Either way the 448 MBPP test rows stay
# held out and are used for evaluation only.
CODE_VARIANT=${CODE_VARIANT:-mbpp_official_k1_t17_strict_lr3e6_s30}
echo "phase2 code variant: $CODE_VARIANT"
bash "$QAOPD_ROOT/opd/launch/run_code_opd_search_w279a8kv16.sh" \
    "$CODE_VARIANT" trainer.total_training_steps="${STEPS:-140}" trainer.save_freq="${SAVE_FREQ:-20}" \
    trainer.max_actor_ckpt_to_keep="$KEEP"
rc=$?
if [[ $rc -ne 0 ]]; then
    echo "UNI2_CODE_FAILED rc=$rc" >&2
    exit $rc
fi
echo "UNI2_CODE_DONE"
