#!/usr/bin/env bash
# Prepare a working QAOPD checkout: submodules, third-party patches, venvs.
# Safe to re-run; each step is skipped if already satisfied.
set -euo pipefail

QAOPD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$QAOPD_ROOT"

VENV_DIR="${VENV_DIR:-$QAOPD_ROOT/venvs}"
PYTHON="${PYTHON:-python3}"
DO_VENVS="${DO_VENVS:-1}"

step() { printf '\n== %s\n' "$*"; }
die()  { printf 'bootstrap failed: %s\n' "$*" >&2; exit 1; }

# --------------------------------------------------------------- submodules
step "submodules"
# Not --recursive: EdgeRazor records example/medical_vit/vit.cpp as a gitlink
# but ships no .gitmodules, so recursing there aborts the whole update and
# leaves verl on its default branch instead of the pinned commit.
if [[ ! -f third_party/verl/setup.py || ! -f third_party/edgerazor/pyproject.toml ]]; then
  git submodule update --init || die "could not init submodules"
fi
for sm in third_party/verl third_party/edgerazor; do
  printf '  %-26s %s\n' "$sm" "$(git -C "$sm" rev-parse --short HEAD)"
done

# ------------------------------------------------------------------ patches
# Applied with `git apply`; `--check -R` tells us whether the patch is already in.
step "third-party patches"
apply_patch() {
  local sm="$1" patch="$QAOPD_ROOT/$2"
  [[ -f "$patch" ]] || die "missing patch $patch"
  # --whitespace=nowarn: these patches are generated from a known-good tree, and
  # the trailing-blank-line notices they produce read like failures.
  if git -C "$sm" apply --check -R "$patch" 2>/dev/null; then
    echo "  $sm: already applied"
  elif git -C "$sm" apply --check --whitespace=nowarn "$patch" 2>/dev/null; then
    git -C "$sm" apply --whitespace=nowarn "$patch"
    echo "  $sm: applied"
  else
    die "$sm: patch neither applies nor is already applied -- is the submodule at the pinned commit?"
  fi
}
apply_patch third_party/verl      third_party/patches/verl-qaopd.patch
apply_patch third_party/edgerazor third_party/patches/edgerazor-qaopd.patch

# --------------------------------------------------------------------- venvs
# Two environments on purpose: the training stack pins vLLM, the evaluation
# stack pins lm-eval, and their dependency closures conflict. They also need
# different interpreters -- 3.12 and 3.10 -- so each is built separately.
CUDA_INDEX=https://download.pytorch.org/whl/cu128
PYPI=${PYPI:-https://pypi.org/simple}

# torch and friends come from the cu128 index, never from PyPI: the lock files
# deliberately exclude them because a `+cu128` local version does not exist on
# PyPI and installing the plain PyPI wheel swaps the whole CUDA stack silently.
TORCH_TRAIN="torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0"
TORCH_EVAL="torch==2.9.1"

# Build one environment and install its editable packages. Returns non-zero
# rather than exiting, so a failure on one environment does not strand the
# other half-finished.
make_env() {
  local name="$1" py="$2" torch_spec="$3"; shift 3
  local target="$VENV_DIR/qaopd-$name"
  local lock="$QAOPD_ROOT/setup/requirements-$name.lock.txt"
  [[ -f "$lock" ]] || { echo "  missing $lock" >&2; return 1; }
  if [[ -x "$target/bin/pip" ]]; then
    echo "  qaopd-$name: exists, skipping"
  else
    echo "  qaopd-$name: creating with $py"
    rm -rf "$target"
    "$py" -m venv "$target" || { echo "  could not create $target with $py" >&2; return 1; }
    "$target/bin/pip" install --quiet --upgrade pip
    # shellcheck disable=SC2086
    "$target/bin/pip" install --quiet $torch_spec --index-url "$CUDA_INDEX" ||
      { echo "  torch install failed for qaopd-$name" >&2; return 1; }
    # --no-deps is required, not an optimisation: the lock is a freeze of a
    # working closure, not a spec pip can solve (see the file's own header).
    # PIP_INDEX_URL is exported rather than passed as -i because pip's
    # build-isolation subprocesses re-read pip.conf and would pick up a mirror.
    PIP_INDEX_URL="$PYPI" "$target/bin/pip" install --quiet --no-deps -r "$lock" ||
      { echo "  pip install failed for qaopd-$name; see $(basename "$lock")" >&2; return 1; }
  fi
  # Editable, and --no-deps: edgerazor declares an unpinned `torch`, and
  # resolving it upgrades the CUDA stack underneath a working env silently.
  # shellcheck disable=SC2086
  "$target/bin/pip" install --quiet --no-deps $(printf -- '-e %s ' "$@") ||
    { echo "  editable install failed for qaopd-$name" >&2; return 1; }
  echo "  qaopd-$name: $(printf '%s ' "$@")"
}

# The evaluation env needs 3.10, and a venv cannot conjure an interpreter.
# Two things make "python3.10 is on PATH" insufficient:
#
#   - Debian and Ubuntu ship the stdlib venv module in a separate
#     pythonX.Y-venv package, so the interpreter can exist and still be
#     unable to create an environment;
#   - edgerazor pins a minimum patch version, and distribution 3.10 is often
#     older than it. pip reports that only after the environment is built and
#     torch is installed into it.
#
# Check both up front.
ER_MIN_PY=$(sed -n 's/.*requires-python[^"]*"[^0-9]*\([0-9.]*\).*/\1/p' \
              third_party/edgerazor/pyproject.toml 2>/dev/null | head -1)
ER_MIN_PY=${ER_MIN_PY:-3.10}

usable_python() {
  local py="$1" probe ok
  command -v "$py" >/dev/null || return 1
  "$py" - "$ER_MIN_PY" <<'PY' >/dev/null 2>&1 || return 1
import sys
need = tuple(int(x) for x in sys.argv[1].split("."))
sys.exit(0 if sys.version_info[:len(need)] >= need else 1)
PY
  probe=$(mktemp -d); "$py" -m venv "$probe/v" >/dev/null 2>&1; ok=$?
  rm -rf "$probe"; return $ok
}

eval_python() {
  local p
  for p in python3.10 python3.11; do
    usable_python "$p" && { command -v "$p"; return; }
  done
  p="$VENV_DIR/py310"
  if [[ ! -x "$p/bin/python3.10" ]] && command -v conda >/dev/null; then
    conda create -y -q -p "$p" "python>=$ER_MIN_PY,<3.12" >/dev/null 2>&1 || true
  fi
  usable_python "$p/bin/python3.10" && { echo "$p/bin/python3.10"; return; }
  return 1
}

if [[ "$DO_VENVS" == "1" ]]; then
  step "virtual environments"
  # Training first and on its own: OPD and QAD both need it, evaluation is a
  # separate step a user can come back to.
  make_env train "$PYTHON" "$TORCH_TRAIN" \
      third_party/verl third_party/edgerazor ||
    die "the training environment is required"

  if PY310=$(eval_python); then
    make_env eval "$PY310" "$TORCH_EVAL" third_party/edgerazor || {
      echo "  qaopd-eval: FAILED -- training still works; evaluation does not."
      EVAL_ENV_MISSING=1
    }
  else
    echo "  qaopd-eval: SKIPPED -- no usable Python (needs >= $ER_MIN_PY, < 3.12)."
    if command -v python3.10 >/dev/null; then
      echo "               found $(python3.10 -V 2>&1). Distribution 3.10 is often"
      echo "               older than edgerazor's minimum, and Debian/Ubuntu also"
      echo "               split the venv module into python3.10-venv."
    else
      echo "               install Python 3.10, then re-run bootstrap."
    fi
    EVAL_ENV_MISSING=1
  fi
else
  step "virtual environments (skipped: DO_VENVS=0)"
fi

# --------------------------------------------------------------------- check
step "check"
[[ -f third_party/verl/verl/utils/qat/edgerazor.py ]] ||
  die "verl patch did not land: verl/utils/qat/edgerazor.py missing"
[[ -f third_party/edgerazor/src/edgerazor/vllm/dynamic_reload.py ]] ||
  die "edgerazor patch did not land: src/edgerazor/vllm/dynamic_reload.py missing"
echo "  third-party patches present"

cat <<EOF

bootstrap complete.

  QAOPD_ROOT = $QAOPD_ROOT
  venvs      = $VENV_DIR/qaopd-{train,eval}

Next: docs/INSTALL.md (weights and data), then docs/QAD.md.
EOF

if [[ -n "${EVAL_ENV_MISSING:-}" ]]; then
    cat <<'EOF'

NOTE: the evaluation environment is not built. Training works; the evaluation
scripts will not. Fix Python 3.10 and re-run this script -- it skips what is
already done.
EOF
fi
