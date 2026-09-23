#!/usr/bin/env bash
# Wire the vendored EdgeRazor QAT trainer for one model/bitwidth combo.
#
# Upstream ships config.py with a "/path/to/your/environment" placeholder and
# selects the model by editing a literal in main.py (its comment refers to a
# run.sh that is not in the tree). This script does all three edits:
#
#   1. config.py  PATH_PREFIX   -> your workspace
#   2. config.py  config_path   -> configs/qad/<combo>.yaml
#   3. main.py    config = ...  -> the class matching <combo>
#
# Usage:
#   bash tools/wire_edgerazor_qat.sh <combo> <workspace>
#   <combo>     a file stem under configs/qad/, e.g. qwen3_1_7b_w2.79
#   <workspace> absolute path whose $workspace/EdgeRazor-QLLM/data holds the
#               training jsonl (see docs/QAD.md)
#
# Optional env vars override fields on the selected class only. Upstream's
# defaults assume the original authors' machine and full 11M-sample corpus:
# teacher/student are hub ids, ds_path points inside a CODE_ROOT tree we do not
# have, and save_steps=1000 is far too coarse when the plan is to stop at a good
# intermediate checkpoint rather than finish two epochs.
#
#   ER_TEACHER ER_STUDENT     local model dirs instead of hub ids
#   ER_LR / ER_SCHED / ER_WARMUP_RATIO
#                             optimiser schedule. Changing ER_SCHED on a resume
#                             is safe: the checkpoint restores last_epoch, not
#                             the lambda, so a run started under
#                             constant_with_warmup can be finished under linear
#                             decay. Verify against the "Learning rate" line and
#                             the first logged learning_rate after resume.
#   ER_DATASETS               comma-separated jsonl (bare names resolve under
#                             $workspace/EdgeRazor-QLLM/data)
#   ER_DS_PATH                deepspeed json
#   ER_EPOCH ER_STEPS ER_SAVE_STEPS ER_PER_DEVICE_BS ER_GRAD_ACC ER_MAX_SEQ_LEN
#   ER_ATTN                   eager | sdpa | flash_attention_2
#   ER_TAG                    tag_name, used for the tensorboard run dir
#
# Edits are in-place under third_party/edgerazor; revert with
#   git -C third_party/edgerazor checkout .
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

COMBO="${1:-}"
WORKSPACE="${2:-}"
if [[ -z "$COMBO" || -z "$WORKSPACE" ]]; then
    echo "usage: $0 <combo> <workspace>" >&2
    echo "available combos:" >&2
    for f in "$HERE"/configs/qad/*.yaml; do echo "  $(basename "$f" .yaml)" >&2; done
    exit 2
fi

YAML="$HERE/configs/qad/$COMBO.yaml"
[[ -f "$YAML" ]] || { echo "no such combo: $COMBO (see $HERE/configs/qad)" >&2; exit 2; }
[[ "$WORKSPACE" = /* ]] || { echo "workspace must be an absolute path" >&2; exit 2; }

# combo -> config.py class name
case "$COMBO" in
    qwen3_06b_*)  CLS=EdgeRazorTrainConfigForQwen3_0_6B ;;
    qwen3_1_7b_*) CLS=EdgeRazorTrainConfigForQwen3_1_7B ;;
    qwen3_4b_*)   CLS=EdgeRazorTrainConfigForQwen3_4B ;;
    *) echo "cannot map combo '$COMBO' to a config.py class -- edit main.py by hand" >&2; exit 2 ;;
esac

ER_SRC="$HERE/third_party/edgerazor/example/edgerazor-llm/src"
CFG="$ER_SRC/config.py"
MAIN="$ER_SRC/main.py"
for f in "$CFG" "$MAIN"; do
    [[ -f "$f" ]] || { echo "missing $f -- run setup/bootstrap.sh first" >&2; exit 1; }
done

PY_BIN="${PY_BIN:-python}"
command -v "$PY_BIN" >/dev/null || PY_BIN=python3
"$PY_BIN" - "$CFG" "$MAIN" "$WORKSPACE" "$YAML" "$CLS" <<'PYEOF'
import os, re, sys
cfg_p, main_p, workspace, yaml, cls = sys.argv[1:6]

cfg = open(cfg_p).read()

# Upstream ships classes for Qwen3-0.6B, Qwen3-1.7B and MobileLLM only, so a
# 4B combo has nothing to wire. Clone the 1.7B class under the wanted name --
# every field below is overwritten anyway, and the two differ only in the
# per-device batch, which the caller sets. Without this the run dies later with
# an ImportError that says nothing about the missing config.
if f"class {cls}:" not in cfg:
    src = "EdgeRazorTrainConfigForQwen3_1_7B"
    m = re.search(rf'^class {src}:.*?(?=^class |\Z)', cfg, flags=re.M | re.S)
    if not m:
        sys.exit(f"cannot synthesise {cls}: template class {src} not found")
    cfg = cfg + "\n\n# Added by tools/wire_edgerazor_qat.sh, cloned from " + src + "\n" \
              + m.group(0).replace(f"class {src}:", f"class {cls}:", 1).rstrip() + "\n"
    print(f"  config.py   -> synthesised {cls} from {src}")

# [ \t] rather than \s throughout: \s matches newlines, so a trailing \s*$
# under re.M swallows the following blank line as well as the target line.
cfg, n1 = re.subn(r'^PATH_PREFIX[ \t]*=.*$',
                  f'PATH_PREFIX = "{workspace}"', cfg, count=1, flags=re.M)
# every model class points config_path at the quant yaml; retarget all of them
cfg, n2 = re.subn(r'^([ \t]*config_path[ \t]*=[ \t]*).*$',
                  lambda m: f'{m.group(1)}"{yaml}"', cfg, flags=re.M)

# Scoped overrides: rewrite fields inside the selected class body only, so
# wiring 1.7B never silently moves the 0.6B or MobileLLM classes.
data_root = f"{workspace}/EdgeRazor-QLLM/data"


def datasets_literal(spec):
    paths = [p.strip() for p in spec.split(",") if p.strip()]
    paths = [p if p.startswith("/") else f"{data_root}/{p}" for p in paths]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print("  WARNING: ER_DATASETS entries not on disk:")
        for p in missing:
            print(f"    {p}")
    body = "".join(f'\n        "{p}",' for p in paths)
    return "dataset_path  = [" + body + "\n    ]"


OVERRIDES = [
    ("ER_TEACHER",       "teacher_path",         lambda v: f'teacher_path  = "{v}"'),
    ("ER_STUDENT",       "student_path",         lambda v: f'student_path  = "{v}"'),
    ("ER_DS_PATH",       "ds_path",              lambda v: f'ds_path       = "{v}"'),
    ("ER_EPOCH",         "epoch",                lambda v: f'epoch         = {v}'),
    ("ER_STEPS",         "steps",                lambda v: f'steps         = {v}'),
    ("ER_SAVE_STEPS",    "save_steps",           lambda v: f'save_steps     = {v}'),
    ("ER_PER_DEVICE_BS", "per_device_bs",        lambda v: f'per_device_bs  = {v}'),
    ("ER_GRAD_ACC",      "grad_acc_steps",       lambda v: f'grad_acc_steps = {v}'),
    ("ER_MAX_SEQ_LEN",   "max_seq_len",          lambda v: f'max_seq_len   = {v}'),
    ("ER_LR",            "lr",                   lambda v: f'lr            = {v}'),
    ("ER_SCHED",         "lr_scheduler",         lambda v: f'lr_scheduler  = "{v}"'),
    ("ER_WARMUP_RATIO",  "warmup_ratio",         lambda v: f'warmup_ratio  = {v}'),
    ("ER_ATTN",          "attn_implementation",  lambda v: f'attn_implementation  = "{v}"'),
    ("ER_TAG",           "tag_name",             lambda v: f'tag_name             = "{v}"'),
]

start = cfg.index(f"class {cls}:")
end = cfg.find("\nclass ", start + 1)
end = len(cfg) if end == -1 else end
body = cfg[start:end]

applied = []
for env, field, render in OVERRIDES:
    val = os.environ.get(env)
    if not val:
        continue
    body, n = re.subn(rf'^([ \t]+){field}[ \t]*=[ \t]*.*$',
                      lambda m: m.group(1) + render(val), body, count=1, flags=re.M)
    if not n:
        sys.exit(f"  ERROR: {env} set but no '{field}' line found in {cls}")
    applied.append(f"{field} -> {val}")

if os.environ.get("ER_DATASETS"):
    body, n = re.subn(r'^([ \t]+)dataset_path[ \t]*=[ \t]*\[.*?^\1\]',
                      lambda m: m.group(1) + datasets_literal(os.environ["ER_DATASETS"]),
                      body, count=1, flags=re.M | re.S)
    if not n:
        sys.exit(f"  ERROR: ER_DATASETS set but no 'dataset_path = [...]' block in {cls}")
    applied.append("dataset_path -> " + os.environ["ER_DATASETS"])

cfg = cfg[:start] + body + cfg[end:]
open(cfg_p, 'w').write(cfg)

main = open(main_p).read()
main, n3 = re.subn(r'^config[ \t]*=[ \t]*EdgeRazorTrainConfigFor\w+\(\)[ \t]*$',
                   f'config = {cls}()', main, count=1, flags=re.M)

# main.py imports the config classes by name, not with `import *`, so the import
# line has to track config.py. Rebuild it from the classes config.py actually
# defines rather than appending to it: config.py is a working-tree file that
# gets reverted (`git -C third_party/edgerazor checkout .`) and re-synthesised
# independently of main.py, and an import naming a class that reverting removed
# breaks *every* arm with an ImportError, not just the one being wired.
present = re.findall(r'^class (EdgeRazorTrainConfigFor\w+):', cfg, flags=re.M)
if cls not in present:
    sys.exit(f"  {cls} still missing from config.py after wiring")
n4 = 1
main, n4 = re.subn(r'^from config import .*$',
                   "from config import " + ", ".join(present), main, count=1, flags=re.M)
open(main_p, 'w').write(main)

if not (n1 and n2 and n3 and n4):
    print(f"  WARNING: substitutions applied: PATH_PREFIX={n1} config_path={n2} class={n3} import={n4}")
    print("  a 0 means upstream changed shape -- edit that file by hand")
    sys.exit(1)
print(f"  PATH_PREFIX -> {workspace}")
print(f"  config_path -> {yaml}   ({n2} class(es))")
print(f"  main.py     -> {cls}()  (the import is guaranteed to cover this class)")
for line in applied:
    print(f"  {cls}.{line}")
PYEOF

DATA_DIR="$WORKSPACE/EdgeRazor-QLLM/data"
echo
echo "wired for $COMBO."
if [[ -d "$DATA_DIR" ]]; then
    # Note the filenames: config.py's dataset_path lists the plain
    # *_instruct.jsonl, NOT the *_distill.jsonl that distill_qwen3.sh produces.
    # Upstream ships it that way, so the teacher-regeneration pass is not on the
    # path the trainer actually reads (docs/QAD.md).
    N=$(ls "$DATA_DIR"/*.jsonl 2>/dev/null | wc -l)
    echo "  data dir $DATA_DIR: $N jsonl present"
    [[ "$N" -eq 0 ]] && echo "  -> build them with data_prepare.sh (docs/QAD.md)"
else
    echo "  NOTE: $DATA_DIR does not exist yet"
    echo "  -> create it and put the distilled jsonl there (docs/QAD.md)"
fi
echo
echo "next: COMBO=$COMBO MODEL=<base model> bash scripts/qad/run_qad.sh"
