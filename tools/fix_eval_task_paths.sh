#!/usr/bin/env bash
# Point every eval_tasks/*/*.yaml at the parquet that sits next to it.
#
# Upstream's task YAMLs were committed with `dataset_kwargs.data_files.test`
# still holding an absolute path from the machine they were authored on, so on
# any other machine every task failed to build. The parquet files ARE in the
# repo, one directory per task, so the correct path is always derivable.
#
# The written form is repo-root-relative on purpose. lm_eval hands data_files to
# datasets.load_dataset, which resolves relative paths against the caller's CWD,
# so the eval entry points cd to $QAOPD_ROOT first. An absolute path would work
# from any CWD but would bake this machine's layout into a tracked file and get
# committed by the next person. A relative path is machine-neutral, and running
# from the wrong directory fails loudly with
# "Unable to find '<cwd>/eval_tasks/...'" rather than silently evaluating
# something else.
#
# Idempotent. The evaluation entry points call it with --check before running.
#
# Usage: bash tools/fix_eval_task_paths.sh [--check]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECK=0
[[ "${1:-}" == "--check" ]] && CHECK=1

rc=0
changed=0
for yaml in "$HERE"/eval_tasks/*/*.yaml; do
    dir="$(dirname "$yaml")"
    parquet="$(find "$dir" -maxdepth 1 -name '*.parquet' | head -1)"
    task="$(basename "$dir")"
    # The rewrite below only knows how to repair the shape it was written for:
    # a single `test:` entry pointing at the one parquet in the task directory.
    nsplits="$(grep -cE '^ +(train|validation|test): ' "$yaml")"
    if [[ -z "$parquet" || "$nsplits" -gt 1 ]]; then
        # Not every task follows the one-parquet-per-directory shape this script
        # was written to repair. eval_tasks/edgerazor_qa/ is a group with no data
        # of its own, and eval_tasks/hendrycks_ethics/ keeps one subdirectory per
        # subtask. Those yamls were written by hand and are already repo-root
        # relative, so there is nothing to rewrite -- but every path they declare
        # still has to exist, which is the check that actually matters.
        missing=0
        while read -r rel; do
            [[ -e "$HERE/$rel" ]] || { echo "MISSING $task: $rel" >&2; missing=1; }
        done < <(grep -oE '^ +[a-z]+: +eval_tasks/[^ ]+' "$yaml" | awk '{print $2}')
        if (( missing )); then
            echo "  -> run: bash tools/fetch_qa_eval_data.sh" >&2
            rc=1
        fi
        continue
    fi
    want="eval_tasks/$task/$(basename "$parquet")"
    current="$(sed -n 's/^ *test: *//p' "$yaml" | head -1)"
    [[ "$current" == "$want" ]] && continue
    if (( CHECK )); then
        echo "STALE $task: $current"
        rc=1
    else
        if ! python3 - "$yaml" "$want" <<'PY'
import re, sys
path, want = sys.argv[1], sys.argv[2]
text = open(path).read()
new, n = re.subn(r'(^ *test: ).*$', lambda m: m.group(1) + want, text,
                 count=1, flags=re.M)
if n != 1:
    sys.exit(f"could not rewrite data_files in {path}")
open(path, 'w').write(new)
PY
        then rc=1; continue; fi
        echo "fixed $task -> $want"
        changed=$((changed + 1))
    fi
done

if (( CHECK )); then
    (( rc == 0 )) && echo "EVAL_TASK_PATHS_OK"
else
    echo "EVAL_TASK_PATHS_DONE changed=$changed"
fi
exit $rc
