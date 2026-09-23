#!/usr/bin/env python3
"""Compute the QA9 average from an lm_eval results file.

QA9 is the equal-weight mean of nine short-form QA benchmarks. mmlu counts as
one of the nine, not as its 57 subjects, so the mean is taken over task-level
rows and the mmlu group row is used directly.

Usage:
    python tools/qa9_mean.py <results dir or results*.json> [...]
"""
from __future__ import annotations

import json
import pathlib
import sys

QA9 = [
    "arc_easy",
    "arc_challenge",
    "hellaswag",
    "social_iqa",
    "openbookqa",
    "piqa",
    "winogrande",
    "truthfulqa_mc2",
    "mmlu",
]
# acc_norm where a task reports it, acc otherwise -- the same order
# scripts/eval/qa_suite.sh uses, and the one the paper's QA9 column is on.
# truthfulqa_mc2, winogrande, social_iqa and mmlu have no acc_norm.
METRIC_ORDER = ["acc_norm,none", "acc,none", "exact_match,none"]


def pick(metrics: dict) -> float | None:
    for k in METRIC_ORDER:
        if k in metrics and metrics[k] is not None:
            return float(metrics[k])
    return None


def results_from(path: pathlib.Path) -> dict:
    files = [path] if path.is_file() else sorted(path.rglob("results*.json"))
    merged: dict = {}
    for f in files:
        try:
            merged.update(json.loads(f.read_text()).get("results", {}))
        except Exception:
            continue
    return merged


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    for arg in argv:
        res = results_from(pathlib.Path(arg))
        print(f"== {arg}")
        got, missing = {}, []
        for t in QA9:
            if t in res:
                v = pick(res[t])
                if v is None:
                    missing.append(f"{t}(no metric)")
                else:
                    got[t] = v * 100.0
            else:
                missing.append(t)
        for t in QA9:
            if t in got:
                print(f"   {t:<18} {got[t]:6.2f}")
        if missing:
            print(f"   missing: {', '.join(missing)}")
        if len(got) == len(QA9):
            print(f"   {'QA9':<18} {sum(got.values())/len(QA9):6.2f}")
        else:
            print(f"   QA9 incomplete ({len(got)}/9) -- not averaging")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
