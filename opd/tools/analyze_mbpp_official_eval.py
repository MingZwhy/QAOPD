#!/usr/bin/env python3
"""Summarize and optionally pair two deterministic MBPP execution evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def load_samples(path: Path) -> dict[str, dict]:
    samples: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            ground_truth = row["gts"]
            parsed = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
            canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
            key = hashlib.sha256(canonical.encode()).hexdigest()
            if key in samples:
                raise ValueError(f"Duplicate ground truth at {path}:{line_number}")
            samples[key] = {
                "passed": float(row["score"]) == 1.0,
                "score": float(row["score"]),
                "input_sha256": hashlib.sha256(str(row["input"]).encode()).hexdigest(),
                "output_sha256": hashlib.sha256(str(row["output"]).encode()).hexdigest(),
            }
    return samples


def exact_mcnemar_p_value(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(gains, losses) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def summarize(samples: dict[str, dict]) -> dict:
    total = len(samples)
    passed = sum(sample["passed"] for sample in samples.values())
    return {
        "passed": passed,
        "total": total,
        "accuracy": passed / total if total else 0.0,
        "accuracy_percent": 100.0 * passed / total if total else 0.0,
        "mean_test_fraction": (
            sum(sample["score"] for sample in samples.values()) / total if total else 0.0
        ),
    }


def compare(baseline: dict[str, dict], candidate: dict[str, dict]) -> dict:
    if baseline.keys() != candidate.keys():
        raise ValueError("Baseline and candidate ground-truth sets differ")
    for key in baseline:
        if baseline[key]["input_sha256"] != candidate[key]["input_sha256"]:
            raise ValueError(f"Generation input differs for ground-truth key {key}")
    gains = sum(not baseline[key]["passed"] and candidate[key]["passed"] for key in baseline)
    losses = sum(baseline[key]["passed"] and not candidate[key]["passed"] for key in baseline)
    baseline_summary = summarize(baseline)
    candidate_summary = summarize(candidate)
    return {
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "delta_accuracy": candidate_summary["accuracy"] - baseline_summary["accuracy"],
        "delta_percentage_points": (
            candidate_summary["accuracy_percent"] - baseline_summary["accuracy_percent"]
        ),
        "paired_gains": gains,
        "paired_losses": losses,
        "mcnemar_exact_p": exact_mcnemar_p_value(gains, losses),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--expected_rows", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_samples(args.result)
    if args.expected_rows is not None and len(samples) != args.expected_rows:
        raise ValueError(f"Expected {args.expected_rows} rows, found {len(samples)}")

    report = {
        "result_file": str(args.result.resolve()),
        "result_sha256": hashlib.sha256(args.result.read_bytes()).hexdigest(),
        "summary": summarize(samples),
    }
    if args.baseline is not None:
        baseline = load_samples(args.baseline)
        if args.expected_rows is not None and len(baseline) != args.expected_rows:
            raise ValueError(f"Expected {args.expected_rows} baseline rows, found {len(baseline)}")
        report["baseline_file"] = str(args.baseline.resolve())
        report["comparison"] = compare(baseline, samples)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
