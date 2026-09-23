#!/usr/bin/env python3
"""Select the earliest highest-accuracy MBPP validation checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from analyze_mbpp_official_eval import load_samples, summarize


def parse_steps(value: str) -> list[int]:
    steps = [int(item) for item in value.replace(",", " ").split()]
    if not steps or any(step <= 0 for step in steps) or len(steps) != len(set(steps)):
        raise ValueError("steps must be unique positive integers")
    return sorted(steps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation_dir", type=Path, required=True)
    parser.add_argument("--steps", default="5 10 15 20 25 30")
    parser.add_argument("--expected_rows", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    steps = parse_steps(args.steps)
    curve = []
    for step in [0, *steps]:
        path = args.validation_dir / f"{step}.jsonl"
        samples = load_samples(path)
        if len(samples) != args.expected_rows:
            raise ValueError(f"Expected {args.expected_rows} rows at step {step}, found {len(samples)}")
        curve.append(
            {
                "step": step,
                **summarize(samples),
                "result_file": str(path.resolve()),
                "result_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )

    trained = [row for row in curve if row["step"] > 0]
    selected = min(trained, key=lambda row: (-row["passed"], row["step"]))
    report = {
        "selection_rule": "maximum strict all-tests-pass count; ties choose earliest trained step",
        "expected_rows": args.expected_rows,
        "curve": curve,
        "selected_step": selected["step"],
        "selected_validation_passed": selected["passed"],
        "selected_validation_accuracy": selected["accuracy"],
        "step0_validation_passed": curve[0]["passed"],
        "step0_validation_accuracy": curve[0]["accuracy"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
