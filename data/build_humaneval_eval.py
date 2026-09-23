#!/usr/bin/env python3
"""Build the HumanEval-164 evaluation pool in verl's evaluation format.

Evaluation only -- nothing here is ever trained on. scripts/eval/b1_humaneval.sh
reads this directory, and verl needs both a train and a validation split even
for a val-only run, so the same 164 rows are written to each.

The reward function (opd/rewards/code_execution_reward.py) executes the model's
completion against HumanEval's own `check(candidate)` harness, so the ground
truth carries the stub, the entry point and the test setup rather than a
reference solution.

Usage:
    python data/build_humaneval_eval.py [--output_dir data/humaneval_eval]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from datasets import load_dataset

INSTRUCTION = (
    "Write a solution to the following problem and make sure that it passes the tests:"
)


def build_rows(dataset) -> list[dict]:
    rows = []
    for index, item in enumerate(dataset):
        stub = item["prompt"]
        entry = item["entry_point"]
        content = f"{INSTRUCTION}\n```python\n{stub}\n```\n"
        ground_truth = {
            "function_stub": stub,
            "function_name": entry,
            "tests": [f"check({entry})"],
            "test_setup_code": item["test"],
        }
        rows.append(
            {
                "task_id": item["task_id"],
                "data_source": "mbpp_humaneval",
                "prompt": [{"role": "user", "content": content}],
                "ability": "code",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": json.dumps(ground_truth),
                },
                "extra_info": {
                    "source_dataset": "openai_humaneval",
                    "source_split": "test",
                    "task_id": item["task_id"],
                    "source_task_id": item["task_id"],
                    "function_name": entry,
                    "split": "train",
                    "index": index,
                },
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output_dir",
        default=os.environ.get("DATA_ROOT", "data") + "/humaneval_eval",
    )
    ap.add_argument("--dataset", default="openai/openai_humaneval")
    args = ap.parse_args()

    rows = build_rows(load_dataset(args.dataset, split="test"))
    frame = pd.DataFrame(rows)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation"):
        frame.to_parquet(out / f"{split}.parquet", index=False)
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "source": args.dataset,
                "split": "test",
                "rows": len(frame),
                "use": "evaluation only",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {len(frame)} rows to {out}/{{train,validation}}.parquet")


if __name__ == "__main__":
    main()
