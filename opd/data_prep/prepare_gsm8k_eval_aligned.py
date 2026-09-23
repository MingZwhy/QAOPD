#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def format_fewshot_prompt(question: str, exemplars: list[tuple[str, str]]) -> str:
    blocks = [f"Question: {example_question}\nAnswer: {answer}" for example_question, answer in exemplars]
    blocks.append(f"Question: {question}\nAnswer:")
    return "\n\n".join(blocks)


def sample_fewshot_indices(
    pool_size: int,
    num_fewshot: int,
    seed: int,
    row_index: int,
    excluded_index: int | None = None,
) -> list[int]:
    available = pool_size - (excluded_index is not None)
    if num_fewshot > available:
        raise ValueError(f"Requested {num_fewshot} examples from a pool of {available}")

    rng = np.random.default_rng(seed + row_index)
    indices = rng.choice(available, size=num_fewshot, replace=False)
    if excluded_index is not None:
        indices = indices + (indices >= excluded_index)
    return indices.tolist()


def build_eval_aligned_prompts(
    frame: pd.DataFrame,
    fewshot_frame: pd.DataFrame,
    num_fewshot: int,
    seed: int,
    same_source: bool,
) -> list[list[dict[str, str]]]:
    pool = [(row["question"], row["answer"]) for row in fewshot_frame["extra_info"]]
    prompts = []
    for row_index, extra_info in enumerate(frame["extra_info"]):
        excluded_index = row_index if same_source else None
        indices = sample_fewshot_indices(
            len(pool),
            num_fewshot,
            seed,
            row_index,
            excluded_index=excluded_index,
        )
        exemplars = [pool[index] for index in indices]
        content = format_fewshot_prompt(extra_info["question"], exemplars)
        prompts.append([{"role": "user", "content": content}])
    return prompts


def parse_args():
    parser = argparse.ArgumentParser(description="Create lm-eval-style plain 5-shot GSM8K prompts.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--fewshot_source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_fewshot", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main():
    args = parse_args()
    source = Path(args.input).resolve()
    fewshot_source = Path(args.fewshot_source).resolve()
    output = Path(args.output)

    frame = pd.read_parquet(source)
    fewshot_frame = pd.read_parquet(fewshot_source)
    frame["prompt"] = build_eval_aligned_prompts(
        frame,
        fewshot_frame,
        num_fewshot=args.num_fewshot,
        seed=args.seed,
        same_source=source == fewshot_source,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    print(f"Wrote {len(frame)} eval-aligned rows to {output}")


if __name__ == "__main__":
    main()
