#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd

_OLD_INSTRUCTION = ' Let\'s think step by step and output the final answer after "####".'
STRICT_INSTRUCTION = (
    " Solve the problem step by step. The last line must be exactly `#### ` followed by the numeric answer. "
    "Do not use `Answer:`, `Final Answer:`, `\\boxed{...}`, a Markdown heading, or any text after that line."
)


def rewrite_prompt(prompt):
    messages = [dict(message) for message in prompt]
    user_indices = [index for index, message in enumerate(messages) if message.get("role") == "user"]
    if not user_indices:
        raise ValueError("Prompt does not contain a user message")

    index = user_indices[-1]
    content = messages[index].get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("The final user message has no text content")
    if STRICT_INSTRUCTION in content:
        return messages

    content = content.replace(_OLD_INSTRUCTION, "")
    messages[index]["content"] = content.rstrip() + STRICT_INSTRUCTION
    return messages


def parse_args():
    parser = argparse.ArgumentParser(description="Add an exact final-line contract to verl GSM8K prompts.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    source = Path(args.input)
    output = Path(args.output)
    frame = pd.read_parquet(source)

    frame["prompt"] = frame["prompt"].map(rewrite_prompt)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    print(f"Wrote {len(frame)} strict-format rows to {output}")


if __name__ == "__main__":
    main()
