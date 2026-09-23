"""Rewrite the few-shot continuation rows of unified_math_v1 as single-problem rows.

2,500 of the pool's 7,500 rows (33%) are GSM8K 5-shot continuations: five
complete "Question:/Answer:/#### N" blocks plus a sixth question, 2,970
characters on average, all packed into one user turn. When OPD trains with the
model's native chat template, that third of the data teaches plain-text
continuation inside a chat wrapper -- a format that never occurs at evaluation
time. It is one likely reason why none of the twelve phase-1 checkpoints beat
their starting point on that line.

After rewriting, the rows match the MATH rows (one problem, answer in
\\boxed{}). math_mixed_reward.py already accepts both "#### N" and
"\\boxed{}", so scoring is unchanged.
"""
import re
import sys
import os
from pathlib import Path

import pandas as pd

SRC_DIR = Path(os.environ.get("SRC_DIR", "data/unified_math_v1"))
DST_DIR = Path(os.environ.get("DST_DIR", "data/unified_math_chat_v1"))
# validation.parquet has to be rewritten too. A first version handled only
# train and the run failed immediately with "Required path does not exist:
# .../validation.parquet"; and all 16 validation rows are few-shot, so leaving
# them alone would keep validation and training in different formats.
SPLITS = ("train", "validation")
INSTR = "Solution: Please reason step by step, and put your final answer within \\boxed{}."


def last_question(text: str) -> str | None:
    """Return the problem statement of the last 'Question: ... Answer:' block."""
    parts = re.split(r"(?m)^Question:\s*", text)
    if len(parts) < 2:
        return None
    tail = parts[-1]
    # The last block looks like "<problem>\nAnswer:" -- no answer follows, which
    # is exactly the question the model is meant to complete.
    m = re.split(r"(?m)^Answer:\s*$", tail)
    q = m[0] if m else tail
    q = q.rstrip()
    if q.endswith("Answer:"):
        q = q[: -len("Answer:")].rstrip()
    return q or None


def convert(split: str) -> int:
    src = SRC_DIR / f"{split}.parquet"
    dst = DST_DIR / f"{split}.parquet"
    if not src.exists():
        print(f"  {split}: source missing, skipped")
        return 0
    d = pd.read_parquet(src)
    rewritten = skipped = 0
    prompts = []
    for _, r in d.iterrows():
        c = r["prompt"][0]["content"]
        if "Question:" in c and "####" in c:
            q = last_question(c)
            if q is None:
                skipped += 1
                prompts.append(r["prompt"])
                continue
            prompts.append([{"role": "user", "content": f"Problem:\n{q}\n\n{INSTR}"}])
            rewritten += 1
        else:
            prompts.append(r["prompt"])
    d["prompt"] = prompts
    dst.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(dst, index=False)

    old = [len(r["prompt"][0]["content"]) for _, r in pd.read_parquet(src).iterrows()]
    new = [len(p[0]["content"]) for p in prompts]
    left = sum(1 for p in prompts if "####" in p[0]["content"])
    print(f"  {split}: rewritten {rewritten}, skipped {skipped}, total {len(d)}, "
          f"mean length {sum(old)//len(old)} -> {sum(new)//len(new)}, few-shot left {left}")
    return left


def main() -> int:
    bad = sum(convert(s) for s in SPLITS)
    ex = pd.read_parquet(DST_DIR / "train.parquet").iloc[0]["prompt"][0]["content"]
    print(f"  sample: {ex[:160]!r}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
