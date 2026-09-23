#!/usr/bin/env python3
"""Build a mixed GSM8K + MATH training set for OPD.

Motivation: GSM8K-only OPD reaches 44.43% on GSM8K (above the 41.62% FP-0.6B
baseline) but only 15.4% on MATH-500 (FP-0.6B is 27.2%). The GSM8K number is
partly GSM8K-specific over-fitting. Mixing MATH train (which is disjoint from
MATH-500, a MATH test subset) should give a model that improves on both
benchmarks without the lopsided GSM8K spike.

Design decisions:
  - keep GSM8K rows in their existing `####`/plain-5shot format (the validated
    recipe); MATH rows keep `\\boxed{}`. The unified reward math_mixed_reward.py
    judges both.
  - GSM8K answer = ground_truth string (a number); MATH answer = the boxed
    content of its solution.
  - MATH-500 rows are dropped by problem-text hash, so the evaluation set
    cannot appear in training.

Usage: build_gsm8k_math_mix.py <math_train.parquet> [--math_frac 1.0]
"""
import argparse
import hashlib
import json
import os
import re

import pandas as pd

ROOT = os.environ.get("QAOPD_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("DATA_ROOT") or f"{ROOT}/data"
GSM8K = f"{DATA_ROOT}/gsm8k_train_eval_aligned_5shot.parquet"
# The evaluation copy that ships with the repo, used only to build the
# exclusion set below.
MATH500 = f"{ROOT}/eval_tasks/math500/math500-test.parquet"
OUT = f"{DATA_ROOT}/gsm8k_math_mix_v1"
PROMPT_TMPL = ("Problem:\n{problem}\n\nSolution: Please reason step by step, "
               "and put your final answer within \\boxed{{}}.")


def _last_boxed(text):
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    start = text.find("{", idx)
    if start == -1:
        return None
    depth = 0
    for pos in range(start, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:pos]
    return None


def _norm(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("math_train")
    ap.add_argument("--math_frac", type=float, default=1.0,
                    help="fraction of MATH train to include (1.0 = all)")
    args = ap.parse_args()

    # exclusion set: MATH-500 problems (never train on the eval set)
    m500 = pd.read_parquet(MATH500)
    forbid = {_norm(p) for p in m500["problem"]}

    gsm = pd.read_parquet(GSM8K)
    math = pd.read_parquet(args.math_train)
    # locate columns
    prob_col = "problem" if "problem" in math.columns else ("question" if "question" in math.columns else math.columns[0])
    sol_col = "solution" if "solution" in math.columns else ("answer" if "answer" in math.columns else None)
    ans_col = "answer" if "answer" in math.columns else None
    print(f"MATH columns: {list(math.columns)} -> problem={prob_col} solution={sol_col} answer={ans_col}")

    rows = []
    # --- GSM8K rows: keep as-is (already in training format) ---
    for _, r in gsm.iterrows():
        rows.append({
            "data_source": "openai/gsm8k",
            "prompt": r["prompt"],
            "ability": "math",
            "reward_model": r["reward_model"],
            "extra_info": {"split": "train", "index": len(rows), "benchmark": "gsm8k"},
        })
    n_gsm = len(rows)

    # --- MATH rows ---
    kept = dropped_500 = dropped_noans = 0
    math = math.sample(frac=args.math_frac, random_state=20260722) if args.math_frac < 1.0 else math
    for _, r in math.iterrows():
        prob = str(r[prob_col])
        if _norm(prob) in forbid:
            dropped_500 += 1
            continue
        # answer: prefer explicit answer col, else boxed content of solution
        if ans_col and pd.notna(r.get(ans_col)) and str(r[ans_col]).strip():
            ans = str(r[ans_col]).strip()
        elif sol_col:
            b = _last_boxed(str(r[sol_col]))
            ans = b if b is not None else None
        else:
            ans = None
        if not ans:
            dropped_noans += 1
            continue
        rows.append({
            "data_source": "hendrycks_math",
            "prompt": [{"role": "user", "content": PROMPT_TMPL.format(problem=prob)}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": ans},
            "extra_info": {"split": "train", "index": len(rows), "benchmark": "math"},
        })
        kept += 1

    df = pd.DataFrame(rows)
    for i in range(len(df)):
        df.at[i, "extra_info"] = {**df.at[i, "extra_info"], "index": i}
    os.makedirs(OUT, exist_ok=True)
    df.to_parquet(f"{OUT}/train.parquet")
    # tiny placeholder validation (trainer needs a val file; real eval is external)
    df.head(16).to_parquet(f"{OUT}/validation.parquet")

    json.dump({
        "gsm8k_rows": n_gsm, "math_rows": kept, "total": len(df),
        "math_dropped_in_math500": dropped_500, "math_dropped_no_answer": dropped_noans,
        "math_source": args.math_train, "math_frac": args.math_frac,
        "note": "GSM8K keeps #### format, MATH keeps boxed; unified math_mixed_reward judges both; MATH-500 excluded by problem hash",
        "train_sha256": hashlib.sha256(open(f"{OUT}/train.parquet", "rb").read()).hexdigest()[:32],
    }, open(f"{OUT}/manifest.json", "w"), indent=1)

    print(f"\nmix: GSM8K {n_gsm} + MATH {kept} = {len(df)}")
    print(f"  MATH dropped (in MATH-500): {dropped_500}  no answer: {dropped_noans}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
