#!/usr/bin/env python
import os
"""Teacher-solvability filter for DAPO-17k.

Runs the OPD teacher (Qwen3-1.7B, bf16, greedy) on every DAPO
prompt in the exact deployment format used for training rollouts and evals
(Minerva 4-shot scaffold, non-thinking plain completion, 1024 new tokens,
stop "Problem:"). Keeps problems whose teacher answer matches gold.

Rationale: on the raw set both OPD attempts collapsed (boxed 91%->12% /
87%->16% by step ~20): competition problems the teacher cannot solve tersely
make its natural continuation long exploratory rambling, so the K1 term drags
the student away from the concluding format, while correct=0/32 leaves GRPO
with no positive signal. Teacher-solvable-in-format is the necessary
condition for K1 to pull in the right direction (rethinking-opd: the teacher
must offer learnable signal at student-visited states).

Usage: CUDA_VISIBLE_DEVICES=0 python filter_dapo_teacher.py --shard 0 --nshards 2
"""
import argparse, json, os, sys

ROOT = os.environ.get("QAOPD_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("DATA_ROOT") or f"{ROOT}/data"
MODELS_DIR = os.environ.get("MODELS_DIR") or f"{ROOT}/models"
sys.path.insert(0, f"{ROOT}/opd/rewards")
import math_mixed_reward as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_parquet(f"{DATA_ROOT}/dapo_math_17k_minerva/train.parquet")
    df = df.iloc[args.shard::args.nshards].reset_index(drop=True)
    prompts = [p[0]["content"] for p in df["prompt"]]
    golds = [rm["ground_truth"] for rm in df["reward_model"]]
    print(f"shard {args.shard}/{args.nshards}: {len(prompts)} prompts", flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model=f"{MODELS_DIR}/Qwen3-1.7B", dtype="bfloat16",
              max_model_len=4096, gpu_memory_utilization=0.85)
    sp = SamplingParams(temperature=0.0, max_tokens=1024, stop=["Problem:"])

    B = 512
    kept = []
    for i in range(0, len(prompts), B):
        outs = llm.generate(prompts[i:i + B], sp, use_tqdm=False)
        for j, o in enumerate(outs):
            text = o.outputs[0].text
            pred = R._extract(text)
            if pred is not None and R._equiv(pred, golds[i + j]):
                kept.append({"row": int(df.index[i + j]), "teacher_len": len(text)})
        print(f"[{min(i+B,len(prompts))}/{len(prompts)}] kept={len(kept)}", flush=True)

    with open(args.out, "w") as f:
        json.dump({"shard": args.shard, "kept_rows": [k["row"] for k in kept],
                   "n": len(prompts), "kept": len(kept)}, f)
    print(f"TEACHER_FILTER_DONE shard={args.shard} kept={len(kept)}/{len(prompts)}", flush=True)


if __name__ == "__main__":
    main()
