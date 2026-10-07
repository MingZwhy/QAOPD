#!/usr/bin/env python3
"""MBPP (official test split, 448) and HumanEval (164) on a deployment export.

The protocol is the one b1_test448.sh and b1_humaneval.sh apply to latent checkpoints: the
code-OPD chat template, greedy decoding of at most 512 tokens stopped at the closing code
fence, and a problem counts only when every test passes. The difference is the model: the
export is loaded through its own EdgeRazor runtime (trust_remote_code), which keeps the
baked-in weights as they are, instead of a latent checkpoint being quantized inside the
training framework. Prompts are decoded in batches.

    python scripts/eval/code_export.py --model <export dir> [--bench mbpp humaneval]
        [--batch_size 32] [--out <dir>]
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList, StopStringCriteria

ROOT = Path(os.environ.get("QAOPD_ROOT") or Path(__file__).resolve().parents[2])
DATA_ROOT = Path(os.environ.get("DATA_ROOT") or ROOT / "data")
sys.path.insert(0, str(ROOT / "opd/rewards"))
sys.path.insert(0, str(ROOT / "opd/tools"))
from mbpp_execution_reward import compute_score  # noqa: E402
from validate_qat_latent_checkpoint import find_prequantized_evidence  # noqa: E402

BENCHES = {
    "mbpp": (DATA_ROOT / "mbpp_official_protocol_v1/test.parquet", 448),
    "humaneval": (DATA_ROOT / "humaneval_eval/validation.parquet", 164),
}
CHAT_TEMPLATE = (ROOT / "opd/humaneval_opd_chat_template.jinja").read_text()
MAX_PROMPT_TOKENS = 1024
MAX_NEW_TOKENS = 512


def generate(model, tokenizer, prompts: list[str], batch_size: int) -> list[str]:
    # Longest first, so each batch pads little and an out-of-memory error shows up at once.
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    stop = StoppingCriteriaList([StopStringCriteria(tokenizer, ["```"])])
    responses: list[str] = [""] * len(prompts)
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        enc = tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True,
                        add_special_tokens=False).to(model.device)
        if enc["input_ids"].shape[1] > MAX_PROMPT_TOKENS:
            raise ValueError(f"prompt longer than {MAX_PROMPT_TOKENS} tokens")
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                 stopping_criteria=stop, pad_token_id=tokenizer.pad_token_id)
        texts = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for i, text in zip(idx, texts):
            responses[i] = text
        print(f"  {min(start + batch_size, len(order))}/{len(order)}", flush=True)
    return responses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="deployment export directory")
    parser.add_argument("--bench", nargs="+", default=list(BENCHES), choices=list(BENCHES))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out", type=Path, help="directory for per-problem results")
    args = parser.parse_args()

    if not find_prequantized_evidence(args.model):
        sys.exit(f"{args.model} is a latent checkpoint: loaded as it is, it would be scored at full "
                 "precision. Score latent checkpoints with scripts/eval/b1_test448.sh and b1_humaneval.sh.")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, dtype=torch.bfloat16)
    model.to("cuda").eval()

    # Exports written by eval_unified.sh all share the name model_<label>; their parent tells them apart.
    model_dir = args.model.resolve()
    out_dir = args.out or ROOT / "experiments/code_export" / model_dir.parent.name / model_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for bench in args.bench:
        path, expected = BENCHES[bench]
        rows = pd.read_parquet(path).to_dict("records")
        if len(rows) != expected:
            sys.exit(f"{path} has {len(rows)} rows, expected {expected}")
        prompts = [tokenizer.apply_chat_template([dict(m) for m in row["prompt"]], chat_template=CHAT_TEMPLATE,
                                                 add_generation_prompt=True, tokenize=False, enable_thinking=False)
                   for row in rows]
        start = time.time()
        print(f"{bench}: {len(rows)} problems, batch {args.batch_size}", flush=True)
        responses = generate(model, tokenizer, prompts, args.batch_size)

        def score(i: int) -> float:
            row = rows[i]
            return compute_score(row["data_source"], responses[i], row["reward_model"]["ground_truth"],
                                 row.get("extra_info"), require_all_tests=True)

        with ThreadPoolExecutor(max_workers=8) as pool:
            scores = list(pool.map(score, range(len(rows))))
        with open(out_dir / f"{bench}.jsonl", "w") as handle:
            for row, response, value in zip(rows, responses, scores):
                handle.write(json.dumps({"task_id": str(row["task_id"]), "response": response,
                                         "score": value}) + "\n")
        passed = int(sum(scores))
        summary[bench] = {"passed": passed, "total": len(rows), "seconds": round(time.time() - start)}
        print(f"CODE {bench} passed={passed}/{len(rows)}", flush=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"results in {out_dir}")


if __name__ == "__main__":
    main()
