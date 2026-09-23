# Evaluate the released weights

The fastest path: download a recovered checkpoint and score it. No training,
no corpus to build. Expect about two hours for one model across every
benchmark, most of it MBPP and HumanEval.

Prerequisite: [INSTALL.md](INSTALL.md), through `bootstrap.sh`.

## 1 · Download

```bash
export HF_TOKEN=<your token>     # the hub rate-limits anonymous downloads
bash tools/fetch_weights.sh qwen3_1_7b_w2.79
```

Run it with no arguments to see every arm. It picks the repository and the
destination; `--qad` fetches the training starting point instead, where one is
published. Or do it by hand:

```bash
hf download MingZwhy/Qwen3-1.7B-W2.79-QAOPD --local-dir models/q17_w279
```

A token matters here: without one the hub cuts off multi-GB downloads partway
with a 429 that reads like a network error. `hf download` resumes, so re-run
after setting it.

| Model | Width | Repository |
|---|---|---|
| Qwen3-0.6B | W2.79 | `MingZwhy/Qwen3-0.6B-W2.79-QAOPD` |
| Qwen3-0.6B | W1.88 | `MingZwhy/Qwen3-0.6B-W1.88-QAOPD` |
| Qwen3-1.7B | W2.79 | `MingZwhy/Qwen3-1.7B-W2.79-QAOPD` |
| Qwen3-1.7B | W1.88 | `MingZwhy/Qwen3-1.7B-W1.88-QAOPD` |
| Qwen3-4B | W2.79 | `MingZwhy/Qwen3-4B-W2.79-QAOPD` |
| Qwen3-4B | W1.88 | `MingZwhy/Qwen3-4B-W1.88-QAOPD` |

These are **deployment exports**: the low-bit weights are baked in, so
`from_pretrained` gives you the quantized model and the scores are the real
ones. The `-QAD` repositories are the other form and are for training only;
[CHECKPOINTS.md](CHECKPOINTS.md) explains why mixing them up is silent.

## 2 · Score

Set `BITWIDTH` to match the checkpoint. It selects the quantizer config, and
the scripts refuse to run if it disagrees with the weights.

```bash
export BITWIDTH=w2.79
export CKPT=models/q17_w279
```

**Mathematics** — GSM8K 5-shot strict-match and MATH-500 4-shot:

```bash
bash scripts/eval/eval_unified.sh
```

**AMC23** — 40 problems, so a single greedy pass is meaningless; this runs
avg@16:

```bash
bash scripts/eval/amc23_avg16_parallel.sh
```

**QA9** — the equal-weight mean of nine likelihood-scored benchmarks:

```bash
MODEL=$CKPT TAG=q17_w279 GPUS=0,1,2,3 bash scripts/eval/qa_suite.sh
```

**Code** — MBPP on the held-out 448 and HumanEval-164, both greedy pass@1:

```bash
bash scripts/eval/b1_test448.sh
bash scripts/eval/b1_humaneval.sh
```

## 3 · Read the output

Each script prints a marker line with the score:

```
UNIEVAL <tag> step<N> gsm8k: 0.5413
UNIEVAL <tag> step<N> math500: 0.4960
B1T448 step=<N> passed=233/448
HE step=<N> passed=97/164
QA_AVG tag=<tag> avg=51.81 over=9
```

Do not trust a trailing `_DONE` on its own. Check the row count matches the
benchmark — 448, 164, 40 — because a short run that produced fewer rows still
reports a score.

`QA_AVG ... over=9` is the number that belongs in the QA9 column. If it says
`over=70`, the mean included MMLU's 57 subject rows and is several points low;
`tools/qa9_mean.py` recomputes it correctly from a saved results file.

## What you should get

The values in the [results table](../README.md#results). Small deviations are
normal across hardware and batch shape: in our own cross-machine checks the
mathematics scores reproduced to the digit, and the code scores moved by one
or two problems out of 448 and 164.

Two protocol details that move numbers more than hardware does:

- **GSM8K must be strict-match.** lm-eval reports both strict and flexible;
  the flexible number is higher and is not what the table reports.
- **Align "GPU count × batch size" before comparing across machines.** Matched,
  the scores reproduce exactly; unmatched, a two-point difference means
  nothing.
