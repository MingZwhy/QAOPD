# Evaluation

| Benchmark | Script | Protocol |
|---|---|---|
| GSM8K, MATH-500 | `scripts/eval/eval_unified.sh` | lm-eval standard |
| AMC23 | `scripts/eval/amc23_avg16_parallel.sh` | avg@16 over 40 problems |
| MBPP | `scripts/eval/b1_test448.sh` | pass@1, deterministic, 448 held-out problems |
| HumanEval | `scripts/eval/b1_humaneval.sh` | pass@1, deterministic, 164 problems |
| QA9 | `scripts/eval/qa_suite.sh` | likelihood-scored multiple choice, 9 benchmarks |
| BF16 reference | `scripts/eval/fp_ceiling.sh` | the unquantized ceiling for retention |

## QA9

QA9 is the equal-weight mean of ARC-Easy, ARC-Challenge, HellaSwag, SIQA,
OpenBookQA, PIQA, WinoGrande, TruthfulQA-mc2 and MMLU, with `acc_norm` where a
task reports it and `acc` otherwise. MMLU counts as **one** of the nine: it
expands into 57 subject rows, so a mean over everything lm-eval prints weights
it 57/65 and lands several points low. `qa_suite.sh` averages the nine
task-level rows; `tools/qa9_mean.py` recomputes the same number from a saved
results file.

`QA_TASKS=edgerazor_qa` runs the twelve-task group EdgeRazor reports instead,
for a like-for-like comparison against it.

## Which artifact to evaluate

Mathematics and QA run on a **deployment export**; code runs on the
**trainable (latent) checkpoint** with quantization re-applied. This is not
interchangeable — see [CHECKPOINTS.md](CHECKPOINTS.md).

## Reading the output

Do not trust a trailing success marker alone. Check that the result file exists
and that the row count matches the benchmark (448, 164, 40, …). A short run
that produced fewer rows can otherwise be read as a score.
