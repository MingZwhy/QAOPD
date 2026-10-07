# Training and evaluation pools

Every training pool is decontaminated against the benchmarks it is evaluated
on: no evaluation split appears in any training pool.

| Directory | Rows | Used by | Contents |
|---|---|---|---|
| `unified_math_v1/` | train 7,500 · val 16 | OPD phase 1 | GSM8K train 2,500 + MATH level 1–3 train 2,500 + DAPO-Math subset 2,500, screened against GSM8K / MATH-500 / AMC23 |
| `mbpp_official_protocol_v1/` | train 329 · val 82 · test 448 | OPD phase 2, MBPP eval | MBPP official splits. The 448 test rows are held out of training throughout |
| `code_opd_24k/` | train 25,685 · val 82 | OPD phase 2 | 24,040 OpenCoder educational_instruct and KodCode-V1 problems + MBPP official train ×5 |
| `code_opd_16k/` | train 17,629 · val 82 | OPD phase 2 | 15,984 of the `code_opd_24k` problems + MBPP official train ×5 |
| `code_opd_12k/` | train 13,372 · val 82 | OPD phase 2 | the 11,727 `code_opd_16k` problems BF16 Qwen3-1.7B solved in all four of its samples + MBPP official train ×5 |
| `code_opd_18k/` | train 19,830 · val 82 | OPD phase 2 | 18,185 further OpenCoder and KodCode-V1 problems, none of them in `code_opd_24k` + MBPP official train ×5 |
| `humaneval_eval/` | 164 | HumanEval eval | HumanEval-164 in verl evaluation format, evaluation only |

The parquet files are not redistributed here. Build them with:

```bash
# phase-1 mathematics pool
python data/build_gsm8k_math_mix.py
python data/filter_dapo_teacher.py       # teacher-solvable DAPO subset
python data/build_math_chat_pool.py     # chat-formatted view

# MBPP official splits: training rows for the code pools, and the evaluation split
python opd/data_prep/prepare_mbpp_official_protocol.py \
    --prepared_dir <raw mbpp> --output_dir data/mbpp_official_protocol_v1

# phase-2 code pools (needs the MBPP splits; downloads about 2.6 GB of sources)
python data/build_code_pools.py

# HumanEval evaluation set
python data/build_humaneval_eval.py
```

The code pools are listed row by row in [`code_pools/`](code_pools/), one
`task_id` per line in training order: `opc_<seq_id>` for OpenCoder
educational_instruct, `kod_<question_id>` for KodCode-V1, `mbpp_<task_id>` for
the MBPP official train split. `build_code_pools.py` fetches the two source
datasets, turns each listed problem into a function stub, docstring and tests,
and checks every pool against its expected digest. Every
problem's reference solution passes its tests. Run the script with the training
environment's python: the stubs come from `ast.unparse`, whose output has
changed between Python releases.

`opd/data_prep/prepare_kodcode_profile_curriculum.py` builds `data/kodcode_v1`,
a smaller KodCode pool for the launcher's `kodcode_k1_t17_strict_lr3e6_s30`
variant; `scripts/opd/run_code.sh` does not use it.

Point the pipeline at a different location with `DATA_ROOT`.
