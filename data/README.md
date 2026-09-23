# Training and evaluation pools

Every training pool is decontaminated against the benchmarks it is evaluated
on: no evaluation split appears in any training pool.

| Directory | Rows | Used by | Contents |
|---|---|---|---|
| `unified_math_v1/` | train 7,500 · val 16 | OPD phase 1 | GSM8K train 2,500 + MATH level 1–3 train 2,500 + DAPO-Math subset 2,500, screened against GSM8K / MATH-500 / AMC23 |
| `mbpp_official_protocol_v1/` | train 329 · val 82 · test 448 | OPD phase 2, MBPP eval | MBPP official splits. The 448 test rows are held out of training throughout |
| `humaneval_eval/` | 164 | HumanEval eval | HumanEval-164 in verl evaluation format, evaluation only |

The parquet files are not redistributed here. Build them with:

```bash
# phase-1 mathematics pool
python data/build_gsm8k_math_mix.py
python data/filter_dapo_teacher.py       # teacher-solvable DAPO subset
python data/build_math_chat_pool.py     # chat-formatted view

# phase-2 code pool and the MBPP evaluation split
python opd/data_prep/prepare_mbpp_official_protocol.py \
    --prepared_dir <raw mbpp> --output_dir data/mbpp_official_protocol_v1

# HumanEval evaluation set
python data/build_humaneval_eval.py
```

`opd/data_prep/prepare_kodcode_profile_curriculum.py` builds the optional
KodCode extension of the code pool, into whatever `--output_dir` you give it;
the launchers look for `data/kodcode_v1`. Going from 329 to 905 prompts did
not change downstream results in our runs, so the MBPP official train split
alone is a reasonable default.

Point the pipeline at a different location with `DATA_ROOT`.
