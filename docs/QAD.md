# Stage 1 — quantization-aware distillation

QAD is not a contribution of this work. We reuse EdgeRazor's mixed-precision QAD
to obtain a low-bit checkpoint that is worth sampling from: round-to-nearest at
these widths collapses to zero on every generative benchmark, so on-policy
recovery has nothing to bootstrap from without this stage.

## Configurations

`configs/qad/` holds one file per model and width:

```
qwen3_06b_w2.79.yaml    qwen3_06b_w1.88.yaml
qwen3_1_7b_w2.79.yaml   qwen3_1_7b_w1.88.yaml
qwen3_4b_w2.79.yaml     qwen3_4b_w1.88.yaml
```

They differ in exactly one field: `w_mixed_precision_prop`, 0.50 for W2.79 and
0.125 for W1.88. Everything else — INT1.58 ternary blocks of 256 input channels,
INT4 embedding and output head, INT8 activations, INT8 KV cache — is shared.

## Corpus

The distillation corpus is EdgeRazor's, built from public HuggingFace datasets
by upstream's own script. Nothing is redistributed here.

```bash
WORKSPACE=runs/qad_workspace bash scripts/qad/prepare_qad_data.sh
```

| File | Source | Rows | In the mixture |
|---|---|---|---|
| `task_0.2M_instruct.jsonl` | mixed downstream QA (ARC, HellaSwag, BoolQ, PIQA, WinoGrande, SIQA, OpenBookQA, ethics) | 243,428 | ×6 → 30.0% |
| `am_1.4M_instruct.jsonl` | `a-m-team/AM-DeepSeek-R1-Distilled-1.4M` | 1,400,000 | ×2 → 57.5% |
| `tulu_0.6M_instruct.jsonl` | `allenai/tulu-v3.1-mix-preview-4096-OLMoE` | 608,042 | ×1 → 12.5% |

**4,868,610 rows, so one epoch is 6,339 steps** at 768 sequences per step. The
per-arm budgets below are whole epochs of this corpus.

The repeats are symlinks to one physical file — the trainer opens each name
separately, so the exposure repeats without the disk cost.

AM is doubled and `BAAI/Infinity-Instruct` (Gen) left out because AM is by far
the densest source of code and mathematics and Gen the thinnest:

| | code | mathematics |
|---|---:|---:|
| `am_1.4M` | 30.5% | 38.7% |
| `tulu_0.6M` | 25.8% | 22.9% |
| `ii_7M` | 14.4% | 24.1% |
| `ii_gen_1.4M` | 9.5% | 14.2% |

Gen was the largest single block of the original four-file mixture. Replacing
it with a second pass over AM takes the corpus from 14.7% code / 18.0%
mathematics to **20.8% / 25.1%**, and makes it smaller. `DATASETS` can still
name `ii_gen_1.4M` or `ii_7M` if you want the older mixture; `TASK_UPSAMPLE`
and `AM_UPSAMPLE` change the repeat counts.

The download is about 6.9 GB and takes a couple of hours.
`DATASETS=task_0.2M` builds only the small file, which is enough to check that
the launcher runs.

The build happens in a throwaway venv, deliberately: it installs its own
`datasets` release, and doing that inside `venvs/qaopd-{train,eval}` would
rewrite a pinned environment.

Several of the sources `task_0.2M` draws on are still script-based datasets,
which `datasets>=4` refuses to run, and four are named by canonical ids the hub
has since retired. Our patch to `data_prepare.sh` remaps those ids and falls
back to each dataset's auto-converted parquet branch, which holds the same
rows, retrying when the hub rate-limits. That patch is applied by
`setup/bootstrap.sh`; without it the builder dies on its first entry.

## Running

```bash
COMBO=qwen3_1_7b_w2.79 \
MODEL=models/Qwen3-1.7B \
WORKSPACE=runs/qad_workspace \
NGPU=8 \
bash scripts/qad/run_qad.sh
```

The teacher and student both start from `MODEL`. The objective is an online
logit KL against the frozen BF16 teacher plus a small task loss.

QAD runs in the **evaluation** environment, not the training one: it is a
DeepSpeed job, and deepspeed and bitsandbytes are pinned there. `QAD_ENV`
overrides that.

`COMBO` and `MODEL` are the only two things you have to supply. Everything
below has a default.

### Shared across all six arms

```
batch      768 sequences per step, on 8 GPUs
           Qwen3-0.6B   12 per device x 8 accumulation
           Qwen3-1.7B    6 per device x 16
           Qwen3-4B      6 per device x 16
sequence   1024
lr         2e-5, constant_with_warmup
```

Every arm takes the same 768-sequence optimizer step; only the split between
per-device batch and accumulation differs, which is mathematically the same
step. Fewer, larger micro-steps are faster whenever they fit — on Qwen3-4B,
`6 x 16` beat `3 x 32` by 19%, and on Qwen3-0.6B, `12 x 8` beats `6 x 16` by
17%. The ceiling is the KD logits tensor (`bs x seq x vocab x 4 B`): 0.6B at
`bs 24` asks for another 7 GiB and dies, and 1.7B and 4B already peak at 42.6
and 52.6 of 71.1 GiB at `bs 6`. All measured on 72 GB cards; on smaller cards,
lower `ER_PER_DEVICE_BS` and raise `ER_GRAD_ACC` to keep the product at 96.

Keep the schedule constant. Extending a `constant_with_warmup` run costs
nothing — warmup is long past and the rate never changes — whereas any cosine
variant derives its whole curve from the total step count, so a later extension
would not continue the same schedule. Checkpoints are selected by scanning, not
by decaying to the endpoint. `run_qad.sh` rejects any other scheduler: the
trainer always passes `min_lr`, which only `constant_with_warmup` and
`cosine_with_min_lr` tolerate.

### Per arm

| `COMBO` | steps | epochs | save every | keep |
|---|---:|---:|---:|---:|
| `qwen3_06b_w2.79` | 12,800 | 2 | 100 | all |
| `qwen3_06b_w1.88` | 12,800 | 2 | 100 | all |
| `qwen3_1_7b_w2.79` | 6,400 | 1 | 100 | all |
| `qwen3_1_7b_w1.88` | 12,800 | 2 | 100 | all |
| `qwen3_4b_w2.79` | 9,600 | 1.5 | 200 | 10 |
| `qwen3_4b_w1.88` | 6,400 | 1 | 200 | 10 |

A Qwen3-4B checkpoint is about 31 GB against 13 GB for 1.7B, so the dense
every-100 policy would need a couple of terabytes there; 4B halves the
frequency and caps what stays on local disk. Raise `ER_SAVE_TOTAL_LIMIT` if you
have the room, and set it to 0 before extending a run — upstream's cap deletes
the early checkpoints, which are often the ones a scan picks.

`ER_STEPS` overrides the budget. Extending later is a resume:
`ER_RESUME=1 ER_STEPS=<larger> ER_SAVE_TOTAL_LIMIT=0`.

### How long it takes

Measured on 8 × RTX PRO 5000 (72 GB) at the shape above, sdpa, ZeRO-3, with the
teacher resident alongside the student:

| model | s/step | its default budget | wall clock |
|---|---:|---:|---:|
| Qwen3-0.6B | 15.5 | 12,800 | 55 h |
| Qwen3-1.7B | 24.6 | 6,400 / 12,800 | 44 h / 87 h |
| Qwen3-4B | 46.6 | 9,600 / 6,400 | 124 h / 83 h |

All six arms come to roughly **450 GPU-node hours**. On two 8-GPU nodes running
two arms at a time that is about nine days, and the two 4B arms are nearly half
of it. A single arm is two to five days, so run them detached with a watchdog
rather than in a shell you have to keep open.

### The wiring assertion

`run_qad.sh` re-reads `config.py` after wiring and refuses to launch if the
quantizer config, schedule, learning rate or batch did not land. That file is a
working-tree artifact which keeps whatever the previous run wrote and clears
nothing, so a field the wiring does not set survives into the next arm. The
quantizer config is the one that matters most: the two widths differ by a
single field, and training the wrong one produces a healthy-looking checkpoint
of the wrong model.

## Choosing the checkpoint to hand to OPD

Save frequently and **do not assume the training endpoint is the best starting
point**. In our runs the useful checkpoint was often well before the end, and
the QAD curve is not monotone on the generative benchmarks. `ER_SAVE_STEPS`
controls save density; `ER_SAVE_TOTAL_LIMIT=0` keeps every checkpoint.

Evaluate several candidates with `scripts/eval/eval_unified.sh` and pick on the
metric you care about, then pass that checkpoint to stage 2 as `STUDENT_MODEL`.

The budgets above are sized so that the useful region is inside them with room
to scan around it, not so that the endpoint is the answer. On the arms we ran,
the picked checkpoint sat at roughly 50–95% of the budget.
