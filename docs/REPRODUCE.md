# Reproduce from scratch

The whole chain: BF16 model → QAD → OPD mathematics → OPD code → evaluation.
This is days of GPU time. If you only want the numbers, start at
[EVALUATE_RELEASED.md](EVALUATE_RELEASED.md) instead.

```
BF16 model
   └─ QAD          6,400–12,800 steps      44–124 h on 8 GPUs
        └─ OPD ①   mathematics, ~80 steps   a few hours on 2 GPUs
             └─ OPD ②  code, ~120 steps     a few hours on 2 GPUs
                  └─ evaluate
```

Prerequisite: [INSTALL.md](INSTALL.md), through `bootstrap.sh`, and the base
models under `models/`.

---

## Stage 0 · Build the corpora

QAD and OPD use different data, built by different scripts.

```bash
# QAD: ~6.9 GB, a couple of hours
WORKSPACE=runs/qad_workspace bash scripts/qad/prepare_qad_data.sh

# OPD: the mathematics and code pools
python data/build_gsm8k_math_mix.py
python data/filter_dapo_teacher.py
python data/build_math_chat_pool.py
python opd/data_prep/prepare_mbpp_official_protocol.py \
    --prepared_dir <raw mbpp> --output_dir data/mbpp_official_protocol_v1
python data/build_humaneval_eval.py
```

Details in [QAD.md](QAD.md#corpus) and [`data/README.md`](../data/README.md).

---

## Stage 1 · QAD

```bash
COMBO=qwen3_1_7b_w2.79 \
MODEL=models/Qwen3-1.7B \
WORKSPACE=runs/qad_workspace \
NGPU=8 \
bash scripts/qad/run_qad.sh
```

`COMBO` and `MODEL` are the only required arguments; the step budget, save
policy and batch shape all default per arm. Budgets and measured wall-clock
are in [QAD.md](QAD.md#per-arm).

This is the long stage — 44 hours for Qwen3-1.7B W2.79, up to 124 for
Qwen3-4B W2.79. Run it detached.

**Watch the loss.** Every step prints `loss`, `train/loss_total`,
`train/loss_task`, `train/loss_dist` and its components, plus `grad_norm` and
`learning_rate`. When it finishes, or at any point during it:

```bash
python tools/loss_curve.py runs/qad_workspace/EdgeRazor-QLLM/train
```

That writes `loss_curve.csv` and `loss_curve.png` — total/task/distillation,
the distillation breakdown, and grad norm — from the full log history in
`trainer_state.json`.

### Choosing the checkpoint

**Do not assume the endpoint is best.** The QAD curve is not monotone on the
generative benchmarks and the useful checkpoint is often well before the end.
Scan candidates and pick:

```bash
BITWIDTH=w2.79 RUN=runs/qad_workspace/EdgeRazor-QLLM/train \
    bash scripts/eval/scan_math_ckpts.sh
```

Then hand the one you picked to stage 2.

---

## Stage 2 · OPD, mathematics

```bash
BITWIDTH=w2.79 \
STUDENT_MODEL=<the QAD checkpoint you picked> \
STEPS=80 \
bash scripts/opd/run_math.sh
```

Corpus `unified_math_v1`. Learning rate 3e-6, 8 prompts per optimizer step,
G=4 completions each. Checkpoint every 20 steps and select on validation —
the optimum here is frequently early too.

Skipping stage 1 is fine for Qwen3-0.6B: the QAD starting points are on the
Hub.

```bash
huggingface-cli download MingZwhy/Qwen3-0.6B-W2.79-QAD --local-dir models/w279_qad
BITWIDTH=w2.79 STUDENT_MODEL=models/w279_qad bash scripts/opd/run_math.sh
```

---

## Stage 3 · OPD, code

```bash
BITWIDTH=w2.79 \
STUDENT_MODEL=<the phase-1 checkpoint you picked> \
STEPS=120 \
bash scripts/opd/run_code.sh
```

Trains on the MBPP official *train* split; the 448 test rows stay held out.
Learning rate 3e-6, 4 prompts per step, G=4.

More on both stages in [OPD.md](OPD.md).

---

## Stage 4 · Evaluate

Same commands as for the released weights, pointed at your own checkpoint —
see [EVALUATE_RELEASED.md](EVALUATE_RELEASED.md#2--score). One thing changes:
your OPD output is a **latent** checkpoint, so

- mathematics and QA run on a **deployment export**; `eval_unified.sh`
  produces one for you from a latent checkpoint,
- the code evaluations run on the **latent** checkpoint directly, because they
  re-apply quantization themselves.

Feeding an export to the code scripts double-quantizes and reports a wrong
number without erroring. [CHECKPOINTS.md](CHECKPOINTS.md) has the guard rails.

---

## What to expect

The mathematics benchmarks are the ones to judge by: they are where
quantization does the most damage and where on-policy recovery does the most
work. Compare against the [results table](../README.md#results).
