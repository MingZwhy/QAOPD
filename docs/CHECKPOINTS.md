# Checkpoint forms

Two artifact forms exist and they are not interchangeable.

**Latent / trainable checkpoint.** BF16 master weights, with quantization
re-applied at training or rollout time. This is what QAD writes and what OPD
resumes from and produces. Use it to continue training and for the code
evaluations.

**Deployment export.** Low-bit weights baked in, plus EdgeRazor runtime
metadata. Produced by the export step in `opd/launch/evaluate_opd_qad_w279a8.sh`.
Use it for lm-eval mathematics and QA, and for serving.

## Why this matters

Both mistakes run to completion without error:

- Training from an export starts from already-rounded weights and silently
  discards the master-weight precision the method depends on.
- Evaluating a latent checkpoint as an ordinary model measures the BF16 master
  weights, not the quantized model — the score is real but it is not the
  quantized model's score.

`opd/tools/validate_qat_latent_checkpoint.py` checks which form a directory
holds before you use it.

## Naming

Directory and mode names may carry a legacy `kv8` tag. It reflects the QAD
stage, where the KV cache is INT8. During OPD and evaluation the KV cache is
16-bit. The authoritative setting is the `kv_cache_function` field of the
config in `configs/opd/`, which is absent there by design.
