"""Experimental masked-token KD: skip vocabulary work at ignored positions.

Keep the original [batch, sequence] reduction layout and per-example weighting.
No model/attention packing, shift changes, or confidence-coefficient changes.
Enabled only by the opt-in QAD fastpath.
"""
import copy
import torch
import torch.nn.functional as F


def install():
    from edgerazor.kd.util import distill_function as native
    original_kld = native.compute_kld
    original_confidence = native.compute_teacher_confidence

    def supported(logits, labels, cfg):
        return (logits.ndim == 3 and labels is not None
                and tuple(labels.shape) == tuple(logits.shape[:2]) and not cfg.is_router_logits)

    def sparse_kld(input_logits, target_logits, labels, kd_config_loss):
        cfg = kd_config_loss
        if not supported(input_logits, labels, cfg):
            return original_kld(input_logits, target_logits, labels, cfg)
        mask = labels.eq(cfg.padding_id)
        indices = (~mask).reshape(-1).nonzero().flatten()
        vocab = input_logits.shape[-1]
        # Separate selections retain the native two-KL autograd branches.
        x = input_logits.reshape(-1, vocab).index_select(0, indices).unsqueeze(0)
        y = target_logits.reshape(-1, vocab).index_select(0, indices).unsqueeze(0)
        element_cfg = copy.copy(cfg)
        element_cfg.reduction = 'none'
        if indices.numel():
            values = original_kld(x, y, None, element_cfg).reshape(-1)
        else:
            values = x.sum(dim=-1).reshape(-1)
        # Sum along exactly the original axes, including zero padding slots.
        raw = values.new_zeros(labels.numel()).index_copy(0, indices, values).view_as(labels)
        valid = (~mask).sum(dim=-1)
        if cfg.reduction == 'sum':
            return raw.sum()
        if cfg.reduction == 'mean':
            return raw.sum() / valid.sum().clamp(min=1)
        if cfg.reduction == 'batch_mean':
            return (raw.sum(dim=-1) / valid.clamp(min=1)).mean()
        if cfg.reduction == 'none':
            return raw
        raise ValueError(cfg.reduction)

    def sparse_confidence(teacher_logits, labels, kd_config_loss):
        cfg = kd_config_loss
        if not supported(teacher_logits, labels, cfg) or not cfg.use_entropy:
            return original_confidence(teacher_logits, labels, cfg)
        mask = labels.eq(cfg.padding_id)
        indices = (~mask).reshape(-1).nonzero().flatten()
        values = teacher_logits.reshape(-1, teacher_logits.shape[-1]).index_select(0, indices)
        probs = F.softmax(values, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        entropy = entropy.new_zeros(labels.numel()).index_copy(0, indices, entropy).view_as(labels)
        lengths = (~mask).sum(dim=-1).float().clamp(min=1)
        sample_average = entropy.sum(dim=-1) / lengths
        maximum = torch.log(torch.tensor(cfg.confidence_k, dtype=torch.float, device=values.device))
        return (1.0 - (sample_average / maximum).mean()).clamp(min=0.0, max=1.0)

    native.compute_kld = sparse_kld
    native.compute_teacher_confidence = sparse_confidence

    def restore():
        native.compute_kld = original_kld
        native.compute_teacher_confidence = original_confidence
    return restore
