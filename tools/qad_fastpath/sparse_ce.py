"""Causal CE that skips ignored vocabulary rows.

Preserve native shifted labels, FP32 logits, and the exact valid-token divisor.
Do not replace a microbatch mean with a different global token weighting.
"""
import torch
import torch.nn.functional as F


def sparse_causal_loss(logits, labels, vocab_size, num_items_in_batch=None,
                       ignore_index=-100, shift_labels=None, **kwargs):
    from transformers.loss.loss_utils import fixed_cross_entropy
    if shift_labels is None:
        shift_labels = F.pad(labels, (0, 1), value=ignore_index)[..., 1:].contiguous()
    targets = shift_labels.reshape(-1).to(logits.device)
    indices = targets.ne(ignore_index).nonzero().flatten()
    selected = logits.reshape(-1, vocab_size).index_select(0, indices).float()
    targets = targets.index_select(0, indices)
    # Empty-label microbatches retain native CE's NaN mean / zero sum behavior.
    return fixed_cross_entropy(selected, targets, num_items_in_batch, ignore_index, **kwargs)


def install(model):
    original = model.loss_function
    from transformers.loss.loss_utils import ForCausalLMLoss
    assert original is ForCausalLMLoss, 'Unexpected task loss implementation'
    model.loss_function = sparse_causal_loss

    def restore():
        model.loss_function = original
    return restore


def fixture_check(device='cpu', full_vocabulary=False):
    from transformers.loss.loss_utils import ForCausalLMLoss
    generator = torch.Generator(device=device).manual_seed(7281)
    cases = []
    shapes = [(3, 17, 251), (6, 37, 1024)]
    if full_vocabulary:
        shapes += [(2, 1024, 151936), (6, 1024, 151936)]
    dtypes = [torch.float32, torch.bfloat16]
    for shape in shapes:
        for dtype in dtypes:
            if shape[-1] == 151936 and dtype != torch.bfloat16:
                continue  # formal model supplies BF16 logits, not FP32 logits
            for explicit_shift in [False, True]:
                logits = torch.randn(shape, generator=generator, device=device, dtype=dtype)
                labels = torch.randint(shape[-1], shape[:2], generator=generator, device=device)
                ignore = torch.rand(shape[:2], generator=generator, device=device) < .7
                labels[ignore] = -100
                # Include both a completely ignored row and ignored first/last labels.
                labels[0] = -100
                labels[:, 0] = labels[:, -1] = -100
                for denominator in [None, 500]:
                    kwargs = dict(vocab_size=shape[-1], num_items_in_batch=denominator)
                    if explicit_shift:
                        kwargs['shift_labels'] = F.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()
                    logits.requires_grad_(True)
                    native = ForCausalLMLoss(logits=logits, labels=labels, **kwargs)
                    native.backward()
                    wanted = logits.grad.detach().clone()
                    logits.grad = None
                    candidate = sparse_causal_loss(logits=logits, labels=labels, **kwargs)
                    candidate.backward()
                    difference = float(abs(native.detach() - candidate.detach()))
                    torch.testing.assert_close(native, candidate, rtol=2e-6, atol=2e-6)
                    torch.testing.assert_close(wanted, logits.grad, rtol=0, atol=0)
                    cases.append(dict(shape=shape, dtype=str(dtype), explicit_shift=explicit_shift,
                                      denominator=denominator, loss_absolute_error=difference,
                                      gradients_identical=True))
                    logits.grad = None
                    del native, candidate, wanted
                del logits, labels
    logits = torch.randn(2, 4, 17, device=device, generator=generator).requires_grad_()
    labels = torch.full((2, 4), -100, device=device)
    for denominator in [None, 500]:
        values, gradients = [], []
        for function in [ForCausalLMLoss, sparse_causal_loss]:
            logits.grad = None
            loss = function(logits, labels, 17, num_items_in_batch=denominator)
            loss.backward()
            values.append(loss.detach())
            gradients.append(logits.grad.detach().clone())
        torch.testing.assert_close(*values, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(*gradients, rtol=0, atol=0)
        cases.append(dict(all_labels_ignored=True, denominator=denominator,
                          native_nan_mean_preserved=denominator is None, gradients_identical=True))
    return dict(status='PASS', device=device, full_vocabulary=full_vocabulary, cases=cases)
