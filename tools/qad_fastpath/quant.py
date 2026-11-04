"""BF16 INT8 activation quantization with native intermediate rounding."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _quant_int8(X, Y, nblocks, EPS: tl.constexpr, BLOCK: tl.constexpr, ROWS: tl.constexpr,
                STE: tl.constexpr = False):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK)
    offsets = rows[:, None] * BLOCK + cols[None, :]
    values = tl.load(X + offsets, rows[:, None] < nblocks, other=0).to(tl.float32)
    maximum = tl.max(tl.abs(values), axis=1)
    epsilon = tl.full((), EPS, tl.float32).to(tl.bfloat16).to(tl.float32)
    maximum = tl.maximum(maximum, epsilon)
    # PyTorch materializes each BF16 intermediate, including the division.
    scale = tl.div_rn(maximum, 127.0).to(tl.bfloat16).to(tl.float32)
    normalized = tl.div_rn(values, scale[:, None]).to(tl.bfloat16).to(tl.float32)
    rounded = libdevice.nearbyint(normalized)
    quantized = tl.minimum(tl.maximum(rounded, -127.0), 127.0) * scale[:, None]
    if STE:
        # Native STE also rounds the subtraction and final addition in BF16.
        quantized = quantized.to(tl.bfloat16).to(tl.float32)
        delta = (quantized - values).to(tl.bfloat16).to(tl.float32)
        quantized = values + delta
    tl.store(Y + offsets, quantized.to(tl.bfloat16), rows[:, None] < nblocks)


def fused_activation_quant(x, epsilon=1e-5, block_size=256):
    from edgerazor.qat.util.quant_function import state_quant_uniform_symmetric_absmax_per_block_int8
    if (not x.is_cuda or x.dtype != torch.bfloat16 or not x.is_contiguous()
            or block_size not in (128, 256) or x.shape[-1] % block_size):
        return state_quant_uniform_symmetric_absmax_per_block_int8(
            x=x, epsilon=epsilon, block_size=block_size)
    out = torch.empty_like(x)
    nblocks = x.numel() // block_size
    _quant_int8[(triton.cdiv(nblocks, 8),)](x, out, nblocks, epsilon, block_size, 8,
                                          num_warps=4, enable_fp_fusion=False)
    return out


class IdentityGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, source, computed):
        return computed

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None


def fused_activation_ste(x, epsilon=1e-5, block_size=256):
    if (not x.is_cuda or x.dtype != torch.bfloat16 or not x.is_contiguous()
            or block_size not in (128, 256) or x.shape[-1] % block_size):
        return x + (fused_activation_quant(x, epsilon, block_size) - x).detach()
    out = torch.empty_like(x)
    nblocks = x.numel() // block_size
    _quant_int8[(triton.cdiv(nblocks, 8),)](x, out, nblocks, epsilon, block_size, 8, True,
                                          num_warps=4, enable_fp_fusion=False)
    return IdentityGradient.apply(x, out)


def install(model):
    replacements = []
    for module in model.modules():
        function = getattr(module, 'a_quant_function', None)
        if function is not None and function.__name__ == 'state_quant_uniform_symmetric_absmax_per_block_int8':
            replacements.append((module, function))
            module.a_quant_function = fused_activation_quant
    assert replacements

    def restore():
        for module, function in replacements:
            module.a_quant_function = function
    return restore, len(replacements)


def fixture_check():
    from edgerazor.qat.util.quant_function import state_quant_uniform_symmetric_absmax_per_block_int8 as native
    # Fixture generation must not perturb resumed training RNG state.
    generator = torch.Generator(device='cuda').manual_seed(3047)
    cases = []
    for shape in [(2, 17, 2048), (2, 1024, 6144), (1, 31, 256)]:
        for block in [128, 256]:
            for magnitude in [0.0, 1e-7, 1.0, 100.0]:
                x = (torch.randn(shape, device='cuda', generator=generator) * magnitude).to(torch.bfloat16)
                a = native(x=x, block_size=block)
                b = fused_activation_quant(x, block_size=block)
                if not torch.equal(a, b):
                    mismatch = a != b
                    raise AssertionError(dict(shape=shape, block=block, magnitude=magnitude,
                        mismatches=int(mismatch.sum()), maximum=float((a.float() - b.float()).abs().max())))
                torch.testing.assert_close(x + (a - x).detach(),
                                           fused_activation_ste(x, block_size=block), rtol=0, atol=0)
                cases.append(dict(shape=shape, block=block, magnitude=magnitude, identical=True))
    # Exercise rounding ties and extremal values using the same BF16 inputs.
    x = torch.linspace(-127, 127, 256, device='cuda', dtype=torch.bfloat16).repeat(256, 1)
    torch.testing.assert_close(native(x=x, block_size=256), fused_activation_quant(x, block_size=256),
                               rtol=0, atol=0)
    return dict(status='PASS', GPU_tested=True, bitwise_identical=True, cases=cases)
