"""CPU equivalence and cache lifecycle tests; CUDA fixtures require opt-in."""
import copy
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from qad_fastpath.reuse import cache_quantized_weights_within_step, omit_unused_teacher_loss


class FastpathTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_weight_cache_across_three_optimizer_updates(self):
        from edgerazor import EdgeRazor
        config = yaml.safe_load((ROOT / 'configs/qad/qwen3_1_7b_w2.79.yaml').read_text())['qat_configuration']
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                torch.manual_seed(73)
                model = torch.nn.Sequential(torch.nn.Linear(256, 256, bias=False)).to(dtype)
                native = EdgeRazor(qat_config=copy.deepcopy(config)).quantize(model).train()
                cached = copy.deepcopy(native)
                keys = set(cached.state_dict())
                quantizer = cached[0].w_quant_function
                cached[0].w_quant_function = Mock(wraps=quantizer)
                cached[0].w_quant_function.__name__ = quantizer.__name__
                clear, restore, count = cache_quantized_weights_within_step(cached)
                self.assertEqual(count, 1)
                try:
                    for step in range(3):
                        clear()
                        native.zero_grad(set_to_none=True)
                        cached.zero_grad(set_to_none=True)
                        for _ in range(3):
                            x = torch.randn(2, 4, 256).to(dtype)
                            target = torch.randn(2, 4, 256)
                            a, b = native(x), cached(x)
                            torch.testing.assert_close(a, b, rtol=0, atol=0)
                            ((a.float()-target).square().mean()/3).backward()
                            ((b.float()-target).square().mean()/3).backward()
                        torch.testing.assert_close(native[0].weight.grad, cached[0].weight.grad, rtol=0, atol=0)
                        self.assertEqual(cached[0].w_quant_function.call_count, step + 1)
                        with torch.no_grad():
                            for a, b in zip(native.parameters(), cached.parameters()):
                                a.add_(a.grad, alpha=-.01)
                                b.add_(b.grad, alpha=-.01)
                        self.assertEqual(set(cached.state_dict()), keys)
                        cache = cached[0].__dict__['_qaopd_step_quantized_weight']
                        self.assertFalse(cache.requires_grad)
                        self.assertIsNone(cache.grad_fn)
                    # Evaluation bypasses the training cache, even before clear.
                    native.eval(); cached.eval()
                    torch.testing.assert_close(native(x), cached(x), rtol=0, atol=0)
                finally:
                    restore()
                self.assertNotIn('_qaopd_step_quantized_weight', cached[0].__dict__)

    def test_teacher_wrapper_is_reversible(self):
        class Teacher:
            def forward(self, **kwargs):
                return kwargs
        teacher = Teacher()
        original = teacher.forward
        restore = omit_unused_teacher_loss(teacher)
        result = teacher.forward(labels='unused', use_cache=True, input_ids='tokens')
        self.assertEqual(result, dict(use_cache=False, input_ids='tokens'))
        restore()
        self.assertEqual(teacher.forward, original)

    def test_fused_linear_matches_native_after_weight_updates(self):
        from edgerazor import EdgeRazor
        from qad_fastpath.linear import install
        config = yaml.safe_load((ROOT / 'configs/qad/qwen3_1_7b_w2.79.yaml').read_text())['qat_configuration']
        for dtype in (torch.float32, torch.bfloat16):
            torch.manual_seed(91)
            native = EdgeRazor(qat_config=copy.deepcopy(config)).quantize(
                torch.nn.Sequential(torch.nn.Linear(256, 256, bias=False)).to(dtype)).train()
            fused = copy.deepcopy(native)
            clear_weight, restore_weight, _ = cache_quantized_weights_within_step(fused)
            restore_linear, clear_linear, _ = install(fused)
            try:
                for _ in range(3):
                    clear_weight(); clear_linear()
                    native.zero_grad(set_to_none=True); fused.zero_grad(set_to_none=True)
                    for _ in range(2):
                        x = torch.randn(2, 3, 256).to(dtype)
                        a, b = native(x), fused(x)
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                        a.float().square().mean().backward()
                        b.float().square().mean().backward()
                    for a, b in zip(native.parameters(), fused.parameters()):
                        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
                        with torch.no_grad():
                            a.add_(a.grad, alpha=-.01); b.add_(b.grad, alpha=-.01)
            finally:
                restore_linear(); restore_weight()
            self.assertFalse(any('_qaopd_step_effective_weight' in m.__dict__ for m in fused.modules()))

    def test_masked_ce_preserves_native_loss_and_gradients(self):
        from qad_fastpath.sparse_ce import fixture_check
        self.assertEqual(fixture_check()['status'], 'PASS')

    def test_masked_kd_preserves_reductions_and_confidence(self):
        from edgerazor.kd.util import distill_function as native
        from qad_fastpath.sparse_kd import install
        torch.manual_seed(7)
        teacher = torch.randn(3, 7, 31)
        inputs = torch.randn(3, 7, 31)
        labels = torch.randint(31, (3, 7))
        labels[0] = -100
        labels[1, 3:] = -100
        for reduction in ('sum', 'mean', 'batch_mean', 'none'):
            for all_ignored in (False, True):
                for entropy in (False, True):
                    with self.subTest(reduction=reduction, all_ignored=all_ignored, entropy=entropy):
                        cfg = SimpleNamespace(reduction=reduction, temperature=1.3,
                            padding_id=-100, is_router_logits=False,
                            use_entropy=entropy, confidence_k=16)
                        target = torch.full_like(labels, -100) if all_ignored else labels
                        a, b = inputs.clone().requires_grad_(), inputs.clone().requires_grad_()
                        expected = native.compute_kld_confidence(a, teacher, target, cfg)
                        confidence = native.compute_teacher_confidence(teacher, target, cfg)
                        expected.sum().backward()
                        original = native.compute_kld
                        restore = install()
                        try:
                            actual = native.compute_kld_confidence(b, teacher, target, cfg)
                            actual.sum().backward()
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                            torch.testing.assert_close(b.grad, a.grad, rtol=0, atol=0)
                            torch.testing.assert_close(native.compute_teacher_confidence(teacher, target, cfg), confidence, rtol=0, atol=0)
                            # Unmasked paths keep the native implementation.
                            torch.testing.assert_close(native.compute_kld(inputs, teacher, None, cfg), original(inputs, teacher, None, cfg), rtol=0, atol=0)
                        finally:
                            restore()
                        self.assertIs(native.compute_kld, original)

    def test_quantizer_cpu_fallback_and_ste_gradient(self):
        from qad_fastpath.quant import fused_activation_quant, fused_activation_ste
        from edgerazor.qat.util.quant_function import state_quant_uniform_symmetric_absmax_per_block_int8 as native
        for dtype in (torch.float32, torch.bfloat16):
            x = torch.randn(2, 256).to(dtype).requires_grad_()
            torch.testing.assert_close(fused_activation_quant(x), native(x), rtol=0, atol=0)
            y = fused_activation_ste(x)
            torch.testing.assert_close(y, x + (native(x)-x).detach(), rtol=0, atol=0)
            y.sum().backward()
            torch.testing.assert_close(x.grad, torch.ones_like(x), rtol=0, atol=0)

    @unittest.skipUnless(os.environ.get('QAOPD_TEST_CUDA') == '1', 'set QAOPD_TEST_CUDA=1 on an allocated GPU')
    def test_cuda_rounding_fixtures(self):
        from qad_fastpath.quant import fixture_check
        self.assertEqual(fixture_check()['status'], 'PASS')


if __name__ == '__main__':
    unittest.main()
