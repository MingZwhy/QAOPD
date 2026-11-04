"""CPU regression for the custom QAD loss under Trainer's DeepSpeed path.

Run after bootstrap: venvs/qaopd-eval/bin/python -m unittest discover -s tests
The backward adapter is captured here; this test does not launch DeepSpeed.
"""
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
from accelerate.utils import DistributedType
from transformers import Qwen3Config, Qwen3ForCausalLM, TrainingArguments

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'third_party/edgerazor/example/edgerazor-llm/src'))
from trainer import EdgeRazorTrainer


class GradientAccumulationTest(unittest.TestCase):
    def gradients(self, accumulation):
        torch.manual_seed(41)
        cfg = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                          num_hidden_layers=1, num_attention_heads=2,
                          num_key_value_heads=2, head_dim=8,
                          max_position_embeddings=32, use_cache=False)
        model = Qwen3ForCausalLM(cfg)
        teacher = copy.deepcopy(model).requires_grad_(False)
        # The normalization contract concerns any microbatch-mean objective.
        # Use task CE to isolate it from the batch-dependent KD coefficient.
        objective = SimpleNamespace(compute_loss=lambda student_outputs, **kw:
            (student_outputs.loss, {'task_loss': student_outputs.loss}))
        with tempfile.TemporaryDirectory() as directory:
            args = TrainingArguments(output_dir=directory, use_cpu=True,
                per_device_train_batch_size=4 // accumulation,
                gradient_accumulation_steps=accumulation, max_grad_norm=0,
                report_to=[], remove_unused_columns=False, optim='adamw_torch')
            trainer = EdgeRazorTrainer(model=model, teacher_model=teacher,
                                       edgerazor=objective, args=args)
            trainer.current_gradient_accumulation_steps = accumulation
            trainer.accelerator.state.distributed_type = DistributedType.DEEPSPEED
            def backward(loss, **kwargs):
                self.assertIs(kwargs.get('scale_wrt_gas'), False)
                loss.backward()
            trainer.accelerator.backward = backward
            tokens = torch.tensor([[1,2,3,4,5,6,7,8], [2,4,6,8,10,12,14,16],
                                   [3,5,7,9,11,13,15,17], [4,6,8,10,12,14,16,18]])
            for ids in tokens.chunk(accumulation):
                trainer.training_step(model, dict(input_ids=ids, labels=ids.clone(),
                    attention_mask=torch.ones_like(ids)), num_items_in_batch=torch.tensor(28))
            return torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])

    def test_accumulated_gradient_matches_one_batch(self):
        torch.set_num_threads(2)
        reference = self.gradients(1)
        for accumulation in (2, 4):
            with self.subTest(accumulation=accumulation):
                torch.testing.assert_close(self.gradients(accumulation), reference,
                                           rtol=3e-5, atol=3e-6)


if __name__ == '__main__':
    unittest.main()
