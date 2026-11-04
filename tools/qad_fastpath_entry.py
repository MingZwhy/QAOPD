"""Launch the pinned EdgeRazor trainer with explicitly requested compute reuse."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'third_party/edgerazor/example/edgerazor-llm/src'


def main():
    from qad_fastpath.options import Options
    if os.environ.get('COMBO') != 'qwen3_1_7b_w2.79':
        raise ValueError('The QAD fastpath has only been checked on qwen3_1_7b_w2.79')
    options = Options.from_environment()
    sys.path.insert(0, str(SOURCE))
    import config
    import trainer as native
    from transformers import TrainerCallback
    from qad_fastpath.reuse import cache_quantized_weights_within_step, omit_unused_teacher_loss

    old_checkpointing = config.EdgeRazorTrainConfigForQwen3_1_7B.grad_chkpt
    if options.no_checkpoint:
        config.EdgeRazorTrainConfigForQwen3_1_7B.grad_chkpt = False

    class ClearWeights(TrainerCallback):
        def __init__(self, clears):
            self.clears = clears

        def clear(self):
            for clear in self.clears:
                clear()

        def on_train_begin(self, args, state, control, **kwargs):
            self.clear()  # Also covers checkpoint resumes and a second train().

        def on_step_begin(self, args, state, control, **kwargs):
            self.clear()

        def on_step_end(self, args, state, control, **kwargs):
            self.clear()  # An eval/save must never see a pre-update weight cache.

        def on_train_end(self, args, state, control, **kwargs):
            self.clear()

    cleanups = []
    class FastTrainer(native.EdgeRazorTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # The custom objective is a microbatch mean; Qwen's **kwargs support
            # does not mean this Trainer consumes num_items_in_batch.
            self.model_accepts_loss_kwargs = False
            if getattr(self.model.config, 'model_type', None) != 'qwen3':
                raise ValueError('The QAD fastpath requires a Qwen3 student')
            cleanups.append(omit_unused_teacher_loss(self.teacher_model))
            clears = []
            if options.weight_cache:
                clear, restore, _ = cache_quantized_weights_within_step(self.model)
                clears.append(clear)
                cleanups.append(restore)
            if options.fused_quant:
                from qad_fastpath.quant import fixture_check, install
                fixture_check()
                restore, _ = install(self.model)
                cleanups.append(restore)
            if options.sparse_kd:
                from qad_fastpath.sparse_kd import install
                cleanups.append(install())
            if options.fused_linear:
                from qad_fastpath.linear import install
                restore, clear, _ = install(self.model)
                cleanups.append(restore)
                clears.append(clear)
            if options.sparse_ce:
                from qad_fastpath.sparse_ce import install
                cleanups.append(install(self.model))
            self.add_callback(ClearWeights(clears))
            print('QAOPD_QAD_FASTPATH ' + json.dumps(asdict(options)), flush=True)

    original = native.EdgeRazorTrainer
    native.EdgeRazorTrainer = FastTrainer
    try:
        runpy.run_path(str(SOURCE / 'main.py'), run_name='__main__')
    finally:
        native.EdgeRazorTrainer = original
        config.EdgeRazorTrainConfigForQwen3_1_7B.grad_chkpt = old_checkpointing
        for restore in reversed(cleanups):
            restore()


if __name__ == '__main__':
    main()
