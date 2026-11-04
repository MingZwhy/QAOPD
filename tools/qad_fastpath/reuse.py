"""Reuse work while the student's latent weights are unchanged within a step."""
import functools
import types


def omit_unused_teacher_loss(teacher):
    original = teacher.forward

    @functools.wraps(original)
    def forward(*args, **kwargs):
        kwargs.pop('labels', None)
        kwargs['use_cache'] = False
        return original(*args, **kwargs)

    teacher.forward = forward
    return lambda: setattr(teacher, 'forward', original)


def cache_quantized_weights_within_step(model):
    """Reuse deterministic quantized weights until the next optimizer update.

    ZeRO may gather/release latent parameters between micro-batches. Each cache
    is a detached full quantized tensor, never a parameter view. Call clear()
    at EVERY optimizer-step begin, including immediately after a resume.
    Activation quantization and the original STE in forward stay untouched.
    """
    modules = []
    allowed = {
        'weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse',
        'weight_quant_uniform_symmetric_absmax_per_block_int4',
    }
    for module in model.modules():
        if not hasattr(module, '_weight_quant') or not hasattr(module, 'w_quant_function'):
            continue
        assert module.w_quant_function.__name__ in allowed, module.w_quant_function
        original = module._weight_quant

        def make_cached(original):
            def cached(self, replace_self=False):
                if replace_self or not self.training:
                    return original(replace_self=replace_self)
                if '_qaopd_step_quantized_weight' not in self.__dict__:
                    self.__dict__['_qaopd_step_quantized_weight'] = original(replace_self=False).detach()
                return self.__dict__['_qaopd_step_quantized_weight']
            return cached

        module._weight_quant = types.MethodType(make_cached(original), module)
        modules.append((module, original))
    assert modules, 'No supported QAT modules found'

    def clear():
        for module, _ in modules:
            module.__dict__.pop('_qaopd_step_quantized_weight', None)

    def restore():
        clear()
        for module, original in modules:
            module._weight_quant = original

    return clear, restore, len(modules)
