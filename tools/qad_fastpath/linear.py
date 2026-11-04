"""Native QLinear forward with fused STE and cached effective weights."""
import types
import torch
import torch.nn.functional as F
from .quant import IdentityGradient, fused_activation_ste


def install(model):
    from edgerazor.qat.module.qlinear import QLinear, TnLinearKVCacheQuant
    replacements = []
    supported = {'state_quant_uniform_symmetric_absmax_per_block_int8', 'fused_activation_quant'}
    for module in model.modules():
        if type(module) not in (QLinear, TnLinearKVCacheQuant):
            continue
        if module.a_quant_function is not None:
            assert module.a_quant_function.__name__ in supported
        kv = getattr(module, 'kv_quant_function', None)
        if kv is not None:
            assert kv.__name__ in supported
        original = module.forward

        def make_forward(original):
            def forward(self, x):
                if not self.training:
                    return original(x)
                if '_qaopd_step_effective_weight' not in self.__dict__:
                    with torch.no_grad():
                        q = self._weight_quant(replace_self=False)
                        effective = self.weight + (q - self.weight)
                    self.__dict__['_qaopd_step_effective_weight'] = effective.detach()
                    # All subsequent forwards in this step use effective. The
                    # raw quantized cache is redundant and otherwise duplicates
                    # nearly the full model on GPU. QEmbedding keeps its cache.
                    self.__dict__.pop('_qaopd_step_quantized_weight', None)
                weight = IdentityGradient.apply(self.weight, self.__dict__['_qaopd_step_effective_weight'])
                if self.a_quant_function is not None:
                    x = fused_activation_ste(x, **self.a_kwargs)
                output = F.linear(x, weight, self.bias)
                if getattr(self, 'kv_quant_function', None) is not None:
                    output = fused_activation_ste(output, **self.kv_kwargs)
                return output
            return forward

        module.forward = types.MethodType(make_forward(original), module)
        replacements.append((module, original))
    assert replacements

    def clear():
        for module, _ in replacements:
            module.__dict__.pop('_qaopd_step_effective_weight', None)

    def restore():
        clear()
        for module, original in replacements:
            module.forward = original
    return restore, clear, len(replacements)
