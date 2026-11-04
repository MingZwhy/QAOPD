"""Parse opt-in flags without importing the training stack."""
from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Options:
    weight_cache: bool = True
    sparse_kd: bool = True
    sparse_ce: bool = True
    no_checkpoint: bool = False
    fused_quant: bool = False
    fused_linear: bool = False

    @classmethod
    def from_environment(cls, env=None):
        env = os.environ if env is None else env
        def flag(name, default):
            key = 'QAOPD_QAD_' + name
            value = env.get(key, '1' if default else '0')
            if value not in ('0', '1'):
                raise ValueError(f'{key} must be 0 or 1, got {value!r}')
            return value == '1'
        no_checkpoint = flag('NO_CHECKPOINT', False)
        fused_quant = flag('FUSED_QUANT', no_checkpoint)
        result = cls(weight_cache=flag('WEIGHT_CACHE', True),
                     sparse_kd=flag('SPARSE_KD', True),
                     sparse_ce=flag('SPARSE_CE', True),
                     no_checkpoint=no_checkpoint, fused_quant=fused_quant,
                     fused_linear=flag('FUSED_LINEAR', fused_quant))
        if result.fused_quant and not result.no_checkpoint:
            raise ValueError('FUSED_QUANT currently requires NO_CHECKPOINT=1')
        if result.fused_linear and not (result.fused_quant and result.weight_cache):
            raise ValueError('FUSED_LINEAR requires FUSED_QUANT=1 and WEIGHT_CACHE=1')
        return result
