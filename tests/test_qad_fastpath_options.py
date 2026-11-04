from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from qad_fastpath.options import Options


class OptionsTest(unittest.TestCase):
    def test_default_keeps_gradient_checkpointing(self):
        options = Options.from_environment({})
        self.assertFalse(options.no_checkpoint)
        self.assertFalse(options.fused_quant)
        self.assertFalse(options.fused_linear)
        self.assertTrue(options.weight_cache and options.sparse_kd and options.sparse_ce)

    def test_explicit_memory_tradeoff_enables_full_compute_path(self):
        options = Options.from_environment({'QAOPD_QAD_NO_CHECKPOINT': '1'})
        self.assertTrue(all(vars(options).values()))

    def test_individual_features_can_be_disabled(self):
        options = Options.from_environment({'QAOPD_QAD_NO_CHECKPOINT':'1',
            'QAOPD_QAD_FUSED_QUANT':'0', 'QAOPD_QAD_FUSED_LINEAR':'0',
            'QAOPD_QAD_WEIGHT_CACHE':'0', 'QAOPD_QAD_SPARSE_CE':'0',
            'QAOPD_QAD_SPARSE_KD':'0'})
        self.assertEqual(options, Options(False, False, False, True, False, False))

    def test_rejects_invalid_flags_and_unsupported_combinations(self):
        for env in ({'QAOPD_QAD_WEIGHT_CACHE':'yes'},
                    {'QAOPD_QAD_FUSED_QUANT':'1'},
                    {'QAOPD_QAD_NO_CHECKPOINT':'1','QAOPD_QAD_WEIGHT_CACHE':'0'}):
            with self.subTest(env=env), self.assertRaises(ValueError):
                Options.from_environment(env)


if __name__ == '__main__':
    unittest.main()
