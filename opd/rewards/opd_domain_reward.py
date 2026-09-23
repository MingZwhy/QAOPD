"""Diagnostic reward router for pure OPD domain-mixture experiments."""

from examples.on_policy_distillation_trainer.code_syntax_reward import compute_score as code_score
from examples.on_policy_distillation_trainer.gsm8k_boxed_reward import compute_score as gsm8k_score


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if data_source == "openai/gsm8k":
        return gsm8k_score(data_source, solution_str, ground_truth, extra_info=extra_info)
    if data_source in {"apps", "codecontests", "codeforces", "taco"}:
        return code_score(data_source, solution_str, ground_truth, extra_info=extra_info)
    return 0.0
