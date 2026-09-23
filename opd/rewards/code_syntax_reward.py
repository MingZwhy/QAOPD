"""Non-executing diagnostic reward for code-only pure OPD runs."""

import ast
import re

_PYTHON_BLOCK_RE = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_python_blocks(solution_str: str) -> list[str]:
    return [match.strip() for match in _PYTHON_BLOCK_RE.findall(solution_str or "")]


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Return 1 for one syntactically valid Python block, 0.5 for invalid syntax, otherwise 0."""
    if data_source not in {"apps", "codecontests", "codeforces", "taco"}:
        return 0.0

    blocks = extract_python_blocks(solution_str)
    if len(blocks) != 1 or not blocks[0]:
        return 0.0
    try:
        ast.parse(blocks[0])
    except SyntaxError:
        return 0.5
    return 1.0
