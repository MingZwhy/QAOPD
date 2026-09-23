"""Restricted stdin/stdout execution reward for verifiable Python programs."""

import ast
import json
import resource
import subprocess
import sys
import tempfile
from collections.abc import Sequence

from examples.on_policy_distillation_trainer.code_syntax_reward import extract_python_blocks

ALLOWED_MODULES = {
    "array",
    "bisect",
    "collections",
    "copy",
    "dataclasses",
    "decimal",
    "fractions",
    "functools",
    "heapq",
    "itertools",
    "math",
    "operator",
    "random",
    "re",
    "statistics",
    "string",
    "sys",
    "typing",
}
DENIED_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "eval",
    "exec",
    "exit",
    "getattr",
    "globals",
    "help",
    "locals",
    "open",
    "quit",
    "setattr",
    "vars",
}
ALLOWED_SYS_ATTRIBUTES = {
    "exit",
    "getrecursionlimit",
    "maxsize",
    "set_int_max_str_digits",
    "setrecursionlimit",
    "stderr",
    "stdin",
    "stdout",
    "version_info",
}
MAX_CAPTURE_BYTES = 1024 * 1024


def _safe_tree(code: str) -> ast.AST | None:
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None

    sys_aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root not in ALLOWED_MODULES:
                    return None
                if root == "sys":
                    sys_aliases.add(alias.asname or root)
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if not root or root not in ALLOWED_MODULES:
                return None
            if root == "sys" and any(alias.name not in ALLOWED_SYS_ATTRIBUTES for alias in node.names):
                return None
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in DENIED_CALLS:
            return None
        elif isinstance(node, ast.Name) and node.id == "__builtins__":
            return None

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in sys_aliases
            and node.attr not in ALLOWED_SYS_ATTRIBUTES
        ):
            return None
    return tree


def _limit_child() -> None:
    memory = 768 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_CAPTURE_BYTES, MAX_CAPTURE_BYTES))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))


def _normalize_output(value: str) -> list[str]:
    return value.split()


def run_case(code: str, stdin: str, expected: str, timeout: float = 2.5) -> bool:
    """Run one test in an isolated child with bounded resources and output."""
    env = {"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PYTHONHASHSEED": "0"}
    try:
        with tempfile.TemporaryDirectory(prefix="verl-code-reward-") as cwd:
            with tempfile.TemporaryFile() as stdout:
                result = subprocess.run(
                    [sys.executable, "-I", "-S", "-c", code],
                    input=stdin,
                    text=True,
                    stdout=stdout,
                    stderr=subprocess.DEVNULL,
                    timeout=timeout,
                    cwd=cwd,
                    env=env,
                    preexec_fn=_limit_child,
                    check=False,
                )
                if result.returncode != 0:
                    return False
                stdout.seek(0)
                actual = stdout.read(MAX_CAPTURE_BYTES).decode("utf-8", errors="replace")
    except (OSError, subprocess.TimeoutExpired):
        return False
    return _normalize_output(actual) == _normalize_output(expected)


def score_program(
    code: str,
    inputs: Sequence[str],
    outputs: Sequence[str],
    *,
    max_cases: int = 3,
    timeout: float = 2.5,
    syntax_score: float = 0.1,
) -> float:
    if _safe_tree(code) is None or not inputs or len(inputs) != len(outputs):
        return 0.0
    case_count = min(len(inputs), max_cases) if max_cases > 0 else len(inputs)
    passed = sum(run_case(code, str(inputs[i]), str(outputs[i]), timeout) for i in range(case_count))
    return syntax_score + (1.0 - syntax_score) * passed / case_count


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Return syntax shaping plus the fraction of bounded execution tests passed."""
    del extra_info
    if data_source not in {"apps", "codecontests", "codeforces", "taco"}:
        return 0.0
    blocks = extract_python_blocks(solution_str)
    if len(blocks) != 1 or not blocks[0]:
        return 0.0
    try:
        tests = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
        return score_program(
            blocks[0],
            tests["inputs"],
            tests["outputs"],
            max_cases=int(kwargs.get("max_cases", 3)),
            timeout=float(kwargs.get("timeout", 2.5)),
            syntax_score=float(kwargs.get("syntax_score", 0.1)),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return 0.0
