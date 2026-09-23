"""Unit-test reward for HumanEval-aligned MBPP function completions."""

import ast
import json
import resource
import subprocess
import sys
import warnings

DENIED_MODULES = {
    "ctypes",
    "http",
    "importlib",
    "multiprocessing",
    "os",
    "pathlib",
    "requests",
    "resource",
    "shutil",
    "signal",
    "socket",
    "subprocess",
    "urllib",
}
DENIED_CALLS = {"__import__", "compile", "eval", "exec", "exit", "input", "open", "quit"}
RUNNER = r"""
import json
import sys

payload = json.loads(sys.stdin.read())
namespace = {}
result = {"passed": 0, "total": len(payload["tests"])}
try:
    if payload["setup"]:
        exec(compile(payload["setup"], "<mbpp-setup>", "exec"), namespace)
    exec(compile(payload["code"], "<candidate>", "exec"), namespace)
    for test in payload["tests"]:
        try:
            exec(compile(test, "<mbpp-test>", "exec"), namespace)
            result["passed"] += 1
        except BaseException:
            pass
except BaseException:
    pass
print(json.dumps(result))
"""


def extract_completion(solution_str: str) -> str:
    completion = (solution_str or "").split("```", 1)[0]
    if completion.lstrip().startswith("python\n"):
        completion = completion.lstrip()[len("python\n") :]
    return completion.rstrip() + "\n"


def is_safe_candidate(code: str) -> bool:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".", 1)[0] in DENIED_MODULES for alias in node.names):
                return False
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".", 1)[0] in DENIED_MODULES:
                return False
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in DENIED_CALLS:
            return False
    return True


def _limit_child():
    memory = 768 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))


def run_tests(code: str, setup: str, tests: list[str], timeout: float = 3.0) -> float:
    if not tests or not is_safe_candidate(code):
        return 0.0
    payload = json.dumps({"code": code, "setup": setup, "tests": tests})
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-c", RUNNER],
            input=payload,
            text=True,
            capture_output=True,
            timeout=timeout,
            preexec_fn=_limit_child,
            check=False,
        )
        parsed = json.loads(result.stdout.strip().splitlines()[-1])
        total = int(parsed["total"])
        return float(parsed["passed"]) / total if total else 0.0
    except (IndexError, json.JSONDecodeError, KeyError, subprocess.TimeoutExpired, ValueError):
        return 0.0


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    require_all_tests: bool = False,
    **kwargs,
):
    if data_source not in {"apps_humaneval", "kodcode_humaneval", "mbpp_humaneval"}:
        return 0.0
    try:
        reference = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
        completion = extract_completion(solution_str)
        code = reference["function_stub"] + completion
        score = run_tests(code, reference.get("test_setup_code", ""), reference["tests"])
        return float(score == 1.0) if require_all_tests else score
    except (KeyError, TypeError, json.JSONDecodeError):
        return 0.0
