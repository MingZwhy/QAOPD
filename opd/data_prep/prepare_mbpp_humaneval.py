#!/usr/bin/env python3
"""Prepare MBPP splits for HumanEval-aligned on-policy distillation."""

import argparse
import ast
import builtins
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd

INSTRUCTION = "Write a solution to the following problem and make sure that it passes the tests:"
ASSISTANT_PREFIX = "<think>\n\n</think>\n\nHere is the completed function:\n```python\n"
ASSISTANT_PREFIX_NO_THINK = "Here is the completed function:\n```python\n"


def _called_names(test_list: list[str]) -> Counter:
    names = Counter()
    for test in test_list:
        try:
            tree = ast.parse(test)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                names[node.func.id] += 1
    return names


def _example_lines(test_list: list[str], function_name: str) -> list[str]:
    examples = []
    for test in test_list:
        try:
            statement = ast.parse(test).body[0]
        except (SyntaxError, IndexError):
            continue
        if not isinstance(statement, ast.Assert) or not isinstance(statement.test, ast.Compare):
            continue
        comparison = statement.test
        if len(comparison.ops) != 1 or not isinstance(comparison.ops[0], ast.Eq):
            continue
        if len(comparison.comparators) != 1:
            continue
        call = comparison.left
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id != function_name:
            continue
        examples.extend([f">>> {ast.unparse(call)}", ast.unparse(comparison.comparators[0])])
        if len(examples) >= 4:
            break
    return examples


def build_function_stub(code: str, description: str, test_list: list[str]) -> tuple[str, str]:
    tree = ast.parse(code)
    functions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if not functions:
        raise ValueError("No top-level function found")

    called = _called_names(test_list)
    candidates = [(called[name], -node.lineno, name, node) for name, node in functions.items() if called[name] > 0]
    if not candidates:
        raise ValueError("Tests do not call a top-level function")
    _, _, function_name, function = max(candidates)

    imports = [ast.unparse(node) for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    doc_lines = description.strip().splitlines()
    examples = _example_lines(test_list, function_name)
    if examples:
        doc_lines.extend(examples)

    stub_function = type(function)(
        name=function.name,
        args=function.args,
        body=[ast.Pass()],
        decorator_list=function.decorator_list,
        returns=function.returns,
        type_comment=function.type_comment,
        type_params=getattr(function, "type_params", []),
    )
    ast.fix_missing_locations(stub_function)
    signature_source = ast.unparse(stub_function)
    header, marker, suffix = signature_source.rpartition("\n    pass")
    if not marker or suffix:
        raise ValueError("Could not render function signature")

    escaped_lines = [line.replace('"""', '\\"\\"\\"') for line in doc_lines]
    docstring = f'    """ {escaped_lines[0]}'
    if len(escaped_lines) > 1:
        docstring += "\n" + "\n".join(f"    {line}" for line in escaped_lines[1:])
    function_source = f'{header}\n{docstring}\n    """'
    sections = ["\n".join(imports), function_source] if imports else [function_source]
    stub = "\n\n\n".join(sections).rstrip() + "\n\n"
    ast.parse(stub)
    return stub, function_name


def build_prompt(function_stub: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": f"{INSTRUCTION}\n```python\n{function_stub}```\n"}]


def build_reference_answer(
    code: str,
    function_stub: str,
    function_name: str,
    assistant_prefix: str = ASSISTANT_PREFIX,
) -> str:
    """Build a correct assistant message whose prefix matches the generation prompt."""
    tree = ast.parse(code)
    function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
    )
    body = list(function.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    if not body:
        raise ValueError("Reference function has no executable body")

    body_source = "\n".join(
        "\n".join(f"    {line}" for line in ast.unparse(statement).splitlines()) for statement in body
    )
    full_code = f"{function_stub}{body_source}\n"
    ast.parse(full_code)
    return f"{assistant_prefix}{full_code}```\n"


def has_external_dependencies(code: str, function_name: str, tests: list[str], setup: str) -> bool:
    tree = ast.parse(code)
    function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
    )
    imported = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.asname or alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.update(alias.asname or alias.name for alias in node.names)

    local = {arg.arg for arg in (*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs)}
    if function.args.vararg:
        local.add(function.args.vararg.arg)
    if function.args.kwarg:
        local.add(function.args.kwarg.arg)
    local.update(
        node.id for node in ast.walk(function) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    )
    allowed = set(dir(builtins)) | imported | local | {function_name}
    function_loads = {
        node.id for node in ast.walk(function) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    if function_loads - allowed:
        return True

    setup_names = set()
    try:
        setup_tree = ast.parse(setup)
        setup_names.update(
            node.id for node in ast.walk(setup_tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        )
        for node in setup_tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                setup_names.add(node.name)
            elif isinstance(node, ast.Import):
                setup_names.update(alias.asname or alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                setup_names.update(alias.asname or alias.name for alias in node.names)
    except SyntaxError:
        return True
    test_allowed = set(dir(builtins)) | imported | setup_names | {function_name}
    for test in tests:
        try:
            test_tree = ast.parse(test)
        except SyntaxError:
            return True
        loads = {
            node.id for node in ast.walk(test_tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        if loads - test_allowed:
            return True
    return False


def _digest(task_id: int, seed: int) -> str:
    return hashlib.sha256(f"{seed}\0{task_id}".encode()).hexdigest()


def prepare_rows(
    source_rows: list[dict],
    tokenizer,
    chat_template: str,
    val_size: int,
    max_prompt_length: int,
    seed: int,
    excluded_task_ids: set[int] | None = None,
    teacher_forcing: bool = False,
    assistant_prefix: str = ASSISTANT_PREFIX,
) -> tuple[list[dict], list[dict], dict]:
    prepared = []
    counters = Counter(input_rows=len(source_rows))
    seen_task_ids = set()
    excluded_task_ids = excluded_task_ids or set()
    for row in source_rows:
        task_id = int(row["task_id"])
        if task_id in excluded_task_ids:
            counters["excluded_task_id"] += 1
            continue
        if task_id in seen_task_ids:
            counters["duplicate_task_id"] += 1
            continue
        seen_task_ids.add(task_id)
        raw_tests = row.get("test_list")
        tests = list(raw_tests) if raw_tests is not None else []
        if not row.get("code") or not row.get("text") or not tests:
            counters["missing_fields"] += 1
            continue
        try:
            stub, function_name = build_function_stub(row["code"], row["text"], tests)
        except (SyntaxError, ValueError):
            counters["unsupported_program"] += 1
            continue
        setup = row.get("test_setup_code") or ""
        if has_external_dependencies(row["code"], function_name, tests, setup):
            counters["external_dependencies"] += 1
            continue
        prompt = build_prompt(stub)
        prompt_ids = tokenizer.apply_chat_template(
            prompt,
            chat_template=chat_template,
            add_generation_prompt=True,
            tokenize=True,
        )
        if len(prompt_ids) > max_prompt_length:
            counters["overlong_prompt"] += 1
            continue
        extra_info = {
            "source_dataset": "google-research-datasets/mbpp",
            "source_split": row.get("_source_split", "train"),
            "task_id": task_id,
            "function_name": function_name,
        }
        prepared_row = {
            "task_id": task_id,
            "prompt_tokens": len(prompt_ids),
            "data_source": "mbpp_humaneval",
            "prompt": prompt,
            "ability": "code",
            "reward_model": {
                "style": "rule",
                "ground_truth": json.dumps(
                    {
                        "function_stub": stub,
                        "function_name": function_name,
                        "tests": tests,
                        "test_setup_code": setup,
                    }
                ),
            },
            "extra_info": extra_info,
        }
        if teacher_forcing:
            try:
                extra_info["answer"] = build_reference_answer(
                    row["code"],
                    stub,
                    function_name,
                    assistant_prefix,
                )
            except (SyntaxError, StopIteration, ValueError):
                counters["unsupported_reference"] += 1
                continue
            prepared_row["agent_name"] = "teacher_forcing_agent"
        prepared.append(prepared_row)

    prepared.sort(key=lambda row: _digest(row["task_id"], seed))
    if len(prepared) <= val_size:
        raise ValueError(f"Only {len(prepared)} eligible rows remain for val_size={val_size}")
    validation, train = prepared[:val_size], prepared[val_size:]
    for split, rows in (("validation", validation), ("train", train)):
        for index, row in enumerate(rows):
            row["extra_info"]["split"] = split
            row["extra_info"]["index"] = index

    lengths = sorted(row.pop("prompt_tokens") for row in prepared)
    manifest = {
        "seed": seed,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "filter_counts": dict(counters),
        "prompt_tokens": {
            "min": lengths[0],
            "p50": lengths[len(lengths) // 2],
            "p90": lengths[int(0.9 * (len(lengths) - 1))],
            "max": lengths[-1],
        },
        "protocol": "lm-eval humaneval_instruct user prompt and assistant function-prefix prefill",
        "trajectory_mode": "teacher_forcing" if teacher_forcing else "on_policy",
        "assistant_prefix": assistant_prefix if teacher_forcing else None,
    }
    return train, validation, manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--chat_template", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--val_size", type=int, default=32)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--exclude_task_id", action="append", type=int, default=[])
    parser.add_argument("--teacher_forcing", action="store_true")
    parser.add_argument(
        "--omit_thinking_prefix",
        action="store_true",
        help="Build fixed answers for a generation template without Qwen3 think tags.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    from transformers import AutoTokenizer

    input_paths = [Path(path) for path in args.input]
    source = []
    for input_path in input_paths:
        source_split = input_path.stem.split("-", 1)[0]
        records = pd.read_parquet(input_path).to_dict("records")
        for row in records:
            row["_source_split"] = source_split
        source.extend(records)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=False)
    chat_template = Path(args.chat_template).read_text()
    train, validation, manifest = prepare_rows(
        source,
        tokenizer,
        chat_template,
        args.val_size,
        args.max_prompt_length,
        args.seed,
        set(args.exclude_task_id),
        args.teacher_forcing,
        ASSISTANT_PREFIX_NO_THINK if args.omit_thinking_prefix else ASSISTANT_PREFIX,
    )
    manifest.update(
        {
            "inputs": [str(path.resolve()) for path in input_paths],
            "tokenizer": str(Path(args.tokenizer).resolve()),
            "chat_template": str(Path(args.chat_template).resolve()),
            "excluded_task_ids": sorted(args.exclude_task_id),
        }
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train).to_parquet(output / "train.parquet", index=False)
    pd.DataFrame(validation).to_parquet(output / "validation.parquet", index=False)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
