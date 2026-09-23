#!/usr/bin/env python3
"""Build a short-code KodCode curriculum with HumanEval-like prompt features."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

if __package__:
    from .prepare_mbpp_humaneval import build_prompt
else:
    from prepare_mbpp_humaneval import build_prompt


@dataclass(frozen=True)
class Example:
    call: ast.Call
    expected: ast.expr

    def render(self) -> tuple[str, str]:
        return f">>> {ast.unparse(self.call)}", ast.unparse(self.expected)


def _digest(seed: int, salt: str, task_id: int) -> str:
    return hashlib.sha256(f"{seed}\0{salt}\0{task_id}".encode()).hexdigest()


def _find_function(tree: ast.Module, function_name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one top-level function named {function_name!r}")
    return matches[0]


def extract_reference_body(answer: str, function_stub: str, function_name: str) -> tuple[str, int]:
    """Return the executable suffix and its non-comment line count."""
    opening = answer.find("```python\n")
    closing = answer.rfind("```")
    if opening < 0 or closing <= opening:
        raise ValueError("teacher answer has no complete Python fence")
    code = answer[opening + len("```python\n") : closing]
    if not code.startswith(function_stub):
        raise ValueError("teacher answer does not repeat the exact function stub")
    tree = ast.parse(code)
    function = _find_function(tree, function_name)
    body = list(function.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        if isinstance(body[0].value.value, str):
            body = body[1:]
    if not body:
        raise ValueError("teacher answer has no executable function body")
    suffix = code[len(function_stub) :]
    line_count = sum(bool(line.strip()) and not line.lstrip().startswith("#") for line in suffix.splitlines())
    return suffix, line_count


def _literal_value(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (TypeError, ValueError):
        return _UNSUPPORTED


_UNSUPPORTED = object()


def extract_direct_examples(tests: list[str], function_name: str) -> list[Example]:
    """Extract deterministic ``function(literals) == literal`` assertions."""
    examples: list[Example] = []
    seen = set()
    for source in tests:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assert) or not isinstance(node.test, ast.Compare):
                continue
            comparison = node.test
            if (
                len(comparison.ops) != 1
                or not isinstance(comparison.ops[0], ast.Eq)
                or len(comparison.comparators) != 1
            ):
                continue
            call = comparison.left
            expected = comparison.comparators[0]
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
                continue
            if call.func.id != function_name or any(isinstance(arg, ast.Starred) for arg in call.args):
                continue
            if any(keyword.arg is None for keyword in call.keywords):
                continue
            literal_nodes = [*call.args, *(keyword.value for keyword in call.keywords), expected]
            if any(_literal_value(value) is _UNSUPPORTED for value in literal_nodes):
                continue
            rendered = (ast.unparse(call), ast.unparse(expected))
            if rendered in seen:
                continue
            seen.add(rendered)
            examples.append(Example(call=call, expected=expected))
    return examples


def _value_type(value: Any) -> str | None:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if value is None:
        return None
    if isinstance(value, list):
        element_types = {_value_type(item) for item in value}
        element_types.discard(None)
        if not value or len(element_types) != 1:
            return "list"
        return f"list[{element_types.pop()}]"
    if isinstance(value, tuple):
        item_types = [_value_type(item) for item in value]
        if not value or any(item_type is None for item_type in item_types):
            return "tuple"
        return f"tuple[{', '.join(item_types)}]"
    if isinstance(value, set):
        element_types = {_value_type(item) for item in value}
        element_types.discard(None)
        if not value or len(element_types) != 1:
            return "set"
        return f"set[{element_types.pop()}]"
    if isinstance(value, dict):
        key_types = {_value_type(item) for item in value}
        value_types = {_value_type(item) for item in value.values()}
        key_types.discard(None)
        value_types.discard(None)
        if not value or len(key_types) != 1 or len(value_types) != 1:
            return "dict"
        return f"dict[{key_types.pop()}, {value_types.pop()}]"
    return None


def infer_annotations(
    function_stub: str,
    function_name: str,
    examples: list[Example],
) -> tuple[dict[str, str], str] | None:
    """Infer a complete positional signature from consistent literal examples."""
    tree = ast.parse(function_stub)
    function = _find_function(tree, function_name)
    parameters = [*function.args.posonlyargs, *function.args.args]
    if function.args.vararg or function.args.kwarg or function.args.kwonlyargs:
        return None
    if not parameters or not examples:
        return None

    observations: dict[str, list[Any]] = {parameter.arg: [] for parameter in parameters}
    returns: list[Any] = []
    for example in examples:
        if example.call.keywords or len(example.call.args) != len(parameters):
            return None
        values = [_literal_value(argument) for argument in example.call.args]
        expected = _literal_value(example.expected)
        if expected is _UNSUPPORTED or any(value is _UNSUPPORTED for value in values):
            return None
        for parameter, value in zip(parameters, values, strict=True):
            observations[parameter.arg].append(value)
        returns.append(expected)

    inferred = {}
    for parameter, values in observations.items():
        value_types = {_value_type(value) for value in values}
        if None in value_types or len(value_types) != 1:
            return None
        inferred[parameter] = value_types.pop()
    return_types = {_value_type(value) for value in returns}
    if None in return_types or len(return_types) != 1:
        return None
    return inferred, return_types.pop()


def has_annotations(function_stub: str, function_name: str) -> bool:
    function = _find_function(ast.parse(function_stub), function_name)
    parameters = [*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs]
    return function.returns is not None or any(parameter.annotation is not None for parameter in parameters)


def has_doctest(function_stub: str, function_name: str) -> bool:
    function = _find_function(ast.parse(function_stub), function_name)
    return ">>>" in (ast.get_docstring(function, clean=False) or "")


def augment_stub(
    function_stub: str,
    function_name: str,
    *,
    examples: list[Example],
    annotations: tuple[dict[str, str], str] | None,
) -> str:
    tree = ast.parse(function_stub)
    function = _find_function(tree, function_name)

    docstring = ast.get_docstring(function, clean=False)
    if not docstring:
        raise ValueError("function stub has no docstring")
    if examples:
        rendered_examples = [line for example in examples for line in example.render()]
        docstring = f"{docstring.rstrip()}\n" + "\n".join(rendered_examples)
        function.body[0] = ast.Expr(value=ast.Constant(value=docstring))

    if annotations is not None:
        parameter_types, return_type = annotations
        parameters = [*function.args.posonlyargs, *function.args.args]
        for parameter in parameters:
            if parameter.annotation is None:
                parameter.annotation = ast.parse(parameter_types[parameter.arg], mode="eval").body
        if function.returns is None:
            function.returns = ast.parse(return_type, mode="eval").body

    ast.fix_missing_locations(tree)
    rendered = ast.unparse(tree).rstrip() + "\n\n"
    ast.parse(rendered)
    return rendered


def _load_paired_rows(on_policy_dir: Path, teacher_forcing_dir: Path) -> list[dict]:
    paired = []
    for split in ("train", "validation"):
        on_policy = pd.read_parquet(on_policy_dir / f"{split}.parquet")
        teacher_forcing = pd.read_parquet(teacher_forcing_dir / f"{split}.parquet")
        on_by_id = {int(row["task_id"]): row for row in on_policy.to_dict("records")}
        tf_by_id = {int(row["task_id"]): row for row in teacher_forcing.to_dict("records")}
        if set(on_by_id) != set(tf_by_id):
            raise ValueError(f"{split} task IDs differ between on-policy and teacher-forcing data")
        for task_id in sorted(on_by_id):
            on_row = on_by_id[task_id]
            tf_row = tf_by_id[task_id]
            answer = tf_row["extra_info"].get("answer")
            if not isinstance(answer, str) or not answer:
                raise ValueError(f"task {task_id} has no teacher-forcing answer")
            paired.append({"row": on_row, "answer": answer, "source_split": split})
    return paired


def _pick_fraction(
    candidates: list[dict],
    *,
    total_rows: int,
    fraction: float,
    already_present: int,
    seed: int,
    salt: str,
) -> set[int]:
    target = max(0, round(total_rows * fraction) - already_present)
    ranked = sorted(candidates, key=lambda item: _digest(seed, salt, item["task_id"]))
    return {item["task_id"] for item in ranked[:target]}


def prepare_curriculum(
    paired_rows: list[dict],
    *,
    tokenizer,
    chat_template: str,
    max_body_lines: int,
    max_response_tokens: int,
    max_prompt_length: int,
    val_size: int,
    doctest_fraction: float,
    annotation_fraction: float,
    max_examples: int,
    seed: int,
) -> tuple[list[dict], list[dict], dict]:
    if not 0.0 <= doctest_fraction <= 1.0 or not 0.0 <= annotation_fraction <= 1.0:
        raise ValueError("profile fractions must lie in [0, 1]")

    candidates = []
    counters = Counter(input_rows=len(paired_rows))
    for paired in paired_rows:
        row = paired["row"]
        ground_truth = json.loads(row["reward_model"]["ground_truth"])
        function_stub = ground_truth["function_stub"]
        function_name = ground_truth["function_name"]
        try:
            _, body_lines = extract_reference_body(
                paired["answer"],
                function_stub,
                function_name,
            )
            response_tokens = len(tokenizer(paired["answer"], add_special_tokens=False)["input_ids"])
            examples = extract_direct_examples(ground_truth["tests"], function_name)
            annotations = infer_annotations(function_stub, function_name, examples)
        except (KeyError, SyntaxError, TypeError, ValueError):
            counters["unsupported_row"] += 1
            continue
        if body_lines > max_body_lines:
            counters["over_body_line_limit"] += 1
            continue
        if response_tokens > max_response_tokens:
            counters["over_response_token_limit"] += 1
            continue
        candidates.append(
            {
                "task_id": int(row["task_id"]),
                "row": row,
                "ground_truth": ground_truth,
                "body_lines": body_lines,
                "response_tokens": response_tokens,
                "examples": examples,
                "annotations": annotations,
                "already_doctest": has_doctest(function_stub, function_name),
                "already_annotated": has_annotations(function_stub, function_name),
            }
        )

    if len(candidates) <= val_size:
        raise ValueError(f"only {len(candidates)} rows remain for val_size={val_size}")

    doctest_ids = _pick_fraction(
        [candidate for candidate in candidates if candidate["examples"] and not candidate["already_doctest"]],
        total_rows=len(candidates),
        fraction=doctest_fraction,
        already_present=sum(candidate["already_doctest"] for candidate in candidates),
        seed=seed,
        salt="doctest",
    )
    annotation_ids = _pick_fraction(
        [
            candidate
            for candidate in candidates
            if candidate["annotations"] is not None and not candidate["already_annotated"]
        ],
        total_rows=len(candidates),
        fraction=annotation_fraction,
        already_present=sum(candidate["already_annotated"] for candidate in candidates),
        seed=seed,
        salt="annotation",
    )

    prepared = []
    for candidate in candidates:
        row = dict(candidate["row"])
        ground_truth = dict(candidate["ground_truth"])
        function_name = ground_truth["function_name"]
        add_examples = candidate["examples"][:max_examples] if candidate["task_id"] in doctest_ids else []
        annotations = candidate["annotations"] if candidate["task_id"] in annotation_ids else None
        function_stub = augment_stub(
            ground_truth["function_stub"],
            function_name,
            examples=add_examples,
            annotations=annotations,
        )
        prompt = build_prompt(function_stub)
        prompt_tokens = len(
            tokenizer.apply_chat_template(
                prompt,
                chat_template=chat_template,
                add_generation_prompt=True,
                tokenize=True,
            )
        )
        if prompt_tokens > max_prompt_length:
            counters["over_augmented_prompt_limit"] += 1
            continue

        ground_truth["function_stub"] = function_stub
        row["prompt"] = prompt
        row["reward_model"] = {**row["reward_model"], "ground_truth": json.dumps(ground_truth)}
        row.pop("agent_name", None)
        extra_info = dict(row["extra_info"])
        extra_info.pop("answer", None)
        extra_info.update(
            {
                "profile_reference_body_lines": candidate["body_lines"],
                "profile_reference_response_tokens": candidate["response_tokens"],
                "profile_doctest_added": bool(add_examples),
                "profile_annotations_added": annotations is not None,
                "profile_prompt_tokens": prompt_tokens,
            }
        )
        row["extra_info"] = extra_info
        prepared.append(row)

    if len(prepared) <= val_size:
        raise ValueError(f"only {len(prepared)} rows remain after prompt filtering")
    prepared.sort(key=lambda row: _digest(seed, "split", int(row["task_id"])))
    validation, train = prepared[:val_size], prepared[val_size:]
    for split, rows in (("validation", validation), ("train", train)):
        for index, row in enumerate(rows):
            row["extra_info"] = {**row["extra_info"], "split": split, "index": index}

    prompt_lengths = sorted(row["extra_info"]["profile_prompt_tokens"] for row in prepared)
    response_lengths = sorted(row["extra_info"]["profile_reference_response_tokens"] for row in prepared)
    body_lengths = sorted(row["extra_info"]["profile_reference_body_lines"] for row in prepared)
    manifest = {
        "seed": seed,
        "input_rows": len(paired_rows),
        "selected_rows": len(prepared),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "max_body_lines": max_body_lines,
        "max_response_tokens": max_response_tokens,
        "max_prompt_length": max_prompt_length,
        "doctest_fraction_target": doctest_fraction,
        "annotation_fraction_target": annotation_fraction,
        "doctest_rows": sum(
            ">>>" in json.loads(row["reward_model"]["ground_truth"])["function_stub"] for row in prepared
        ),
        "annotated_rows": sum(
            has_annotations(
                json.loads(row["reward_model"]["ground_truth"])["function_stub"],
                json.loads(row["reward_model"]["ground_truth"])["function_name"],
            )
            for row in prepared
        ),
        "filter_counts": dict(counters),
        "body_lines": _length_summary(body_lengths),
        "response_tokens": _length_summary(response_lengths),
        "prompt_tokens": _length_summary(prompt_lengths),
        "trajectory_mode": "on_policy",
        "reference_usage": "teacher answers filter KodCode difficulty only; output rows contain no answers",
    }
    return train, validation, manifest


def _length_summary(lengths: list[int]) -> dict[str, int]:
    return {
        "min": lengths[0],
        "p50": lengths[len(lengths) // 2],
        "p90": lengths[int(0.9 * (len(lengths) - 1))],
        "max": lengths[-1],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--on_policy_dir", required=True, type=Path)
    parser.add_argument("--teacher_forcing_dir", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--chat_template", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--max_body_lines", type=int, default=5)
    parser.add_argument("--max_response_tokens", type=int, default=192)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--val_size", type=int, default=32)
    parser.add_argument("--doctest_fraction", type=float, default=0.46)
    parser.add_argument("--annotation_fraction", type=float, default=0.37)
    parser.add_argument("--max_examples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260717)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer

    if args.output_dir.exists():
        raise ValueError(f"output directory already exists: {args.output_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=False)
    chat_template = args.chat_template.read_text(encoding="utf-8")
    paired_rows = _load_paired_rows(args.on_policy_dir, args.teacher_forcing_dir)
    train, validation, manifest = prepare_curriculum(
        paired_rows,
        tokenizer=tokenizer,
        chat_template=chat_template,
        max_body_lines=args.max_body_lines,
        max_response_tokens=args.max_response_tokens,
        max_prompt_length=args.max_prompt_length,
        val_size=args.val_size,
        doctest_fraction=args.doctest_fraction,
        annotation_fraction=args.annotation_fraction,
        max_examples=args.max_examples,
        seed=args.seed,
    )
    manifest.update(
        {
            "on_policy_dir": str(args.on_policy_dir.resolve()),
            "teacher_forcing_dir": str(args.teacher_forcing_dir.resolve()),
            "tokenizer": str(Path(args.tokenizer).resolve()),
            "chat_template": str(args.chat_template.resolve()),
        }
    )

    args.output_dir.mkdir(parents=True)
    pd.DataFrame(train).to_parquet(args.output_dir / "train.parquet", index=False)
    pd.DataFrame(validation).to_parquet(args.output_dir / "validation.parquet", index=False)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
