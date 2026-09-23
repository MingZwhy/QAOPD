#!/usr/bin/env python3
"""Audit that two EdgeRazor exports form a fair KV16/KV8 pair."""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any

LOAD_SMOKE_PATTERN = re.compile(r"KV quantization verified: enabled=(True|False), updates=(\d+)")


def _load_qat_config(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    try:
        return config["edgerazor_config"]["qat_configuration"]
    except KeyError as error:
        raise ValueError(f"Missing EdgeRazor QAT configuration in {config_path}") from error


def _kv_enabled(qat_config: dict[str, Any]) -> bool:
    return "kv_cache" in qat_config["select"]["target_types"]


def _without_kv_settings(qat_config: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(qat_config)
    normalized["select"]["target_types"] = [name for name in normalized["select"]["target_types"] if name != "kv_cache"]
    normalized["function"]["kv_cache_function"] = None
    normalized["function"]["kv_block_size"] = -1
    normalized["function"].pop("kv_mixed_precision_prop", None)
    return normalized


def _files_equal(first: Path, second: Path, chunk_size: int = 8 * 1024 * 1024) -> bool:
    if first.stat().st_size != second.stat().st_size:
        return False
    with first.open("rb") as left, second.open("rb") as right:
        while True:
            left_chunk = left.read(chunk_size)
            right_chunk = right.read(chunk_size)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def _load_smoke_result(path: Path) -> tuple[bool, int]:
    matches = LOAD_SMOKE_PATTERN.findall(path.read_text(encoding="utf-8"))
    if not matches:
        raise ValueError(f"No KV runtime verification found in {path}")
    enabled, updates = matches[-1]
    return enabled == "True", int(updates)


def _load_eval_result(root: Path, task: str) -> tuple[dict[str, Any], Path]:
    matches: list[tuple[int, Path, dict[str, Any]]] = []
    for path in root.rglob("results_*.json"):
        try:
            with path.open(encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if task in data.get("results", {}):
            matches.append((path.stat().st_mtime_ns, path, data))
    if not matches:
        raise ValueError(f"No {task} result found under {root}")
    _, path, data = max(matches, key=lambda item: (item[0], str(item[1])))
    return data, path


def _load_eval_samples(
    result_path: Path,
    task: str,
) -> dict[tuple[int, str], dict[str, Any]]:
    sample_files = sorted(result_path.parent.glob(f"samples_{task}_*.jsonl"))
    if not sample_files:
        raise ValueError(f"No {task} samples found beside {result_path}")
    samples: dict[tuple[int, str], dict[str, Any]] = {}
    with sample_files[-1].open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (int(row["doc_id"]), str(row.get("filter", "none")))
            samples[key] = row
    return samples


def _audit_task_protocol(
    kv16_root: Path,
    kv8_root: Path,
    task: str,
) -> tuple[int, str]:
    kv16_result, kv16_path = _load_eval_result(kv16_root, task)
    kv8_result, kv8_path = _load_eval_result(kv8_root, task)

    protocol_fields = (
        "batch_size",
        "limit",
        "gen_kwargs",
        "random_seed",
        "numpy_seed",
        "torch_seed",
        "fewshot_seed",
    )
    for field in protocol_fields:
        kv16_value = kv16_result.get("config", {}).get(field)
        kv8_value = kv8_result.get("config", {}).get(field)
        if kv16_value != kv8_value:
            raise ValueError(f"{task} {field} differs between KV16 and KV8: {kv16_value!r} != {kv8_value!r}")

    kv16_samples = _load_eval_samples(kv16_path, task)
    kv8_samples = _load_eval_samples(kv8_path, task)
    if set(kv16_samples) != set(kv8_samples):
        raise ValueError(f"{task} sample keys differ between KV16 and KV8")
    for sample_key in sorted(kv16_samples):
        for field in ("doc_hash", "prompt_hash", "target_hash", "arguments"):
            if kv16_samples[sample_key].get(field) != kv8_samples[sample_key].get(field):
                raise ValueError(f"{task} {field} differs for sample={sample_key}")

    doc_count = len({doc_id for doc_id, _ in kv16_samples})
    return doc_count, str(kv16_result["config"]["batch_size"])


def audit_pair(
    kv16_model: Path,
    kv8_model: Path,
    kv16_log: Path,
    kv8_log: Path,
    kv16_eval_root: Path | None = None,
    kv8_eval_root: Path | None = None,
    kv16_generation_eval_root: Path | None = None,
    kv8_generation_eval_root: Path | None = None,
) -> str:
    kv16_qat = _load_qat_config(kv16_model)
    kv8_qat = _load_qat_config(kv8_model)

    if _kv_enabled(kv16_qat):
        raise ValueError("KV16 export unexpectedly enables the kv_cache target")
    if not _kv_enabled(kv8_qat):
        raise ValueError("KV8 export does not enable the kv_cache target")
    kv8_function = kv8_qat["function"].get("kv_cache_function")
    if not isinstance(kv8_function, str) or "int8" not in kv8_function:
        raise ValueError(f"KV8 export does not use an INT8 KV function: {kv8_function!r}")
    if _without_kv_settings(kv16_qat) != _without_kv_settings(kv8_qat):
        raise ValueError("Non-KV QAT configuration differs between the two exports")

    kv16_weights = kv16_model / "model.safetensors"
    kv8_weights = kv8_model / "model.safetensors"
    if not _files_equal(kv16_weights, kv8_weights):
        raise ValueError("KV16 and KV8 model.safetensors are not byte-identical")

    kv16_runtime_enabled, kv16_updates = _load_smoke_result(kv16_log)
    kv8_runtime_enabled, kv8_updates = _load_smoke_result(kv8_log)
    if kv16_runtime_enabled or kv16_updates != 0:
        raise ValueError(f"KV16 runtime mismatch: enabled={kv16_runtime_enabled}, updates={kv16_updates}")
    if not kv8_runtime_enabled or kv8_updates <= 0:
        raise ValueError(f"KV8 runtime mismatch: enabled={kv8_runtime_enabled}, updates={kv8_updates}")

    eval_audits: list[tuple[str, int, str]] = []
    if (kv16_eval_root is None) != (kv8_eval_root is None):
        raise ValueError("KV16 and KV8 evaluation roots must be provided together")
    if kv16_eval_root is not None and kv8_eval_root is not None:
        sample_count, batch_size = _audit_task_protocol(
            kv16_eval_root,
            kv8_eval_root,
            "humaneval_instruct",
        )
        eval_audits.append(("HumanEval", sample_count, batch_size))

    if (kv16_generation_eval_root is None) != (kv8_generation_eval_root is None):
        raise ValueError("KV16 and KV8 generation evaluation roots must be provided together")
    if kv16_generation_eval_root is not None and kv8_generation_eval_root is not None:
        for task, label in (
            ("gsm8k", "GSM8K"),
            ("ifeval_qwen3_instruct", "IFEval"),
        ):
            sample_count, batch_size = _audit_task_protocol(
                kv16_generation_eval_root,
                kv8_generation_eval_root,
                task,
            )
            eval_audits.append((label, sample_count, batch_size))

    size_bytes = kv16_weights.stat().st_size
    lines = [
        "# KV Pair Audit",
        "",
        "| Check | Result |",
        "| --- | --- |",
        f"| Latent/quantized weight artifact | Byte-identical ({size_bytes} bytes) |",
        "| Non-KV QAT configuration | Identical |",
        f"| KV16 runtime | Disabled, `{kv16_updates}` cache updates |",
        f"| KV8 runtime | `{kv8_function}`, `{kv8_updates}` cache updates |",
    ]
    for label, sample_count, batch_size in eval_audits:
        lines.append(f"| {label} protocol | Identical ({sample_count} samples, batch `{batch_size}`) |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kv16-model", required=True, type=Path)
    parser.add_argument("--kv8-model", required=True, type=Path)
    parser.add_argument("--kv16-log", required=True, type=Path)
    parser.add_argument("--kv8-log", required=True, type=Path)
    parser.add_argument("--kv16-eval-root", type=Path)
    parser.add_argument("--kv8-eval-root", type=Path)
    parser.add_argument("--kv16-generation-eval-root", type=Path)
    parser.add_argument("--kv8-generation-eval-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = audit_pair(
        args.kv16_model.resolve(),
        args.kv8_model.resolve(),
        args.kv16_log.resolve(),
        args.kv8_log.resolve(),
        args.kv16_eval_root.resolve() if args.kv16_eval_root else None,
        args.kv8_eval_root.resolve() if args.kv8_eval_root else None,
        args.kv16_generation_eval_root.resolve() if args.kv16_generation_eval_root else None,
        args.kv8_generation_eval_root.resolve() if args.kv8_generation_eval_root else None,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(args.output.resolve())
    else:
        print(report)


if __name__ == "__main__":
    main()
