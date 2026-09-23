#!/usr/bin/env python3
"""Verify that a checkpoint contains latent weights suitable for QAT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PREQUANTIZED_STATES = {
    "prequantized",
    "prequantized_deployment",
    "quantized",
    "quantized_deployment",
}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _is_true(value: Any) -> bool:
    return value is True or (
        isinstance(value, str) and value.strip().lower() in {"1", "true", "yes"}
    )


def _config_evidence(config: dict[str, Any], config_path: Path) -> list[str]:
    evidence: list[str] = []

    quantization_config = config.get("quantization_config")
    if isinstance(quantization_config, dict) and _is_true(
        quantization_config.get("is_w_quantized")
    ):
        evidence.append(f"{config_path}: quantization_config.is_w_quantized=true")

    edgerazor_config = config.get("edgerazor_config")
    if isinstance(edgerazor_config, dict):
        qat_config = edgerazor_config.get("qat_configuration", edgerazor_config)
        if isinstance(qat_config, dict):
            function = qat_config.get("function")
            if isinstance(function, dict) and _is_true(function.get("is_w_quantized")):
                evidence.append(
                    f"{config_path}: "
                    "edgerazor_config.qat_configuration.function.is_w_quantized=true"
                )

    state = config.get("opd_qad_weight_state")
    if isinstance(state, str) and state.strip().lower() in PREQUANTIZED_STATES:
        evidence.append(f"{config_path}: opd_qad_weight_state={state!r}")

    return evidence


def find_prequantized_evidence(
    model_dir: Path,
    *,
    _visited: set[Path] | None = None,
) -> list[str]:
    model_dir = model_dir.expanduser().resolve()
    visited = _visited if _visited is not None else set()
    if model_dir in visited:
        return []
    visited.add(model_dir)

    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise ValueError(f"Checkpoint has no config.json: {model_dir}")

    evidence = _config_evidence(_load_json(config_path), config_path)

    manifest_path = model_dir / "trainable_view.json"
    if manifest_path.is_file():
        manifest = _load_json(manifest_path)
        state = manifest.get("weight_state")
        if isinstance(state, str) and state.strip().lower() in PREQUANTIZED_STATES:
            evidence.append(f"{manifest_path}: weight_state={state!r}")

        for source_key in ("source", "source_weights"):
            source_value = manifest.get(source_key)
            if not isinstance(source_value, str):
                continue
            source = Path(source_value).expanduser().resolve()
            source_dir = source if source.is_dir() else source.parent
            if source_dir != model_dir and (source_dir / "config.json").is_file():
                evidence.extend(
                    find_prequantized_evidence(source_dir, _visited=visited)
                )

    weights_path = model_dir / "model.safetensors"
    if weights_path.is_symlink():
        weight_source = weights_path.resolve().parent
        if weight_source != model_dir and (weight_source / "config.json").is_file():
            evidence.extend(
                find_prequantized_evidence(weight_source, _visited=visited)
            )

    return evidence


def validate_latent_checkpoint(model_dir: Path, purpose: str) -> None:
    evidence = find_prequantized_evidence(model_dir)
    if not evidence:
        print(f"Validated latent QAT checkpoint for {purpose}: {model_dir.resolve()}")
        return

    details = "\n".join(f"  - {item}" for item in evidence)
    raise ValueError(
        f"{purpose} requires latent, unquantized master weights, but "
        f"{model_dir.resolve()} contains deployment-time quantized weights:\n"
        f"{details}\n"
        "Re-quantizing these weights is not idempotent and changes the model. "
        "Use the latent checkpoint from before replace_quantized_weights()."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--purpose", default="QAT")
    args = parser.parse_args()
    validate_latent_checkpoint(args.model, args.purpose)


if __name__ == "__main__":
    main()
