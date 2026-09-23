#!/usr/bin/env python3
"""Build a matched OPD evaluation table from lm-eval result directories."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class EvalRun:
    label: str
    root: Path
    humaneval: float | None = None
    ifeval: float | None = None
    gsm_strict: float | None = None
    gsm_flexible: float | None = None
    main_average: float | None = None
    humaneval_samples: dict[int, dict[str, Any]] | None = None
    gsm_samples: dict[str, dict[int, dict[str, Any]]] | None = None


def _load_latest_result(root: Path, task: str) -> tuple[dict[str, Any], Path] | None:
    matches: list[tuple[int, Path, dict[str, Any]]] = []
    for path in root.rglob("results_*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if task in data.get("results", {}):
            matches.append((path.stat().st_mtime_ns, path, data))
    if not matches:
        return None
    _, path, data = max(matches, key=lambda item: (item[0], str(item[1])))
    return data, path


def _load_humaneval_samples(result_path: Path) -> dict[int, dict[str, Any]] | None:
    files = sorted(result_path.parent.glob("samples_humaneval_instruct_*.jsonl"))
    if not files:
        return None
    samples: dict[int, dict[str, Any]] = {}
    with files[-1].open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            samples[int(row["doc_id"])] = row
    return samples


def _load_gsm_samples(result_path: Path) -> dict[str, dict[int, dict[str, Any]]] | None:
    files = sorted(result_path.parent.glob("samples_gsm8k_*.jsonl"))
    if not files:
        return None
    samples: dict[str, dict[int, dict[str, Any]]] = {}
    with files[-1].open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            filter_name = str(row["filter"])
            doc_id = int(row["doc_id"])
            filter_samples = samples.setdefault(filter_name, {})
            if doc_id in filter_samples:
                raise ValueError(f"Duplicate GSM8K sample for filter={filter_name}, doc_id={doc_id}")
            filter_samples[doc_id] = row
    return samples


def _metric(metrics: dict[str, Any], name: str) -> float:
    value = metrics[name]
    if not isinstance(value, (int, float)):
        raise TypeError(f"Metric {name} is not numeric: {value!r}")
    return float(value)


def _main_average(data: dict[str, Any]) -> float:
    results = data["results"]
    values = [
        _metric(results["arc_easy"], "acc_norm,none"),
        _metric(results["arc_challenge"], "acc_norm,none"),
        _metric(results["hellaswag"], "acc_norm,none"),
        _metric(results["boolq"], "acc,none"),
        _metric(results["social_iqa"], "acc,none"),
        _metric(results["openbookqa"], "acc_norm,none"),
        _metric(results["piqa"], "acc_norm,none"),
        _metric(results["winogrande"], "acc,none"),
        _metric(results["truthfulqa_mc2"], "acc,none"),
        _metric(results["mmlu"], "acc,none"),
    ]
    ethics = [
        _metric(results[name], "acc,none")
        for name in (
            "ethics_cm",
            "ethics_deontology",
            "ethics_justice",
            "ethics_utilitarianism",
            "ethics_virtue",
        )
    ]
    values.append(sum(ethics) / len(ethics))
    return sum(values) / len(values)


def load_run(label: str, root: Path) -> EvalRun:
    run = EvalRun(label=label, root=root)

    humaneval_result = _load_latest_result(root, "humaneval_instruct")
    if humaneval_result is not None:
        data, path = humaneval_result
        run.humaneval = _metric(data["results"]["humaneval_instruct"], "pass@1,create_test")
        run.humaneval_samples = _load_humaneval_samples(path)

    ifeval_result = _load_latest_result(root, "ifeval_qwen3_instruct")
    if ifeval_result is not None:
        data, _ = ifeval_result
        run.ifeval = _metric(
            data["results"]["ifeval_qwen3_instruct"],
            "prompt_level_strict_acc,none",
        )

    gsm_result = _load_latest_result(root, "gsm8k")
    if gsm_result is not None:
        data, path = gsm_result
        metrics = data["results"]["gsm8k"]
        run.gsm_strict = _metric(metrics, "exact_match,strict-match")
        run.gsm_flexible = _metric(metrics, "exact_match,flexible-extract")
        run.gsm_samples = _load_gsm_samples(path)

    main_result = _load_latest_result(root, "arc_easy")
    if main_result is not None:
        data, _ = main_result
        run.main_average = _main_average(data)
    return run


def exact_mcnemar_p(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(gains, losses) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def paired_humaneval(
    baseline: dict[int, dict[str, Any]],
    candidate: dict[int, dict[str, Any]],
) -> dict[str, int | float]:
    if set(baseline) != set(candidate):
        raise ValueError("HumanEval doc_id sets differ")
    gains = losses = both_correct = both_wrong = 0
    for doc_id in sorted(baseline):
        before = baseline[doc_id]
        after = candidate[doc_id]
        for key in ("doc_hash", "prompt_hash", "target_hash", "arguments"):
            if before.get(key) != after.get(key):
                raise ValueError(f"HumanEval {key} differs for doc_id={doc_id}")
        before_ok = float(before["pass@1"]) > 0.5
        after_ok = float(after["pass@1"]) > 0.5
        if not before_ok and after_ok:
            gains += 1
        elif before_ok and not after_ok:
            losses += 1
        elif before_ok:
            both_correct += 1
        else:
            both_wrong += 1
    return {
        "gains": gains,
        "losses": losses,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "mcnemar_p": exact_mcnemar_p(gains, losses),
    }


def paired_gsm8k(
    baseline: dict[str, dict[int, dict[str, Any]]],
    candidate: dict[str, dict[int, dict[str, Any]]],
    filter_name: str,
) -> dict[str, int | float]:
    if filter_name not in baseline or filter_name not in candidate:
        raise ValueError(f"Missing GSM8K filter: {filter_name}")
    before_samples = baseline[filter_name]
    after_samples = candidate[filter_name]
    if set(before_samples) != set(after_samples):
        raise ValueError(f"GSM8K doc_id sets differ for filter={filter_name}")

    gains = losses = both_correct = both_wrong = 0
    for doc_id in sorted(before_samples):
        before = before_samples[doc_id]
        after = after_samples[doc_id]
        for key in ("doc_hash", "prompt_hash", "target_hash", "arguments"):
            if before.get(key) != after.get(key):
                raise ValueError(f"GSM8K {key} differs for filter={filter_name}, doc_id={doc_id}")
        before_ok = float(before["exact_match"]) > 0.5
        after_ok = float(after["exact_match"]) > 0.5
        if not before_ok and after_ok:
            gains += 1
        elif before_ok and not after_ok:
            losses += 1
        elif before_ok:
            both_correct += 1
        else:
            both_wrong += 1
    return {
        "gains": gains,
        "losses": losses,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "mcnemar_p": exact_mcnemar_p(gains, losses),
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.2f}"


def render_markdown(baseline: EvalRun, candidates: list[EvalRun]) -> str:
    lines = [
        "# OPD Evaluation Comparison",
        "",
        "Scores are percentages. The main average contains 11 non-generation tasks;",
        "the five Hendrycks ethics subsets are averaged into one task.",
        "",
        "| Run | HumanEval | HE delta | Paired gains/losses | McNemar p | "
        "IFEval strict | GSM strict | GSM flexible | Main avg |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    all_runs = [baseline, *candidates]
    for run in all_runs:
        delta = "n/a"
        paired = "n/a"
        p_value = "n/a"
        if run.humaneval is not None and baseline.humaneval is not None:
            delta = f"{100 * (run.humaneval - baseline.humaneval):+.2f}"
        if run is not baseline and baseline.humaneval_samples and run.humaneval_samples:
            comparison = paired_humaneval(baseline.humaneval_samples, run.humaneval_samples)
            paired = f"{comparison['gains']}/{comparison['losses']}"
            p_value = f"{comparison['mcnemar_p']:.4g}"
        lines.append(
            f"| {run.label} | {_pct(run.humaneval)} | {delta} | {paired} | {p_value} | "
            f"{_pct(run.ifeval)} | {_pct(run.gsm_strict)} | {_pct(run.gsm_flexible)} | "
            f"{_pct(run.main_average)} |"
        )
    lines.extend(
        [
            "",
            "HumanEval paired comparisons require identical `doc_hash`, `prompt_hash`,",
            "`target_hash`, and generation `arguments` for every sample.",
            "",
        ]
    )
    gsm_comparisons = [
        run for run in candidates if baseline.gsm_samples is not None and run.gsm_samples is not None
    ]
    if gsm_comparisons:
        lines.extend(
            [
                "## GSM8K Paired Comparisons",
                "",
                "| Run | Strict gains/losses | Strict McNemar p | "
                "Flexible gains/losses | Flexible McNemar p |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for run in gsm_comparisons:
            assert baseline.gsm_samples is not None and run.gsm_samples is not None
            strict = paired_gsm8k(baseline.gsm_samples, run.gsm_samples, "strict-match")
            flexible = paired_gsm8k(baseline.gsm_samples, run.gsm_samples, "flexible-extract")
            lines.append(
                f"| {run.label} | {strict['gains']}/{strict['losses']} | "
                f"{strict['mcnemar_p']:.4g} | {flexible['gains']}/{flexible['losses']} | "
                f"{flexible['mcnemar_p']:.4g} |"
            )
        lines.extend(
            [
                "",
                "GSM8K paired comparisons apply the same protocol-integrity checks as HumanEval.",
                "",
            ]
        )
    return "\n".join(lines)


def _run_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must use LABEL=EVAL_ROOT")
    label, raw_path = value.split("=", 1)
    path = Path(raw_path).expanduser().resolve()
    if not label or not path.is_dir():
        raise argparse.ArgumentTypeError(f"invalid run specification: {value}")
    return label, path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=_run_spec)
    parser.add_argument("--run", action="append", default=[], type=_run_spec)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    baseline = load_run(*args.baseline)
    candidates = [load_run(*spec) for spec in args.run]
    report = render_markdown(baseline, candidates)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(args.output.resolve())
    else:
        print(report)


if __name__ == "__main__":
    main()
