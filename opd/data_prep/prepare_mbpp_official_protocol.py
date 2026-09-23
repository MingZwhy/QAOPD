#!/usr/bin/env python3
"""Restore official MBPP train/validation/test boundaries from prepared rows."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import pandas as pd


OFFICIAL_SPLITS = ("train", "validation", "test")


def jsonable(value):
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ground_truth_key(row: dict) -> str:
    ground_truth = row["reward_model"]["ground_truth"]
    parsed = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def build_official_splits(prepared_dir: Path) -> tuple[dict[str, list[dict]], dict]:
    source_paths = [prepared_dir / "train.parquet", prepared_dir / "validation.parquet"]
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    source = pd.concat([pd.read_parquet(path) for path in source_paths], ignore_index=True)
    rows = source.to_dict("records")
    task_ids = [str(row["task_id"]) for row in rows]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Prepared source contains duplicate task IDs")

    splits: dict[str, list[dict]] = {name: [] for name in OFFICIAL_SPLITS}
    omitted_counts: dict[str, int] = {}
    for source_row in rows:
        row = copy.deepcopy(source_row)
        source_split = str(row["extra_info"]["source_split"])
        if source_split not in splits:
            omitted_counts[source_split] = omitted_counts.get(source_split, 0) + 1
            continue
        row["extra_info"]["protocol_split"] = source_split
        splits[source_split].append(row)

    key_sets: dict[str, set[str]] = {}
    prompt_sets: dict[str, set[str]] = {}
    for split, split_rows in splits.items():
        split_rows.sort(key=lambda row: int(row["task_id"]))
        for index, row in enumerate(split_rows):
            row["extra_info"]["split"] = split
            row["extra_info"]["index"] = index
        keys = {ground_truth_key(row) for row in split_rows}
        prompts = {
            hashlib.sha256(
                json.dumps(
                    jsonable(row["prompt"]), sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()
            for row in split_rows
        }
        if len(keys) != len(split_rows):
            raise ValueError(f"{split} contains duplicate canonical ground truth")
        if len(prompts) != len(split_rows):
            raise ValueError(f"{split} contains duplicate prompts")
        key_sets[split] = keys
        prompt_sets[split] = prompts

    for index, left in enumerate(OFFICIAL_SPLITS):
        for right in OFFICIAL_SPLITS[index + 1 :]:
            if overlap := key_sets[left] & key_sets[right]:
                raise ValueError(f"{left}/{right} ground-truth overlap: {len(overlap)}")
            if overlap := prompt_sets[left] & prompt_sets[right]:
                raise ValueError(f"{left}/{right} prompt overlap: {len(overlap)}")

    manifest = {
        "protocol": "mbpp_official_train_validation_test_humaneval_style_v1",
        "prepared_source": str(prepared_dir.resolve()),
        "prepared_source_files": {
            path.name: {"rows": len(pd.read_parquet(path)), "sha256": file_sha256(path)}
            for path in source_paths
        },
        "counts": {split: len(rows) for split, rows in splits.items()},
        "omitted_source_splits": omitted_counts,
        "split_rule": "extra_info.source_split from the original google-research-datasets/mbpp split",
        "human_eval_used_for_selection": False,
    }
    return splits, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output already exists: {args.output_dir}")

    splits, manifest = build_official_splits(args.prepared_dir)
    args.output_dir.mkdir(parents=True)
    for split, rows in splits.items():
        pd.DataFrame(rows).to_parquet(args.output_dir / f"{split}.parquet", index=False)
    manifest["output_sha256"] = {
        split: file_sha256(args.output_dir / f"{split}.parquet") for split in OFFICIAL_SPLITS
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
