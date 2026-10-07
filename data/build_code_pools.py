#!/usr/bin/env python3
"""Build the code OPD pools from OpenCoder, KodCode-V1 and the MBPP train split.

Each pool is listed row by row in data/code_pools/<pool>.txt, one task_id per line, in
training order:

    opc_<seq_id>        OpenCoder-LLM/opc-sft-stage2, educational_instruct subset
    kod_<question_id>   KodCode/KodCode-V1
    mbpp_<task_id>      MBPP official train split (data/mbpp_official_protocol_v1)

The script downloads the two source datasets (about 2.6 GB), turns every listed problem into
the MBPP-protocol row the code reward and chat template read -- a function stub with its
docstring as the prompt, the tests as ground truth -- and writes data/<pool>/train.parquet,
with the MBPP official validation split beside it as validation.parquet for the trainer.

Run it with the training environment's python. The stubs and test harnesses are produced by
ast.unparse, whose formatting has changed between Python releases, so each pool is checked
against its expected digest and a mismatch is reported.

    python data/build_code_pools.py [--pools code_opd_24k ...] \\
        [--opc <educational_instruct parquet>] [--kodcode <dir of KodCode-V1 parquet files>]

Build data/mbpp_official_protocol_v1 first (data/README.md).
"""

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import sys
import warnings
from pathlib import Path

import pandas as pd

# Some KodCode test files hold non-raw strings such as "\|", which ast.parse warns about.
warnings.filterwarnings("ignore", category=SyntaxWarning)

ROOT = Path(os.environ.get("QAOPD_ROOT") or Path(__file__).resolve().parents[1])
DATA_ROOT = Path(os.environ.get("DATA_ROOT") or ROOT / "data")
sys.path.insert(0, str(ROOT / "opd/data_prep"))
from prepare_mbpp_humaneval import build_function_stub, build_prompt  # noqa: E402

LISTS = ROOT / "data/code_pools"
MBPP = DATA_ROOT / "mbpp_official_protocol_v1"
OPC_REPO, OPC_FILE = "OpenCoder-LLM/opc-sft-stage2", "educational_instruct/train-00000-of-00001.parquet"
KOD_REPO = "KodCode/KodCode-V1"
KOD_COLUMNS = ["subset", "question_id", "solution", "test", "test_info"]

# Expected sha256 over the canonical JSON of every train row of each pool.
DIGESTS = {
    "code_opd_24k": "2515f7bc953d32c7636a99ab2bc195f7c1b14884370117731976cb8945092703",
    "code_opd_16k": "77b37d0b29f60e92d00124163fd8d258293219483de12d9671f84dddcd3dbc87",
    "code_opd_12k": "ec9e854b3413a359a4328f8121e129508f4f0ff7e8b403136ded820e714972a2",
    "code_opd_18k": "ef419046aca1c6fccf34d1a4faefdb21a1fbba858f6bf97ad5428529b0e93fc9",
}
EXTRA_KEYS = ("source_dataset", "task_id", "function_name", "subset", "split")


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def convert_pytest(test_code: str) -> tuple[str, list[str]]:
    """A KodCode pytest file as (setup code, ['test_x()', ...]) for the code reward's runner."""
    tree = ast.parse(test_code)
    body, calls = [], []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "solution":
            continue
        if isinstance(node, ast.Import) and any(alias.name in {"solution", "pytest"} for alias in node.names):
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            if node.decorator_list or node.args.args or node.args.vararg or node.args.kwonlyargs:
                raise ValueError("fixture or parametrized test")
            calls.append(f"{node.name}()")
        body.append(node)
    setup = ast.unparse(ast.Module(body=body, type_ignores=[]))
    if "pytest" in setup or "solution" in words(setup) or "capsys" in setup or "monkeypatch" in setup:
        raise ValueError("uses pytest machinery")
    if not calls:
        raise ValueError("no tests")
    return setup + "\n", calls


def example_asserts(test_code: str, name: str) -> list[str]:
    """Literal `assert name(...) == value` lines from a test file, shown in the docstring."""
    examples = []
    for node in ast.walk(ast.parse(test_code)):
        if isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare):
            call = node.test.left
            if (len(node.test.ops) == 1 and isinstance(node.test.ops[0], ast.Eq)
                    and isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == name):
                try:
                    for value in [*call.args, node.test.comparators[0]]:
                        ast.literal_eval(value)
                except (ValueError, TypeError, SyntaxError):
                    continue
                source = ast.unparse(node)
                if len(source) <= 160:
                    examples.append(source)
    return examples


def opc_problem(row: dict) -> dict:
    raw_tests = row["testcase"]
    tests = json.loads(raw_tests) if isinstance(raw_tests, str) else [str(t) for t in raw_tests]
    stub, name = build_function_stub(row["code"], " ".join(row["instruction"].split()), tests)
    return {"source": "opc", "source_id": str(row["seq_id"]), "name": name, "stub": stub, "tests": tests,
            "setup": "", "subset": "educational_instruct"}


def kod_problem(row: dict) -> dict:
    info = json.loads(row["test_info"]) if isinstance(row["test_info"], str) else list(row["test_info"])
    name = info[0]["function_name"]
    doc = (info[0].get("docstring") or "").strip()
    setup, calls = convert_pytest(row["test"])
    examples = example_asserts(row["test"], name)[:2]
    stub, _ = build_function_stub(row["solution"], doc, examples or [f"assert {name}()"])
    return {"source": "kod", "source_id": row["question_id"], "name": name, "stub": stub, "tests": calls,
            "setup": setup, "subset": row["subset"]}


def problem_row(problem: dict) -> dict:
    task_id = f"{problem['source']}_{problem['source_id']}"
    ground_truth = {"function_stub": problem["stub"], "function_name": problem["name"],
                    "tests": problem["tests"], "test_setup_code": problem["setup"]}
    return {
        "task_id": task_id,
        "data_source": "mbpp_humaneval" if problem["source"] == "opc" else "kodcode_humaneval",
        "prompt": build_prompt(problem["stub"]),
        "ability": "code",
        "reward_model": {"style": "rule", "ground_truth": json.dumps(ground_truth)},
        "extra_info": {"source_dataset": problem["source"], "task_id": task_id, "function_name": problem["name"],
                       "subset": problem["subset"], "split": "train"},
    }


def mbpp_row(row: dict) -> dict:
    task_id = f"mbpp_{row['task_id']}"
    extra = {key: str(row["extra_info"].get(key, "")) for key in EXTRA_KEYS}
    extra.update(source_dataset="mbpp", task_id=task_id)
    return {"task_id": task_id, "data_source": row["data_source"], "prompt": [dict(m) for m in row["prompt"]],
            "ability": "code", "reward_model": dict(row["reward_model"]), "extra_info": extra}


def canonical(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return [canonical(v) for v in value]
    if isinstance(value, dict):
        return {k: canonical(v) for k, v in value.items()}
    return value


def digest(rows: list[dict]) -> str:
    h = hashlib.sha256()
    for row in rows:
        h.update(json.dumps(canonical(row), sort_keys=True).encode())
    return h.hexdigest()


def fetch(args) -> tuple[Path, list[Path]]:
    if args.opc and args.kodcode:
        return args.opc, sorted(args.kodcode.glob("*.parquet"))
    from huggingface_hub import hf_hub_download, list_repo_files
    opc = args.opc or Path(hf_hub_download(OPC_REPO, OPC_FILE, repo_type="dataset"))
    if args.kodcode:
        return opc, sorted(args.kodcode.glob("*.parquet"))
    shards = [f for f in list_repo_files(KOD_REPO, repo_type="dataset")
              if f.startswith("data/") and f.endswith(".parquet")]
    return opc, sorted(Path(hf_hub_download(KOD_REPO, f, repo_type="dataset")) for f in shards)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pools", nargs="+", default=list(DIGESTS), choices=list(DIGESTS))
    parser.add_argument("--opc", type=Path, help="local educational_instruct parquet (skips the download)")
    parser.add_argument("--kodcode", type=Path, help="local directory of KodCode-V1 parquet files")
    parser.add_argument("--out_root", type=Path, default=DATA_ROOT)
    args = parser.parse_args()
    if not (MBPP / "train.parquet").is_file():
        sys.exit(f"missing {MBPP}/train.parquet -- build the MBPP official splits first (data/README.md)")

    lists = {pool: (LISTS / f"{pool}.txt").read_text().split() for pool in args.pools}
    wanted = {t for ids in lists.values() for t in ids}
    opc_path, kod_paths = fetch(args)

    problems = {}
    opc_ids = {int(t[4:]) for t in wanted if t.startswith("opc_")}
    for row in pd.read_parquet(opc_path).to_dict("records"):
        if row["seq_id"] in opc_ids:
            problems[f"opc_{row['seq_id']}"] = problem_row(opc_problem(row))
    kod_ids = {t[4:] for t in wanted if t.startswith("kod_")}
    for path in kod_paths:
        frame = pd.read_parquet(path, columns=KOD_COLUMNS)
        for row in frame[frame["question_id"].isin(kod_ids)].to_dict("records"):
            problems[f"kod_{row['question_id']}"] = problem_row(kod_problem(row))
    for row in pd.read_parquet(MBPP / "train.parquet").to_dict("records"):
        problems[f"mbpp_{row['task_id']}"] = mbpp_row(row)
    missing = sorted(wanted - problems.keys())
    if missing:
        sys.exit(f"{len(missing)} listed problems not found in the sources, e.g. {missing[:5]}")

    for pool, ids in lists.items():
        rows = []
        for index, task_id in enumerate(ids):
            row = json.loads(json.dumps(problems[task_id]))
            row["extra_info"]["index"] = index
            rows.append(row)
        out = args.out_root / pool
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(out / "train.parquet", index=False)
        shutil.copy(MBPP / "validation.parquet", out / "validation.parquet")
        got = digest(rows)
        status = "matches the expected digest" if got == DIGESTS[pool] else \
            f"DIFFERS from the expected digest ({DIGESTS[pool][:12]}); check the python version"
        print(f"{pool}: {len(rows)} rows -> {out}  sha256 {got[:12]} {status}")


if __name__ == "__main__":
    main()
