#!/usr/bin/env python3
"""Extract a training run's loss history to CSV and plot it.

Reads `trainer_state.json`, which the HF trainer rewrites at every checkpoint
and which carries the whole `log_history` -- every logged step, not a
downsample. Point this at a run directory and it finds the newest checkpoint
itself.

The QAD trainer logs, per step:

    loss              the optimizer step's loss, as HF reports it
    train/loss_total  the same quantity from EdgeRazor's own accounting
    train/loss_task   next-token cross-entropy against the data
    train/loss_dist   the distillation term against the frozen teacher
    train/loss_dist_N its per-component breakdown
    grad_norm, learning_rate

Usage:
    python tools/loss_curve.py runs/qad_workspace/EdgeRazor-QLLM/train
    python tools/loss_curve.py <run> --out results/qad_1_7b_w2.79
    python tools/loss_curve.py <run> --no-plot          # CSV only
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

PREFERRED = [
    "loss",
    "train/loss_total",
    "train/loss_task",
    "train/loss_dist",
    "grad_norm",
    "learning_rate",
]


def find_state(run: pathlib.Path) -> pathlib.Path:
    """trainer_state.json from the highest-numbered checkpoint under `run`."""
    if run.is_file():
        return run
    direct = run / "trainer_state.json"
    if direct.is_file():
        return direct
    cks = [
        (int(p.name.split("-")[-1]), p)
        for p in run.glob("checkpoint-*")
        if p.name.split("-")[-1].isdigit() and (p / "trainer_state.json").is_file()
    ]
    if not cks:
        sys.exit(f"no trainer_state.json under {run} (nor in any checkpoint-*)")
    return max(cks)[1] / "trainer_state.json"


def history(state: pathlib.Path) -> list[dict]:
    rows = json.loads(state.read_text()).get("log_history", [])
    # The final entry of a finished run is the train summary, not a step.
    return [r for r in rows if "step" in r and "loss" in r]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run dir, checkpoint dir, or trainer_state.json")
    ap.add_argument("--out", help="output stem (default: <run>/loss_curve)")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    state = find_state(pathlib.Path(args.run))
    rows = history(state)
    if not rows:
        sys.exit(f"{state} has no step entries yet")

    stem = pathlib.Path(args.out) if args.out else state.parent.parent / "loss_curve"
    stem.parent.mkdir(parents=True, exist_ok=True)

    cols = [c for c in PREFERRED if any(c in r for r in rows)]
    cols += sorted({k for r in rows for k in r} - set(cols) - {"step", "epoch"})
    with open(f"{stem}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step"] + cols)
        for r in rows:
            w.writerow([r["step"]] + [r.get(c, "") for c in cols])

    print(f"source  {state}")
    print(f"steps   {rows[0]['step']} .. {rows[-1]['step']}  ({len(rows)} logged)")
    print(f"csv     {stem}.csv")

    if args.no_plot:
        return 0
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("plot    skipped (no matplotlib; the CSV is written)")
        return 0

    steps = [r["step"] for r in rows]
    panels = [c for c in ("train/loss_total", "train/loss_task", "train/loss_dist")
              if any(c in r for r in rows)] or ["loss"]
    extra = sorted(c for c in cols if c.startswith("train/loss_dist_"))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for name in panels:
        axes[0].plot(steps, [r.get(name) for r in rows], lw=1, label=name.split("/")[-1])
    axes[0].set_title("loss"); axes[0].set_xlabel("step"); axes[0].legend(frameon=False)

    for name in extra:
        axes[1].plot(steps, [r.get(name) for r in rows], lw=1, label=name.split("/")[-1])
    axes[1].set_title("distillation components"); axes[1].set_xlabel("step")
    if extra:
        axes[1].legend(frameon=False)

    if "grad_norm" in cols:
        axes[2].plot(steps, [r.get("grad_norm") for r in rows], lw=1, color="tab:red")
        axes[2].set_yscale("log")
    axes[2].set_title("grad norm"); axes[2].set_xlabel("step")

    for ax in axes:
        ax.grid(alpha=0.25, lw=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.savefig(f"{stem}.png", dpi=150)
    print(f"plot    {stem}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
