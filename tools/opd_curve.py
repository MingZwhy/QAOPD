#!/usr/bin/env python3
"""Extract an OPD run's metrics to CSV and plot them.

verl prints every metric it tracks to the console once per optimizer step, as
`step:N key:value key:value ...`. This reads that back out of a training log,
so nothing has to be reconfigured and a run that has already finished is still
recoverable.

What matters most on these runs:

    critic/score/mean          the task verifier's score on sampled rollouts
    critic/rewards/mean        the reward actually optimized
    actor/pg_loss              policy-gradient term
    actor/distill_loss         token-level distillation against the teacher
    actor/entropy              rollout entropy; a collapse here precedes a
                               collapse in everything else
    response_length/mean       generations running to the cap is the failure
    response_length/clip_ratio mode quantization causes

Usage:
    python tools/opd_curve.py runs/<run>/train.log
    python tools/opd_curve.py <log> --out results/opd_1_7b_w2.79
    python tools/opd_curve.py <log> --no-plot
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import re
import sys

STEP = re.compile(r"step:(\d+)\s")
PAIR = re.compile(r"([A-Za-z_][\w/]*):(-?[\d.]+(?:e-?\d+)?)")

PANELS = [
    ("reward", ["critic/score/mean", "critic/rewards/mean"]),
    ("loss", ["actor/pg_loss", "actor/distill_loss", "actor/kl_loss"]),
    ("entropy", ["actor/entropy"]),
    ("response length", ["response_length/mean", "response_length/clip_ratio"]),
]


def parse(log: pathlib.Path) -> dict[int, dict[str, float]]:
    """verl emits several partial metric lines per step; merge them by step."""
    steps: dict[int, dict[str, float]] = {}
    for line in log.read_text(errors="replace").splitlines():
        m = STEP.search(line)
        if not m:
            continue
        row = steps.setdefault(int(m.group(1)), {})
        for k, v in PAIR.findall(line[m.end():]):
            try:
                row[k] = float(v)
            except ValueError:
                pass
    return steps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--out")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    log = pathlib.Path(args.log)
    if not log.is_file():
        sys.exit(f"no such log: {log}")
    steps = parse(log)
    if not steps:
        sys.exit(f"{log} has no `step:N key:value` lines yet")

    order = sorted(steps)
    cols: list[str] = []
    for _, keys in PANELS:
        cols += [k for k in keys if any(k in steps[s] for s in order)]
    cols += sorted({k for s in order for k in steps[s]} - set(cols))

    stem = pathlib.Path(args.out) if args.out else log.with_suffix("")
    stem = stem.parent / (stem.name + ("" if args.out else "_metrics"))
    stem.parent.mkdir(parents=True, exist_ok=True)

    with open(f"{stem}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step"] + cols)
        for s in order:
            w.writerow([s] + [steps[s].get(c, "") for c in cols])

    print(f"source  {log}")
    print(f"steps   {order[0]} .. {order[-1]}  ({len(order)} logged, {len(cols)} metrics)")
    print(f"csv     {stem}.csv")
    for k in ("critic/score/mean", "critic/rewards/mean", "actor/entropy"):
        vals = [steps[s][k] for s in order if k in steps[s]]
        if vals:
            print(f"  {k:<26} {vals[0]:8.4f} -> {vals[-1]:8.4f}")

    if args.no_plot:
        return 0
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("plot    skipped (no matplotlib; the CSV is written)")
        return 0

    live = [(t, [k for k in keys if any(k in steps[s] for s in order)])
            for t, keys in PANELS]
    live = [(t, ks) for t, ks in live if ks]
    fig, axes = plt.subplots(1, len(live), figsize=(4.4 * len(live), 3.8),
                             constrained_layout=True, squeeze=False)
    for ax, (title, keys) in zip(axes[0], live):
        for k in keys:
            xs = [s for s in order if k in steps[s]]
            ax.plot(xs, [steps[s][k] for s in xs], lw=1.2,
                    label=k.split("/", 1)[-1])
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(frameon=False, fontsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.savefig(f"{stem}.png", dpi=150)
    print(f"plot    {stem}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
