#!/usr/bin/env python3
"""
Plot the R4 ablation from ablation/results/summary.csv.

Two panels (named / described questions) on one shared y-scale; x = chunk size,
one line per citation budget k with its 95% paper-clustered bootstrap CI band.
Writes ablation/results/ablation.png (and .pdf for the report).

    python3 ablation/plot_ablation.py            # k = 1, 3, 5
    python3 ablation/plot_ablation.py --ks 1 2 3
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SHIPPED_SIZE, SHIPPED_K = 400, 3

# Light-mode tokens from the dataviz reference palette (print target).
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
# k is ordered, so it takes an ordinal one-hue ramp (blue, no lighter than step 250).
BLUE_STEPS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]


def ramp(n: int) -> list[str]:
    if n == 1:
        return [BLUE_STEPS[2]]
    return [BLUE_STEPS[round(i * (len(BLUE_STEPS) - 1) / (n - 1))] for i in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5])
    args = ap.parse_args()

    rows = list(csv.DictReader((RESULTS / "summary.csv").open()))
    for r in rows:
        for key in ("size", "k", "n"):
            r[key] = int(r[key])
        for key in ("hit_rate", "ci_lo", "ci_hi"):
            r[key] = float(r[key])
    sizes = sorted({r["size"] for r in rows})
    colors = dict(zip(args.ks, ramp(len(args.ks))))
    panels = [("named", "Named questions"), ("described", "Described questions")]

    shown = [r for r in rows if r["group"] in dict(panels) and r["k"] in args.ks]
    y_lo = max(0.0, math.floor(min(r["ci_lo"] for r in shown) * 10) / 10 - 0.05)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK_2,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True, facecolor=SURFACE)

    for ax, (group, label) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        g = [r for r in rows if r["group"] == group]
        n = g[0]["n"] if g else 0
        ends = {}
        for k in args.ks:
            pts = sorted((r for r in g if r["k"] == k), key=lambda r: r["size"])
            xs = [r["size"] for r in pts]
            ax.fill_between(xs, [r["ci_lo"] for r in pts], [r["ci_hi"] for r in pts],
                            color=colors[k], alpha=0.10, linewidth=0)
            ax.plot(xs, [r["hit_rate"] for r in pts], color=colors[k], linewidth=2,
                    solid_capstyle="round", solid_joinstyle="round",
                    marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5,
                    label=f"k = {k}" + (" (shipped)" if k == SHIPPED_K else ""))
            ends[k] = (xs[-1], pts[-1]["hit_rate"])
        # Direct end-labels only where they don't collide (shipped k first);
        # the legend carries identity for the rest.
        placed: list[float] = []
        for k in sorted(ends, key=lambda k: k != SHIPPED_K):
            x, y = ends[k]
            if all(abs(y - p) >= 0.03 for p in placed):
                ax.annotate(f"k={k}", (x, y), xytext=(6, 0), textcoords="offset points",
                            va="center", fontsize=8, color=INK_2)
                placed.append(y)
        ax.axvline(SHIPPED_SIZE, color=GRID, linewidth=1, zorder=0)
        ax.text(SHIPPED_SIZE, y_lo + 0.01, " shipped corpus", fontsize=7, color=MUTED, va="bottom")
        ax.set_title(f"{label} (n = {n})", fontsize=9.5, color=INK, loc="left")
        ax.set_xticks(sizes)
        ax.set_xlim(sizes[0] - 40, sizes[-1] + 90)
        ax.set_ylim(y_lo, 1.015)  # headroom so markers at 1.0 aren't clipped
        ax.set_yticks([t / 10 for t in range(math.ceil(y_lo * 10), 11)])
        ax.set_xlabel("Chunk size (tokens)")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)

    axes[0].set_ylabel("Paper hit-rate (gold paper cited)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(args.ks), frameon=False, fontsize=8,
               bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("Hit-rate vs. chunk size, by citation budget k", x=0.01, ha="left",
                 fontsize=10.5, color=INK, fontweight="semibold")
    fig.text(0.01, 0.005, "BM25 over 108 papers. Bands: 95% bootstrap CI, resampling papers.",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(RESULTS / f"ablation.{ext}", dpi=200, facecolor=SURFACE)
    print(f"wrote {RESULTS / 'ablation.png'} and .pdf")


if __name__ == "__main__":
    main()
