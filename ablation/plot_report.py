#!/usr/bin/env python3
"""
The report's ablation figure: hit-rate on all 216 ablation questions (named and described)
vs chunk size, one line per citation budget k = 1, 3, 5, from ablation/results/summary.csv.

Writes report/figures/ablation_report.{png,pdf}.

    python3 ablation/plot_report.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "report" / "figures"
SIZES = [150, 250, 400, 600, 800]
SHIPPED_SIZE, SHIPPED_K = 400, 3

SURFACE, INK, INK_2, MUTED = "#ffffff", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
K_COLORS = {1: "#86b6ef", 3: "#2a78d6", 5: "#104281"}  # ordinal one-hue ramp: k is ordered


def style_axis(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    ax.set_xticks(range(len(SIZES)))
    ax.set_xticklabels([str(s) for s in SIZES])
    ax.set_xlim(-0.35, len(SIZES) - 0.65)
    ax.set_xlabel("Chunk size (tokens)")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)


def line(ax, ys, color, label, labels_below=True, show_values=True):
    xs = range(len(ys))
    ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label=label,
            solid_capstyle="round", solid_joinstyle="round")
    if show_values:
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.0f}", (x, y), xytext=(0, -11 if labels_below else 6),
                        textcoords="offset points", ha="center", fontsize=7, color=INK_2)


def main() -> None:
    rows = [r for r in csv.DictReader((HERE / "results" / "summary.csv").open())
            if r["group"] == "all_new"]
    hit = {(int(r["size"]), int(r["k"])): 100 * float(r["hit_rate"]) for r in rows}

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK_2,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    })
    fig, a = plt.subplots(figsize=(4.2, 2.8), facecolor=SURFACE)

    style_axis(a, "Hit-rate@k")
    for k in (1, 3, 5):
        line(a, [hit[(s, k)] for s in SIZES], K_COLORS[k],
             f"k = {k}" + (" (ours)" if k == SHIPPED_K else ""),
             labels_below=(k != 5), show_values=(k != 5))
    a.set_ylim(85, 101.5)
    a.set_yticks([85, 90, 95, 100])
    a.set_ylabel("% gold paper cited")
    a.legend(loc="lower right", frameon=False, fontsize=7.5)

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"ablation_report.{ext}", dpi=250, facecolor=SURFACE)
    print(f"wrote {OUT / 'ablation_report.png'} and .pdf")


if __name__ == "__main__":
    main()
