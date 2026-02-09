#!/usr/bin/env python3
"""Visualize pairwise evaluation results as a horizontal bar chart.

Usage:
  python -m src.mampfsearch.evaluation.visualize_pairwise --csv path/to/pairwise_results.csv --out winners.png

Dependencies: pandas, matplotlib
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_COLORS = {
    "Multi-Entity": "#2b8cbe",
    "Unstructured": "#f03b20",
    "TIE": "#969696",
}


def plot_winner_counts(
    csv_path: Path, out_path: Path | None = None, title: str | None = None
):
    df = pd.read_csv(csv_path)
    if "winner" not in df.columns:
        raise ValueError("CSV does not contain a 'winner' column")

    # Map raw winner strings into the three categories we care about.
    def map_category(s: str) -> str:
        if not isinstance(s, str):
            return "INVALID"
        t = s.strip().lower()
        if t == "":
            return "INVALID"
        if "multi" in t or "entity" in t:
            return "Multi-Entity"
        if "unstruct" in t or "unstructured" in t:
            return "Unstructured"
        if t == "tie" or t == "tied":
            return "TIE"
        return s

    winners = df["winner"].fillna("")
    mapped = winners.astype(str).apply(map_category)

    # Order: put Multi-Entity on the left
    cats = ["Multi-Entity", "Unstructured", "TIE"]
    counts = [int((mapped == c).sum()) for c in cats]
    total = sum(counts)

    colors = [DEFAULT_COLORS.get(c, None) for c in cats]

    fig, ax = plt.subplots(figsize=(8, 6))

    if total <= 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
    else:
        x_pos = list(range(len(cats)))
        ax.bar(x_pos, counts, color=colors, width=0.6)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(cats, fontsize=13)
        ax.tick_params(axis="x", which="major", pad=8)

        for i, v in enumerate(counts):
            pct = 100.0 * v / total if total > 0 else 0.0
            ax.text(
                i,
                v + max(1, total * 0.01),
                f"{v} ({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=12,
            )

        ax.set_ylabel("Count of pairs", fontsize=13)
        # Add explicit pad so title doesn't overlap the figure edge
        ax.set_title(
            title or f"Pairwise winners ({total} comparisons)", fontsize=16, pad=24
        )

    # Increase padding and explicitly reserve space at the top to avoid clipping
    plt.tight_layout(pad=0.8)
    plt.subplots_adjust(top=1.2)

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
        print(f"Saved chart to {out_path}")
    else:
        plt.show()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Visualize pairwise evaluation winner counts"
    )
    parser.add_argument(
        "--csv",
        required=False,
        default="src/mampfsearch/evaluation/Results/18100a-lecture-15-multicam_360p_16_9/gpt-oss_20b_20260208_143908/pairwise_results.csv",
        help="Path to pairwise_results.csv",
    )
    parser.add_argument(
        "--out",
        required=False,
        help="Output image path (PNG). If omitted, show interactively",
    )
    parser.add_argument("--title", required=False, help="Optional chart title")
    args = parser.parse_args(argv)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(2)

    out_path = Path(args.out) if args.out else None
    plot_winner_counts(csv_path, out_path, args.title)


if __name__ == "__main__":
    main()
