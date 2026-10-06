from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_all import RUNS, computed, verdicts

BLUE = "#4a6fa5"
GRAY = "#c9c9c9"
RUST = "#b5513a"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
})


def computed_claims() -> list[str]:
    claims = sorted(p.name for p in (RUNS / "frozen").iterdir() if p.is_dir())
    return [c for c in claims if computed(c) and verdicts("frozen-search", c)]


def variance_figure(out: Path) -> None:
    rows = []
    for c in computed_claims():
        tools, search = verdicts("frozen", c, None), verdicts("frozen-search", c, None)
        n = min(len(tools), len(search))
        rows.append((st.pstdev(tools[:n]), st.pstdev(search[:n])))
    rows.sort(key=lambda r: (r[1], r[0]))
    x = np.arange(len(rows))
    tools = np.array([r[0] for r in rows])
    search = np.array([r[1] for r in rows])
    fig, ax = plt.subplots(figsize=(3.2, 2.0))
    ax.vlines(x, np.minimum(tools, search), np.maximum(tools, search), color=GRAY, lw=1.2, zorder=1)
    ax.scatter(x, search, s=16, color="white", edgecolor="#7a7a7a", lw=0.9, zorder=2, label="Search only")
    ax.scatter(x, tools, s=16, color=BLUE, zorder=3, label="With tools")
    ax.axhline(search.mean(), color="#7a7a7a", lw=0.6, ls=":")
    ax.axhline(tools.mean(), color=BLUE, lw=0.6, ls=":")
    ax.set_xticks([])
    ax.set_xlabel(f"Computed claims (n={len(rows)})")
    ax.set_ylabel("SD across 5 runs")
    ax.set_ylim(-0.08, 1.5)
    ax.legend(loc="lower center", frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 1.02), ncol=2, borderaxespad=0)
    fig.tight_layout()
    fig.savefig(out / "computed_variance.pdf")
    plt.close(fig)


def edit_rows(edit_mode: str, base_mode: str) -> list[dict]:
    rows = []
    for c in computed_claims():
        path = RUNS / edit_mode / c / "r1.json"
        base = verdicts(base_mode, c, None)
        if not path.exists() or not base:
            continue
        edited = json.loads(path.read_text())["verdict"]
        if edited is not None:
            rows.append({"edited": edited, "base": base, "base_mean": st.fmean(base)})
    return rows


def edit_figure(out: Path) -> None:
    edits = [("edit-quantifier", "frozen", "Quantifier made universal, with tools"),
             ("edit-quantifier-search", "frozen-search", "Quantifier made universal, search only")]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.2), sharey=True)
    bins = np.arange(-4.25, 2.26, 0.5)
    for ax, (edit_mode, base_mode, title) in zip(axes, edits):
        rows = edit_rows(edit_mode, base_mode)
        shifts = np.array([r["edited"] - r["base_mean"] for r in rows])
        below = np.array([r["edited"] < min(r["base"]) for r in rows])
        ax.hist([shifts[below], shifts[~below]], bins=bins, stacked=True, color=[RUST, GRAY],
                label=["below every unedited run", "within the unedited range"])
        ax.axvline(0, color="black", lw=0.6)
        ax.axvline(shifts.mean(), color=RUST, lw=1, ls="--")
        ax.set_title(f"{title} (n={len(rows)})")
        ax.set_xlabel("Score change after the edit (Likert steps)")
        ax.text(shifts.mean() - 0.1, 0.62, f"mean {shifts.mean():+.2f}", ha="right", va="top", fontsize=7,
                color=RUST, transform=ax.get_xaxis_transform())
    axes[0].set_ylabel("Claims")
    axes[1].legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(out / "computed_edits.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    variance_figure(args.out)
    edit_figure(args.out)


if __name__ == "__main__":
    main()
