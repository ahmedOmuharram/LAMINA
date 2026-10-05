#!/usr/bin/env python3
"""
Generate plots for all hypothesis results.

Creates visualizations for H1-H5 hypothesis testing results.
Outputs to benchmarks/plots/

Usage:
    python generate_plots.py
"""

import json
import sys
from pathlib import Path
import numpy as np

# Try to import matplotlib with a non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from common.logging_utils import load_run_json, extract_labels_from_run
from common.metrics import compute_qwk, bootstrap_kappa_diff_ci

# Output directory
PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Color palette - distinctive and accessible
COLORS = {
    'with_tools': '#2ecc71',      # Green
    'no_tools': '#e74c3c',        # Red
    'no_policy': '#95a5a6',       # Gray
    'policy': '#3498db',          # Blue
    'oracle': '#9b59b6',          # Purple
    'crisp': '#1abc9c',           # Teal
    'ambiguous': '#e67e22',       # Orange
    'fr': '#e67e22',              # Orange for FR
    'qwfr': '#d35400',            # Darker orange for QWFR
    'multi_call': '#2980b9',      # Dark blue
    'single_call': '#f39c12',     # Yellow
    'small_model': '#e74c3c',     # Red
    'large_model': '#2ecc71',     # Green
    'ci_band': '#bdc3c7',         # Light gray
}

# Plot style settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def load_summary(hypothesis: str) -> dict:
    """Load summary JSON for a hypothesis."""
    summary_path = Path(__file__).parent / hypothesis / f"summary_{hypothesis}_metrics.json"
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)
    return {}


def plot_h1_qwk():
    """H1: QWK comparison bar chart."""
    summary = load_summary("h1")
    if not summary:
        print("H1 summary not found, skipping H1 plots")
        return
    
    per_run = summary.get("per_run", {})
    models = ["gpt-4o-mini", "gpt-4o", "gpt-5.1"]
    model_labels = ["4o-mini", "4o", "5.1"]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(models))
    width = 0.35
    
    qwk_tools = []
    qwk_no_tools = []
    
    for model in models:
        tools_key = f"h1_{model}_with-tools_policy"
        no_tools_key = f"h1_{model}_no-tools_policy"
        qwk_tools.append(per_run.get(tools_key, {}).get("kappa", 0))
        qwk_no_tools.append(per_run.get(no_tools_key, {}).get("kappa", 0))
    
    bars1 = ax.bar(
        x - width/2,
        qwk_no_tools,
        width,
        label='No Tools',
        color=COLORS['no_tools'],
        edgecolor='white',
        linewidth=1,
    )
    bars2 = ax.bar(
        x + width/2,
        qwk_tools,
        width,
        label='With Tools',
        color=COLORS['with_tools'],
        edgecolor='white',
        linewidth=1,
    )
    
    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.02,
            f'{bar.get_height():.3f}',
            ha='center',
            va='bottom',
            fontsize=9,
        )
    for bar in bars2:
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.02,
            f'{bar.get_height():.3f}',
            ha='center',
            va='bottom',
            fontsize=9,
        )
    
    ax.axhspan(0.0, 0.33, alpha=0.12, color='gray', label='Chance-level (95% band)')
    
    ax.set_ylabel('Quadratic Weighted Kappa (QWK)')
    ax.set_xlabel('Model')
    ax.set_title('H1: Tools vs No-Tools Agreement')
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.set_ylim(0, 0.7)
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "h1_qwk_comparison.png", dpi=150, bbox_inches='tight')
    plt.savefig(PLOTS_DIR / "h1_qwk_comparison.pdf", bbox_inches='tight')
    plt.close()
    print(f"Saved: h1_qwk_comparison.png")


def plot_h1_delta():
    """H1: Δκ with CIs."""
    summary = load_summary("h1")
    if not summary:
        return
    
    models = ["gpt-4o-mini", "gpt-4o", "gpt-5.1"]
    model_labels = ["4o-mini", "4o", "5.1"]
    comparisons = summary.get("comparisons", {})
    
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(models))
    
    deltas = []
    ci_lowers = []
    ci_uppers = []
    
    for model in models:
        comp = comparisons.get(model, {})
        bootstrap = comp.get("bootstrap", {})
        deltas.append(bootstrap.get("delta_hat", 0))
        ci_lowers.append(bootstrap.get("ci_lower", 0))
        ci_uppers.append(bootstrap.get("ci_upper", 0))
    
    errors = [
        [d - l for d, l in zip(deltas, ci_lowers)],
        [u - d for d, u in zip(deltas, ci_uppers)],
    ]
    
    bars = ax.bar(x, deltas, width=0.6, color=COLORS['with_tools'], edgecolor='white', linewidth=1)
    ax.errorbar(x, deltas, yerr=errors, fmt='none', color='black', capsize=5, capthick=2)
    
    ax.axhline(y=0.25, color='green', linestyle='--', alpha=0.7, label='Threshold (Δκ ≥ 0.25)')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    
    for bar, d, l, u in zip(bars, deltas, ci_lowers, ci_uppers):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            u + 0.03,
            f'{d:.3f}\n[{l:.3f}, {u:.3f}]',
            ha='center',
            va='bottom',
            fontsize=8,
        )
    
    ax.set_ylabel('Δκ (Tools - No Tools)')
    ax.set_xlabel('Model')
    ax.set_title('H1: Effect of Tools (with 95% CI)')
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.set_ylim(-0.35, 0.7)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "h1_delta_kappa.png", dpi=150, bbox_inches='tight')
    plt.savefig(PLOTS_DIR / "h1_delta_kappa.pdf", bbox_inches='tight')
    plt.close()
    print(f"Saved: h1_delta_kappa.png")


def plot_h2_qwk():
    """H2: QWK by router."""
    summary = load_summary("h2")
    if not summary:
        print("H2 summary not found, skipping H2 plots")
        return
    
    per_run = summary.get("per_run", {})
    routers = ["no-policy", "policy", "oracle"]
    router_labels = ["No-Policy", "Policy", "Oracle"]
    colors = [COLORS['no_policy'], COLORS['policy'], COLORS['oracle']]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(routers))
    
    qwks = []
    for router in routers:
        run_key = f"h2_gpt-4o_{router}"
        qwks.append(per_run.get(run_key, {}).get("kappa", 0))
    
    bars = ax.bar(x, qwks, color=colors, edgecolor='white', linewidth=1)
    
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.02,
            f'{bar.get_height():.3f}',
            ha='center',
            va='bottom',
            fontsize=10,
        )
    
    ax.set_ylabel('Quadratic Weighted Kappa (QWK)')
    ax.set_xlabel('Router Configuration')
    ax.set_title('H2: Routing Quality (gpt-4o)')
    ax.set_xticks(x)
    ax.set_xticklabels(router_labels)
    ax.set_ylim(0, 0.7)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "h2_qwk_routing.png", dpi=150, bbox_inches='tight')
    plt.savefig(PLOTS_DIR / "h2_qwk_routing.pdf", bbox_inches='tight')
    plt.close()
    print(f"Saved: h2_qwk_routing.png")


def plot_h2_miss_rate():
    """H2: Tool miss rates."""
    summary = load_summary("h2")
    if not summary:
        return
    
    per_run = summary.get("per_run", {})
    routers = ["no-policy", "policy", "oracle"]
    router_labels = ["No-Policy", "Policy", "Oracle"]
    colors = [COLORS['no_policy'], COLORS['policy'], COLORS['oracle']]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(routers))
    
    miss_rates = []
    for router in routers:
        run_key = f"h2_gpt-4o_{router}"
        miss_rates.append(per_run.get(run_key, {}).get("miss_rate", 0) * 100)
    
    bars = ax.bar(x, miss_rates, color=colors, edgecolor='white', linewidth=1)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.5,
            f'{height:.1f}%',
            ha='center',
            va='bottom',
            fontsize=10,
        )
    
    ax.axhline(y=5, color='red', linestyle='--', alpha=0.7, label='5% threshold')
    
    ax.set_ylabel('Tool Miss Rate (%)')
    ax.set_xlabel('Router Configuration')
    ax.set_title('H2: Tool Miss Rates')
    ax.set_xticks(x)
    ax.set_xticklabels(router_labels)
    ax.set_ylim(0, 15)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "h2_miss_rate.png", dpi=150, bbox_inches='tight')
    plt.savefig(PLOTS_DIR / "h2_miss_rate.pdf", bbox_inches='tight')
    plt.close()
    print(f"Saved: h2_miss_rate.png")


def plot_h3_qwk_by_type():
    """H3b: QWK by claim type."""
    summary = load_summary("h3")
    if not summary:
        print("H3 summary not found, skipping H3 plots")
        return
    
    crisp = summary.get("crisp", {})
    ambiguous = summary.get("ambiguous", {})
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    groups = ["Precise\n(n=15)", "Vague\n(n=22)"]
    qwks = [crisp.get("kappa", 0), ambiguous.get("kappa", 0)]
    colors = [COLORS['crisp'], COLORS['ambiguous']]
    
    bars = ax.bar(groups, qwks, color=colors, edgecolor='white', linewidth=1)
    
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.02,
            f'{bar.get_height():.3f}',
            ha='center',
            va='bottom',
            fontsize=11,
        )
    
    ax.set_ylabel('Quadratic Weighted Kappa (QWK)')
    ax.set_title('H3b: Agreement by Claim Type')
    ax.set_ylim(0, 0.8)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "h3b_qwk_by_type.png", dpi=150, bbox_inches='tight')
    plt.savefig(PLOTS_DIR / "h3b_qwk_by_type.pdf", bbox_inches='tight')
    plt.close()
    print(f"Saved: h3b_qwk_by_type.png")


def plot_h3_flip_rates():
    """H3a: Flip rates for ambiguous claims."""
    summary = load_summary("h3")
    if not summary:
        return
    
    ambiguous = summary.get("ambiguous", {})
    fr = ambiguous.get("FR", 0)
    qwfr = ambiguous.get("QWFR", 0)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    metrics = ["Flip Rate\n(FR)", "Quad-Weighted\nFlip Rate (QWFR)"]
    values = [fr, qwfr]
    thresholds = [0.20, 0.05]
    bar_colors = [COLORS['fr'], COLORS['qwfr']]  # Different colors
    
    bars = ax.bar(metrics, values, color=bar_colors, edgecolor='white', linewidth=1)
    
    for bar, thresh in zip(bars, thresholds):
        # Value label
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f'{bar.get_height():.3f}',
            ha='center',
            va='bottom',
            fontsize=11,
        )
        # Threshold line - use dark blue to contrast with orange bars
        ax.hlines(
            thresh,
            bar.get_x(),
            bar.get_x() + bar.get_width(),
            colors='#1a1a2e',
            linestyles='--',
            linewidth=2,
        )
        # Threshold value label
        ax.text(
            bar.get_x() + bar.get_width() + 0.05,
            thresh,
            f'{thresh:.2f}',
            ha='left',
            va='center',
            fontsize=9,
            color='#1a1a2e',
        )
    
    ax.set_ylabel('Rate')
    ax.set_title('H3a: Flip Rates (Vague Claims)')
    ax.set_ylim(0, 0.6)
    ax.grid(axis='y', alpha=0.3)
    
    ax.plot([], [], '--', color='#1a1a2e', label='Threshold')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "h3a_flip_rates.png", dpi=150, bbox_inches='tight')
    plt.savefig(PLOTS_DIR / "h3a_flip_rates.pdf", bbox_inches='tight')
    plt.close()
    print(f"Saved: h3a_flip_rates.png")


def plot_h3_qwk_range():
    """H3: QWK range across interpretations."""
    summary = load_summary("h3")
    if not summary:
        return
    
    ambiguous = summary.get("ambiguous", {})
    kappa_best = ambiguous.get("kappa_best", 0)
    kappa_worst = ambiguous.get("kappa_worst", 0)
    kappa_actual = ambiguous.get("kappa", 0)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Horizontal bar showing range
    ax.barh(
        0,
        kappa_best - kappa_worst,
        left=kappa_worst,
        height=0.4,
        color=COLORS['ci_band'],
        edgecolor='gray',
        linewidth=1,
    )
    ax.plot(kappa_actual, 0, 'o', markersize=15, color=COLORS['ambiguous'], zorder=5)
    ax.plot(kappa_worst, 0, '|', markersize=20, color='red', mew=3)
    ax.plot(kappa_best, 0, '|', markersize=20, color='green', mew=3)
    
    ax.set_xlim(-0.1, 1.0)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel('Quadratic Weighted Kappa (QWK)')
    ax.set_title('H3: QWK Range Across Interpretations')
    ax.set_yticks([])
    ax.grid(axis='x', alpha=0.3)
    
    # Create legend with vertical spacing using custom handler
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['ambiguous'], 
               markersize=12, label=f'Actual κ = {kappa_actual:.3f}'),
        Line2D([0], [0], marker='|', color='red', markersize=15, mew=3,
               linestyle='None', label=f'Worst = {kappa_worst:.3f}'),
        Line2D([0], [0], marker='|', color='green', markersize=15, mew=3,
               linestyle='None', label=f'Best = {kappa_best:.3f}'),
        mpatches.Patch(facecolor=COLORS['ci_band'], edgecolor='gray', label='Possible range'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9, 
              labelspacing=1.0, handletextpad=1.0)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "h3_qwk_range.png", dpi=150, bbox_inches='tight')
    plt.savefig(PLOTS_DIR / "h3_qwk_range.pdf", bbox_inches='tight')
    plt.close()
    print(f"Saved: h3_qwk_range.png")


def plot_h4_multi_call():
    """H4: Multi-call vs single-call comparison."""
    summary = load_summary("h4")
    if not summary:
        print("H4 summary not found, skipping H4 plots")
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    multi = summary.get("multi_call", {})
    single = summary.get("single_call", {})
    comparison = summary.get("comparison", {})
    
    configs = ["Single-Call", "Multi-Call"]
    qwks = [single.get("kappa", 0), multi.get("kappa", 0)]
    colors = [COLORS['single_call'], COLORS['multi_call']]
    
    x = np.arange(len(configs))
    bars = ax.bar(x, qwks, color=colors, edgecolor='white', linewidth=1, width=0.5)
    
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.02,
            f'{bar.get_height():.3f}',
            ha='center',
            va='bottom',
            fontsize=11,
        )
    
    # comparison has delta_hat, ci_lower, ci_upper directly (not nested under bootstrap)
    delta = comparison.get("delta_hat", 0)
    ci_lower = comparison.get("ci_lower", 0)
    ci_upper = comparison.get("ci_upper", 0)
    
    # Simple diagonal arrow from single-call to multi-call
    single_height = qwks[0]
    multi_height = qwks[1]
    
    ax.annotate(
        '',
        xy=(1, multi_height),  # Arrow head at multi-call bar top
        xytext=(0, single_height),  # Arrow tail at single-call bar top
        arrowprops=dict(
            arrowstyle='->', 
            color='gray', 
            lw=1.5,
        ),
    )
    
    # Text label for the delta (above the arrow)
    ax.text(
        0.5, 0.58,
        f'Δκ = {delta:.3f}\n95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]',
        ha='center',
        va='bottom',
        fontsize=9,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
    )
    
    ax.set_ylabel('Quadratic Weighted Kappa (QWK)')
    ax.set_xlabel('Tool Call Configuration')
    ax.set_title('H4: Multi-call vs Single-call (gpt-4o)')
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylim(0, 0.7)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "h4_multi_call.png", dpi=150, bbox_inches='tight')
    plt.savefig(PLOTS_DIR / "h4_multi_call.pdf", bbox_inches='tight')
    plt.close()
    print(f"Saved: h4_multi_call.png")


def plot_h5_qwk():
    """H5: QWK by model size with/without tools."""
    summary = load_summary("h5")
    if not summary:
        print("H5 summary not found, skipping H5 plots")
        return
    
    per_run = summary.get("per_run", {})
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    models = ["gpt-4o-mini", "gpt-5.1"]
    model_labels = ["4o-mini\n(small)", "5.1\n(large)"]
    x = np.arange(len(models))
    width = 0.35
    
    qwk_tools = []
    qwk_no_tools = []
    
    for model in models:
        # H5 uses H1 runs, so keys are h1_* with _policy suffix
        tools_key = f"h1_{model}_with-tools_policy"
        no_tools_key = f"h1_{model}_no-tools_policy"
        qwk_tools.append(per_run.get(tools_key, {}).get("kappa", 0))
        qwk_no_tools.append(per_run.get(no_tools_key, {}).get("kappa", 0))
    
    bars1 = ax.bar(
        x - width/2, qwk_no_tools, width,
        label='No Tools', color=COLORS['no_tools'], edgecolor='white', linewidth=1,
    )
    bars2 = ax.bar(
        x + width/2, qwk_tools, width,
        label='With Tools', color=COLORS['with_tools'], edgecolor='white', linewidth=1,
    )
    
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Quadratic Weighted Kappa (QWK)')
    ax.set_xlabel('Model Size')
    ax.set_title('H5: Model Size Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.set_ylim(0, 0.7)
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "h5_qwk_model_size.png", dpi=150, bbox_inches='tight')
    plt.savefig(PLOTS_DIR / "h5_qwk_model_size.pdf", bbox_inches='tight')
    plt.close()
    print(f"Saved: h5_qwk_model_size.png")


def plot_h5_gap():
    """H5: Gap shrinkage with tools."""
    summary = load_summary("h5")
    if not summary:
        return
    
    # Gap data is under "analysis" key
    analysis = summary.get("analysis", {})
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Direct keys, not nested
    gap_no_tools = analysis.get("gap_no_tools", 0)
    gap_with_tools = analysis.get("gap_tools", 0)
    shrinkage_val = analysis.get("delta_gap", 0)
    
    configs = ["No Tools", "With Tools"]
    gaps = [gap_no_tools, gap_with_tools]
    colors = [COLORS['no_tools'], COLORS['with_tools']]
    
    bars = ax.bar(configs, gaps, color=colors, edgecolor='white', linewidth=1, width=0.5)
    
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f'{bar.get_height():.3f}',
            ha='center',
            va='bottom',
            fontsize=11,
        )
    
    if gap_no_tools:
        # Curved arrow from no-tools to with-tools
        ax.annotate(
            '',
            xy=(1, gap_with_tools),  # Arrow head at with-tools bar top
            xytext=(0, gap_no_tools),  # Arrow tail at no-tools bar top
            arrowprops=dict(
                arrowstyle='->',
                color='gray',
                lw=1.5,
                connectionstyle='arc3,rad=-0.3',
            ),
        )
        
        # Text label along the curve - clarify it's kappa gap
        ax.text(
            0.5, 0.18,
            f'Δκ gap\n{((shrinkage_val/gap_no_tools)*100):.0f}% smaller',
            ha='center',
            va='center',
            fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
        )
    
    ax.set_ylabel('Performance Gap (Large - Small Model)')
    ax.set_title('H5: Gap Between Model Sizes')
    ax.set_ylim(0, 0.35)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "h5_gap_shrinkage.png", dpi=150, bbox_inches='tight')
    plt.savefig(PLOTS_DIR / "h5_gap_shrinkage.pdf", bbox_inches='tight')
    plt.close()
    print(f"Saved: h5_gap_shrinkage.png")


def plot_summary_overview():
    """Create an overview summary plot showing all hypothesis verdicts."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    hypotheses = [
        ("H1a/b", "Tools improve agreement", "NOT SUPPORTED\n(CIs overlap 0)"),
        ("H2a", "Policy routing helps QWK", "NOT SUPPORTED\n(CI overlaps 0)"),
        ("H2b", "Miss rate < 5%", "SUPPORTED\n(0% miss rate)"),
        ("H2c", "Oracle headroom", "INCONCLUSIVE\n(CI overlaps 0)"),
        ("H3a", "Ambiguity causes flips", "SUPPORTED\n(FR ≥ 0.20, QWFR ≥ 0.05)"),
        ("H3b", "Ambiguity lowers QWK", "NOT SUPPORTED\n(wide CI)"),
        ("H4", "Multi-call helps", "NOT SUPPORTED\n(small effect)"),
        ("H5a/b/c", "Tools compress model gap", "NOT SUPPORTED\n(CIs overlap 0)"),
    ]
    
    y_pos = np.arange(len(hypotheses))
    colors = []
    for _, _, verdict in hypotheses:
        if verdict.startswith("SUPPORTED"):
            colors.append('#2ecc71')
        elif verdict.startswith("NOT SUPPORTED"):
            colors.append('#e74c3c')
        else:
            colors.append('#f39c12')
    
    bars = ax.barh(y_pos, [1]*len(hypotheses), color=colors, edgecolor='white', alpha=0.7)
    
    # Push labels away from the left edge
    for i, (h_id, description, verdict) in enumerate(hypotheses):
        ax.text(0.03, i, f"{h_id}: {description}", va='center', ha='left', fontsize=10, fontweight='bold')
        ax.text(0.97, i, verdict, va='center', ha='right', fontsize=9)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(hypotheses) - 0.5)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.invert_yaxis()
    ax.set_title('Summary: Hypothesis Verdicts on Drop-4 (N=37)', fontsize=14, fontweight='bold')
    
    # Remove left spine to avoid overlap
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "summary_overview.png", dpi=150, bbox_inches='tight')
    plt.savefig(PLOTS_DIR / "summary_overview.pdf", bbox_inches='tight')
    plt.close()
    print(f"Saved: summary_overview.png")


def plot_h3_flip_example_alzn():
    """H3: Al-Zn eutectic flip example - "~" approximation tolerance."""
    fig, ax = plt.subplots(figsize=(7, 5))
    
    claim_title = 'Al-Zn Eutectic: "~Al15Zn75" tolerance'
    mappings = ['±1%\n(tight)', '±5%\n(moderate)', '±10%\n(loose)']
    predictions = [-2, 2, 2]  # From actual run data
    gold = 2
    
    x = np.arange(len(mappings))
    colors = ['#2ecc71' if p > 0 else '#e74c3c' if p < 0 else '#95a5a6' for p in predictions]
    
    # Draw lollipop chart
    for i, p in enumerate(predictions):
        ax.plot([i, i], [0, p], color=colors[i], linewidth=3, zorder=1)
    ax.scatter(x, predictions, c=colors, s=200, zorder=2, edgecolors='white', linewidths=2)
    
    # Value labels
    for i, p in enumerate(predictions):
        label = {2: '+2', 1: '+1', 0: '0', -1: '-1', -2: '-2'}[p]
        offset = 0.3 if p >= 0 else -0.3
        ax.text(i, p + offset, label, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Gold label line
    ax.axhline(y=gold, color='gold', linestyle='--', linewidth=2, label=f'Gold label (+{gold})')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    ax.set_xticks(x)
    ax.set_xticklabels(mappings, fontsize=10)
    ax.set_ylabel('Predicted Label')
    ax.set_xlabel('Interpretation of "~" (approximately)')
    ax.set_ylim(-2.8, 2.8)
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_yticklabels(['-2\nInfeasible', '-1', '0', '+1', '+2\nFeasible'])
    ax.set_title(claim_title, fontsize=11)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "h3_flip_alzn.png", dpi=150, bbox_inches='tight')
    plt.savefig(PLOTS_DIR / "h3_flip_alzn.pdf", bbox_inches='tight')
    plt.close()
    print(f"Saved: h3_flip_alzn.png")


def plot_h3_flip_example_fe2o3():
    """H3: Fe2O3 Al doping flip example - "stronger magnet" definition."""
    fig, ax = plt.subplots(figsize=(7, 5))
    
    claim_title = 'Fe₂O₃ Al Doping: "stronger magnet" definition'
    mappings = ['Higher\nMs', 'Higher\nCoercivity', 'Higher\nPull Force', 'Higher\nRemanence']
    predictions = [-1, 1, -2, -2]  # From actual run data
    gold = 2
    
    x = np.arange(len(mappings))
    colors = ['#2ecc71' if p > 0 else '#e74c3c' if p < 0 else '#95a5a6' for p in predictions]
    
    # Draw lollipop chart
    for i, p in enumerate(predictions):
        ax.plot([i, i], [0, p], color=colors[i], linewidth=3, zorder=1)
    ax.scatter(x, predictions, c=colors, s=200, zorder=2, edgecolors='white', linewidths=2)
    
    # Value labels
    for i, p in enumerate(predictions):
        label = {2: '+2', 1: '+1', 0: '0', -1: '-1', -2: '-2'}[p]
        offset = 0.3 if p >= 0 else -0.3
        ax.text(i, p + offset, label, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Gold label line
    ax.axhline(y=gold, color='gold', linestyle='--', linewidth=2, label=f'Gold label (+{gold})')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    ax.set_xticks(x)
    ax.set_xticklabels(mappings, fontsize=9)
    ax.set_ylabel('Predicted Label')
    ax.set_xlabel('Interpretation of "stronger magnet"')
    ax.set_ylim(-2.8, 2.8)
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_yticklabels(['-2\nInfeasible', '-1', '0', '+1', '+2\nFeasible'])
    ax.set_title(claim_title, fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "h3_flip_fe2o3.png", dpi=150, bbox_inches='tight')
    plt.savefig(PLOTS_DIR / "h3_flip_fe2o3.pdf", bbox_inches='tight')
    plt.close()
    print(f"Saved: h3_flip_fe2o3.png")


def main():
    """Generate all hypothesis plots."""
    print(f"Generating plots in {PLOTS_DIR}...")
    print()
    
    # H1: Two separate plots
    plot_h1_qwk()
    plot_h1_delta()
    
    # H2: Two separate plots
    plot_h2_qwk()
    plot_h2_miss_rate()
    
    # H3: Five separate plots
    plot_h3_qwk_by_type()
    plot_h3_flip_rates()
    plot_h3_qwk_range()
    plot_h3_flip_example_alzn()
    plot_h3_flip_example_fe2o3()
    
    # H4: One plot
    plot_h4_multi_call()
    
    # H5: Two separate plots
    plot_h5_qwk()
    plot_h5_gap()
    
    # Summary
    plot_summary_overview()
    
    print()
    print(f"Done! All plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
