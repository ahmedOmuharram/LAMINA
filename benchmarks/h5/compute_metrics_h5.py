#!/usr/bin/env python3
"""
H5: With tools enabled, bigger models have diminishing returns

Compute metrics script to analyze gap shrinkage between small and large models.

Note: This hypothesis reuses H1 runs, no separate run script needed.

Usage:
    python compute_metrics_h5.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.logging_utils import (
    load_run_json,
    extract_labels_from_run,
    save_summary_json,
)
from common.metrics import (
    compute_qwk,
    bootstrap_gap_shrinkage_ci,
)


H1_RUNS_DIR = Path(__file__).parent.parent / "h1" / "runs"
SUMMARY_OUTPUT = Path(__file__).parent / "summary_h5_metrics.json"

# Models to compare (small vs large)
SMALL_MODEL = "gpt-4o-mini"
LARGE_MODEL = "gpt-5.1"


def get_h1_run_path(model: str, tools_enabled: bool) -> Path:
    """Get path to H1 run JSON file."""
    tools_str = "with-tools" if tools_enabled else "no-tools"
    return H1_RUNS_DIR / f"h1_{model}_{tools_str}_policy.json"


def compute_h5_metrics():
    """Compute all H5 metrics and save summary."""
    
    summary = {
        "per_run": {},
        "analysis": {},
    }
    
    # Load all required H1 runs
    required_runs = [
        (SMALL_MODEL, False),
        (SMALL_MODEL, True),
        (LARGE_MODEL, False),
        (LARGE_MODEL, True),
    ]
    
    run_data = {}
    
    for model, tools in required_runs:
        tools_str = "with-tools" if tools else "no-tools"
        run_id = f"h1_{model}_{tools_str}_policy"
        run_path = get_h1_run_path(model, tools)
        
        if not run_path.exists():
            print(f"Warning: Run not found: {run_path}")
            continue
        
        print(f"Loading {run_id}...")
        data = load_run_json(run_path)
        run_data[run_id] = data
        
        # Extract labels and compute QWK
        gold_labels, pred_labels = extract_labels_from_run(data)
        kappa = compute_qwk(gold_labels, pred_labels)
        
        summary["per_run"][run_id] = {
            "model": model,
            "tools_enabled": tools,
            "kappa": kappa,
            "n_examples": len(gold_labels),
        }
        
        print(f"  κ={kappa:.3f}")
    
    # Check we have all required runs
    required_ids = [
        f"h1_{SMALL_MODEL}_no-tools_policy",
        f"h1_{SMALL_MODEL}_with-tools_policy",
        f"h1_{LARGE_MODEL}_no-tools_policy",
        f"h1_{LARGE_MODEL}_with-tools_policy",
    ]
    
    if not all(rid in run_data for rid in required_ids):
        print("Error: Missing required H1 runs for H5 analysis")
        return summary
    
    # Extract predictions
    gold, pred_small_no = extract_labels_from_run(run_data[f"h1_{SMALL_MODEL}_no-tools_policy"])
    _, pred_large_no = extract_labels_from_run(run_data[f"h1_{LARGE_MODEL}_no-tools_policy"])
    _, pred_small_tools = extract_labels_from_run(run_data[f"h1_{SMALL_MODEL}_with-tools_policy"])
    _, pred_large_tools = extract_labels_from_run(run_data[f"h1_{LARGE_MODEL}_with-tools_policy"])
    
    # Compute gap shrinkage with bootstrap CIs
    print("\nComputing gap shrinkage...")
    
    gap_analysis = bootstrap_gap_shrinkage_ci(
        y_true=gold,
        y_pred_small_no_tools=pred_small_no,
        y_pred_large_no_tools=pred_large_no,
        y_pred_small_tools=pred_small_tools,
        y_pred_large_tools=pred_large_tools,
        n_boot=10_000,
        ci=0.95,
    )
    
    # Determine if hypothesis is supported
    # Support if:
    # 1. Gap_no_tools CI lower >= 0.25 (clear scale gap without tools)
    # 2. Gap_tools <= 0.10 AND CI includes 0 (tools equalize performance)
    # 3. Δ_gap CI lower > 0 (gap shrinks with tools)
    
    gap_no_tools = gap_analysis["gap_no_tools"]
    gap_no_tools_ci = gap_analysis["gap_no_tools_ci"]
    gap_tools = gap_analysis["gap_tools"]
    gap_tools_ci = gap_analysis["gap_tools_ci"]
    delta_gap = gap_analysis["delta_gap"]
    delta_gap_ci = gap_analysis["delta_gap_ci"]
    
    # Condition 1: Clear scale gap without tools
    condition_1_a = gap_no_tools_ci["lower"] >= 0.25
    condition_1 = condition_1_a
    
    # Condition 2: Gap closes with tools (point est <= 0.10 AND CI includes 0)
    condition_2_a = gap_tools <= 0.10
    condition_2_b = gap_tools_ci["lower"] <= 0 <= gap_tools_ci["upper"]
    condition_2 = condition_2_a and condition_2_b
    
    # Condition 3: Gap shrinkage is statistically significant
    condition_3 = delta_gap_ci["lower"] > 0
    
    hypothesis_supported = condition_1 and condition_2 and condition_3
    
    conditions = {
        "gap_no_tools_large": {
            "criterion": "Gap(no-tools) CI lower ≥ 0.25 (clear scale gap)",
            "value": gap_no_tools_ci["lower"],
            "threshold": 0.25,
            "passed": condition_1,
        },
        "gap_tools_small": {
            "criterion": "Gap(tools) ≤ 0.10",
            "value": gap_tools,
            "threshold": 0.10,
            "passed": condition_2_a,
        },
        "gap_tools_includes_zero": {
            "criterion": "Gap(tools) CI includes 0",
            "value": f"[{gap_tools_ci['lower']:.3f}, {gap_tools_ci['upper']:.3f}]",
            "passed": condition_2_b,
        },
        "gap_shrinks": {
            "criterion": "Δ_gap CI lower > 0 (gap shrinks)",
            "value": delta_gap_ci["lower"],
            "passed": condition_3,
        },
    }
    
    summary["analysis"] = {
        "small_model": SMALL_MODEL,
        "large_model": LARGE_MODEL,
        "gap_no_tools": gap_no_tools,
        "gap_no_tools_ci": gap_no_tools_ci,
        "gap_tools": gap_tools,
        "gap_tools_ci": gap_tools_ci,
        "delta_gap": delta_gap,
        "delta_gap_ci": delta_gap_ci,
        "conditions": conditions,
        "condition_1_met": condition_1,
        "condition_2_met": condition_2,
        "condition_3_met": condition_3,
        "hypothesis_supported": hypothesis_supported,
        "interpretation": interpret_h5_result(gap_analysis, condition_1, condition_2, condition_3),
    }
    
    print(f"\n  Gap (no tools) = {gap_no_tools:.3f} [{gap_no_tools_ci['lower']:.3f}, {gap_no_tools_ci['upper']:.3f}]")
    print(f"  Gap (tools)    = {gap_tools:.3f} [{gap_tools_ci['lower']:.3f}, {gap_tools_ci['upper']:.3f}]")
    print(f"  Δ_gap          = {delta_gap:.3f} [{delta_gap_ci['lower']:.3f}, {delta_gap_ci['upper']:.3f}]")
    
    print(f"\n  Condition checks:")
    for cond_name, cond in conditions.items():
        status = "✓" if cond["passed"] else "✗"
        if isinstance(cond["value"], str):
            print(f"    [{status}] {cond['criterion']}: {cond['value']}")
        elif "threshold" in cond:
            print(f"    [{status}] {cond['criterion']}: {cond['value']:.3f} (threshold: {cond['threshold']})")
        else:
            print(f"    [{status}] {cond['criterion']}: {cond['value']:.3f}")
    
    overall_status = "✓ SUPPORTED" if hypothesis_supported else "✗ NOT SUPPORTED"
    print(f"\n  → H5 Hypothesis: {overall_status}")
    
    # Overall verdict
    summary["overall_verdict"] = {
        "h5_supported": hypothesis_supported,
        "notes": "H5 tests whether tools narrow the performance gap between small and large models",
    }
    
    # Save summary
    save_summary_json(SUMMARY_OUTPUT, summary)
    print(f"\nSaved summary to {SUMMARY_OUTPUT}")
    
    return summary


def interpret_h5_result(
    gap_analysis: dict,
    condition_1: bool,
    condition_2: bool,
    condition_3: bool,
) -> str:
    """Generate interpretation text for H5 result.
    
    Effect size heuristic for Drop-4 (N=37):
    - |Δκ| < 0.20: indistinguishable from noise
    - 0.20 ≤ |Δκ| < 0.35: borderline, requires careful CI inspection
    - |Δκ| ≥ 0.35: very likely a real effect
    """
    parts = []
    
    gap_no = gap_analysis['gap_no_tools']
    gap_tools = gap_analysis['gap_tools']
    delta_gap = gap_analysis['delta_gap']
    
    # Gap without tools interpretation
    if gap_no >= 0.35:
        parts.append(f"Large scale gap without tools: Δκ={gap_no:.2f} ≥ 0.35 (very likely real)")
    elif gap_no >= 0.20:
        parts.append(f"Borderline scale gap without tools: Δκ={gap_no:.2f} in [0.20, 0.35)")
    else:
        parts.append(f"No clear scale gap without tools: Δκ={gap_no:.2f} < 0.20 (noise range)")
    
    # Gap with tools interpretation
    if condition_2:
        parts.append(f"Gap closed with tools: Δκ={gap_tools:.2f} ≤ 0.10, CI includes 0")
    elif gap_tools < 0.20:
        parts.append(f"Gap reduced to noise level: Δκ={gap_tools:.2f} < 0.20")
    elif gap_tools < gap_no:
        parts.append(f"Gap reduced but not eliminated: Δκ={gap_tools:.2f}")
    else:
        parts.append(f"Gap not reduced with tools: Δκ={gap_tools:.2f}")
    
    # Gap shrinkage interpretation
    if condition_3:
        if delta_gap >= 0.35:
            parts.append(f"Strong gap shrinkage: Δ={delta_gap:.2f} ≥ 0.35")
        elif delta_gap >= 0.20:
            parts.append(f"Borderline gap shrinkage: Δ={delta_gap:.2f}, CI confirms significance")
        else:
            parts.append(f"Small but significant gap shrinkage: Δ={delta_gap:.2f}")
    else:
        parts.append(f"Gap shrinkage not statistically significant: Δ={delta_gap:.2f}")
    
    return "; ".join(parts)


def print_summary_table(summary: dict):
    """Print a formatted summary table."""
    
    print("\n" + "=" * 80)
    print("H5 SUMMARY: Diminishing Returns with Tools")
    print("=" * 80)
    
    # Per-run QWK values
    print("\nPer-Run QWK:")
    print("-" * 80)
    print(f"{'Model':<15} {'No Tools':>12} {'With Tools':>12}")
    print("-" * 80)
    
    per_run = summary.get("per_run", {})
    
    for model in [SMALL_MODEL, LARGE_MODEL]:
        no_tools_id = f"h1_{model}_no-tools_policy"
        tools_id = f"h1_{model}_with-tools_policy"
        
        kappa_no = per_run.get(no_tools_id, {}).get("kappa", 0)
        kappa_tools = per_run.get(tools_id, {}).get("kappa", 0)
        
        print(f"{model:<15} {kappa_no:>12.3f} {kappa_tools:>12.3f}")
    
    # Gap analysis
    if "analysis" in summary:
        ana = summary["analysis"]
        
        print("\n\nGap Analysis:")
        print("-" * 80)
        
        print(f"Gap (no tools): {ana['gap_no_tools']:.3f} "
              f"[{ana['gap_no_tools_ci']['lower']:.3f}, {ana['gap_no_tools_ci']['upper']:.3f}]")
        print(f"Gap (tools):    {ana['gap_tools']:.3f} "
              f"[{ana['gap_tools_ci']['lower']:.3f}, {ana['gap_tools_ci']['upper']:.3f}]")
        print(f"Δ_gap:          {ana['delta_gap']:.3f} "
              f"[{ana['delta_gap_ci']['lower']:.3f}, {ana['delta_gap_ci']['upper']:.3f}]")
        
        print(f"\nCondition Checks:")
        conditions = ana.get("conditions", {})
        for cond_name, cond in conditions.items():
            status = "✓" if cond["passed"] else "✗"
            if isinstance(cond["value"], str):
                print(f"  [{status}] {cond['criterion']}: {cond['value']}")
            elif "threshold" in cond:
                print(f"  [{status}] {cond['criterion']}: {cond['value']:.3f} (threshold: {cond['threshold']})")
            else:
                print(f"  [{status}] {cond['criterion']}: {cond['value']:.3f}")
        
        print(f"\nInterpretation: {ana['interpretation']}")
    
    # Overall verdict
    print("\n" + "=" * 80)
    verdict = summary.get("overall_verdict", {})
    supported = "YES" if verdict.get("h5_supported") else "NO"
    print(f"H5 Hypothesis supported: {supported}")
    print("=" * 80)


if __name__ == "__main__":
    summary = compute_h5_metrics()
    print_summary_table(summary)

