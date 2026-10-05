#!/usr/bin/env python3
"""
H1: Tools improve agreement

Compute metrics script to analyze runs and compute bootstrap CIs.

Usage:
    python compute_metrics_h1.py
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.logging_utils import (
    load_run_json,
    extract_labels_from_run,
    compute_run_stats,
    save_summary_json,
)
from common.metrics import (
    compute_qwk,
    compute_accuracy,
    compute_mae,
    bootstrap_kappa_diff_ci,
)


RUNS_DIR = Path(__file__).parent / "runs"
SUMMARY_OUTPUT = Path(__file__).parent / "summary_h1_metrics.json"

# Expected run configurations
MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-5.1"]
TOOLS_CONDITIONS = ["with-tools", "no-tools"]


def get_run_path(model: str, tools_condition: str) -> Path:
    """Get path to run JSON file."""
    return RUNS_DIR / f"h1_{model}_{tools_condition}_policy.json"


def compute_h1_metrics():
    """Compute all H1 metrics and save summary."""
    
    summary = {
        "per_run": {},
        "comparisons": {},
    }
    
    # Load all runs and compute per-run metrics
    run_data = {}
    
    for model in MODELS:
        for tools_cond in TOOLS_CONDITIONS:
            run_id = f"h1_{model}_{tools_cond}_policy"
            run_path = get_run_path(model, tools_cond)
            
            if not run_path.exists():
                print(f"Warning: Run not found: {run_path}")
                continue
            
            print(f"Loading {run_id}...")
            data = load_run_json(run_path)
            run_data[run_id] = data
            
            # Extract labels
            gold_labels, pred_labels = extract_labels_from_run(data)
            
            # Compute metrics
            kappa = compute_qwk(gold_labels, pred_labels)
            accuracy = compute_accuracy(gold_labels, pred_labels)
            mae = compute_mae(gold_labels, pred_labels)
            
            # Compute run stats
            stats = compute_run_stats(data)
            
            summary["per_run"][run_id] = {
                "kappa": kappa,
                "accuracy": accuracy,
                "mae": mae,
                "n_examples": len(gold_labels),
                "avg_latency_ms": stats.get("avg_total_latency_ms", 0),
                "avg_tool_calls": stats.get("avg_tool_calls_per_example", 0),
            }
            
            print(f"  κ={kappa:.3f}, Acc={accuracy:.3f}, MAE={mae:.3f}")
    
    # Compute comparisons between tools vs no-tools for each model
    for model in MODELS:
        run_id_no_tools = f"h1_{model}_no-tools_policy"
        run_id_tools = f"h1_{model}_with-tools_policy"
        
        if run_id_no_tools not in run_data or run_id_tools not in run_data:
            print(f"Skipping comparison for {model}: missing runs")
            continue
        
        print(f"\nComparing {model}: tools vs no-tools...")
        
        # Extract labels from both runs
        gold_no, pred_no = extract_labels_from_run(run_data[run_id_no_tools])
        gold_tools, pred_tools = extract_labels_from_run(run_data[run_id_tools])
        
        # Verify gold labels match
        assert gold_no == gold_tools, "Gold labels don't match between runs!"
        
        # Compute bootstrap CI for Δκ
        bootstrap_result = bootstrap_kappa_diff_ci(
            y_true=gold_no,
            y_pred_a=pred_no,  # System A = no-tools
            y_pred_b=pred_tools,  # System B = with-tools
            n_boot=10_000,
            ci=0.95,
        )
        
        # Determine if hypothesis is supported
        # Support if: CI lower bound > 0 AND delta_hat >= 0.25 (for 4o-mini and 4o)
        # For 5.1: CI lower bound > 0 is sufficient (no minimum effect size)
        delta_hat = bootstrap_result["delta_hat"]
        ci_lower = bootstrap_result["ci_lower"]
        ci_upper = bootstrap_result["ci_upper"]
        
        # Condition checks per hypothesis spec
        ci_positive = ci_lower > 0
        threshold_met = delta_hat >= 0.25
        
        if model == "gpt-5.1":
            # For largest model: only require CI lower > 0
            hypothesis_supported = ci_positive
            conditions = {
                "ci_positive": {
                    "criterion": "95% CI lower bound > 0",
                    "value": ci_lower,
                    "passed": ci_positive,
                },
            }
        else:
            # For smaller models: require both conditions
            hypothesis_supported = ci_positive and threshold_met
            conditions = {
                "ci_positive": {
                    "criterion": "95% CI lower bound > 0",
                    "value": ci_lower,
                    "passed": ci_positive,
                },
                "threshold_met": {
                    "criterion": "Δκ ≥ 0.25",
                    "value": delta_hat,
                    "threshold": 0.25,
                    "passed": threshold_met,
                },
            }
        
        summary["comparisons"][model] = {
            "baseline_run": run_id_no_tools,
            "tools_run": run_id_tools,
            "bootstrap": bootstrap_result,
            "conditions": conditions,
            "hypothesis_supported": hypothesis_supported,
            "interpretation": interpret_h1_result(delta_hat, ci_lower, model),
        }
        
        print(f"  Δκ = {delta_hat:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
        print(f"\n  Condition checks for {model}:")
        for cond_name, cond in conditions.items():
            status = "✓" if cond["passed"] else "✗"
            print(f"    [{status}] {cond['criterion']}: {cond['value']:.3f}")
        status = "✓ SUPPORTED" if hypothesis_supported else "✗ NOT SUPPORTED"
        print(f"  → {status}")
    
    # Overall H1 verdict
    all_supported = all(
        comp.get("hypothesis_supported", False)
        for comp in summary["comparisons"].values()
    )
    summary["overall_verdict"] = {
        "h1_supported": all_supported,
        "notes": "H1 is supported if tools improve QWK for all tested models",
    }
    
    # Save summary
    save_summary_json(SUMMARY_OUTPUT, summary)
    print(f"\nSaved summary to {SUMMARY_OUTPUT}")
    
    return summary


def interpret_h1_result(delta_hat: float, ci_lower: float, model: str) -> str:
    """Generate interpretation text for H1 result.
    
    Effect size heuristic for Drop-4 (N=37):
    - |Δκ| < 0.20: indistinguishable from noise
    - 0.20 ≤ |Δκ| < 0.35: borderline, requires careful CI inspection
    - |Δκ| ≥ 0.35: very likely a real effect
    """
    abs_delta = abs(delta_hat)
    
    if ci_lower > 0:
        # Statistically significant (CI excludes 0)
        if abs_delta >= 0.35:
            return f"Strong effect: Δκ={delta_hat:.2f} ≥ 0.35, very likely real improvement for {model}"
        elif abs_delta >= 0.20:
            return f"Borderline effect: 0.20 ≤ Δκ={delta_hat:.2f} < 0.35, but CI confirms significance for {model}"
        else:
            return f"Small effect: Δκ={delta_hat:.2f} < 0.20 (noise range), but CI excludes 0 for {model}"
    else:
        # Not statistically significant (CI includes 0)
        if abs_delta >= 0.35:
            return f"Large effect Δκ={delta_hat:.2f} but CI includes 0 - inconclusive, need more data for {model}"
        elif abs_delta >= 0.20:
            return f"Borderline effect: Δκ={delta_hat:.2f} in [0.20, 0.35), CI includes 0 - inconclusive for {model}"
        else:
            return f"No effect: Δκ={delta_hat:.2f} < 0.20, indistinguishable from noise for {model}"


def print_summary_table(summary: dict):
    """Print a formatted summary table."""
    
    print("\n" + "=" * 80)
    print("H1 SUMMARY: Tools vs No-Tools")
    print("=" * 80)
    
    # Per-run metrics
    print("\nPer-Run Metrics:")
    print("-" * 80)
    print(f"{'Run ID':<45} {'κ':>8} {'Acc':>8} {'MAE':>8} {'Tools':>8}")
    print("-" * 80)
    
    for run_id, metrics in summary.get("per_run", {}).items():
        print(f"{run_id:<45} {metrics['kappa']:>8.3f} {metrics['accuracy']:>8.3f} "
              f"{metrics['mae']:>8.3f} {metrics['avg_tool_calls']:>8.1f}")
    
    # Comparisons
    print("\n\nComparisons (tools - no-tools):")
    print("-" * 80)
    
    for model, comp in summary.get("comparisons", {}).items():
        bs = comp["bootstrap"]
        delta = bs["delta_hat"]
        ci_lower = bs["ci_lower"]
        ci_upper = bs["ci_upper"]
        
        print(f"\n{model}:")
        print(f"  Δκ = {delta:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
        
        # Show conditions
        conditions = comp.get("conditions", {})
        for cond_name, cond in conditions.items():
            status = "✓" if cond["passed"] else "✗"
            if "threshold" in cond:
                print(f"  [{status}] {cond['criterion']}: {cond['value']:.3f} (threshold: {cond['threshold']})")
            else:
                print(f"  [{status}] {cond['criterion']}: {cond['value']:.3f}")
        
        supported = "✓ SUPPORTED" if comp["hypothesis_supported"] else "✗ NOT SUPPORTED"
        print(f"  → {supported}")
    
    # Overall verdict
    print("\n" + "=" * 80)
    verdict = summary.get("overall_verdict", {})
    overall = "SUPPORTED" if verdict.get("h1_supported") else "NOT SUPPORTED"
    print(f"H1 Overall: {overall}")
    print("=" * 80)


if __name__ == "__main__":
    summary = compute_h1_metrics()
    print_summary_table(summary)

