#!/usr/bin/env python3
"""
H2: Better routing helps

Compute metrics script to analyze router configurations.

Usage:
    python compute_metrics_h2.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.dataset import load_metadata
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
    compute_tool_miss_rate,
    bootstrap_miss_rate_ci,
)


RUNS_DIR = Path(__file__).parent / "runs"
H1_RUNS_DIR = Path(__file__).parent.parent / "h1" / "runs"
SUMMARY_OUTPUT = Path(__file__).parent / "summary_h2_metrics.json"

MODEL = "gpt-4o"
# H2 compares: no-policy vs policy vs oracle
# Policy run is reused from H1 (h1_gpt-4o_with-tools_policy.json)
ROUTERS = ["no-policy", "policy", "oracle"]


def get_run_path(router: str) -> Path:
    """Get path to run JSON file.
    
    Note: 'policy' router uses H1's with-tools policy run (same configuration).
    """
    if router == "policy":
        # Reuse H1's policy run - same configuration as H2 policy
        return H1_RUNS_DIR / f"h1_{MODEL}_with-tools_policy.json"
    else:
    return RUNS_DIR / f"h2_{MODEL}_{router}.json"


def compute_h2_metrics():
    """Compute all H2 metrics and save summary."""
    
    metadata = load_metadata()
    
    summary = {
        "per_run": {},
        "comparisons": {},
    }
    
    # Load all runs and compute per-run metrics
    run_data = {}
    
    for router in ROUTERS:
        run_id = f"h2_{MODEL}_{router}"
        run_path = get_run_path(router)
        
        if not run_path.exists():
            print(f"Warning: Run not found: {run_path}")
            continue
        
        # Policy run is reused from H1
        source_note = " (from H1)" if router == "policy" else ""
        print(f"Loading {run_id}{source_note}...")
        data = load_run_json(run_path)
        run_data[run_id] = data
        
        # Extract labels
        gold_labels, pred_labels = extract_labels_from_run(data)
        
        # Compute metrics
        kappa = compute_qwk(gold_labels, pred_labels)
        accuracy = compute_accuracy(gold_labels, pred_labels)
        mae = compute_mae(gold_labels, pred_labels)
        
        # Compute tool miss rate
        examples = data.get("examples", [])
        miss_rate = compute_tool_miss_rate(examples, metadata)
        miss_rate_ci = bootstrap_miss_rate_ci(examples, metadata)
        
        # Compute run stats
        stats = compute_run_stats(data)
        
        summary["per_run"][run_id] = {
            "router": router,
            "source": "H1" if router == "policy" else "H2",
            "kappa": kappa,
            "accuracy": accuracy,
            "mae": mae,
            "n_examples": len(gold_labels),
            "miss_rate": miss_rate,
            "miss_rate_ci": miss_rate_ci,
            "avg_latency_ms": stats.get("avg_total_latency_ms", 0),
            "avg_tool_calls": stats.get("avg_tool_calls_per_example", 0),
        }
        
        print(f"  κ={kappa:.3f}, Acc={accuracy:.3f}, MissRate={miss_rate:.3f}")
    
    # Comparison 1: Policy vs No-Policy
    run_id_no_policy = f"h2_{MODEL}_no-policy"
    run_id_policy = f"h2_{MODEL}_policy"
    
    if run_id_no_policy in run_data and run_id_policy in run_data:
        print(f"\nComparing Policy vs No-Policy...")
        
        gold_no, pred_no = extract_labels_from_run(run_data[run_id_no_policy])
        gold_policy, pred_policy = extract_labels_from_run(run_data[run_id_policy])
        
        assert gold_no == gold_policy, "Gold labels don't match!"
        
        bootstrap_result = bootstrap_kappa_diff_ci(
            y_true=gold_no,
            y_pred_a=pred_no,  # System A = no-policy
            y_pred_b=pred_policy,  # System B = policy
            n_boot=10_000,
            ci=0.95,
        )
        
        # Individual condition checks per hypothesis spec
        policy_miss_ci = summary["per_run"][run_id_policy]["miss_rate_ci"]
        policy_miss_rate = summary["per_run"][run_id_policy]["miss_rate"]
        
        delta_hat = bootstrap_result["delta_hat"]
        ci_lower = bootstrap_result["ci_lower"]
        ci_upper = bootstrap_result["ci_upper"]
        
        # Condition 1: CI lower > 0
        ci_positive = ci_lower > 0
        # Condition 2: Δκ ≥ 0.25
        threshold_met = delta_hat >= 0.25
        # Condition 3: Miss rate upper CI < 5%
        miss_rate_ok = policy_miss_ci["ci_upper"] < 0.05
        
        hypothesis_supported = ci_positive and threshold_met and miss_rate_ok
        
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
            "miss_rate_ok": {
                "criterion": "Policy miss-rate upper CI < 5%",
                "value": policy_miss_ci["ci_upper"],
                "threshold": 0.05,
                "passed": miss_rate_ok,
            },
        }
        
        summary["comparisons"]["policy_vs_no_policy"] = {
            "baseline_run": run_id_no_policy,
            "comparison_run": run_id_policy,
            "bootstrap": bootstrap_result,
            "conditions": conditions,
            "hypothesis_supported": hypothesis_supported,
            "interpretation": interpret_policy_result(delta_hat, ci_lower, miss_rate_ok),
        }
        
        print(f"  Δκ = {delta_hat:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
        print(f"\n  Condition checks:")
        for cond_name, cond in conditions.items():
            status = "✓" if cond["passed"] else "✗"
            if "threshold" in cond:
                print(f"    [{status}] {cond['criterion']}: {cond['value']:.3f} (threshold: {cond['threshold']})")
            else:
                print(f"    [{status}] {cond['criterion']}: {cond['value']:.3f}")
        status = "✓ SUPPORTED" if hypothesis_supported else "✗ NOT SUPPORTED"
        print(f"  → Policy vs No-Policy: {status}")
    
    # Comparison 2: Oracle vs Policy
    run_id_oracle = f"h2_{MODEL}_oracle"
    
    if run_id_policy in run_data and run_id_oracle in run_data:
        print(f"\nComparing Oracle vs Policy...")
        
        gold_policy, pred_policy = extract_labels_from_run(run_data[run_id_policy])
        gold_oracle, pred_oracle = extract_labels_from_run(run_data[run_id_oracle])
        
        assert gold_policy == gold_oracle, "Gold labels don't match!"
        
        bootstrap_result = bootstrap_kappa_diff_ci(
            y_true=gold_policy,
            y_pred_a=pred_policy,  # System A = policy
            y_pred_b=pred_oracle,  # System B = oracle
            n_boot=10_000,
            ci=0.95,
        )
        
        # Condition: CI lower > 0 (oracle provides headroom)
        delta_hat = bootstrap_result["delta_hat"]
        ci_lower = bootstrap_result["ci_lower"]
        ci_upper = bootstrap_result["ci_upper"]
        has_headroom = ci_lower > 0
        
        conditions = {
            "ci_positive": {
                "criterion": "95% CI lower bound > 0 (headroom exists)",
                "value": ci_lower,
                "passed": has_headroom,
            },
        }
        
        # Additional info: is headroom small?
        headroom_small = delta_hat < 0.05
        
        summary["comparisons"]["oracle_vs_policy"] = {
            "baseline_run": run_id_policy,
            "comparison_run": run_id_oracle,
            "bootstrap": bootstrap_result,
            "conditions": conditions,
            "has_headroom": has_headroom,
            "headroom_small": headroom_small,
            "interpretation": interpret_oracle_result(delta_hat, ci_lower),
        }
        
        print(f"  Δκ = {delta_hat:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
        print(f"\n  Condition checks:")
        status = "✓" if has_headroom else "✗"
        print(f"    [{status}] 95% CI lower bound > 0: {ci_lower:.3f}")
        if headroom_small:
            print(f"    [!] Headroom is small (Δκ < 0.05) - routing may not be the bottleneck")
        headroom_status = "✓ YES" if has_headroom else "✗ NO"
        print(f"  → Oracle shows headroom: {headroom_status}")
    
    # Overall H2 verdict
    policy_supported = summary["comparisons"].get("policy_vs_no_policy", {}).get("hypothesis_supported", False)
    has_oracle_headroom = summary["comparisons"].get("oracle_vs_policy", {}).get("has_headroom", False)
    
    summary["overall_verdict"] = {
        "h2_policy_supported": policy_supported,
        "h2_oracle_headroom": has_oracle_headroom,
        "notes": "H2 tests whether policy router improves over no-policy, with oracle as upper bound",
    }
    
    # Save summary
    save_summary_json(SUMMARY_OUTPUT, summary)
    print(f"\nSaved summary to {SUMMARY_OUTPUT}")
    
    return summary


def interpret_policy_result(delta_hat: float, ci_lower: float, miss_rate_ok: bool) -> str:
    """Generate interpretation text for policy vs no-policy comparison.
    
    Effect size heuristic for Drop-4 (N=37):
    - |Δκ| < 0.20: indistinguishable from noise
    - 0.20 ≤ |Δκ| < 0.35: borderline, requires careful CI inspection
    - |Δκ| ≥ 0.35: very likely a real effect
    """
    if not miss_rate_ok:
        return "Policy router has too high miss rate (>5% upper CI)"
    
    abs_delta = abs(delta_hat)
    
    if ci_lower > 0:
        if abs_delta >= 0.35:
            return f"Strong effect: Δκ={delta_hat:.2f} ≥ 0.35, very likely real improvement from policy routing"
        elif abs_delta >= 0.20:
            return f"Borderline effect: 0.20 ≤ Δκ={delta_hat:.2f} < 0.35, but CI confirms significance"
        else:
            return f"Small effect: Δκ={delta_hat:.2f} < 0.20 (noise range), but CI excludes 0"
    else:
        if abs_delta >= 0.35:
            return f"Large effect Δκ={delta_hat:.2f} but CI includes 0 - inconclusive, need more data"
        elif abs_delta >= 0.20:
            return f"Borderline effect: Δκ={delta_hat:.2f} in [0.20, 0.35), CI includes 0 - inconclusive"
        else:
            return f"No effect: Δκ={delta_hat:.2f} < 0.20, indistinguishable from noise"


def interpret_oracle_result(delta_hat: float, ci_lower: float) -> str:
    """Generate interpretation text for oracle vs policy comparison.
    
    Effect size heuristic for Drop-4 (N=37):
    - |Δκ| < 0.20: indistinguishable from noise
    - 0.20 ≤ |Δκ| < 0.35: borderline, requires careful CI inspection
    - |Δκ| ≥ 0.35: very likely a real effect
    """
    abs_delta = abs(delta_hat)
    
    if ci_lower > 0:
        if abs_delta >= 0.35:
            return f"Large headroom: Δκ={delta_hat:.2f} ≥ 0.35, routing is a major bottleneck"
        elif abs_delta >= 0.20:
            return f"Moderate headroom: Δκ={delta_hat:.2f} in [0.20, 0.35), CI confirms significance"
        else:
            return f"Small headroom: Δκ={delta_hat:.2f} < 0.20, but CI excludes 0"
    else:
        if abs_delta < 0.20:
            return f"No headroom: Δκ={delta_hat:.2f} < 0.20, routing likely not the main bottleneck"
        elif abs_delta < 0.35:
            return f"Borderline headroom: Δκ={delta_hat:.2f} in [0.20, 0.35), but CI includes 0 - inconclusive"
        else:
            return f"Large headroom Δκ={delta_hat:.2f} but CI includes 0 - inconclusive, need more data"


def print_summary_table(summary: dict):
    """Print a formatted summary table."""
    
    print("\n" + "=" * 80)
    print("H2 SUMMARY: Routing Quality")
    print("=" * 80)
    
    # Per-run metrics
    print("\nPer-Run Metrics:")
    print("-" * 80)
    print(f"{'Router':<15} {'κ':>8} {'Acc':>8} {'MAE':>8} {'MissRate':>10} {'Tools':>8}")
    print("-" * 80)
    
    for run_id, metrics in summary.get("per_run", {}).items():
        print(f"{metrics['router']:<15} {metrics['kappa']:>8.3f} {metrics['accuracy']:>8.3f} "
              f"{metrics['mae']:>8.3f} {metrics['miss_rate']:>10.3f} {metrics['avg_tool_calls']:>8.1f}")
    
    # Comparisons
    print("\n\nComparisons:")
    print("-" * 80)
    
    for comp_name, comp in summary.get("comparisons", {}).items():
        bs = comp["bootstrap"]
        delta = bs["delta_hat"]
        ci_lower = bs["ci_lower"]
        ci_upper = bs["ci_upper"]
        
        print(f"\n{comp_name}:")
        print(f"  Δκ = {delta:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
        
        # Show conditions
        conditions = comp.get("conditions", {})
        for cond_name, cond in conditions.items():
            status = "✓" if cond["passed"] else "✗"
            if "threshold" in cond:
                print(f"  [{status}] {cond['criterion']}: {cond['value']:.3f} (threshold: {cond['threshold']})")
            else:
                print(f"  [{status}] {cond['criterion']}: {cond['value']:.3f}")
        
        if comp_name == "policy_vs_no_policy":
            supported = "✓ SUPPORTED" if comp.get("hypothesis_supported") else "✗ NOT SUPPORTED"
            print(f"  → {supported}")
        else:
            headroom = "✓ YES" if comp.get("has_headroom") else "✗ NO"
            print(f"  → Headroom exists: {headroom}")
        
        print(f"  Interpretation: {comp['interpretation']}")
    
    # Overall verdict
    print("\n" + "=" * 80)
    verdict = summary.get("overall_verdict", {})
    policy_ok = "YES" if verdict.get("h2_policy_supported") else "NO"
    headroom = "YES" if verdict.get("h2_oracle_headroom") else "NO"
    print(f"Policy vs No-Policy supported: {policy_ok}")
    print(f"Oracle shows headroom: {headroom}")
    print("=" * 80)


if __name__ == "__main__":
    summary = compute_h2_metrics()
    print_summary_table(summary)

