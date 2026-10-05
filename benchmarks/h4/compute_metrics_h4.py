"""
H4 Metrics: Multiple tool calls help

Computes metrics for H4 hypothesis:
- Δκ = κ(multi-call) - κ(single-call)

Multi-call baseline is taken from H1 with-tools run.
Single-call is from H4 run.

Support if:
- 95% bootstrap CI for Δκ has lower bound > 0
- Δκ ≥ 0.25
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmarks.common.metrics import (
    compute_qwk,
    compute_accuracy,
    compute_mae,
    bootstrap_kappa_diff_ci,
)


RUNS_DIR = Path(__file__).parent / "runs"
H1_RUNS_DIR = Path(__file__).parent.parent / "h1" / "runs"


def load_h1_multi_call(model: str = "gpt-4o") -> Dict[str, Any]:
    """Load H1 with-tools run as multi-call baseline."""
    run_file = H1_RUNS_DIR / f"h1_{model}_with-tools_policy.json"
    
    if not run_file.exists():
        raise FileNotFoundError(f"H1 multi-call baseline not found: {run_file}")
    
    with open(run_file) as f:
        return json.load(f)


def load_h4_single_call(model: str = "gpt-4o") -> Dict[str, Any]:
    """Load H4 single-call run."""
    run_file = RUNS_DIR / f"h4_{model}_single-call.json"
    
    if not run_file.exists():
        raise FileNotFoundError(f"H4 single-call run not found: {run_file}")
    
    with open(run_file) as f:
        return json.load(f)


def compute_h4_metrics(model: str = "gpt-4o") -> Dict[str, Any]:
    """
    Compute all H4 metrics.
    
    Returns:
        Dictionary with:
        - multi_call: metrics for multi-call configuration (from H1)
        - single_call: metrics for single-call configuration (from H4)
        - comparison: Δκ and bootstrap CI
        - hypothesis_supported: bool
    """
    # Load runs
    try:
        multi_run = load_h1_multi_call(model)
        single_run = load_h4_single_call(model)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return {"error": str(e)}
    
    # Extract predictions
    multi_examples = multi_run["examples"]
    single_examples = single_run["examples"]
    
    # Get gold and predicted labels
    multi_gold = [ex["gold_label"] for ex in multi_examples]
    multi_pred = [ex.get("predicted_label", 0) for ex in multi_examples]
    
    single_gold = [ex["gold_label"] for ex in single_examples]
    single_pred = [ex.get("predicted_label", 0) for ex in single_examples]
    
    # Compute per-configuration metrics
    multi_kappa = compute_qwk(multi_gold, multi_pred)
    multi_acc = compute_accuracy(multi_gold, multi_pred)
    multi_mae = compute_mae(multi_gold, multi_pred)
    
    single_kappa = compute_qwk(single_gold, single_pred)
    single_acc = compute_accuracy(single_gold, single_pred)
    single_mae = compute_mae(single_gold, single_pred)
    
    print(f"Multi-call (H1):  κ={multi_kappa:.3f}, Acc={multi_acc:.3f}, MAE={multi_mae:.3f}")
    print(f"Single-call (H4): κ={single_kappa:.3f}, Acc={single_acc:.3f}, MAE={single_mae:.3f}")
    
    # Tool call statistics
    # H1 format uses "tool_calls" list
    multi_avg_tools = sum(len(ex.get("tool_calls", [])) for ex in multi_examples) / len(multi_examples)
    single_avg_tools = sum(ex.get("num_tool_calls", 0) for ex in single_examples) / len(single_examples)
    
    # For H1, count successful as those without error field in tool_calls
    multi_successful = 0
    for ex in multi_examples:
        for tc in ex.get("tool_calls", []):
            if tc.get("error") is None:
                multi_successful += 1
    multi_successful /= len(multi_examples)
    
    single_successful = sum(ex.get("successful_tool_calls", 0) for ex in single_examples) / len(single_examples)
    
    # Bootstrap comparison
    # Create claim-level predictions for bootstrap
    multi_by_claim = {ex["claim_id"]: (ex["gold_label"], ex.get("predicted_label", 0)) 
                     for ex in multi_examples}
    single_by_claim = {ex["claim_id"]: (ex["gold_label"], ex.get("predicted_label", 0)) 
                      for ex in single_examples}
    
    # Align claims
    common_claims = set(multi_by_claim.keys()) & set(single_by_claim.keys())
    
    if len(common_claims) < len(multi_examples):
        print(f"Warning: Only {len(common_claims)} claims in common between H1 and H4 runs")
    
    multi_pairs = [(multi_by_claim[c][0], multi_by_claim[c][1]) for c in common_claims]
    single_pairs = [(single_by_claim[c][0], single_by_claim[c][1]) for c in common_claims]
    
    # Bootstrap: Δκ = κ(multi-call) - κ(single-call)
    bootstrap_result = bootstrap_kappa_diff_ci(
        y_true=[p[0] for p in multi_pairs],
        y_pred_a=[p[1] for p in single_pairs],  # single-call (baseline)
        y_pred_b=[p[1] for p in multi_pairs],   # multi-call (treatment)
        n_boot=10000,
        ci=0.95,
    )
    
    # Determine support
    delta_hat = bootstrap_result["delta_hat"]
    ci_lower = bootstrap_result["ci_lower"]
    ci_upper = bootstrap_result["ci_upper"]
    
    # Support if: CI lower > 0 AND Δκ ≥ 0.25
    ci_positive = ci_lower > 0
    delta_sufficient = delta_hat >= 0.25
    hypothesis_supported = ci_positive and delta_sufficient
    
    print(f"\nComparing Multi-call vs Single-call...")
    print(f"  Δκ = {delta_hat:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
    print(f"\n  Condition checks:")
    print(f"    [{'✓' if ci_positive else '✗'}] 95% CI lower bound > 0: {ci_lower:.3f}")
    print(f"    [{'✓' if delta_sufficient else '✗'}] Δκ ≥ 0.25: {delta_hat:.3f}")
    status = "✓ SUPPORTED" if hypothesis_supported else "✗ NOT SUPPORTED"
    print(f"  → H4 Hypothesis: {status}")
    
    summary = {
        "model": model,
        "multi_call": {
            "source": "H1 with-tools run",
            "kappa": multi_kappa,
            "accuracy": multi_acc,
            "mae": multi_mae,
            "n_claims": len(multi_examples),
            "avg_tool_calls": multi_avg_tools,
            "avg_successful_calls": multi_successful,
        },
        "single_call": {
            "source": "H4 single-call run",
            "kappa": single_kappa,
            "accuracy": single_acc,
            "mae": single_mae,
            "n_claims": len(single_examples),
            "avg_tool_calls": single_avg_tools,
            "avg_successful_calls": single_successful,
        },
        "comparison": {
            "delta_hat": delta_hat,
            "ci_lower": ci_lower,
            "ci_upper": bootstrap_result["ci_upper"],
            "std_error": bootstrap_result["std_error_delta_hat"],
        },
        "conditions": {
            "ci_lower_positive": ci_positive,
            "delta_sufficient": delta_sufficient,
        },
        "hypothesis_supported": hypothesis_supported,
        "interpretation": interpret_h4_result(delta_hat, ci_lower),
    }
    
    # Save summary
    summary_path = Path(__file__).parent / "summary_h4_metrics.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSaved summary to {summary_path}")
    
    return summary


def interpret_h4_result(delta_hat: float, ci_lower: float) -> str:
    """Generate interpretation text for H4 result.
    
    Effect size heuristic for Drop-4 (N=37):
    - |Δκ| < 0.20: indistinguishable from noise
    - 0.20 ≤ |Δκ| < 0.35: borderline, requires careful CI inspection
    - |Δκ| ≥ 0.35: very likely a real effect
    """
    abs_delta = abs(delta_hat)
    
    if ci_lower > 0:
        if abs_delta >= 0.35:
            return f"Strong effect: Δκ={delta_hat:.2f} ≥ 0.35, multi-call very likely helps"
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
            return f"No effect: Δκ={delta_hat:.2f} < 0.20, single call likely sufficient"


def print_h4_summary(summary: Dict[str, Any]):
    """Print formatted H4 summary."""
    if "error" in summary:
        print(f"Error: {summary['error']}")
        return
    
    print("\n" + "="*70)
    print("H4 SUMMARY: Multiple Tool Calls")
    print("="*70)
    
    print("\nPer-Configuration Metrics:")
    print("-"*70)
    print(f"{'Configuration':<15} {'κ':<8} {'Acc':<8} {'MAE':<8} {'N':<6} {'Avg Tools':<12} {'Avg Success':<12}")
    print("-"*70)
    
    multi = summary["multi_call"]
    single = summary["single_call"]
    
    print(f"{'Multi-call':<15} {multi['kappa']:.3f}    {multi['accuracy']:.3f}    {multi['mae']:.3f}    {multi['n_claims']:<6} {multi['avg_tool_calls']:.2f}         {multi['avg_successful_calls']:.2f}")
    print(f"{'Single-call':<15} {single['kappa']:.3f}    {single['accuracy']:.3f}    {single['mae']:.3f}    {single['n_claims']:<6} {single['avg_tool_calls']:.2f}         {single['avg_successful_calls']:.2f}")
    
    print(f"\n  (Multi-call source: {multi['source']})")
    print(f"  (Single-call source: {single['source']})")
    
    print("\nComparison (Multi - Single):")
    print("-"*70)
    comp = summary["comparison"]
    print(f"  Δκ = {comp['delta_hat']:.3f} [{comp['ci_lower']:.3f}, {comp['ci_upper']:.3f}]")
    
    # Effect size interpretation
    interpretation = interpret_h4_result(comp['delta_hat'], comp['ci_lower'])
    print(f"  Interpretation: {interpretation}")
    
    print("\nCondition Checks:")
    conds = summary["conditions"]
    
    print(f"  [{'✓' if conds['ci_lower_positive'] else '✗'}] 95% CI lower bound > 0: {comp['ci_lower']:.3f}")
    print(f"  [{'✓' if conds['delta_sufficient'] else '✗'}] Δκ ≥ 0.25: {comp['delta_hat']:.3f}")
    
    print("\n" + "-"*70)
    if summary["hypothesis_supported"]:
        print("  → H4 Hypothesis: ✓ SUPPORTED")
        print("    Multiple tool calls provide significant improvement over single-call")
    else:
        print("  → H4 Hypothesis: ✗ NOT SUPPORTED")
        if not conds["ci_lower_positive"]:
            print("    Effect is not statistically significant (CI includes 0)")
        if not conds["delta_sufficient"]:
            print("    Effect size is below threshold (Δκ < 0.25)")
        print("    A single well-chosen tool call may be sufficient for most claims")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4o", help="Model to analyze")
    args = parser.parse_args()
    
    summary = compute_h4_metrics(args.model)
    print_h4_summary(summary)
