#!/usr/bin/env python3
"""
H3: Ambiguity induces flips and lowers agreement

Compute metrics script to analyze FR, QWFR, and QWK differences.

Usage:
    python compute_metrics_h3.py
"""

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.dataset import load_claims
from common.logging_utils import (
    load_run_json,
    save_summary_json,
)
from common.metrics import (
    compute_qwk,
    compute_accuracy,
    compute_mae,
    flip_rate,
    quadratic_weighted_flip_rate,
    aggregate_fr_qwfr,
)


RUNS_DIR = Path(__file__).parent / "runs"
H1_RUNS_DIR = Path(__file__).parent.parent / "h1" / "runs"
SUMMARY_OUTPUT = Path(__file__).parent / "summary_h3_metrics.json"

MODEL = "gpt-4o"
ROUTER = "policy"


def get_h3_run_path() -> Path:
    """Get path to H3 run JSON file (demystified runs for FR/QWFR)."""
    return RUNS_DIR / f"h3_{MODEL}_{ROUTER}.json"


def get_h1_run_path() -> Path:
    """Get path to H1 run JSON file (original texts for QWK comparison)."""
    return H1_RUNS_DIR / f"h1_{MODEL}_with-tools_{ROUTER}.json"


def split_examples_by_group(examples: List[dict], claims: List[dict]) -> dict:
    """
    Split examples into crisp and ambiguous groups.
    
    Args:
        examples: List of example dicts from run
        claims: List of claim dicts for reference
        
    Returns:
        Dict with 'crisp' and 'ambiguous' keys, each containing:
        - 'examples': list of examples
        - 'per_claim_labels': dict[claim_id] -> list of labels
    """
    # Build claim lookup
    claim_lookup = {c["id"]: c for c in claims}
    
    result = {
        "crisp": {"examples": [], "per_claim_labels": defaultdict(list)},
        "ambiguous": {"examples": [], "per_claim_labels": defaultdict(list)},
    }
    
    for ex in examples:
        claim_id = ex["claim_id"]
        claim = claim_lookup.get(claim_id, {})
        group = claim.get("group", "crisp")
        
        result[group]["examples"].append(ex)
        result[group]["per_claim_labels"][claim_id].append(ex["predicted_label"])
    
    return result


def bootstrap_qwk_diff_ci(
    gold_labels_a: List[int],
    pred_labels_a: List[int],
    gold_labels_b: List[int],
    pred_labels_b: List[int],
    n_boot: int = 10_000,
    ci: float = 0.95,
    random_state: int = 0,
) -> dict:
    """
    Bootstrap CI for difference in QWK between two groups (crisp vs ambiguous).
    
    This uses a stratified bootstrap that resamples ALL claims together, then
    splits by group. This maintains the correlation structure and gives more
    stable estimates than independent resampling.
    """
    gold_a = np.asarray(gold_labels_a, dtype=int)
    pred_a = np.asarray(pred_labels_a, dtype=int)
    gold_b = np.asarray(gold_labels_b, dtype=int)
    pred_b = np.asarray(pred_labels_b, dtype=int)
    
    n_a = len(gold_a)
    n_b = len(gold_b)
    n_total = n_a + n_b
    
    rng = np.random.default_rng(random_state)
    
    # Point estimates
    kappa_a = compute_qwk(gold_a, pred_a)
    kappa_b = compute_qwk(gold_b, pred_b)
    delta_hat = kappa_a - kappa_b  # Crisp - Ambiguous
    
    # Combine all data with group indicators
    all_gold = np.concatenate([gold_a, gold_b])
    all_pred = np.concatenate([pred_a, pred_b])
    all_group = np.array([0] * n_a + [1] * n_b)  # 0 = crisp, 1 = ambiguous
    
    # Stratified bootstrap: resample within each group
    diffs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        # Resample within each stratum
        idx_a = rng.integers(0, n_a, size=n_a)
        idx_b = rng.integers(0, n_b, size=n_b)
        
        kappa_a_b = compute_qwk(gold_a[idx_a], pred_a[idx_a])
        kappa_b_b = compute_qwk(gold_b[idx_b], pred_b[idx_b])
        diffs[b] = kappa_a_b - kappa_b_b
    
    std_error = diffs.std(ddof=1)
    alpha = 1.0 - ci
    ci_lower, ci_upper = np.percentile(
        diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)]
    )
    
    return {
        "kappa_crisp": float(kappa_a),
        "kappa_ambiguous": float(kappa_b),
        "delta_hat": float(delta_hat),
        "std_error": float(std_error),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
    }


def bootstrap_fr_diff_ci(
    per_claim_labels_a: Dict[str, List[int]],
    per_claim_labels_b: Dict[str, List[int]],
    metric: str = "FR",  # "FR" or "QWFR"
    n_boot: int = 10_000,
    ci: float = 0.95,
    random_state: int = 0,
) -> dict:
    """
    Bootstrap CI for difference in FR or QWFR between two groups.
    """
    metric_fn = flip_rate if metric == "FR" else quadratic_weighted_flip_rate
    
    # Compute per-claim metrics for each group
    values_a = [metric_fn(labels) for labels in per_claim_labels_a.values()]
    values_b = [metric_fn(labels) for labels in per_claim_labels_b.values()]
    
    # For crisp claims with single prediction, FR/QWFR = 0
    if not values_a:
        values_a = [0.0]
    if not values_b:
        values_b = [0.0]
    
    values_a = np.asarray(values_a)
    values_b = np.asarray(values_b)
    
    n_a = len(values_a)
    n_b = len(values_b)
    
    rng = np.random.default_rng(random_state)
    
    # Point estimates
    mean_a = float(values_a.mean())
    mean_b = float(values_b.mean())
    delta_hat = mean_b - mean_a  # Ambiguous - Crisp
    
    # Bootstrap
    diffs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx_a = rng.integers(0, n_a, size=n_a)
        idx_b = rng.integers(0, n_b, size=n_b)
        
        mean_a_b = values_a[idx_a].mean()
        mean_b_b = values_b[idx_b].mean()
        diffs[b] = mean_b_b - mean_a_b
    
    alpha = 1.0 - ci
    ci_lower, ci_upper = np.percentile(
        diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)]
    )
    
    return {
        f"{metric}_crisp": mean_a,
        f"{metric}_ambiguous": mean_b,
        "delta_hat": float(delta_hat),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
    }


def compute_h3_metrics():
    """Compute all H3 metrics and save summary.
    
    This uses TWO run files:
    - H1 run (original claim texts): For QWK comparison between crisp and ambiguous claims
    - H3 run (demystified texts): For FR/QWFR analysis of ambiguous claims
    
    The key insight is that κ(Crisp) vs κ(Ambiguous) should compare the model's
    performance on original claim texts (both crisp and ambiguous), not on
    demystified versions. H3 demystified runs are only for measuring flip rate.
    """
    
    claims = load_claims()
    
    # Load H1 run for QWK comparison (original texts)
    h1_path = get_h1_run_path()
    if not h1_path.exists():
        print(f"Error: H1 run not found: {h1_path}")
        return None
    
    print(f"Loading H1 run (for QWK) from {h1_path}...")
    h1_data = load_run_json(h1_path)
    h1_examples = h1_data.get("examples", [])
    
    # Load H3 run for FR/QWFR (demystified texts)
    h3_path = get_h3_run_path()
    if not h3_path.exists():
        print(f"Error: H3 run not found: {h3_path}")
        return None
    
    print(f"Loading H3 run (for FR/QWFR) from {h3_path}...")
    h3_data = load_run_json(h3_path)
    h3_examples = h3_data.get("examples", [])
    
    # Split H1 examples by group (for QWK comparison)
    h1_groups = split_examples_by_group(h1_examples, claims)
    
    # Split H3 examples by group (for FR/QWFR, only ambiguous matters)
    h3_groups = split_examples_by_group(h3_examples, claims)
    
    print(f"\nH1 (for QWK): Crisp: {len(h1_groups['crisp']['examples'])} examples, "
          f"{len(h1_groups['crisp']['per_claim_labels'])} claims")
    print(f"H1 (for QWK): Ambiguous: {len(h1_groups['ambiguous']['examples'])} examples, "
          f"{len(h1_groups['ambiguous']['per_claim_labels'])} claims")
    print(f"H3 (for FR/QWFR): Ambiguous: {len(h3_groups['ambiguous']['examples'])} examples, "
          f"{len(h3_groups['ambiguous']['per_claim_labels'])} claims")
    
    summary = {
        "crisp": {},
        "ambiguous": {},
        "comparisons": {},
    }
    
    # =========================================================================
    # Crisp claims: QWK from H1 (original claim texts)
    # =========================================================================
    crisp_h1_examples = h1_groups["crisp"]["examples"]
    if crisp_h1_examples:
        crisp_gold = [ex["gold_label"] for ex in crisp_h1_examples]
        crisp_pred = [ex["predicted_label"] for ex in crisp_h1_examples]
        
        summary["crisp"] = {
            "n_claims": len(h1_groups["crisp"]["per_claim_labels"]),
            "n_examples": len(crisp_h1_examples),
            "kappa": compute_qwk(crisp_gold, crisp_pred),
            "accuracy": compute_accuracy(crisp_gold, crisp_pred),
            "mae": compute_mae(crisp_gold, crisp_pred),
            "FR": 0.0,  # No flip rate for crisp claims
            "QWFR": 0.0,
        }
        print(f"\nCrisp (H1): κ={summary['crisp']['kappa']:.3f}")
    
    # =========================================================================
    # Ambiguous claims: 
    #   - QWK from H1 (original ambiguous texts like "~0.5 V")
    #   - FR/QWFR from H3 (demystified runs)
    #   - Best/Worst QWK from H3 (across different interpretations)
    # =========================================================================
    amb_h1_examples = h1_groups["ambiguous"]["examples"]
    amb_h3_examples = h3_groups["ambiguous"]["examples"]
    amb_h3_per_claim = h3_groups["ambiguous"]["per_claim_labels"]
    
    if amb_h1_examples:
        # QWK is based on H1 predictions (original ambiguous texts)
        # H1 has one prediction per claim for ambiguous claims
        amb_gold = [ex["gold_label"] for ex in amb_h1_examples]
        amb_pred = [ex["predicted_label"] for ex in amb_h1_examples]
        
        kappa_h1 = compute_qwk(amb_gold, amb_pred)
        
        # For "best" and "worst" QWK, use H3 runs (across different demystified versions)
        # This shows the range of possible outcomes depending on interpretation
        amb_best = {}
        amb_worst = {}
        for ex in amb_h3_examples:
            claim_id = ex["claim_id"]
            gold = ex["gold_label"]
            pred = ex["predicted_label"]
            error = abs(pred - gold)
            
            if claim_id not in amb_best or error < amb_best[claim_id]["error"]:
                amb_best[claim_id] = {"ex": ex, "error": error}
            if claim_id not in amb_worst or error > amb_worst[claim_id]["error"]:
                amb_worst[claim_id] = {"ex": ex, "error": error}
        
        amb_best_gold = [v["ex"]["gold_label"] for v in amb_best.values()]
        amb_best_pred = [v["ex"]["predicted_label"] for v in amb_best.values()]
        amb_worst_gold = [v["ex"]["gold_label"] for v in amb_worst.values()]
        amb_worst_pred = [v["ex"]["predicted_label"] for v in amb_worst.values()]
        
        kappa_best = compute_qwk(amb_best_gold, amb_best_pred) if amb_best else None
        kappa_worst = compute_qwk(amb_worst_gold, amb_worst_pred) if amb_worst else None
        
        # Compute FR and QWFR from H3 demystified runs
        fr_qwfr = aggregate_fr_qwfr(dict(amb_h3_per_claim)) if amb_h3_per_claim else {"FR": 0.0, "QWFR": 0.0}
        
        summary["ambiguous"] = {
            "n_claims": len(amb_h1_examples),
            "n_examples_h1": len(amb_h1_examples),
            "n_examples_h3": len(amb_h3_examples),
            "kappa": kappa_h1,  # H1: original ambiguous text
            "kappa_best": kappa_best,  # H3: best demystified interpretation
            "kappa_worst": kappa_worst,  # H3: worst demystified interpretation
            "accuracy": compute_accuracy(amb_gold, amb_pred),
            "mae": compute_mae(amb_gold, amb_pred),
            "FR": fr_qwfr["FR"],
            "QWFR": fr_qwfr["QWFR"],
        }
        print(f"Ambiguous (H1 QWK, H3 FR/QWFR):")
        print(f"  κ(H1 original)={kappa_h1:.3f}")
        if kappa_best is not None:
            print(f"  κ_best(H3)={kappa_best:.3f}, κ_worst(H3)={kappa_worst:.3f}")
        print(f"  FR={fr_qwfr['FR']:.3f}, QWFR={fr_qwfr['QWFR']:.3f}")
    
    # Comparisons
    if crisp_h1_examples and amb_h1_examples:
        # QWK difference: Crisp - Ambiguous (both from H1, original texts)
        print("\nComparing Crisp vs Ambiguous (both from H1 original texts)...")
        
        # For QWK comparison, use H1 examples (original claim texts)
        crisp_gold_list = [ex["gold_label"] for ex in crisp_h1_examples]
        crisp_pred_list = [ex["predicted_label"] for ex in crisp_h1_examples]
        
        amb_gold_list = [ex["gold_label"] for ex in amb_h1_examples]
        amb_pred_list = [ex["predicted_label"] for ex in amb_h1_examples]
        
        qwk_diff = bootstrap_qwk_diff_ci(
            crisp_gold_list, crisp_pred_list,
            amb_gold_list, amb_pred_list,
        )
        
        # FR difference: Ambiguous - Crisp
        # Crisp claims from H1 have single predictions (FR=0)
        # Ambiguous claims from H3 have multiple predictions per claim
        fr_diff = bootstrap_fr_diff_ci(
            h1_groups["crisp"]["per_claim_labels"],  # Single prediction per claim
            h3_groups["ambiguous"]["per_claim_labels"],  # Multiple (demystified) predictions
            metric="FR",
        )
        
        # QWFR difference
        qwfr_diff = bootstrap_fr_diff_ci(
            h1_groups["crisp"]["per_claim_labels"],
            h3_groups["ambiguous"]["per_claim_labels"],
            metric="QWFR",
        )
        
        # Determine if hypothesis is supported
        # Support if:
        # - QWFR(Amb) - QWFR(Crisp) >= 0.05 with CI > 0, OR
        # - FR(Amb) - FR(Crisp) >= 0.20 with CI > 0
        # AND
        # - κ(Crisp) - κ(Amb) >= 0.25 with CI > 0
        
        # Condition 1a: QWFR difference
        qwfr_threshold_met = qwfr_diff["delta_hat"] >= 0.05
        qwfr_ci_positive = qwfr_diff["ci_lower"] > 0
        qwfr_passed = qwfr_threshold_met and qwfr_ci_positive
        
        # Condition 1b: FR difference
        fr_threshold_met = fr_diff["delta_hat"] >= 0.20
        fr_ci_positive = fr_diff["ci_lower"] > 0
        fr_passed = fr_threshold_met and fr_ci_positive
        
        # Condition 2: QWK difference
        qwk_threshold_met = qwk_diff["delta_hat"] >= 0.25
        qwk_ci_positive = qwk_diff["ci_lower"] > 0
        qwk_passed = qwk_threshold_met and qwk_ci_positive
        
        flip_condition = qwfr_passed or fr_passed
        hypothesis_supported = flip_condition and qwk_passed
        
        summary["comparisons"] = {
            "qwk_diff": qwk_diff,
            "fr_diff": fr_diff,
            "qwfr_diff": qwfr_diff,
            # Individual condition details
            "conditions": {
                "qwfr": {
                    "criterion": "QWFR(Amb) - QWFR(Crisp) ≥ 0.05 with CI > 0",
                    "delta": qwfr_diff["delta_hat"],
                    "threshold": 0.05,
                    "threshold_met": qwfr_threshold_met,
                    "ci_lower": qwfr_diff["ci_lower"],
                    "ci_positive": qwfr_ci_positive,
                    "passed": qwfr_passed,
                },
                "fr": {
                    "criterion": "FR(Amb) - FR(Crisp) ≥ 0.20 with CI > 0",
                    "delta": fr_diff["delta_hat"],
                    "threshold": 0.20,
                    "threshold_met": fr_threshold_met,
                    "ci_lower": fr_diff["ci_lower"],
                    "ci_positive": fr_ci_positive,
                    "passed": fr_passed,
                },
                "qwk": {
                    "criterion": "κ(Crisp) - κ(Amb) ≥ 0.25 with CI > 0",
                    "delta": qwk_diff["delta_hat"],
                    "threshold": 0.25,
                    "threshold_met": qwk_threshold_met,
                    "ci_lower": qwk_diff["ci_lower"],
                    "ci_positive": qwk_ci_positive,
                    "passed": qwk_passed,
                },
            },
            "flip_condition_met": flip_condition,  # QWFR OR FR
            "qwk_condition_met": qwk_passed,
            "hypothesis_supported": hypothesis_supported,
            "interpretation": interpret_h3_result(qwk_diff, fr_diff, qwfr_diff),
        }
        
        # Detailed condition output
        print("\n  Condition checks:")
        
        qwfr_status = "✓ PASS" if qwfr_passed else "✗ FAIL"
        print(f"    [1a] QWFR(Amb) - QWFR(Crisp) ≥ 0.05:  {qwfr_diff['delta_hat']:.3f} ≥ 0.05? "
              f"{'Yes' if qwfr_threshold_met else 'No'}, CI [{qwfr_diff['ci_lower']:.3f}, {qwfr_diff['ci_upper']:.3f}] > 0? "
              f"{'Yes' if qwfr_ci_positive else 'No'} → {qwfr_status}")
        
        fr_status = "✓ PASS" if fr_passed else "✗ FAIL"
        print(f"    [1b] FR(Amb) - FR(Crisp) ≥ 0.20:     {fr_diff['delta_hat']:.3f} ≥ 0.20? "
              f"{'Yes' if fr_threshold_met else 'No'}, CI [{fr_diff['ci_lower']:.3f}, {fr_diff['ci_upper']:.3f}] > 0? "
              f"{'Yes' if fr_ci_positive else 'No'} → {fr_status}")
        
        flip_status = "✓ PASS" if flip_condition else "✗ FAIL"
        print(f"    [1]  Flip condition (1a OR 1b):      {flip_status}")
        
        qwk_status = "✓ PASS" if qwk_passed else "✗ FAIL"
        print(f"    [2]  κ(Crisp) - κ(Amb) ≥ 0.25:       {qwk_diff['delta_hat']:.3f} ≥ 0.25? "
              f"{'Yes' if qwk_threshold_met else 'No'}, CI [{qwk_diff['ci_lower']:.3f}, {qwk_diff['ci_upper']:.3f}] > 0? "
              f"{'Yes' if qwk_ci_positive else 'No'} → {qwk_status}")
        
        overall_status = "✓ SUPPORTED" if hypothesis_supported else "✗ NOT SUPPORTED"
        print(f"\n  H3 Hypothesis ([1] AND [2]): {overall_status}")
    
    # Save summary
    save_summary_json(SUMMARY_OUTPUT, summary)
    print(f"\nSaved summary to {SUMMARY_OUTPUT}")
    
    return summary


def interpret_h3_result(qwk_diff: dict, fr_diff: dict, qwfr_diff: dict) -> str:
    """Generate interpretation text for H3 results.
    
    Effect size heuristic for Drop-4 (N=37):
    - |Δκ| < 0.20: indistinguishable from noise
    - 0.20 ≤ |Δκ| < 0.35: borderline, requires careful CI inspection
    - |Δκ| ≥ 0.35: very likely a real effect
    """
    parts = []
    
    qwk_delta = qwk_diff["delta_hat"]
    qwk_ci_positive = qwk_diff["ci_lower"] > 0
    abs_qwk = abs(qwk_delta)
    
    # QWK interpretation
    if qwk_delta > 0:
        if abs_qwk >= 0.35:
            effect = "strong" if qwk_ci_positive else "large but inconclusive"
        elif abs_qwk >= 0.20:
            effect = "borderline" if qwk_ci_positive else "borderline, CI includes 0"
        else:
            effect = "small (noise range)"
        parts.append(f"Crisp > Ambiguous QWK by {qwk_delta:.2f} ({effect})")
    else:
        parts.append(f"Ambiguous > Crisp QWK by {-qwk_delta:.2f} (unexpected)")
    
    # Flip rate interpretation
    if fr_diff["delta_hat"] > 0:
        parts.append(f"Ambiguous FR higher by {fr_diff['delta_hat']:.2f}")
    
    if qwfr_diff["delta_hat"] > 0:
        parts.append(f"Ambiguous QWFR higher by {qwfr_diff['delta_hat']:.2f}")
    
    return "; ".join(parts)


def print_summary_table(summary: dict):
    """Print a formatted summary table."""
    
    print("\n" + "=" * 80)
    print("H3 SUMMARY: Ambiguity Analysis")
    print("=" * 80)
    
    # Per-group metrics
    print("\nPer-Group Metrics:")
    print("-" * 80)
    print(f"{'Group':<15} {'κ':>8} {'Acc':>8} {'MAE':>8} {'FR':>8} {'QWFR':>8} {'Claims':>8}")
    print("-" * 80)
    
    for group in ["crisp", "ambiguous"]:
        if group in summary:
            m = summary[group]
            kappa = m.get('kappa_canonical', m.get('kappa', 0))
            print(f"{group:<15} {kappa:>8.3f} {m['accuracy']:>8.3f} "
                  f"{m['mae']:>8.3f} {m['FR']:>8.3f} {m['QWFR']:>8.3f} {m['n_claims']:>8}")
    
    # Show best/worst for ambiguous if available
    if "ambiguous" in summary:
        amb = summary["ambiguous"]
        if "kappa_best" in amb and amb["kappa_best"] is not None:
            print("\nAmbiguous QWK breakdown:")
            print(f"  κ (H1 original text):         {amb['kappa']:.3f}  ← used for hypothesis test")
            print(f"  κ_best (H3 best demystified): {amb['kappa_best']:.3f}")
            print(f"  κ_worst (H3 worst demystified): {amb['kappa_worst']:.3f}")
            print(f"  H3 Range: {amb['kappa_worst']:.3f} → {amb['kappa_best']:.3f} "
                  f"(Δ = {amb['kappa_best'] - amb['kappa_worst']:.3f})")
    
    # Comparisons
    if "comparisons" in summary:
        comp = summary["comparisons"]
        conditions = comp.get("conditions", {})
        
        print("\n\nH3 Hypothesis Conditions:")
        print("-" * 80)
        
        # Show each condition with pass/fail
        if conditions:
            qwfr_c = conditions.get("qwfr", {})
            fr_c = conditions.get("fr", {})
            qwk_c = conditions.get("qwk", {})
            
            qwfr_status = "✓" if qwfr_c.get("passed") else "✗"
            print(f"[1a] ΔQWFR ≥ 0.05 & CI > 0:  {qwfr_c.get('delta', 0):.3f} "
                  f"[{qwfr_c.get('ci_lower', 0):.3f}]  {qwfr_status}")
            
            fr_status = "✓" if fr_c.get("passed") else "✗"
            print(f"[1b] ΔFR ≥ 0.20 & CI > 0:    {fr_c.get('delta', 0):.3f} "
                  f"[{fr_c.get('ci_lower', 0):.3f}]  {fr_status}")
            
            flip_status = "✓" if comp.get("flip_condition_met") else "✗"
            print(f"[1]  Flip (1a OR 1b):        {flip_status}")
            
            qwk_status = "✓" if qwk_c.get("passed") else "✗"
            print(f"[2]  Δκ ≥ 0.25 & CI > 0:     {qwk_c.get('delta', 0):.3f} "
                  f"[{qwk_c.get('ci_lower', 0):.3f}]  {qwk_status}")
        else:
            # Fallback for old format
            qwk = comp.get("qwk_diff", {})
            print(f"Δκ (Crisp - Amb) = {qwk.get('delta_hat', 0):.3f} "
                  f"[{qwk.get('ci_lower', 0):.3f}, {qwk.get('ci_upper', 0):.3f}]")
            
            fr = comp.get("fr_diff", {})
            print(f"ΔFR (Amb - Crisp) = {fr.get('delta_hat', 0):.3f} "
                  f"[{fr.get('ci_lower', 0):.3f}, {fr.get('ci_upper', 0):.3f}]")
            
            qwfr = comp.get("qwfr_diff", {})
            print(f"ΔQWFR (Amb - Crisp) = {qwfr.get('delta_hat', 0):.3f} "
                  f"[{qwfr.get('ci_lower', 0):.3f}, {qwfr.get('ci_upper', 0):.3f}]")
        
        print(f"\nInterpretation: {comp.get('interpretation', 'N/A')}")
    
    # Overall verdict
    print("\n" + "=" * 80)
    if "comparisons" in summary:
        supported = "YES" if summary["comparisons"].get("hypothesis_supported") else "NO"
        print(f"H3 Hypothesis supported: {supported}")
    print("=" * 80)


if __name__ == "__main__":
    summary = compute_h3_metrics()
    if summary:
        print_summary_table(summary)

