"""
Metrics module for LAMINA Evaluation Framework.

Contains implementations for:
- Quadratic Weighted Kappa (QWK)
- Accuracy and MAE
- Flip-Rate (FR) and Quadratic-Weighted Flip-Rate (QWFR)
- Bootstrap confidence intervals for Δκ
"""

from typing import Any, Dict, List, Sequence

import numpy as np
from sklearn.metrics import cohen_kappa_score


def compute_qwk(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """
    Compute Quadratic Weighted Kappa for labels in {-2, -1, 0, 1, 2}.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        QWK score in [-1, 1]
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    
    if len(y_true) == 0:
        return 0.0
    
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def compute_accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """
    Compute exact match accuracy.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Accuracy in [0, 1]
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if len(y_true) == 0:
        return 0.0
    
    return float((y_true == y_pred).mean())


def compute_mae(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """
    Compute Mean Absolute Error.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        MAE (lower is better)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    if len(y_true) == 0:
        return 0.0
    
    return float(np.abs(y_true - y_pred).mean())


def flip_rate(labels: List[int]) -> float:
    """
    Compute flip-rate for a single claim across multiple mappings.
    
    FR(c) = (2 / n(n-1)) * sum_{i<j} 1[y_i != y_j]
    
    Args:
        labels: List of predicted labels for same claim under different mappings
        
    Returns:
        FR(c) in [0, 1]
    """
    labels = np.asarray(labels, dtype=int)
    n = len(labels)
    
    if n < 2:
        return 0.0
    
    num_pairs = n * (n - 1) / 2.0
    diffs = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] != labels[j]:
                diffs += 1
    
    return float(2.0 * diffs / (n * (n - 1)))


def quadratic_weighted_flip_rate(labels: List[int]) -> float:
    """
    Compute quadratic-weighted flip-rate for a single claim.
    
    QWFR(c) = (2 / n(n-1)) * sum_{i<j} ((|y_i - y_j| / (K-1))^2)
    
    where labels are shifted to [0, 4] and K=5.
    
    Args:
        labels: List of predicted labels for same claim under different mappings
        
    Returns:
        QWFR(c) in [0, 1]
    """
    labels = np.asarray(labels, dtype=int)
    n = len(labels)
    
    if n < 2:
        return 0.0
    
    # Shift labels from {-2, -1, 0, 1, 2} to {0, 1, 2, 3, 4}
    shifted = labels + 2
    K = 5
    
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = abs(int(shifted[i]) - int(shifted[j]))
            w = (d / (K - 1)) ** 2
            total += w
    
    return float(2.0 * total / (n * (n - 1)))


def aggregate_fr_qwfr(per_claim_labels: Dict[str, List[int]]) -> Dict[str, float]:
    """
    Aggregate FR and QWFR over multiple claims.
    
    Args:
        per_claim_labels: dict[claim_id] -> List[int] of labels across mappings
        
    Returns:
        {'FR': float, 'QWFR': float}
    """
    fr_values = []
    qwfr_values = []
    
    for claim_id, labels in per_claim_labels.items():
        fr_values.append(flip_rate(labels))
        qwfr_values.append(quadratic_weighted_flip_rate(labels))
    
    return {
        "FR": float(np.mean(fr_values)) if fr_values else 0.0,
        "QWFR": float(np.mean(qwfr_values)) if qwfr_values else 0.0,
    }


def bootstrap_kappa_diff_ci(
    y_true: Sequence[int],
    y_pred_a: Sequence[int],
    y_pred_b: Sequence[int],
    n_boot: int = 10_000,
    ci: float = 0.95,
    random_state: int = 0,
) -> Dict[str, Any]:
    """
    Bootstrap CI for Δκ = κ_B - κ_A between two systems A and B.
    
    Args:
        y_true: Ground truth labels
        y_pred_a: Predictions from system A
        y_pred_b: Predictions from system B
        n_boot: Number of bootstrap samples
        ci: Confidence interval level
        random_state: Random seed
        
    Returns:
        Dictionary with kappa values, delta, std error, and CI bounds
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred_a = np.asarray(y_pred_a, dtype=int)
    y_pred_b = np.asarray(y_pred_b, dtype=int)
    
    assert len(y_true) == len(y_pred_a) == len(y_pred_b), \
        f"Length mismatch: {len(y_true)}, {len(y_pred_a)}, {len(y_pred_b)}"
    
    n = len(y_true)
    rng = np.random.default_rng(random_state)
    
    # Point estimate
    kappa_a = compute_qwk(y_true, y_pred_a)
    kappa_b = compute_qwk(y_true, y_pred_b)
    delta_hat = kappa_b - kappa_a
    
    # Bootstrap
    diffs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        kappa_a_b = compute_qwk(y_true[idx], y_pred_a[idx])
        kappa_b_b = compute_qwk(y_true[idx], y_pred_b[idx])
        diffs[b] = kappa_b_b - kappa_a_b
    
    std_error = diffs.std(ddof=1)
    alpha = 1.0 - ci
    ci_lower, ci_upper = np.percentile(
        diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)]
    )
    
    return {
        "kappa_a": float(kappa_a),
        "kappa_b": float(kappa_b),
        "delta_hat": float(delta_hat),
        "std_error_delta_hat": float(std_error),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "n_boot": n_boot,
        "ci": ci,
    }


def bootstrap_gap_shrinkage_ci(
    y_true: Sequence[int],
    y_pred_small_no_tools: Sequence[int],
    y_pred_large_no_tools: Sequence[int],
    y_pred_small_tools: Sequence[int],
    y_pred_large_tools: Sequence[int],
    n_boot: int = 10_000,
    ci: float = 0.95,
    random_state: int = 0,
) -> Dict[str, Any]:
    """
    Bootstrap CI for gap shrinkage between small and large models.
    
    Computes:
    - Gap_no_tools = κ_large_no_tools - κ_small_no_tools
    - Gap_tools = κ_large_tools - κ_small_tools
    - Δ_gap = Gap_no_tools - Gap_tools
    
    Args:
        y_true: Ground truth labels
        y_pred_small_no_tools: Small model predictions without tools
        y_pred_large_no_tools: Large model predictions without tools
        y_pred_small_tools: Small model predictions with tools
        y_pred_large_tools: Large model predictions with tools
        n_boot: Number of bootstrap samples
        ci: Confidence interval level
        random_state: Random seed
        
    Returns:
        Dictionary with gaps and bootstrap CIs
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred_small_no_tools = np.asarray(y_pred_small_no_tools, dtype=int)
    y_pred_large_no_tools = np.asarray(y_pred_large_no_tools, dtype=int)
    y_pred_small_tools = np.asarray(y_pred_small_tools, dtype=int)
    y_pred_large_tools = np.asarray(y_pred_large_tools, dtype=int)
    
    n = len(y_true)
    rng = np.random.default_rng(random_state)
    
    # Point estimates
    kappa_small_no_tools = compute_qwk(y_true, y_pred_small_no_tools)
    kappa_large_no_tools = compute_qwk(y_true, y_pred_large_no_tools)
    kappa_small_tools = compute_qwk(y_true, y_pred_small_tools)
    kappa_large_tools = compute_qwk(y_true, y_pred_large_tools)
    
    gap_no_tools = kappa_large_no_tools - kappa_small_no_tools
    gap_tools = kappa_large_tools - kappa_small_tools
    delta_gap = gap_no_tools - gap_tools
    
    # Bootstrap for gap_no_tools
    gaps_no_tools = np.empty(n_boot, dtype=float)
    gaps_tools = np.empty(n_boot, dtype=float)
    delta_gaps = np.empty(n_boot, dtype=float)
    
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        
        k_small_no = compute_qwk(y_true[idx], y_pred_small_no_tools[idx])
        k_large_no = compute_qwk(y_true[idx], y_pred_large_no_tools[idx])
        k_small_t = compute_qwk(y_true[idx], y_pred_small_tools[idx])
        k_large_t = compute_qwk(y_true[idx], y_pred_large_tools[idx])
        
        gaps_no_tools[b] = k_large_no - k_small_no
        gaps_tools[b] = k_large_t - k_small_t
        delta_gaps[b] = gaps_no_tools[b] - gaps_tools[b]
    
    alpha = 1.0 - ci
    
    return {
        "kappa_small_no_tools": float(kappa_small_no_tools),
        "kappa_large_no_tools": float(kappa_large_no_tools),
        "kappa_small_tools": float(kappa_small_tools),
        "kappa_large_tools": float(kappa_large_tools),
        "gap_no_tools": float(gap_no_tools),
        "gap_tools": float(gap_tools),
        "delta_gap": float(delta_gap),
        "gap_no_tools_ci": {
            "lower": float(np.percentile(gaps_no_tools, 100 * alpha / 2)),
            "upper": float(np.percentile(gaps_no_tools, 100 * (1 - alpha / 2))),
        },
        "gap_tools_ci": {
            "lower": float(np.percentile(gaps_tools, 100 * alpha / 2)),
            "upper": float(np.percentile(gaps_tools, 100 * (1 - alpha / 2))),
        },
        "delta_gap_ci": {
            "lower": float(np.percentile(delta_gaps, 100 * alpha / 2)),
            "upper": float(np.percentile(delta_gaps, 100 * (1 - alpha / 2))),
        },
        "n_boot": n_boot,
        "ci": ci,
    }


def _get_families_from_tool_calls(tool_calls: List[Dict[str, Any]]) -> set:
    """
    Get all families used by a list of tool calls.
    
    A tool can belong to multiple families (e.g., assess_phase_strength_and_stiffness_claims
    uses both CALPHAD and Materials Project internally). If ANY tool in the call list
    uses the required family, we count it as called.
    
    Args:
        tool_calls: List of tool call dicts with 'tool_name' and optional 'family' fields
        
    Returns:
        Set of all family names used by any tool call
    """
    # Try to import the tool families mapping
    try:
        from .tool_families import get_tool_families, normalize_family
        use_mapping = True
    except ImportError:
        use_mapping = False
    
    families = set()
    for tc in tool_calls:
        tool_name = tc.get("tool_name")
        
        if use_mapping and tool_name:
            # Get families from the mapping (handles multi-family tools)
            tool_families = get_tool_families(tool_name)
            families.update(tool_families)
        
        # Also include the explicitly recorded family (for backwards compatibility)
        recorded_family = tc.get("family")
        if recorded_family:
            if use_mapping:
                # Normalize to canonical names
                families.update(normalize_family(recorded_family))
            else:
                families.add(recorded_family)
    
    return families


def compute_tool_miss_rate(
    run_examples: List[Dict[str, Any]],
    metadata: Dict[str, Dict[str, str]],
) -> float:
    """
    Compute the tool miss rate for a run.
    
    Miss rate = proportion of claims where required tool family was never called.
    A miss only occurs if NO tool from the required family was called.
    Tools can belong to multiple families (e.g., a CALPHAD+MP tool counts for both).
    
    Args:
        run_examples: List of example dicts from run JSON
        metadata: dict[claim_id] -> {'required_tool_family': ...}
        
    Returns:
        Miss rate in [0, 1]
    """
    total = 0
    misses = 0
    
    for ex in run_examples:
        claim_id = ex["claim_id"]
        if claim_id not in metadata:
            continue
            
        required_family = metadata[claim_id].get("required_tool_family")
        if required_family is None:
            continue
            
        total += 1
        families_called = _get_families_from_tool_calls(ex.get("tool_calls", []))
        
        # Check if required family (or its normalized form) was called
        try:
            from .tool_families import normalize_family
            required_families = set(normalize_family(required_family))
        except ImportError:
            required_families = {required_family}
        
        # Miss if none of the required families were called
        if not (required_families & families_called):
            misses += 1
    
    return misses / total if total > 0 else 0.0


def bootstrap_miss_rate_ci(
    run_examples: List[Dict[str, Any]],
    metadata: Dict[str, Dict[str, str]],
    n_boot: int = 10_000,
    ci: float = 0.95,
    random_state: int = 0,
) -> Dict[str, Any]:
    """
    Bootstrap CI for tool miss rate.
    
    Args:
        run_examples: List of example dicts from run JSON
        metadata: dict[claim_id] -> {'required_tool_family': ...}
        n_boot: Number of bootstrap samples
        ci: Confidence interval level
        random_state: Random seed
        
    Returns:
        Dictionary with miss rate and CI bounds
    """
    # Try to import normalization
    try:
        from .tool_families import normalize_family
        use_mapping = True
    except ImportError:
        use_mapping = False
    
    # Build array of miss indicators
    misses = []
    for ex in run_examples:
        claim_id = ex["claim_id"]
        if claim_id not in metadata:
            continue
            
        required_family = metadata[claim_id].get("required_tool_family")
        if required_family is None:
            continue
            
        families_called = _get_families_from_tool_calls(ex.get("tool_calls", []))
        
        # Check if required family was called
        if use_mapping:
            required_families = set(normalize_family(required_family))
        else:
            required_families = {required_family}
        
        is_miss = 1 if not (required_families & families_called) else 0
        misses.append(is_miss)
    
    misses = np.asarray(misses)
    n = len(misses)
    
    if n == 0:
        return {
            "miss_rate": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "n_boot": n_boot,
            "ci": ci,
        }
    
    miss_rate = float(misses.mean())
    
    rng = np.random.default_rng(random_state)
    boot_rates = np.empty(n_boot, dtype=float)
    
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_rates[b] = misses[idx].mean()
    
    alpha = 1.0 - ci
    ci_lower, ci_upper = np.percentile(
        boot_rates, [100 * alpha / 2, 100 * (1 - alpha / 2)]
    )
    
    return {
        "miss_rate": miss_rate,
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "n_boot": n_boot,
        "ci": ci,
    }

