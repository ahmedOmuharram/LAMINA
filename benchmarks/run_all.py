#!/usr/bin/env python3
"""
Run all hypothesis evaluations (H1, H2, H3, H4, H5).

This script:
1. Generates predictions for each hypothesis (if not already done)
2. Computes metrics for each hypothesis
3. Prints a summary of all results

Usage:
    python run_all.py                    # Run all (generate + compute)
    python run_all.py --compute-only     # Only compute metrics (skip generation)
    python run_all.py --generate-only    # Only generate predictions
    python run_all.py --mock             # Use mock runner (for testing)
"""

import argparse
import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add benchmarks directory to path
sys.path.insert(0, str(Path(__file__).parent))


def print_header(text: str, char: str = "=", width: int = 80):
    """Print a formatted header."""
    print()
    print(char * width)
    print(f" {text}")
    print(char * width)


def print_subheader(text: str):
    """Print a subheader."""
    print()
    print(f"--- {text} ---")
    print()


async def run_h1_all(use_mock: bool = False, resume: bool = True):
    """Run H1 for all model/tool configurations."""
    from h1.run_h1 import run_h1
    
    configs = [
        ("gpt-4o", True),
        ("gpt-4o", False),
        ("gpt-4o-mini", True),
        ("gpt-4o-mini", False),
    ]
    
    runs_dir = Path(__file__).parent / "h1" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    for model, tools in configs:
        tools_str = "with-tools" if tools else "no-tools"
        output_path = runs_dir / f"h1_{model}_{tools_str}_policy.json"
        
        if output_path.exists() and not use_mock:
            print(f"  Skipping {model} {tools_str} (already exists)")
            continue
        
        print(f"  Running {model} {tools_str}...")
        await run_h1(
            model_name=model,
            tools_enabled=tools,
            router_type="policy",
            output_path=output_path,
            use_mock=use_mock,
            resume=resume,
        )


async def run_h2_all(use_mock: bool = False, resume: bool = True):
    """Run H2 for all router configurations."""
    from h2.run_h2 import run_h2
    
    routers = ["no-policy", "policy", "oracle"]
    
    runs_dir = Path(__file__).parent / "h2" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    for router in routers:
        output_path = runs_dir / f"h2_gpt-4o_{router}.json"
        
        if output_path.exists() and not use_mock:
            print(f"  Skipping gpt-4o {router} (already exists)")
            continue
        
        print(f"  Running gpt-4o {router}...")
        await run_h2(
            model_name="gpt-4o",
            router_type=router,
            output_path=output_path,
            use_mock=use_mock,
            resume=resume,
        )


async def run_h3_all(use_mock: bool = False):
    """Run H3 ambiguity analysis."""
    from h3.run_h3 import run_h3
    
    runs_dir = Path(__file__).parent / "h3" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = runs_dir / "h3_gpt-4o_policy.json"
    
    if output_path.exists() and not use_mock:
        print("  Skipping H3 (already exists)")
        return
    
    print("  Running H3 gpt-4o policy...")
    await run_h3(
        model_name="gpt-4o",
        router_type="policy",
        output_path=output_path,
        use_mock=use_mock,
    )


async def run_h4_all(use_mock: bool = False):
    """Run H4 single-call mode (multi-call baseline comes from H1)."""
    from h4.run_h4 import run_h4_benchmark
    
    runs_dir = Path(__file__).parent / "h4" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = runs_dir / "h4_gpt-4o_single-call.json"
    
    if output_path.exists() and not use_mock:
        print("  Skipping H4 single-call (already exists)")
        return
    
    print("  Running H4 gpt-4o single-call...")
    print("  (Multi-call baseline uses H1 with-tools run)")
    await run_h4_benchmark(
        model_name="gpt-4o",
        seed=42,
    )


def compute_h1_metrics():
    """Compute H1 metrics."""
    from h1.compute_metrics_h1 import compute_h1_metrics as _compute
    return _compute()


def compute_h2_metrics():
    """Compute H2 metrics."""
    from h2.compute_metrics_h2 import compute_h2_metrics as _compute
    return _compute()


def compute_h3_metrics():
    """Compute H3 metrics."""
    from h3.compute_metrics_h3 import compute_h3_metrics as _compute
    return _compute()


def compute_h4_metrics():
    """Compute H4 metrics."""
    from h4.compute_metrics_h4 import compute_h4_metrics as _compute
    return _compute()


def compute_h5_metrics():
    """Compute H5 metrics."""
    from h5.compute_metrics_h5 import compute_h5_metrics as _compute
    return _compute()


def print_final_summary(results: dict):
    """Print a final summary of all hypothesis results."""
    print_header("FINAL SUMMARY: All Hypotheses", "=", 80)
    
    print()
    print(f"{'Hypothesis':<12} {'Status':<15} {'Key Metric':<40}")
    print("-" * 67)
    
    for h_id, result in results.items():
        if result is None:
            print(f"{h_id:<12} {'SKIPPED':<15} {'(no data)':<40}")
            continue
        
        # Extract hypothesis support status based on result structure
        if h_id == "H1":
            # H1 has overall_verdict.h1_supported
            supported = result.get("overall_verdict", {}).get("h1_supported", None)
            # Get best model's delta
            comparisons = result.get("comparisons", {})
            best_delta = 0
            for model, data in comparisons.items():
                delta = data.get("bootstrap", {}).get("delta_hat", 0)
                if delta > best_delta:
                    best_delta = delta
            key_metric = f"Best Δκ(tools) = {best_delta:.3f}"
            
        elif h_id == "H2":
            # H2 has overall_verdict.h2_policy_supported
            supported = result.get("overall_verdict", {}).get("h2_policy_supported", None)
            delta = result.get("comparisons", {}).get("policy_vs_no_policy", {}).get("bootstrap", {}).get("delta_hat", 0)
            key_metric = f"Δκ(policy) = {delta:.3f}"
            
        elif h_id == "H3":
            # H3 has comparisons.hypothesis_supported
            supported = result.get("comparisons", {}).get("hypothesis_supported", None)
            delta = result.get("comparisons", {}).get("qwk_diff", {}).get("delta_hat", 0)
            fr = result.get("ambiguous", {}).get("FR", 0)
            key_metric = f"Δκ = {delta:.3f}, FR = {fr:.3f}"
            
        elif h_id == "H4":
            # H4 has hypothesis_supported at top level
            supported = result.get("hypothesis_supported", None)
            delta = result.get("comparison", {}).get("delta_hat", 0)
            multi_kappa = result.get("multi_call", {}).get("kappa", 0)
            single_kappa = result.get("single_call", {}).get("kappa", 0)
            key_metric = f"Δκ = {delta:.3f} (multi={multi_kappa:.3f}, single={single_kappa:.3f})"
            
        elif h_id == "H5":
            # H5 has overall_verdict.h5_supported or analysis.hypothesis_supported
            supported = result.get("overall_verdict", {}).get("h5_supported", 
                        result.get("analysis", {}).get("hypothesis_supported", None))
            delta_gap = result.get("analysis", {}).get("delta_gap", 0)
            gap_no_tools = result.get("analysis", {}).get("gap_no_tools", 0)
            if gap_no_tools > 0:
                shrinkage_pct = (delta_gap / gap_no_tools) * 100
            else:
                shrinkage_pct = 0
            key_metric = f"Gap shrinkage = {shrinkage_pct:.1f}%"
        else:
            supported = None
            key_metric = ""
        
        if supported is None:
            status = "N/A"
        elif supported:
            status = "✓ SUPPORTED"
        else:
            status = "✗ NOT SUPPORTED"
        
        print(f"{h_id:<12} {status:<15} {key_metric:<40}")
    
    print()


async def main():
    parser = argparse.ArgumentParser(description="Run all hypothesis evaluations")
    parser.add_argument("--compute-only", action="store_true",
                       help="Only compute metrics (skip prediction generation)")
    parser.add_argument("--generate-only", action="store_true",
                       help="Only generate predictions (skip metrics computation)")
    parser.add_argument("--mock", action="store_true",
                       help="Use mock runner for testing")
    parser.add_argument("--resume", action="store_true", default=True,
                       help="Resume from existing runs (default: True)")
    parser.add_argument("--hypotheses", type=str, default="1,2,3,4,5",
                       help="Comma-separated list of hypotheses to run (default: 1,2,3,4,5)")
    
    args = parser.parse_args()
    
    # Parse hypothesis list
    hypotheses = [f"H{h.strip()}" for h in args.hypotheses.split(",")]
    
    print_header(f"Hypothesis Evaluation - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Running hypotheses: {', '.join(hypotheses)}")
    print(f"Mode: {'Mock' if args.mock else 'Production'}")
    print(f"Compute only: {args.compute_only}")
    print(f"Generate only: {args.generate_only}")
    
    results = {}
    
    # =========================================================================
    # GENERATION PHASE
    # =========================================================================
    if not args.compute_only:
        print_header("PHASE 1: Generating Predictions", "-", 80)
        
        if "H1" in hypotheses:
            print_subheader("H1: Tools vs No-Tools")
            await run_h1_all(use_mock=args.mock, resume=args.resume)
        
        if "H2" in hypotheses:
            print_subheader("H2: Router Configurations")
            await run_h2_all(use_mock=args.mock, resume=args.resume)
        
        if "H3" in hypotheses:
            print_subheader("H3: Ambiguity Analysis")
            await run_h3_all(use_mock=args.mock)
        
        if "H4" in hypotheses:
            print_subheader("H4: Single-Call vs Multi-Call")
            await run_h4_all(use_mock=args.mock)
        
        # H5 reuses H1 runs, no generation needed
    
    # =========================================================================
    # METRICS COMPUTATION PHASE
    # =========================================================================
    if not args.generate_only:
        print_header("PHASE 2: Computing Metrics", "-", 80)
        
        if "H1" in hypotheses:
            print_subheader("H1: Tools vs No-Tools Metrics")
            try:
                results["H1"] = compute_h1_metrics()
            except Exception as e:
                print(f"Error computing H1 metrics: {e}")
                results["H1"] = None
        
        if "H2" in hypotheses:
            print_subheader("H2: Router Comparison Metrics")
            try:
                results["H2"] = compute_h2_metrics()
            except Exception as e:
                print(f"Error computing H2 metrics: {e}")
                results["H2"] = None
        
        if "H3" in hypotheses:
            print_subheader("H3: Ambiguity Analysis Metrics")
            try:
                results["H3"] = compute_h3_metrics()
            except Exception as e:
                print(f"Error computing H3 metrics: {e}")
                results["H3"] = None
        
        if "H4" in hypotheses:
            print_subheader("H4: Single-Call vs Multi-Call Metrics")
            try:
                results["H4"] = compute_h4_metrics()
            except Exception as e:
                print(f"Error computing H4 metrics: {e}")
                results["H4"] = None
        
        if "H5" in hypotheses:
            print_subheader("H5: Model Size Diminishing Returns")
            try:
                results["H5"] = compute_h5_metrics()
            except Exception as e:
                print(f"Error computing H5 metrics: {e}")
                results["H5"] = None
        
        # Print final summary
        print_final_summary(results)
    
    print_header("Done!", "=", 80)
    return results


if __name__ == "__main__":
    asyncio.run(main())

