"""
H4: Multiple tool calls help

Claim: Allowing multiple tool calls yields better QWK than restricting the 
model to a single tool call.

Test: Compare multi-call vs single-call settings using the same model 
and Policy router.

Configurations:
- Multi-call: Reuses H1 with-tools run (unlimited tool calls)
- Single-call: Exactly one tool invocation per claim (max_tool_calls=1)

This script only runs single-call mode. Multi-call baseline comes from H1.
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from benchmarks.common.runners import run_kani_on_claim


def load_claims():
    """Load claims from drop4_claims.json."""
    claims_path = Path(__file__).parent.parent / "data" / "drop4_claims.json"
    with open(claims_path) as f:
        return json.load(f)


def check_h1_exists(model_name: str) -> bool:
    """Check if H1 with-tools run exists (used as multi-call baseline)."""
    h1_path = Path(__file__).parent.parent / "h1" / "runs" / f"h1_{model_name}_with-tools_policy.json"
    return h1_path.exists()


async def run_h4_single_call(
    claims: list,
    model_name: str,
    output_path: Path,
    seed: int = 42,
) -> dict:
    """
    Run H4 single-call configuration (max_tool_calls=1).
    
    Args:
        claims: List of claim dictionaries
        model_name: Model to use (e.g., "gpt-4o")
        output_path: Path to save results (saves progress after each claim)
        seed: Random seed for reproducibility
    
    Returns:
        Run data dictionary with all predictions
    """
    print(f"\n{'='*60}")
    print(f"H4: Running single-call mode with {model_name}")
    print(f"    (Model is limited to 1 successful tool call per claim)")
    print(f"{'='*60}")
    
    examples = []
    
    def save_progress(status: str = "in_progress"):
        """Save current progress to file."""
        run_data = {
            "hypothesis": "H4",
            "description": "Single-call mode (max_tool_calls=1)",
            "status": status,
            "run_metadata": {
                "model": model_name,
                "router_type": "policy",
                "tools_enabled": True,
                "call_mode": "single",
                "max_tool_calls": 1,
                "seed": seed,
                "timestamp": datetime.now().isoformat(),
                "completed": len(examples),
                "total": len(claims),
            },
            "examples": examples,
        }
        with open(output_path, 'w') as f:
            json.dump(run_data, f, indent=2, default=str)
    
    for i, claim in enumerate(claims):
        print(f"\n[{i+1}/{len(claims)}] {claim['id'][:40]}...", end=" ", flush=True)
        
        try:
            pred = await run_kani_on_claim(
                claim=claim,
                model_name=model_name,
                router_type="policy",
                tools_enabled=True,
                max_tool_calls=1,  # Key H4 parameter: limit to 1 successful call
                seed=seed,
            )
            
            # Count successful vs failed tool calls
            successful_calls = sum(1 for tc in (pred.tool_calls or []) if tc.error is None)
            failed_calls = sum(1 for tc in (pred.tool_calls or []) if tc.error is not None)
            
            example = {
                "claim_id": claim["id"],
                "gold_label": claim["gold_label"],
                "predicted_label": pred.predicted_label,
                "reasoning": pred.reasoning,
                "claim_text": pred.claim_text,
                "num_tool_calls": len(pred.tool_calls or []),
                "successful_tool_calls": successful_calls,
                "failed_tool_calls": failed_calls,
                "hit_tool_limit": pred.extra_metadata.get("hit_tool_limit", False),
                "tool_calls": [
                    {
                        "tool_name": tc.tool_name,
                        "family": tc.family,
                        "arguments": tc.arguments,
                        "result": tc.result,
                        "raw_result": tc.raw_result,
                        "error": tc.error,
                    }
                    for tc in (pred.tool_calls or [])
                ],
            }
            
            examples.append(example)
            
            status = "✓" if pred.predicted_label == claim["gold_label"] else "✗"
            n_tools = len(pred.tool_calls or [])
            print(f"pred={pred.predicted_label}, gold={claim['gold_label']} {status} (tools={n_tools}, success={successful_calls})")
            
        except Exception as e:
            print(f"ERROR: {e}")
            examples.append({
                "claim_id": claim["id"],
                "gold_label": claim["gold_label"],
                "predicted_label": 0,
                "error": str(e),
                "success": False,
                "num_tool_calls": 0,
                "successful_tool_calls": 0,
                "failed_tool_calls": 0,
            })
        
        # Save progress after each claim
        save_progress("in_progress")
    
    # Final save with completed status
    save_progress("completed")
    
    return {
        "hypothesis": "H4",
        "description": "Single-call mode (max_tool_calls=1)",
        "status": "completed",
        "run_metadata": {
            "model": model_name,
            "router_type": "policy",
            "tools_enabled": True,
            "call_mode": "single",
            "max_tool_calls": 1,
            "seed": seed,
            "timestamp": datetime.now().isoformat(),
        },
        "examples": examples,
    }


async def run_h4_benchmark(
    model_name: str = "gpt-4o",
    seed: int = 42,
):
    """
    Run the H4 benchmark: single-call mode only.
    Multi-call baseline is taken from H1.
    
    Args:
        model_name: Model to use
        seed: Random seed
    """
    # Check H1 exists
    if not check_h1_exists(model_name):
        print(f"⚠️  Warning: H1 with-tools run not found for {model_name}")
        print(f"   Run H1 first: python benchmarks/h1/run_h1.py --model-name {model_name} --tools-enabled true")
        print(f"   H4 metrics will use H1 as the multi-call baseline.")
        print()
    
    claims = load_claims()
    print(f"Loaded {len(claims)} claims")
    
    runs_dir = Path(__file__).parent / "runs"
    runs_dir.mkdir(exist_ok=True)
    
    single_path = runs_dir / f"h4_{model_name}_single-call.json"
    
    # Run single-call configuration only (saves progress after each claim)
    single_data = await run_h4_single_call(
        claims=claims,
        model_name=model_name,
        output_path=single_path,
        seed=seed,
    )
    
    print(f"\n💾 Saved single-call run to {single_path}")
    
    # Quick summary
    single_correct = sum(1 for ex in single_data["examples"] 
                        if ex.get("predicted_label") == ex.get("gold_label"))
    single_avg_tools = sum(ex.get("num_tool_calls", 0) for ex in single_data["examples"]) / len(claims)
    single_avg_success = sum(ex.get("successful_tool_calls", 0) for ex in single_data["examples"]) / len(claims)
    
    print(f"\n{'='*60}")
    print("H4 QUICK SUMMARY")
    print(f"{'='*60}")
    print(f"Single-call: {single_correct}/{len(claims)} correct ({single_correct/len(claims)*100:.1f}%)")
    print(f"             avg {single_avg_tools:.2f} total calls, {single_avg_success:.2f} successful")
    print(f"\nNote: Multi-call baseline comes from H1 with-tools run.")
    print(f"      Run compute_metrics_h4.py to see full comparison.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run H4 benchmark (single-call mode)")
    parser.add_argument("--model", default="gpt-4o", help="Model to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    asyncio.run(run_h4_benchmark(
        model_name=args.model,
        seed=args.seed,
    ))
