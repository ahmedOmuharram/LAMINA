#!/usr/bin/env python3
"""
H3: Ambiguity induces flips and lowers agreement

Run script to generate predictions for crisp and ambiguous claims with multiple mappings.

Usage:
    python run_h3.py --output runs/h3_gpt-4o_policy.json
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.dataset import load_claims
from common.logging_utils import save_run_json
from common.runners import PredictionRecord, run_kani_on_claim, mock_run_claim


async def run_h3(
    model_name: str = "gpt-4o",
    router_type: str = "policy",
    output_path: Path = None,
    seed: int = 42,
    use_mock: bool = False,
) -> list[PredictionRecord]:
    """
    Run H3 evaluation for ambiguity analysis.
    
    For crisp claims: single run
    For ambiguous claims: one run per mapping
    
    Args:
        model_name: LLM model to use
        router_type: Router type
        output_path: Path to save results
        seed: Random seed
        use_mock: Whether to use mock runner
        
    Returns:
        List of PredictionRecord objects
    """
    claims = load_claims()
    predictions: list[PredictionRecord] = []
    
    run_id = f"h3_{model_name}_{router_type}"
    
    # Calculate total expected runs
    total_runs = sum(
        1 if claim.get("group") == "crisp" or not claim.get("ambiguous_mappings")
        else len(claim.get("ambiguous_mappings", []))
        for claim in claims
    )
    
    print(f"Running H3: model={model_name}, router={router_type}")
    print(f"Processing {len(claims)} claims ({total_runs} total runs)...")
    
    run_count = 0
    
    def save_progress():
        """Save current progress to file."""
        if output_path:
            save_run_json(
                output_path=output_path,
                predictions=predictions,
                hypothesis="H3",
                run_id=run_id,
                model_name=model_name,
                tools_enabled=True,
                router_type=router_type,
                multi_source=True,
                seed=seed,
                notes=f"H3: Ambiguity analysis (in progress: {run_count}/{total_runs})",
            )
    
    for i, claim in enumerate(claims):
        claim_group = claim.get("group", "crisp")
        mappings = claim.get("ambiguous_mappings")
        
        if claim_group == "crisp" or not mappings:
            # Single run for crisp claims
            print(f"  [{i+1}/{len(claims)}] {claim['id'][:30]} (crisp)")
            
            if use_mock:
                pred = mock_run_claim(
                    claim=claim,
                    model_name=model_name,
                    router_type=router_type,
                    tools_enabled=True,
                    multi_source=True,
                    seed=seed + run_count,
                )
            else:
                pred = await run_kani_on_claim(
                    claim=claim,
                    model_name=model_name,
                    router_type=router_type,
                    tools_enabled=True,
                    multi_source=True,
                    seed=seed,
                )
            
            predictions.append(pred)
            run_count += 1
            print(f"    -> Predicted: {pred.predicted_label}, Gold: {pred.gold_label}, Tools: {len(pred.tool_calls)}")
            save_progress()
        else:
            # Multiple runs for ambiguous claims
            print(f"  [{i+1}/{len(claims)}] {claim['id'][:30]} (ambiguous, {len(mappings)} mappings)")
            
            for j, mapping in enumerate(mappings):
                mapping_override = {
                    "mapping_id": mapping["mapping_id"],
                    "description": mapping["description"],
                    "demystified_text": mapping.get("demystified_text"),
                }
                
                if use_mock:
                    pred = mock_run_claim(
                        claim=claim,
                        model_name=model_name,
                        router_type=router_type,
                        tools_enabled=True,
                        multi_source=True,
                        mapping_override=mapping_override,
                        seed=seed + run_count,
                    )
                else:
                    pred = await run_kani_on_claim(
                        claim=claim,
                        model_name=model_name,
                        router_type=router_type,
                        tools_enabled=True,
                        multi_source=True,
                        mapping_override=mapping_override,
                        seed=seed,
                    )
                
                predictions.append(pred)
                run_count += 1
                print(f"    [{j+1}/{len(mappings)}] -> Predicted: {pred.predicted_label}, Gold: {pred.gold_label}")
                save_progress()
    
    print(f"Total runs: {run_count}")
    
    # Final save with completed status
    if output_path:
        save_run_json(
            output_path=output_path,
            predictions=predictions,
            hypothesis="H3",
            run_id=run_id,
            model_name=model_name,
            tools_enabled=True,
            router_type=router_type,
            multi_source=True,
            seed=seed,
            notes="H3: Ambiguity analysis with multiple mappings for ambiguous claims",
        )
        print(f"Saved final results to {output_path}")
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Run H3 evaluation")
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt-4o",
        help="LLM model to use",
    )
    parser.add_argument(
        "--router",
        type=str,
        default="policy",
        help="Router type",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for results JSON",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock runner for testing",
    )
    
    args = parser.parse_args()
    output_path = Path(args.output)
    
    asyncio.run(run_h3(
        model_name=args.model_name,
        router_type=args.router,
        output_path=output_path,
        seed=args.seed,
        use_mock=args.mock,
    ))


if __name__ == "__main__":
    main()

