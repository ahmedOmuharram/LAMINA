#!/usr/bin/env python3
"""
H1: Tools improve agreement

Run script to generate predictions with/without tools for each model.

Usage:
    python run_h1.py --model-name gpt-4o --tools-enabled true --output runs/h1_gpt-4o_with-tools_policy.json
    python run_h1.py --model-name gpt-4o-mini --tools-enabled false --output runs/h1_gpt-4o-mini_no-tools_policy.json
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.dataset import load_claims
from common.logging_utils import save_run_json, load_run_json, dict_to_prediction
from common.runners import PredictionRecord, run_kani_on_claim, mock_run_claim


async def run_h1(
    model_name: str,
    tools_enabled: bool,
    router_type: str = "policy",
    output_path: Path = None,
    seed: int = 42,
    use_mock: bool = False,
    resume: bool = False,
) -> list[PredictionRecord]:
    """
    Run H1 evaluation for a single configuration.
    
    Args:
        model_name: LLM model to use
        tools_enabled: Whether to enable tools
        router_type: Router type (always "policy" for H1)
        output_path: Path to save results
        seed: Random seed
        use_mock: Whether to use mock runner (for testing)
        
    Returns:
        List of PredictionRecord objects
    """
    claims = load_claims()
    predictions: list[PredictionRecord] = []
    completed_ids: set[str] = set()

    tools_str = "with-tools" if tools_enabled else "no-tools"
    run_id = f"h1_{model_name}_{tools_str}_{router_type}"
    
    print(f"Running H1: model={model_name}, tools_enabled={tools_enabled}")
    print(f"Processing {len(claims)} claims...")
    
    # If resume requested and output exists, load existing predictions
    if resume and output_path and output_path.exists():
        existing = load_run_json(output_path)
        for ex in existing.get("examples", []):
            pred = dict_to_prediction(ex)
            predictions.append(pred)
            completed_ids.add(pred.claim_id)
        if completed_ids:
            print(f"Resuming H1 from {len(completed_ids)} completed examples.")
    
    for i, claim in enumerate(claims):
        claim_id = claim["id"]
        if claim_id in completed_ids:
            print(f"  [{i+1}/{len(claims)}] {claim_id[:30]}... (skipping, already completed)")
            continue
        
        print(f"  [{i+1}/{len(claims)}] {claim_id[:30]}...")
        
        if use_mock:
            pred = mock_run_claim(
                claim=claim,
                model_name=model_name,
                router_type=router_type,
                tools_enabled=tools_enabled,
                multi_source=True,
                seed=seed + i,
            )
        else:
            pred = await run_kani_on_claim(
                claim=claim,
                model_name=model_name,
                router_type=router_type,
                tools_enabled=tools_enabled,
                multi_source=True,
                seed=seed,
            )
        
        predictions.append(pred)
        
        # Save incrementally after each claim so progress is visible in real-time
        if output_path:
            save_run_json(
                output_path=output_path,
                predictions=predictions,
                hypothesis="H1",
                run_id=run_id,
                model_name=model_name,
                tools_enabled=tools_enabled,
                router_type=router_type,
                multi_source=True,
                seed=seed,
                notes=f"H1: tools vs no-tools comparison with {router_type} router (in progress: {i+1}/{len(claims)})",
            )
        
        print(f"    -> Predicted: {pred.predicted_label}, Gold: {pred.gold_label}, Tools: {len(pred.tool_calls)}")
    
    # Final save with completed status
    if output_path:
        save_run_json(
            output_path=output_path,
            predictions=predictions,
            hypothesis="H1",
            run_id=run_id,
            model_name=model_name,
            tools_enabled=tools_enabled,
            router_type=router_type,
            multi_source=True,
            seed=seed,
            notes=f"H1: tools vs no-tools comparison with {router_type} router",
        )
        print(f"Saved final results to {output_path}")
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Run H1 evaluation")
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        choices=["gpt-4o-mini", "gpt-4o", "gpt-5.1", "gpt-4.1"],
        help="LLM model to use",
    )
    parser.add_argument(
        "--tools-enabled",
        type=str,
        required=True,
        choices=["true", "false"],
        help="Whether to enable tools",
    )
    parser.add_argument(
        "--router",
        type=str,
        default="policy",
        help="Router type (always 'policy' for H1)",
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file if present (skip completed claims)",
    )
    
    args = parser.parse_args()
    
    tools_enabled = args.tools_enabled.lower() == "true"
    output_path = Path(args.output)
    
    asyncio.run(run_h1(
        model_name=args.model_name,
        tools_enabled=tools_enabled,
        router_type=args.router,
        output_path=output_path,
        seed=args.seed,
        use_mock=args.mock,
        resume=args.resume,
    ))


if __name__ == "__main__":
    main()

