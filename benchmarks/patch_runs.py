#!/usr/bin/env python3
"""
Patch existing benchmark runs by re-running only affected claims.

This is much faster than re-running entire hypotheses when only some tools were buggy.
"""

import asyncio
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.common.runners import run_kani_on_claim
from benchmarks.common.logging_utils import load_run_json

# Claims affected by bugs we fixed
AFFECTED_CLAIMS = {
    # Solute handler bugs (compare_solute_lattice_effects, analyze_solute_lattice_effect)
    "al_mg_lattice_param",
    "al_cu_lattice_param_moderate",
    "al_cu_modulus_increase",
    "al_cu_modulus_decrease",
    "al_mg_modulus_increase",
    
    # SiC phase alias (SIC -> CSI)
    "alsic_carbide_dissolve_1500k",
    "alsic_precipitates_temp",
    
    # Sweep coroutine bug + composition parsing
    "almgzn_tau_phase",
    "almgzn_tau_phase_zn_addition",
    "feal_65at_strength",
}


def load_claims() -> Dict[str, dict]:
    """Load claims from drop4_claims.json."""
    claims_path = Path(__file__).parent / "data" / "drop4_claims.json"
    with open(claims_path) as f:
        data = json.load(f)
    # Handle both list format and dict with "claims" key
    if isinstance(data, list):
        return {c["id"]: c for c in data}
    else:
        return {c["id"]: c for c in data["claims"]}


def identify_claims_to_patch(run_data: dict, affected_ids: set) -> List[str]:
    """Identify which claims in a run need patching."""
    claims_to_patch = []
    
    for ex in run_data.get("examples", []):
        claim_id = ex.get("claim_id")
        if claim_id in affected_ids:
            claims_to_patch.append(claim_id)
    
    return claims_to_patch


async def patch_run_file(
    run_file: Path,
    affected_ids: set,
    dry_run: bool = False,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Patch a single run file by re-running affected claims.
    
    Returns summary of changes made.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {run_file.name}")
    print(f"{'='*60}")
    
    # Load existing run
    run_data = load_run_json(str(run_file))
    
    # Extract run config from run_metadata
    metadata = run_data.get("run_metadata", {})
    model_name = metadata.get("model_name", "gpt-4o")
    router_type = metadata.get("router_type", "policy")
    tools_enabled = metadata.get("tools_enabled", True)
    
    # Load claims
    claims_lookup = load_claims()
    
    # Find claims to patch
    claims_to_patch = identify_claims_to_patch(run_data, affected_ids)
    
    if not claims_to_patch:
        print(f"  No affected claims found in this run.")
        return {"file": str(run_file), "patched": 0, "claims": []}
    
    print(f"  Found {len(claims_to_patch)} claims to patch: {claims_to_patch}")
    print(f"  Model: {model_name}, Router: {router_type}, Tools: {tools_enabled}")
    
    if dry_run:
        print(f"  [DRY RUN] Would patch these claims")
        return {"file": str(run_file), "patched": 0, "claims": claims_to_patch, "dry_run": True}
    
    # Create index of examples by claim_id
    examples_by_id = {}
    for i, ex in enumerate(run_data["examples"]):
        examples_by_id[ex["claim_id"]] = i
    
    # Patch each affected claim
    patched = []
    for claim_id in claims_to_patch:
        if claim_id not in claims_lookup:
            print(f"  ⚠ Claim {claim_id} not found in claims file, skipping")
            continue
        
        claim = claims_lookup[claim_id]
        print(f"  🔄 Re-running: {claim_id}...", end=" ", flush=True)
        
        try:
            # Re-run the claim
            pred = await run_kani_on_claim(
                claim=claim,
                model_name=model_name,
                router_type=router_type,
                tools_enabled=tools_enabled,
                multi_source=True,
                mapping_override=None,
                seed=seed
            )
            
            # Update the example in run_data
            idx = examples_by_id[claim_id]
            old_pred = run_data["examples"][idx].get("predicted_label")
            gold = run_data["examples"][idx].get("gold_label")
            
            run_data["examples"][idx] = {
                "claim_id": claim_id,
                "gold_label": gold,
                "predicted_label": pred.predicted_label,
                "reasoning": pred.reasoning,
                "tool_calls": [
                    {
                        "tool_name": tc.tool_name,
                        "arguments": tc.arguments,
                        "result": tc.result
                    }
                    for tc in (pred.tool_calls or [])
                ],
                "patched_at": datetime.now().isoformat(),
                "previous_prediction": old_pred
            }
            
            status = "✓" if pred.predicted_label == gold else "✗"
            change = f"({old_pred} → {pred.predicted_label})" if old_pred != pred.predicted_label else "(unchanged)"
            print(f"{status} gold={gold}, pred={pred.predicted_label} {change}")
            
            patched.append({
                "claim_id": claim_id,
                "old": old_pred,
                "new": pred.predicted_label,
                "gold": gold,
                "correct": pred.predicted_label == gold
            })
            
        except Exception as e:
            print(f"❌ Error: {e}")
            patched.append({"claim_id": claim_id, "error": str(e)})
    
    # Save patched run
    if patched:
        # Update metadata
        if "run_metadata" not in run_data:
            run_data["run_metadata"] = {}
        run_data["run_metadata"]["patched_at"] = datetime.now().isoformat()
        run_data["run_metadata"]["patched_claims"] = [p["claim_id"] for p in patched if "error" not in p]
        
        # Save to new file with _patched suffix (direct JSON save since we have full run_data)
        patched_file = run_file.parent / f"{run_file.stem}_patched.json"
        with open(patched_file, 'w') as f:
            json.dump(run_data, f, indent=2, default=str)
        print(f"\n  💾 Saved patched run to: {patched_file.name}")
    
    return {
        "file": str(run_file),
        "patched": len([p for p in patched if "error" not in p]),
        "claims": patched
    }


async def main():
    parser = argparse.ArgumentParser(description="Patch affected claims in benchmark runs")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be patched without running")
    parser.add_argument("--hypothesis", type=str, help="Only patch runs for specific hypothesis (h1, h2, h3, h5)")
    parser.add_argument("--file", type=str, help="Patch a specific run file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    print("=" * 60)
    print("BENCHMARK RUN PATCHER")
    print("=" * 60)
    print(f"Affected claims: {len(AFFECTED_CLAIMS)}")
    for claim in sorted(AFFECTED_CLAIMS):
        print(f"  - {claim}")
    
    # Find run files to patch
    if args.file:
        run_files = [Path(args.file)]
    else:
        run_dirs = []
        if args.hypothesis:
            run_dirs = [Path(f"benchmarks/{args.hypothesis}/runs")]
        else:
            run_dirs = [
                Path("benchmarks/h1/runs"),
                Path("benchmarks/h2/runs"),
                Path("benchmarks/h3/runs"),
                Path("benchmarks/h5/runs"),
            ]
        
        run_files = []
        for run_dir in run_dirs:
            if run_dir.exists():
                for f in run_dir.glob("*.json"):
                    # Skip already patched files and invalid files
                    if "_patched" not in f.name and "invalid" not in f.name:
                        run_files.append(f)
    
    print(f"\nFound {len(run_files)} run files to check")
    
    # Patch each run file
    results = []
    for run_file in sorted(run_files):
        result = await patch_run_file(
            run_file,
            AFFECTED_CLAIMS,
            dry_run=args.dry_run,
            seed=args.seed
        )
        results.append(result)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_patched = sum(r["patched"] for r in results)
    files_with_patches = len([r for r in results if r["patched"] > 0])
    
    print(f"Total claims patched: {total_patched}")
    print(f"Files modified: {files_with_patches}")
    
    if args.dry_run:
        print("\n[DRY RUN] No files were actually modified")


if __name__ == "__main__":
    asyncio.run(main())

