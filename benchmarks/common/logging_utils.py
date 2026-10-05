"""
Logging utilities for LAMINA Evaluation Framework.

Handles JSON serialization and file I/O for run results.
"""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .runners import PredictionRecord, ToolCallRecord


def tool_call_to_dict(tc: ToolCallRecord) -> Dict[str, Any]:
    """Convert ToolCallRecord to dictionary."""
    return {
        "tool_name": tc.tool_name,
        "family": tc.family,
        "library": tc.library,
        "arguments": tc.arguments,
        "result": tc.result,
        "raw_result": tc.raw_result,
        "error": tc.error,
        "latency_ms": tc.latency_ms,
    }


def prediction_to_dict(pred: PredictionRecord) -> Dict[str, Any]:
    """Convert PredictionRecord to dictionary."""
    return {
        "claim_id": pred.claim_id,
        "claim_text": pred.claim_text,
        "gold_label": pred.gold_label,
        "predicted_label": pred.predicted_label,
        "reasoning": pred.reasoning,
        "raw_response": pred.raw_response,
        "mapping_id": pred.mapping_id,
        "mapping_description": pred.mapping_description,
        "router_type": pred.router_type,
        "tools_enabled": pred.tools_enabled,
        "multi_source": pred.multi_source,
        "model_name": pred.model_name,
        "extra_metadata": pred.extra_metadata,
        "tool_calls": [tool_call_to_dict(tc) for tc in pred.tool_calls],
    }


def dict_to_tool_call(data: Dict[str, Any]) -> ToolCallRecord:
    """Convert a stored tool call dictionary back into a ToolCallRecord."""
    return ToolCallRecord(
        tool_name=data.get("tool_name", ""),
        family=data.get("family"),
        library=data.get("library"),
        arguments=data.get("arguments", {}),
        result=data.get("result"),
        raw_result=data.get("raw_result"),
        error=data.get("error"),
        latency_ms=data.get("latency_ms", 0.0),
    )


def dict_to_prediction(data: Dict[str, Any]) -> PredictionRecord:
    """Convert a stored example dictionary back into a PredictionRecord."""
    return PredictionRecord(
        claim_id=data.get("claim_id", ""),
        claim_text=data.get("claim_text", ""),
        gold_label=data.get("gold_label", 0),
        predicted_label=data.get("predicted_label", 0),
        reasoning=data.get("reasoning", ""),
        raw_response=data.get("raw_response", ""),
        mapping_id=data.get("mapping_id"),
        mapping_description=data.get("mapping_description"),
        router_type=data.get("router_type", ""),
        tools_enabled=data.get("tools_enabled", False),
        multi_source=data.get("multi_source", True),
        model_name=data.get("model_name", ""),
        extra_metadata=data.get("extra_metadata", {}),
        tool_calls=[dict_to_tool_call(tc) for tc in data.get("tool_calls", [])],
    )


def create_run_metadata(
    hypothesis: str,
    run_id: str,
    model_name: str,
    tools_enabled: bool,
    router_type: str,
    multi_source: bool = True,
    seed: int = 42,
    notes: str = "",
) -> Dict[str, Any]:
    """Create metadata dictionary for a run."""
    return {
        "hypothesis": hypothesis,
        "run_id": run_id,
        "model_name": model_name,
        "tools_enabled": tools_enabled,
        "router_type": router_type,
        "multi_source": multi_source,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "seed": seed,
        "drop4_version": "v1",
        "notes": notes,
    }


def save_run_json(
    output_path: Path,
    predictions: List[PredictionRecord],
    hypothesis: str,
    run_id: str,
    model_name: str,
    tools_enabled: bool,
    router_type: str,
    multi_source: bool = True,
    seed: int = 42,
    notes: str = "",
) -> None:
    """
    Save run results to JSON file.
    
    Args:
        output_path: Path to output JSON file
        predictions: List of PredictionRecord objects
        hypothesis: Hypothesis identifier (e.g., "H1")
        run_id: Unique run identifier
        model_name: LLM model used
        tools_enabled: Whether tools were enabled
        router_type: Router configuration used
        multi_source: Whether multi-source was enabled
        seed: Random seed used
        notes: Additional notes about the run
    """
    run_data = {
        "run_metadata": create_run_metadata(
            hypothesis=hypothesis,
            run_id=run_id,
            model_name=model_name,
            tools_enabled=tools_enabled,
            router_type=router_type,
            multi_source=multi_source,
            seed=seed,
            notes=notes,
        ),
        "examples": [prediction_to_dict(p) for p in predictions],
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(run_data, f, indent=2)


def load_run_json(input_path: Path) -> Dict[str, Any]:
    """
    Load run results from JSON file.
    
    Args:
        input_path: Path to input JSON file
        
    Returns:
        Dictionary with run_metadata and examples
    """
    with open(input_path, "r") as f:
        return json.load(f)


def extract_labels_from_run(run_data: Dict[str, Any]) -> tuple[List[int], List[int]]:
    """
    Extract gold and predicted labels from run data.
    
    Args:
        run_data: Dictionary loaded from run JSON
        
    Returns:
        Tuple of (gold_labels, predicted_labels)
    """
    examples = run_data.get("examples", [])
    gold_labels = [ex["gold_label"] for ex in examples]
    pred_labels = [ex["predicted_label"] for ex in examples]
    return gold_labels, pred_labels


def compute_run_stats(run_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute summary statistics for a run.
    
    Args:
        run_data: Dictionary loaded from run JSON
        
    Returns:
        Dictionary with summary statistics
    """
    examples = run_data.get("examples", [])
    
    if not examples:
        return {}
    
    # Basic counts
    n_examples = len(examples)
    n_correct = sum(
        1 for ex in examples 
        if ex["gold_label"] == ex["predicted_label"]
    )
    
    # Tool usage
    total_tool_calls = sum(len(ex.get("tool_calls", [])) for ex in examples)
    examples_with_tools = sum(
        1 for ex in examples 
        if len(ex.get("tool_calls", [])) > 0
    )
    
    # Latencies
    tool_latencies = []
    for ex in examples:
        for tc in ex.get("tool_calls", []):
            if tc.get("latency_ms"):
                tool_latencies.append(tc["latency_ms"])
    
    avg_tool_latency = sum(tool_latencies) / len(tool_latencies) if tool_latencies else 0
    
    # Per-example total latencies
    total_latencies = [
        ex.get("extra_metadata", {}).get("total_latency_ms", 0)
        for ex in examples
    ]
    avg_total_latency = sum(total_latencies) / len(total_latencies) if total_latencies else 0
    
    return {
        "n_examples": n_examples,
        "n_correct": n_correct,
        "accuracy": n_correct / n_examples if n_examples > 0 else 0,
        "total_tool_calls": total_tool_calls,
        "avg_tool_calls_per_example": total_tool_calls / n_examples if n_examples > 0 else 0,
        "examples_with_tools": examples_with_tools,
        "pct_examples_with_tools": examples_with_tools / n_examples if n_examples > 0 else 0,
        "avg_tool_latency_ms": avg_tool_latency,
        "avg_total_latency_ms": avg_total_latency,
    }


def save_summary_json(
    output_path: Path,
    summary_data: Dict[str, Any],
) -> None:
    """
    Save summary metrics to JSON file.
    
    Args:
        output_path: Path to output JSON file
        summary_data: Dictionary with summary metrics
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(summary_data, f, indent=2)


def load_summary_json(input_path: Path) -> Dict[str, Any]:
    """
    Load summary metrics from JSON file.
    
    Args:
        input_path: Path to input JSON file
        
    Returns:
        Dictionary with summary metrics
    """
    with open(input_path, "r") as f:
        return json.load(f)

