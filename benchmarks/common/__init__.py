"""Common utilities for the LAMINA Evaluation Framework."""

from .metrics import (
    compute_qwk,
    compute_accuracy,
    compute_mae,
    flip_rate,
    quadratic_weighted_flip_rate,
    aggregate_fr_qwfr,
    bootstrap_kappa_diff_ci,
    bootstrap_gap_shrinkage_ci,
)
from .dataset import load_claims, get_crisp_claims, get_ambiguous_claims
from .runners import ToolCallRecord, PredictionRecord, run_kani_on_claim
from .logging_utils import save_run_json, load_run_json, prediction_to_dict

__all__ = [
    # Metrics
    "compute_qwk",
    "compute_accuracy", 
    "compute_mae",
    "flip_rate",
    "quadratic_weighted_flip_rate",
    "aggregate_fr_qwfr",
    "bootstrap_kappa_diff_ci",
    "bootstrap_gap_shrinkage_ci",
    # Dataset
    "load_claims",
    "get_crisp_claims",
    "get_ambiguous_claims",
    # Runners
    "ToolCallRecord",
    "PredictionRecord",
    "run_kani_on_claim",
    # Logging
    "save_run_json",
    "load_run_json",
    "prediction_to_dict",
]

