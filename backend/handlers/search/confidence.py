"""
Confidence Calculation

This module handles confidence score calculation based on result quality and diversity.
"""

import logging
from typing import Any, Dict, List

from ..shared import Confidence

_log = logging.getLogger(__name__)


def calculate_confidence(
    ranked_results: List[Dict[str, Any]],
    enriched_docs: List[Dict[str, Any]]
) -> str:
    """
    Calculate confidence score based on result quality and diversity.
    
    Args:
        ranked_results: List of ranked results
        enriched_docs: List of enriched documents with extracted content
        
    Returns:
        Confidence enum value (HIGH, MEDIUM, or LOW)
    """
    if not ranked_results:
        return Confidence.LOW
    
    # Count unique domains
    unique_domains = len(set(r.get("source_domain", "") for r in ranked_results if r.get("source_domain")))
    
    # Count academic sources
    num_academic = sum(1 for r in ranked_results if r.get("is_academic", False))
    
    # Count successful extractions
    num_extractions = len(enriched_docs)
    
    # HIGH confidence: ≥3 results from ≥2 domains, with at least 1 successful extraction
    if len(ranked_results) >= 3 and unique_domains >= 2 and num_extractions >= 1:
        return Confidence.HIGH
    
    # MEDIUM confidence: ≥2 results or academic sources present
    if len(ranked_results) >= 2 or num_academic >= 1:
        return Confidence.MEDIUM
    
    # LOW confidence: minimal results
    return Confidence.LOW

