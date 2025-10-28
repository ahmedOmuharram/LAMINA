"""
Ranking and Re-ranking

This module handles advanced re-ranking of search results with authority signals.
"""

import logging
from typing import Any, Dict, List, Optional

_log = logging.getLogger(__name__)


def calculate_domain_authority(domain: str) -> float:
    """
    Calculate domain authority multiplier.
    
    Args:
        domain: Domain name (e.g., "arxiv.org")
        
    Returns:
        Authority multiplier (1.0 = neutral, >1.0 = boost)
    """
    domain_lower = domain.lower()
    
    # Government and educational domains
    if domain_lower.endswith(".gov") or domain_lower.endswith(".edu"):
        return 2.5
    
    # High-authority academic sources
    if any(d in domain_lower for d in ["arxiv.org", "nist.gov", "nature.com", "science.org", "ieee.org"]):
        return 2.0
    
    # Academic publishers
    if any(d in domain_lower for d in ["springer.com", "sciencedirect.com", "wiley.com", "acs.org"]):
        return 1.5
    
    # Wikipedia and reputable encyclopedias
    if "wikipedia.org" in domain_lower:
        return 1.3
    
    return 1.0


def calculate_recency_boost(published_at: Optional[str], time_range: str = "") -> float:
    """
    Calculate recency boost multiplier.
    
    Args:
        published_at: Publication date string
        time_range: Time range filter from query (day, week, month, year)
        
    Returns:
        Recency boost multiplier
    """
    if not published_at or not time_range:
        return 1.0
    
    # Simple boost for recent time ranges
    boost_map = {
        "day": 1.5,
        "week": 1.3,
        "month": 1.2,
        "year": 1.1
    }
    
    return boost_map.get(time_range.lower(), 1.0)


def rerank_results(clustered_results: List[Dict[str, Any]], time_range: str = "") -> List[Dict[str, Any]]:
    """
    Re-rank clustered results using authority signals and recency.
    
    Args:
        clustered_results: List of clustered results
        time_range: Time range filter from original query
        
    Returns:
        Re-ranked list of results
    """
    reranked = []
    
    for r in clustered_results:
        base_score = r.get("score", 0)
        domain = r.get("source_domain", "")
        published_at = r.get("published_at")
        
        # Calculate multipliers
        authority_mult = calculate_domain_authority(domain)
        recency_mult = calculate_recency_boost(published_at, time_range)
        
        # Boost for academic sources
        academic_boost = 1.3 if r.get("is_academic", False) else 1.0
        
        # Boost for clustered results (multiple sources pointing to same content)
        cluster_boost = 1.1 if r.get("num_variants", 1) > 1 else 1.0
        
        # Calculate final score
        final_score = base_score * authority_mult * recency_mult * academic_boost * cluster_boost
        
        rr = r.copy()
        rr["final_score"] = final_score
        rr["base_score"] = base_score
        rr["authority_mult"] = authority_mult
        rr["academic_boost"] = academic_boost
        reranked.append(rr)
    
    # Sort by final score
    return sorted(reranked, key=lambda x: x.get("final_score", 0), reverse=True)

