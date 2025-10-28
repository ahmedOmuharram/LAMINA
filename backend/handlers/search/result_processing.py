"""
Result Processing

This module handles normalization, deduplication, and clustering of search results.
"""

import logging
from typing import Any, Dict, List
from urllib.parse import urlparse
from collections import defaultdict

_log = logging.getLogger(__name__)


def normalize_results(raw_results: List[Dict[str, Any]], query: str = "") -> List[Dict[str, Any]]:
    """
    Normalize raw SearXNG results into a standardized schema.
    
    Args:
        raw_results: Raw results from SearXNG
        query: Original search query (for context)
        
    Returns:
        List of normalized result dictionaries
    """
    normalized = []
    for r in raw_results:
        url = r.get("url", "")
        parsed_url = urlparse(url) if url else None
        
        # Determine if academic source
        engine = (r.get("engine") or "").lower()
        is_academic = any(k in engine for k in ["arxiv", "pubmed", "semantic scholar", "scholar"])
        
        # Extract published date
        published_at = r.get("publishedDate") or r.get("published_at") or r.get("pubdate")
        
        # Determine content type
        content_type = r.get("mime") or r.get("content_type") or "text/html"
        
        normalized.append({
            "title": r.get("title") or r.get("title_html") or "",
            "url": url,
            "score": r.get("score", 0),
            "snippet": r.get("content") or r.get("snippet") or r.get("abstract", ""),
            "published_at": published_at,
            "source_domain": parsed_url.netloc if parsed_url else "",
            "engine": r.get("engine"),
            "is_academic": is_academic,
            "is_primary_source": False,  # TODO: Enhance with domain heuristics
            "content_type": content_type,
            "original_result": r  # Keep original for reference
        })
    
    return normalized


def cluster_by_canonical_url(normalized_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Cluster results by canonical URL to eliminate duplicates.
    
    Groups results that point to the same content (e.g., same paper on arXiv + mirrors).
    
    Args:
        normalized_results: List of normalized results
        
    Returns:
        List of clustered results with representative entries
    """
    clusters = defaultdict(list)
    
    for r in normalized_results:
        url = r.get("url", "")
        if not url:
            continue
        
        parsed = urlparse(url)
        # Create canonical key: lowercase host + path without query params
        path = parsed.path.split("?")[0].rstrip("/")
        canonical_key = (parsed.netloc.lower(), path)
        
        clusters[canonical_key].append(r)
    
    clustered = []
    for canonical_key, group in clusters.items():
        # Sort by score and pick the highest as representative
        group_sorted = sorted(group, key=lambda x: x.get("score", 0), reverse=True)
        rep = group_sorted[0].copy()
        
        # Add metadata about clustering
        rep["all_variants"] = [g["url"] for g in group_sorted]
        rep["num_variants"] = len(group_sorted)
        rep["is_clustered"] = len(group_sorted) > 1
        
        # Aggregate is_academic (if any variant is academic, mark as academic)
        rep["is_academic"] = any(g.get("is_academic", False) for g in group_sorted)
        
        clustered.append(rep)
    
    return clustered

