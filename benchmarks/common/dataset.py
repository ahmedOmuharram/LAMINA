"""
Dataset loading utilities for LAMINA Evaluation Framework.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# Default paths relative to benchmarks directory
DATA_DIR = Path(__file__).parent.parent / "data"
CLAIMS_FILE = DATA_DIR / "drop4_claims.json"
METADATA_FILE = DATA_DIR / "drop4_metadata.json"


def load_claims(claims_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Load all claims from the Drop-4 dataset.
    
    Args:
        claims_path: Optional path to claims JSON file
        
    Returns:
        List of claim dictionaries
    """
    path = claims_path or CLAIMS_FILE
    with open(path, "r") as f:
        return json.load(f)


def load_metadata(metadata_path: Optional[Path] = None) -> Dict[str, Dict[str, str]]:
    """
    Load claim metadata (required tool families, etc).
    
    Args:
        metadata_path: Optional path to metadata JSON file
        
    Returns:
        Dictionary mapping claim_id to metadata
    """
    path = metadata_path or METADATA_FILE
    with open(path, "r") as f:
        return json.load(f)


def get_crisp_claims(claims: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """
    Get only crisp claims (explicit numbers, no vague quantifiers).
    
    Args:
        claims: Optional list of claims (loads from file if not provided)
        
    Returns:
        List of crisp claim dictionaries
    """
    if claims is None:
        claims = load_claims()
    
    return [c for c in claims if c.get("group") == "crisp"]


def get_ambiguous_claims(claims: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """
    Get only ambiguous claims (vague quantifiers, approximations).
    
    Args:
        claims: Optional list of claims (loads from file if not provided)
        
    Returns:
        List of ambiguous claim dictionaries
    """
    if claims is None:
        claims = load_claims()
    
    return [c for c in claims if c.get("group") == "ambiguous"]


def get_claim_by_id(claim_id: str, claims: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    """
    Get a specific claim by ID.
    
    Args:
        claim_id: The claim identifier
        claims: Optional list of claims (loads from file if not provided)
        
    Returns:
        Claim dictionary or None if not found
    """
    if claims is None:
        claims = load_claims()
    
    for c in claims:
        if c.get("id") == claim_id:
            return c
    
    return None


def get_gold_labels(claims: Optional[List[Dict[str, Any]]] = None) -> List[int]:
    """
    Extract gold labels from claims in order.
    
    Args:
        claims: Optional list of claims (loads from file if not provided)
        
    Returns:
        List of gold labels
    """
    if claims is None:
        claims = load_claims()
    
    return [c["gold_label"] for c in claims]


def get_claim_ids(claims: Optional[List[Dict[str, Any]]] = None) -> List[str]:
    """
    Extract claim IDs in order.
    
    Args:
        claims: Optional list of claims (loads from file if not provided)
        
    Returns:
        List of claim IDs
    """
    if claims is None:
        claims = load_claims()
    
    return [c["id"] for c in claims]


def split_claims_by_group(
    claims: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Split claims into crisp and ambiguous groups.
    
    Args:
        claims: Optional list of claims (loads from file if not provided)
        
    Returns:
        Dictionary with 'crisp' and 'ambiguous' keys
    """
    if claims is None:
        claims = load_claims()
    
    return {
        "crisp": get_crisp_claims(claims),
        "ambiguous": get_ambiguous_claims(claims),
    }

