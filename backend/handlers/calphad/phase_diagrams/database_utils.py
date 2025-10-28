"""
Database utilities for CALPHAD phase diagrams.

Functions for loading databases, handling elements, and database selection.

NOTE: For database loading and element normalization, use the functions from
shared.calphad_utils (find_tdb_database, load_tdb_database).
This module contains only CALPHAD-specific database utilities.
"""
import re
import logging
from typing import Optional, Dict, Set
from pathlib import Path
from pycalphad import Database

from ...base.constants import EXCLUDE_PHASE_PATTERNS, PHASE_NAME_MAP, ELEMENT_ALIASES

_log = logging.getLogger(__name__)


def is_excluded_phase(name: str) -> bool:
    """
    Check if a phase should be excluded from analysis.
    
    Args:
        name: Phase name
        
    Returns:
        True if phase should be excluded
    """
    up = name.upper()
    return any(re.search(pat, up) for pat in EXCLUDE_PHASE_PATTERNS)


def map_phase_name(name: str) -> str:
    """
    Map database phase name to readable name.
    
    Args:
        name: Phase name from database
        
    Returns:
        Readable phase name (e.g., 'CSI' -> 'SiC')
    """
    return PHASE_NAME_MAP.get(name.upper(), name)

def get_db_elements(db: Database) -> Set[str]:
    """
    Get set of element symbols from database.
    
    Args:
        db: PyCalphad Database instance
        
    Returns:
        Set of uppercase element symbols (excluding 'VA')
    """
    # pycalphad Database has .elements (set of species) incl. 'VA'
    return {el.upper() for el in getattr(db, "elements", set()) if el.upper() != "VA"}


def compose_alias_map(db: Database) -> Dict[str, str]:
    """
    Create element alias mapping for database.
    
    Args:
        db: PyCalphad Database instance
        
    Returns:
        Dictionary mapping aliases to standard element symbols
    """
    elems = get_db_elements(db)
    aliases = {k: v for k, v in ELEMENT_ALIASES.items() if v in elems or v == "VA"}
    
    # Also map bare symbols in any case
    for el in elems:
        aliases[el.lower()] = el
        aliases[el.capitalize()] = el
        aliases[el] = el
    
    return aliases

