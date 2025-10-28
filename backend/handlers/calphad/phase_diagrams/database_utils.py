"""
Database utilities for CALPHAD phase diagrams.

Functions for loading databases, handling elements, and database selection.
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


def upper_symbol(s: str) -> str:
    """
    Normalize element symbol to uppercase.
    
    Args:
        s: Element symbol string
        
    Returns:
        Uppercase element symbol
    """
    s = s.strip()
    if not s:
        return s
    # Allow 'Al', 'al', 'AL' → 'AL'
    return s[:2].capitalize().upper() if len(s) <= 2 else s.upper()


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


def pick_tdb_path(tdb_dir: Path, elements: Optional[list] = None) -> Optional[Path]:
    """
    Pick appropriate thermodynamic database based on elements involved.
    
    Args:
        tdb_dir: Directory containing .tdb files
        elements: List of element symbols (e.g., ['AL', 'SI', 'C'])
    
    Returns:
        Path to appropriate .tdb file or None if not found
    """
    if not tdb_dir.exists():
        _log.warning(f"TDB directory does not exist: {tdb_dir}")
        return None
    
    candidates = sorted([p for p in tdb_dir.glob("*.tdb") if p.is_file()])
    if not candidates:
        _log.warning(f"No .tdb files found in {tdb_dir}")
        return None
    
    # If elements provided, choose database based on system
    if elements:
        elements_upper = [el.upper() for el in elements]
        elements_set = set(elements_upper)
        
        # COST507.tdb for Al-Mg-Zn ternary system (has clean tau phase data)
        if elements_set == {'AL', 'MG', 'ZN'} or elements_set <= {'AL', 'MG', 'ZN', 'VA'}:
            for p in candidates:
                if "COST507" in p.name or "cost507" in p.name.lower():
                    _log.info(f"Selected {p.name} for Al-Mg-Zn system")
                    return p
        
        # COST507.tdb for systems with C, N, B, or Li
        if any(el in elements_upper for el in ['C', 'N', 'B', 'LI']):
            for p in candidates:
                if "COST507" in p.name or "cost507" in p.name.lower():
                    _log.info(f"Selected {p.name} for elements {elements_upper}")
                    return p
    
    # Default: prefer pycal versions for Al systems
    for p in candidates:
        if "pycal" in p.name.lower() and "al" in p.name.lower():
            _log.info(f"Selected {p.name} (default Al database)")
            return p
    
    # Fallback to first candidate
    _log.info(f"Selected {candidates[0].name} (fallback)")
    return candidates[0]


def load_database(tdb_dir: Path, elements: Optional[list] = None) -> Optional[Database]:
    """
    Load appropriate thermodynamic database.
    
    Args:
        tdb_dir: Directory containing .tdb files
        elements: Optional list of element symbols for database selection
        
    Returns:
        PyCalphad Database instance or None if loading fails
    """
    db_path = pick_tdb_path(tdb_dir, elements)
    if not db_path:
        return None
    
    try:
        db = Database(str(db_path))
        _log.info(f"Loaded database: {db_path.name}")
        return db
    except Exception as e:
        _log.error(f"Failed to load database {db_path}: {e}")
        return None

