import re
from typing import Optional
from pathlib import Path
from .consts import STATIC_ALIASES

# exclude clearly non-equilibrium / cluster / GP phases / helper phases by default
EXCLUDE_PHASE_PATTERNS = (
    r'^GP_', r'_GP', r'^CL_', r'_DP$', r'^B_PRIME', r'^PRE_', r'^THETA_PRIME$',
    r'^TH_DP', r'^U1_PHASE$', r'^U2_PHASE$',
    r'^FCCAL$', r'^FCCMG$', r'^FCCSI$',  # Helper phases for thermodynamic calculations only
)

def _is_excluded_phase(name: str) -> bool:
    up = name.upper()
    return any(re.search(pat, up) for pat in EXCLUDE_PHASE_PATTERNS)

def _upper_symbol(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    # Allow 'Al', 'al', 'AL' → 'AL'
    return s[:2].capitalize().upper() if len(s) <= 2 else s.upper()

def _db_elements(db) -> set[str]:
    # pycalphad Database has .elements (set of species) incl. 'VA'
    return {el.upper() for el in getattr(db, "elements", set()) if el.upper() != "VA"}

def _compose_alias_map(db) -> dict:
    elems = _db_elements(db)
    aliases = {k: v for k, v in STATIC_ALIASES.items() if v in elems or v == "VA"}
    # also map bare symbols in any case
    for el in elems:
        aliases[el.lower()] = el
        aliases[el.capitalize()] = el
        aliases[el] = el
    return aliases

def _pick_tdb_path(tdb_dir: Path, elements: Optional[list] = None) -> Optional[Path]:
    """
    Pick appropriate thermodynamic database based on elements involved.
    
    Args:
        tdb_dir: Directory containing .tdb files
        elements: List of element symbols (e.g., ['AL', 'SI', 'C'])
    
    Returns:
        Path to appropriate .tdb file
    """
    if not tdb_dir.exists():
        return None
    
    candidates = sorted([p for p in tdb_dir.glob("*.tdb") if p.is_file()])
    if not candidates:
        return None
    
    # If elements provided, choose database based on system
    if elements:
        elements_upper = [el.upper() for el in elements]
        
        # COST507.tdb for carbon-containing systems
        if 'C' in elements_upper or 'N' in elements_upper or 'B' in elements_upper:
            for p in candidates:
                if "COST507" in p.name or "cost507" in p.name.lower():
                    return p
    
    # Default: prefer pycal versions for Al systems
    for p in candidates:
        if "pycal" in p.name.lower() and "al" in p.name.lower():
            return p
    
    # Fallback to first candidate
    return candidates[0]
