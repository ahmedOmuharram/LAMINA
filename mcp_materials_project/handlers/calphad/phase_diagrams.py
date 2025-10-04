"""
CALPHAD phase diagram generation using pycalphad.

Supports binary phase diagram calculation and plotting for various chemical systems.
"""

import os
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Tuple
import logging
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import re

# exclude clearly non-equilibrium / cluster / GP phases by default
EXCLUDE_PHASE_PATTERNS = (
    r'^GP_', r'_GP', r'^CL_', r'_DP$', r'^B_PRIME', r'^PRE_', r'^THETA_PRIME$',
    r'^TH_DP', r'^U1_PHASE$', r'^U2_PHASE$', r'^FCCAL$'  # FCCAL in this DB is a special Al-only helper
)

def _is_excluded_phase(name: str) -> bool:
    up = name.upper()
    return any(re.search(pat, up) for pat in EXCLUDE_PHASE_PATTERNS)

# Terminal (pure-element) crystal structures for mc_al_v2.037.tdb
# (names taken from the TDB header; include safe alternates where helpful)
_TDB_TERMINAL_BY_ELEMENT = {
    "AL": ["FCC_A1"],
    "CU": ["FCC_A1"],
    "NI": ["FCC_A1"],

    "FE": ["BCC_A2"],
    "CR": ["BCC_A2"],
    # MatCalc uses BCC_A12 for Mn; fall back to BCC_A2 if A12 absent
    "MN": ["BCC_A12", "BCC_A2"],

    "MG": ["HCP_A3"],
    "TI": ["HCP_A3"],
    "ZN": ["HCP_A3", "HCP_ZN"],   # include HCP_ZN (low priority in DB) just in case
    "ZR": ["HCP_A3"],
    "SC": ["HCP_A3"],

    # Silicon is named SI_DIAMOND_A4 in this DB; include generic alias as fallback
    "SI": ["SI_DIAMOND_A4", "DIAMOND_A4"],
}

# Extra "must keep" phases for specific pairs (beyond terminal structures)
# Use frozenset to ignore input order ('AL-ZN' == 'ZN-AL')
_TDB_PAIR_ADDITIONS = {
    frozenset({"AL", "ZN"}): ["HCP_ZN"],     # make sure Zn-side HCP is available
    frozenset({"AL", "SI"}): ["SI_DIAMOND_A4"],
    frozenset({"ZN", "SI"}): ["SI_DIAMOND_A4", "HCP_ZN"],
    # add more pair-specific nudges here if needed
}

def _must_keep_for_system(db, pair: tuple[str, str]) -> list[str]:
    """
    Build the must_keep list for (A,B) based on this TDB:
      - Always LIQUID
      - Terminal structures for A and B (from _TDB_TERMINAL_BY_ELEMENT)
      - Optional pair-specific additions (_TDB_PAIR_ADDITIONS)
    Only keep phases that exist in the loaded database.
    """
    all_phases = set(db.phases.keys())
    A, B = pair
    keep: list[str] = ["LIQUID"]

    for el in (A, B):
        for ph in _TDB_TERMINAL_BY_ELEMENT.get(el, []):
            if ph in all_phases:
                keep.append(ph)

    # pair-specific additions
    extras = _TDB_PAIR_ADDITIONS.get(frozenset({A, B}), [])
    for ph in extras:
        if ph in all_phases:
            keep.append(ph)

    # As a defensive fallback, if none of the classic structures slipped in,
    # include any of them that exist in the DB.
    classics = ("FCC_A1", "HCP_A3", "BCC_A2")
    if not any(p in keep for p in classics):
        for ph in classics:
            if ph in all_phases:
                keep.append(ph)

    # de-duplicate while preserving order
    seen = set()
    ordered = []
    for ph in keep:
        if ph not in seen:
            seen.add(ph)
            ordered.append(ph)
    return ordered

try:
    from pycalphad import Database, binplot
    import pycalphad.variables as v
    from pycalphad.core.solver import Solver
    PYCALPHAD_AVAILABLE = True
except ImportError:
    PYCALPHAD_AVAILABLE = False

from ..base import BaseHandler
from kani.ai_function import ai_function
from typing_extensions import Annotated
from kani import AIParam

_log = logging.getLogger(__name__)

# Periodic masses (g/mol) for elements present in your DB (extendable)
_ATOMIC_MASS = {
    "AL": 26.98154, "CR": 51.996, "CU": 63.546, "FE": 55.847, "MG": 24.305,
    "MN": 54.938, "NI": 58.69, "SC": 44.956, "SI": 28.0855, "TI": 47.88,
    "ZN": 65.38, "ZR": 91.224
}

# Build aliases dynamically from DB elements once loaded; seed with common names
STATIC_ALIASES = {
    "vac": "VA", "va": "VA",
    "aluminum": "AL", "aluminium": "AL",
    "iron": "FE", "magnesium": "MG", "manganese": "MN",
    "nickel": "NI", "silicon": "SI", "titanium": "TI",
    "zinc": "ZN", "zirconium": "ZR", "chromium": "CR", "copper": "CU", "scandium": "SC"
}

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

def _pick_tdb_path(tdb_dir: Path) -> Optional[Path]:
    # prefer a single .tdb; if multiple, prefer one that looks Al-focused and pycal-compatible
    if not tdb_dir.exists():
        return None
    candidates = sorted([p for p in tdb_dir.glob("*.tdb") if p.is_file()])
    if not candidates:
        return None
    # First try pycal versions
    for p in candidates:
        if "pycal" in p.name.lower() and "al" in p.name.lower():
            return p
    # Then try any Al-focused version
    else:
        return "COST507.tdb"

class CalPhadHandler(BaseHandler):
    
    def __init__(self):
        if not PYCALPHAD_AVAILABLE:
            _log.warning("pycalphad not available - CALPHAD functionality disabled")
        self.tdb_dir = Path(__file__).parent.parent.parent.parent / "tdbs"
        if not self.tdb_dir.exists():
            _log.warning(f"TDB directory not found at {self.tdb_dir}")
        self._solver6 = None
    
    def _phase_constituent_elements(self, db, phase: str) -> set[str]:
        cons = db.phases[phase].constituents
        elems = set()
        db_elems = _db_elements(db)
        for sub in cons:
            for sp in sub:
                # 1) try structured
                try:
                    elems |= {el.upper() for el in sp.elements if el.upper() != "VA"}
                    continue
                except Exception:
                    pass
                # 2) fallback: parse species string (e.g., 'AL3MG2SI')
                s = str(sp).upper()
                for el in db_elems:
                    if el in s:
                        elems.add(el)
        return elems  # may be empty if truly unknown; caller treats empty as "exclude"

    def _filter_phases_for_system(self, db, pair: tuple[str, str]) -> list[str]:
        """
        Keep phases whose constituents are subset of {A,B} (+VA).
        Always keep per-system 'must_keep' phases derived from the TDB.
        """
        a, b = pair
        allowed = {a, b}
        all_phases = list(db.phases.keys())

        # NEW: system-aware must_keep built from the TDB + pair-specific nudges
        must_keep = [p for p in _must_keep_for_system(db, pair)]

        chosen = set(must_keep)
        for ph in all_phases:
            if _is_excluded_phase(ph):
                continue
            try:
                elems = self._phase_constituent_elements(db, ph)
            except Exception:
                elems = set()
            # conservative: only include if we can tell it uses only A/B (plus VA)
            if elems and allowed.issubset(elems):
                chosen.add(ph)

        # keep must_keep first in the returned order
        return sorted(chosen, key=lambda s: (s not in must_keep, s))
    
    def _split_phase_instances(self, eq, phase_name, element="ZN", decimals=5, tol=2e-4):
        """
        Return masks for distinct coexisting instances of `phase_name` at one T.
        Uses composition clustering: vertices with the same phase name but
        different X(element) are grouped as #1, #2, ...
        
        Ensures consistent ordering: #1 = lower X(element), #2 = higher X(element)
        """
        import numpy as np

        # mask for this phase across vertices
        phase_vec = np.asarray(eq.Phase.values, dtype=str).ravel()
        vmask = (phase_vec == phase_name)

        if not vmask.any():
            return {}

        # element composition per vertex for this phase
        x_raw = eq.X.sel(component=element).values.ravel().astype(float)
        x_vals = x_raw[vmask]
        np_vals = eq.NP.values.ravel().astype(float)[vmask]

        # guard: if everything is essentially one value → no split
        if np.all(np.isnan(x_vals)):
            return {}

        x_clean = x_vals[~np.isnan(x_vals)]
        if x_clean.size == 0:
            return {}

        # cluster by rounded composition (robust to tiny numerical noise)
        # Use higher precision and tighter tolerance to avoid leakage
        keys = np.round(x_vals, decimals=5)  # was 4 - higher precision
        uniq = np.unique(keys[~np.isnan(keys)])

        # if only one unique composition → no miscibility gap
        if uniq.size <= 1:
            return {}

        # build per-instance masks on the *full* vertex index
        instance_masks = {}
        # Sort by composition ASCENDING: α1 := lower X(Zn), α2 := higher X(Zn)
        uniq_sorted = np.sort(uniq)  # ascending X(Zn) - ensures consistent ordering
        for i, center in enumerate(uniq_sorted, start=1):
            inst_key = f"{phase_name}#{i}"
            # "close to this center" = same rounded value OR within tighter tol
            local_mask = np.zeros_like(vmask, dtype=bool)
            local_mask[vmask] = (np.abs(x_vals - center) < 2e-4) | (np.round(x_vals, decimals=5) == center)  # tighter tolerance
            # only keep if it has any moles
            if np.any(local_mask):
                instance_masks[inst_key] = local_mask

        # sanity: if masks overlap (rare), assign by nearest center
        if len(instance_masks) > 1:
            centers = np.array(uniq_sorted)
            indices = np.where(vmask)[0]
            for idx in indices:
                xv = x_vals[np.where(vmask)[0].tolist().index(idx)]
                if np.isnan(xv):
                    continue
                nearest = np.argmin(np.abs(centers - xv))
                # clear from all, set only nearest
                for k in instance_masks:
                    instance_masks[k][idx] = False
                instance_masks[f"{phase_name}#{nearest+1}"][idx] = True

        return instance_masks
    
    def _split_by_region(self, eq, phase_name):
        """Alternative splitter using Phase_region when available."""
        import numpy as np
        names = np.asarray(eq.Phase.values, dtype=str).ravel()
        if not hasattr(eq, 'Phase_region'):
            return {}
        regions = np.asarray(getattr(eq, "Phase_region").values).ravel()
        mask = (names == phase_name)
        if not mask.any():
            return {}
        regs = np.unique(regions[mask])
        if regs.size <= 1:
            return {}
        out = {}
        for r in regs:
            out[f"{phase_name}#{int(r)}"] = (mask & (regions == r))
        return out
    
    def _normalize_system(self, system: str, db=None) -> Tuple[str, str]:
        """
        Return a canonical (A,B) pair of element symbols where A-B order
        follows the input order if given as 'A-B' or 'AB' or any spelling,
        else alphabetical. We do NOT swap them silently once parsed because
        the second element becomes the x-axis variable.
        """
        raw = system.strip().replace("_", "").replace(" ", "")
        if "-" in raw:
            a, b = raw.split("-", 1)
        elif "—" in raw:
            a, b = raw.split("—", 1)
        else:
            # split like 'AlCu' → ['Al','Cu'] by camel-ish boundary
            import re
            parts = re.findall(r"[A-Za-z]{1,2}", raw)
            if len(parts) >= 2:
                a, b = parts[0], parts[1]
            else:
                # fallback: if only one token, let caller decide later
                a, b = raw, ""
        aliases = _compose_alias_map(db) if db else STATIC_ALIASES
        A = aliases.get(a.lower(), _upper_symbol(a))
        B = aliases.get(b.lower(), _upper_symbol(b)) if b else ""
        return A, B

    def _get_database_path(self, _system_ignored: str = "") -> Optional[Path]:
        """
        No per-system mapping anymore. Just pick a .tdb from self.tdb_dir.
        Prefer one that contains 'al' in the filename (your MatCalc DB).
        """
        return _pick_tdb_path(self.tdb_dir)
    
    def _normalize_elements(self, elements: List[str], db=None) -> List[str]:
        """Normalize element names."""
        aliases = _compose_alias_map(db) if db else STATIC_ALIASES
        normalized = []
        for elem in elements:
            clean = elem.strip().lower()
            if clean in aliases:
                normalized.append(aliases[clean])
            else:
                normalized.append(elem.upper())
        return normalized
    
    def _extract_phase_colors_from_plot(self, axes, phases):
        """Extract the actual colors used by binplot for the phases."""
        phase_colors = {}
        
        # Get all collections (filled contours) from the axes
        collections = axes.collections
        print(f"Found {len(collections)} collections in the plot", flush=True)
        
        # binplot creates collections in the order phases are provided
        for i, phase in enumerate(phases):
            if i < len(collections):
                # Get the facecolor from the collection
                facecolors = collections[i].get_facecolors()
                if len(facecolors) > 0:
                    # Convert to hex color
                    color = facecolors[0][:3]  # RGB only, ignore alpha
                    import matplotlib.colors as mcolors
                    hex_color = mcolors.to_hex(color)
                    phase_colors[phase] = hex_color
                    print(f"Extracted color for {phase}: {hex_color}", flush=True)
        
        # Only use fallbacks if we truly have no colors extracted
        if not phase_colors:
            print("No colors extracted from binplot collections, using fallbacks", flush=True)
            fallback_colors = {
                'LIQUID': '#C10020',    # Dark red
                'FCC_A1': '#00538A',    # Dark blue  
                'HCP_A3': '#F13A13',    # Red-orange
            }
            
            for phase in phases:
                phase_colors[phase] = fallback_colors.get(phase, '#808080')
                print(f"Using fallback color for {phase}: {phase_colors[phase]}", flush=True)
        else:
            print(f"Successfully extracted {len(phase_colors)} colors from binplot", flush=True)
        
        return phase_colors
    
    def _parse_composition(self, system_or_comp: str, composition_type: str = "atomic", db=None) -> Tuple[Tuple[str,str], Optional[float], str]:
        """
        Parse: 'Al-Cu', 'AlCu', 'Al20Cu80', 'Cu7Al93wt%', 'Al' (pure), etc.
        Returns ((A,B), x_B, comp_type_used), where x_B is mole fraction of B (the x-axis element).
        """
        import re
        txt = system_or_comp.strip().replace(" ", "").replace("–", "-").replace("—", "-")

        # detect optional 'wt' tag (case-insensitive)
        comp_type = composition_type.lower()
        if txt.lower().endswith("wt%"):
            comp_type = "weight"
            txt = txt[:-3]
        elif txt.lower().endswith("at%"):
            comp_type = "atomic"
            txt = txt[:-3]

        # Will need DB-aware aliases
        aliases = _compose_alias_map(db) if db else STATIC_ALIASES

        def norm_token(tok):
            return aliases.get(tok.lower(), _upper_symbol(tok))

        # Cases:
        # 1) Pure element: 'Al' or alias
        if re.fullmatch(r"[A-Za-z]{1,2}", txt):
            A = norm_token(txt)
            # pick any other element later; for now treat as pure B=the same → x_B = 1 if B==A else 0
            # but we can't infer (A,B). We return (A,"") so the caller can decide B from context.
            return (A, ""), None if comp_type == "weight" else None, comp_type

        # 2) System name 'Al-Cu' or 'AlCu'
        sys_match = re.fullmatch(r"([A-Za-z]{1,2})-?([A-Za-z]{1,2})$", txt)
        if sys_match:
            a, b = norm_token(sys_match.group(1)), norm_token(sys_match.group(2))
            return (a, b), None, comp_type

        # 3) Compositions like 'Al20Cu80' (decimals allowed)
        comp_match = re.fullmatch(
            r"([A-Za-z]{1,2})(\d+(?:\.\d+)?)"
            r"([A-Za-z]{1,2})(\d+(?:\.\d+)?)", txt)
        if comp_match:
            el1, p1, el2, p2 = comp_match.groups()
            A, B = norm_token(el1), norm_token(el2)
            p1, p2 = float(p1), float(p2)
            total = p1 + p2
            if total <= 0:
                return (A, B), None, comp_type
            p1n, p2n = p1/total, p2/total

            # We always plot x = X(B)
            if comp_type == "weight":
                # convert weight fractions to mole fractions
                mA = _ATOMIC_MASS.get(A)
                mB = _ATOMIC_MASS.get(B)
                if not (mA and mB):
                    # fallback: treat as atomic if mass unknown
                    xB = p2n
                    comp_type = "atomic"
                else:
                    nA = p1n / mA
                    nB = p2n / mB
                    den = nA + nB if (nA + nB) > 0 else 1.0
                    xB = nB / den
            else:
                xB = p2n
                comp_type = "atomic"

            return (A, B), xB, comp_type

        # 4) Fallback: try to pull two symbols anywhere in the string
        els = re.findall(r"[A-Za-z]{1,2}", txt)
        if len(els) >= 2:
            A, B = norm_token(els[0]), norm_token(els[1])
            return (A, B), None, comp_type

        # give up
        return (txt.upper(), ""), None, comp_type
    
    def _add_phase_labels(self, axes, temp_range: Tuple[float, float], phases: List[str], db=None, elements=None, comp_var=None) -> None:
        """Add interactive annotations with calculated key points."""
                    
        # Interactive Annotations with calculated key points
        print("Adding interactive annotations with calculated points...", flush=True)
        
        # Calculate key thermodynamic points dynamically
        if db is not None and elements is not None and comp_var is not None:
            try:
                key_points = self._calculate_key_thermodynamic_points(db, elements, phases, comp_var, temp_range)
                # Store key points for use in analysis
                self._last_key_points = key_points
            except Exception as e:
                print(f"Key point calculation failed: {e}", flush=True)
                key_points = []
                self._last_key_points = []
        else:
            print("Skipping dynamic key point calculation - missing parameters", flush=True)
            key_points = []
            self._last_key_points = []
        
        # Get the composition element symbol for generic labeling
        comp_el = str(getattr(comp_var, "species", getattr(comp_var, "x", "B"))).upper() if comp_var else "B"
        
        # Add annotations for calculated points with improved visibility
        print(f"Adding annotations for {len(key_points)} key points", flush=True)
        for i, point in enumerate(key_points):
            print(f"Processing point {i+1}: {point['type']} at {point.get('composition_pct', 'N/A')}% {comp_el}, {point.get('temperature', 'N/A')}K", flush=True)
            
            # Validate point data
            if 'composition' not in point or 'temperature' not in point:
                print(f"Skipping invalid point {i+1}: missing composition or temperature", flush=True)
                continue
                
            x_pos = point['composition']
            y_pos = point['temperature']
            
            # Ensure coordinates are within plot bounds
            if not (0 <= x_pos <= 1) or not (temp_range[0] <= y_pos <= temp_range[1]):
                print(f"Skipping point {i+1}: coordinates out of bounds ({x_pos}, {y_pos})", flush=True)
                continue
            
            if point['type'] == 'eutectic':
                # Place eutectic annotation with better positioning
                offset_x = -0.2 if x_pos > 0.5 else 0.2  # Move left if on right side
                offset_y = (temp_range[1] - temp_range[0]) * 0.12
                
                axes.annotate(f"Eutectic Point\n{point['composition_pct']:.1f}% {comp_el}\n{point['temperature']:.0f}K", 
                             xy=(x_pos, y_pos), 
                             xytext=(x_pos + offset_x, y_pos + offset_y),
                             fontsize=9,
                             fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.8, edgecolor='red'),
                             arrowprops=dict(arrowstyle='->', color='red', lw=2, connectionstyle="arc3,rad=0.1"))
                print(f"Added eutectic annotation at ({x_pos:.3f}, {y_pos:.0f})", flush=True)
            
            elif point['type'] == 'pure_melting':
                element = point['element']
                element_name = element  # Use element symbol directly
                color = 'orange' if element == elements[0] else 'darkgreen'
                
                # Position annotations clearly outside the plot area
                if element == elements[0]:  # First element (left side) - place to the left outside
                    text_x = -0.15    # Position text outside left boundary (x=0)
                    text_y = y_pos + (temp_range[1] - temp_range[0]) * 0.1  # Slightly above the point
                else:  # Second element (right side) - place to the right outside  
                    text_x = 1.05     # Bring closer to the graph (was 1.15)
                    text_y = y_pos + (temp_range[1] - temp_range[0]) * 0.1  # Slightly above the point
                
                axes.annotate(f"Pure {element_name}\nMelting Point\n{point['temperature']:.0f}K", 
                             xy=(x_pos, y_pos), 
                             xytext=(text_x, text_y),
                             fontsize=8,
                             fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7, edgecolor='black'),
                             arrowprops=dict(arrowstyle='->', color=color, lw=2))
                print(f"Added {element_name} melting point annotation at ({x_pos:.3f}, {y_pos:.0f})", flush=True)
            
            elif point['type'] == 'phase_transition':
                axes.annotate(f"Phase Transition\n{point['composition_pct']:.1f}% {comp_el}\n{point['temperature']:.0f}K", 
                             xy=(x_pos, y_pos), 
                             xytext=(x_pos + 0.12, y_pos + (temp_range[1] - temp_range[0]) * 0.06),
                             fontsize=7,
                             bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.7, edgecolor='orange'),
                             arrowprops=dict(arrowstyle='->', color='orange', lw=1.5))
                print(f"Added phase transition annotation at ({x_pos:.3f}, {y_pos:.0f})", flush=True)
        
        
        # Let binplot handle the legend - it knows the colors it used        
        print("All phase diagram enhancements complete!", flush=True)

    def _calculate_key_thermodynamic_points(self, db, elements, phases, comp_var, temp_range):
        import numpy as np, matplotlib.pyplot as plt
        key_points = []
        ax = plt.gca()
        lines = ax.get_lines()
        if not lines:
            return key_points

        ymin, ymax = ax.get_ylim()
        yspan = ymax - ymin
        top_margin = 0.03 * yspan

        usable = []
        for ln in lines:
            y = np.asarray(ln.get_ydata(), float)
            x = np.asarray(ln.get_xdata(), float)
            if y.size == 0 or np.all(np.isnan(y)):
                continue
            if np.nanmax(y) >= ymax - top_margin:
                continue  # ignore axis frames / top caps
            if np.nanstd(y) < 1e-3:
                continue  # ignore near-horizontal artifacts
            usable.append((x, y))

        if not usable:
            return key_points

        # liquidus candidate = highest average T among usable lines
        avgs = [float(np.nanmean(y)) for x, y in usable]
        x_liq, y_liq = usable[int(np.argmax(avgs))]

        # eutectic = minimum on that liquidus, but avoid the extreme ends
        mask_mid = (x_liq > 0.02) & (x_liq < 0.98)
        if np.any(mask_mid):
            idx = np.nanargmin(y_liq[mask_mid])
            x_e = float(x_liq[mask_mid][idx]); y_e = float(y_liq[mask_mid][idx])
            key_points.append({'type': 'eutectic', 'composition': x_e,
                               'composition_pct': x_e * 100, 'temperature': y_e})

        # pure melt points (left/right, not using top caps)
        all_pts = np.concatenate([np.column_stack((x, y)) for x, y in usable], axis=0)
        left = all_pts[all_pts[:, 0] < 0.1]
        right = all_pts[all_pts[:, 0] > 0.9]
        if left.size:
            xL, yL = left[np.argmax(left[:, 1])]
            key_points.append({'type': 'pure_melting', 'element': elements[0],
                               'composition': float(xL), 'composition_pct': float(xL)*100,
                               'temperature': float(yL)})
        if right.size:
            xR, yR = right[np.argmax(right[:, 1])]
            key_points.append({'type': 'pure_melting', 'element': elements[1],
                               'composition': float(xR), 'composition_pct': float(xR)*100,
                               'temperature': float(yR)})
        return key_points

    def _plot_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        buf = io.BytesIO()
        # Reduce DPI for faster processing and smaller file size
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        buf.seek(0)
        img_data = buf.read()
        print(f"Image data size: {len(img_data)} bytes", flush=True)
        img_base64 = base64.b64encode(img_data).decode('utf-8')
        buf.close()
        print(f"Base64 encoding complete", flush=True)
        return img_base64
    
    def _analyze_visual_content(self, fig, axes, normalized_system: str, phases: List[str], temp_range: Tuple[float, float]) -> str:
        """Analyze the visual content of the generated phase diagram."""
        try:
            visual_analysis = []
            visual_analysis.append("### Visual Analysis of Generated Phase Diagram:")
            
            # Analyze axis information
            xlim = axes.get_xlim()
            ylim = axes.get_ylim()
            xlabel = axes.get_xlabel()
            ylabel = axes.get_ylabel()
            title = axes.get_title()
            
            # Detect plot type and format dimensions correctly
            if "Temperature" in xlabel and "Phase Fraction" in ylabel:
                # Composition-temperature plot: x = temperature, y = phase fraction
                visual_analysis.append(f"- **Temperature Range**: {xlabel} from {xlim[0]:.0f}K to {xlim[1]:.0f}K")
                visual_analysis.append(f"- **Y-Axis Range**: {ylabel} from {ylim[0]:.2f} to {ylim[1]:.2f}")
            elif "Mole Fraction" in xlabel and "Temperature" in ylabel:
                # Binary phase diagram: x = composition, y = temperature
                visual_analysis.append(f"- **Composition Range**: {xlabel} from {xlim[0]:.2f} to {xlim[1]:.2f}")
                visual_analysis.append(f"- **Temperature Range**: {ylabel} from {ylim[0]:.0f}K to {ylim[1]:.0f}K")
            else:
                # Generic fallback
                visual_analysis.append(f"- **X-Axis**: {xlabel} from {xlim[0]:.2f} to {xlim[1]:.2f}")
                visual_analysis.append(f"- **Y-Axis**: {ylabel} from {ylim[0]:.2f} to {ylim[1]:.2f}")
            visual_analysis.append(f"- **Title**: {title}")
            
            # Analyze plot elements
            plot_elements = []
            
            # Count number of plotted elements (lines, patches, etc.)
            lines = axes.get_lines()
            patches = axes.patches
            collections = axes.collections
            
            if lines:
                plot_elements.append(f"{len(lines)} phase boundary lines")
            if patches:
                plot_elements.append(f"{len(patches)} filled regions/patches")
            if collections:
                plot_elements.append(f"{len(collections)} plot collections (contours/fills)")
            
            if plot_elements:
                visual_analysis.append(f"- **Plot Elements**: {', '.join(plot_elements)}")
            
            # Analyze legend if present
            legend = axes.get_legend()
            if legend:
                legend_labels = [t.get_text() for t in legend.get_texts()]
                visual_analysis.append(f"- **Legend Phases**: {', '.join(legend_labels)}")
            
            # Analyze grid and formatting
            if axes.grid:
                visual_analysis.append("- **Grid**: Enabled for better readability")
            
            # Check for annotations/arrows
            if hasattr(axes, 'texts') and axes.texts:
                annotations = len([t for t in axes.texts if t.get_text().strip()])
                if annotations > 0:
                    visual_analysis.append(f"- **Annotations**: {annotations} text labels/annotations present")
            
            # Analyze colors used (from collections and patches)
            colors_used = set()
            for collection in collections:
                if hasattr(collection, 'get_facecolors'):
                    face_colors = collection.get_facecolors()
                    if len(face_colors) > 0:
                        # Convert to hex for readability
                        import matplotlib.colors as mcolors
                        for color in face_colors[:3]:  # Sample first few colors
                            if len(color) >= 3:
                                hex_color = mcolors.to_hex(color[:3])
                                colors_used.add(hex_color)
            
            for patch in patches:
                if hasattr(patch, 'get_facecolor'):
                    color = patch.get_facecolor()
                    if color:
                        import matplotlib.colors as mcolors
                        hex_color = mcolors.to_hex(color[:3] if len(color) >= 3 else color)
                        colors_used.add(hex_color)
            
            if colors_used:
                visual_analysis.append(f"- **Color Scheme**: {len(colors_used)} distinct colors used for phase regions")
            
            # Check for special markers or arrows (like the ones we add for key points)
            arrow_patches = [p for p in patches if 'Arrow' in str(type(p))]
            if arrow_patches:
                visual_analysis.append(f"- **Special Markers**: {len(arrow_patches)} arrows/markers for key points")
            
            return "\n".join(visual_analysis)
            
        except Exception as e:
            return f"Error analyzing visual content: {str(e)}"

    async def _analyze_single_temperature_point(
        self, db, normalized_system: str, phases: List[str],
        target_composition: float, temperature: float,
        original_composition: str, composition_type: str = "atomic",
        A: str = None, B: str = None
    ) -> str:
        """Analyze phase equilibrium at a single temperature point instead of generating a plot."""
        try:
            from pycalphad import equilibrium
            import pycalphad.variables as v
            
            if not (A and B):
                # best-effort parse from normalized_system 'A-B'
                A, B = normalized_system.split('-')

            print(f"CALPHAD: Analyzing single temperature point: {temperature:.0f}K for composition {original_composition}", flush=True)
            
            # Calculate equilibrium at the single temperature
            eq_result = equilibrium(
                db, [A, B, 'VA'], phases,
                {v.T: temperature, v.P: 101325, v.X(B): target_composition}
            )
            
            # Calculate phase fractions using the same method as composition-temperature plot
            phase_fractions_raw = {}
            
            # First get raw moles for each phase
            for phase_name in phases:
                phase_mask = eq_result.Phase == phase_name
                if phase_mask.any():
                    phase_moles = float(eq_result.NP.where(phase_mask, eq_result.NP, 0).sum())
                    phase_fractions_raw[phase_name] = phase_moles
                else:
                    phase_fractions_raw[phase_name] = 0.0
            
            # Normalize to get fractions (should sum to 1)
            total_phase_moles = sum(phase_fractions_raw.values())
            phase_fractions = {}
            
            for phase_name in phases:
                if total_phase_moles > 0:
                    phase_fractions[phase_name] = phase_fractions_raw[phase_name] / total_phase_moles
                else:
                    phase_fractions[phase_name] = 0.0
            
            print(f"CALPHAD: Raw phase moles: {phase_fractions_raw}", flush=True)
            print(f"CALPHAD: Total phase moles: {total_phase_moles:.6f}", flush=True)
            print(f"CALPHAD: Normalized fractions: {phase_fractions}", flush=True)
            
            # Build analysis
            pct_B = target_composition * 100
            pct_A = (1 - target_composition) * 100
            comp_suffix = "at%" if composition_type == "atomic" else "wt%"
            
            analysis_parts = []
            analysis_parts.append(f"# Single Point Analysis: {A}{pct_A:.0f}{B}{pct_B:.0f} ({comp_suffix}) at {temperature:.0f}K ({temperature-273.15:.0f}°C)")
            analysis_parts.append(f"**System**: {normalized_system}")
            analysis_parts.append(f"**Composition**: {original_composition} ({composition_type} percent)")
            analysis_parts.append(f"**Temperature**: {temperature:.0f}K ({temperature-273.15:.0f}°C)")
            analysis_parts.append("")
            
            # Phase equilibrium at this temperature
            analysis_parts.append("## Phase Equilibrium:")
            stable_phases = []
            for phase_name, fraction in phase_fractions.items():
                if fraction > 0.001:  # Only significant phases (>0.1%)
                    phase_label = {
                        'LIQUID': 'Liquid',
                        'FCC_A1': 'FCC (Al-rich)',
                        'HCP_A3': 'HCP (Zn-rich)'
                    }.get(phase_name, phase_name)
                    stable_phases.append((phase_label, fraction))
            
            if stable_phases:
                total_fraction = sum(frac for _, frac in stable_phases)
                analysis_parts.append("**Stable phases at this temperature:**")
                for phase_label, fraction in stable_phases:
                    analysis_parts.append(f"- **{phase_label}**: {fraction:.1%}")
                analysis_parts.append(f"**Total accounted**: {total_fraction:.1%}")
            else:
                analysis_parts.append("**No stable phases detected** (calculation may have failed)")
            
            # Microstructure description
            analysis_parts.append("\n## Microstructural State:")
            if len(stable_phases) == 1:
                phase_label = stable_phases[0][0]
                analysis_parts.append(f"- **Single-phase microstructure**: {phase_label}")
                if 'Liquid' in phase_label:
                    analysis_parts.append("- **State**: Completely molten")
                else:
                    analysis_parts.append("- **State**: Solid single-phase region")
            elif len(stable_phases) == 2:
                phase1, frac1 = stable_phases[0]
                phase2, frac2 = stable_phases[1]
                analysis_parts.append(f"- **Two-phase microstructure**: {phase1} + {phase2}")
                analysis_parts.append(f"- **Phase balance**: {frac1:.1%} {phase1}, {frac2:.1%} {phase2}")
            elif len(stable_phases) > 2:
                analysis_parts.append(f"- **Multi-phase microstructure**: {len(stable_phases)} phases present")
            
            # Note about why no plot was generated
            analysis_parts.append("\n## Note:")
            analysis_parts.append(f"Since the temperature range was essentially a single point ({temperature:.0f}K), ")
            analysis_parts.append("a phase fraction vs. temperature plot would not be meaningful. ")
            analysis_parts.append("Instead, this analysis shows the equilibrium state at the specified temperature.")
            
            # Store the analysis in metadata for display as a tool pill
            analysis_text = "\n".join(analysis_parts)
            
            # Create metadata similar to plot functions
            metadata = {
                "system": normalized_system,
                "analysis_type": "single_temperature_point",
                "composition": {
                    "A_pct": pct_A,
                    "B_pct": pct_B,
                    "mole_fraction_B": target_composition
                },
                "temperature_K": temperature,
                "description": f"Single point equilibrium analysis for {original_composition} at {temperature:.0f}K",
                "analysis": analysis_text,
                "thermodynamic_analysis": analysis_text,
                "visual_analysis": "",  # No visual content for single point
                "stable_phases": [label for label, _ in stable_phases]
            }
            
            # Store metadata for analysis panel display
            setattr(self, '_last_image_metadata', metadata)
            print(f"CALPHAD: Stored single point analysis metadata", flush=True)
            
            # Return simple success message like other functions
            phase_list = ", ".join([label for label, _ in stable_phases]) if stable_phases else "none detected"
            return f"Analyzed equilibrium for {original_composition} at {temperature:.0f}K ({temperature-273.15:.0f}°C). Stable phases: {phase_list}. Single point analysis shows the microstructural state at this specific temperature."
            
        except Exception as e:
            return f"Error analyzing single temperature point: {str(e)}"

    def _analyze_phase_diagram(self, db, normalized_system: str, phases: List[str], temp_range: Tuple[float, float]) -> str:
        """Generate deterministic analysis of the phase diagram using calculated key points."""
        try:
            if normalized_system != "Al-Zn":
                return "Phase diagram analysis available only for Al-Zn system currently."
            
            analysis_parts = []
            analysis_parts.append(f"## Phase Diagram Analysis: {normalized_system} System\n")
            
            # Use the key points that were already calculated in _add_phase_labels
            key_points = getattr(self, '_last_key_points', [])
            
            if key_points:
                analysis_parts.append("### Key Thermodynamic Points:")
                
                # Extract calculated values for pure elements and eutectic
                al_melting_point = None
                zn_melting_point = None
                eutectic_point = None
                
                for point in key_points:
                    if point['type'] == 'pure_melting':
                        if point['element'] == 'AL':
                            al_melting_point = point
                        elif point['element'] == 'ZN':
                            zn_melting_point = point
                    elif point['type'] == 'eutectic':
                        eutectic_point = point
                
                # Report pure melting points
                if al_melting_point:
                    temp = al_melting_point['temperature']
                    analysis_parts.append(f"- **Pure Aluminum**: {temp:.0f}K ({temp-273.15:.0f}°C)")
                
                if zn_melting_point:
                    temp = zn_melting_point['temperature']
                    analysis_parts.append(f"- **Pure Zinc**: {temp:.0f}K ({temp-273.15:.0f}°C)")
                
                if eutectic_point:
                    temp = eutectic_point['temperature']
                    comp = eutectic_point['composition_pct']
                    analysis_parts.append(f"- **Eutectic Point**: {temp:.0f}K ({temp-273.15:.0f}°C) at {comp:.1f}% Zn")
                    analysis_parts.append(f"  - Reaction: L → FCC + HCP")
                    analysis_parts.append(f"  - Composition: Al{100-comp:.0f}Zn{comp:.0f}")
            else:
                # Fallback to approximate values if key points weren't calculated
                analysis_parts.append("### Key Features (Approximate):")
                analysis_parts.append("- **Pure Aluminum**: ~933K (660°C)")
                analysis_parts.append("- **Pure Zinc**: ~693K (420°C)")
                analysis_parts.append("- **Eutectic**: ~655K (382°C) at ~95% Zn")
            
            # Analyze phase regions
            analysis_parts.append("\n### Phase Regions:")
            analysis_parts.append("- **LIQUID**: High-temperature single-phase region where all components are molten")
            analysis_parts.append("- **FCC_A1 (Al-rich)**: Face-centered cubic aluminum-rich solid solution")
            analysis_parts.append("- **HCP_A3 (Zn-rich)**: Hexagonal close-packed zinc-rich solid solution")
            
            # Add system-specific information
            analysis_parts.append("\n### System Characteristics:")
            analysis_parts.append("- **Terminal solid solutions**: Limited solubility between Al and Zn")
            analysis_parts.append("- **Crystal structures**: FCC (Al) and HCP (Zn) are immiscible at low temperatures")
            analysis_parts.append("- **Phase separation**: Clear distinction between Al-rich and Zn-rich phases")
            
            # Processing implications
            analysis_parts.append("\n### Processing Implications:")
            analysis_parts.append("- **Casting**: Liquidus temperatures determine minimum casting temperatures")
            analysis_parts.append("- **Heat treatment**: Two-phase regions allow for precipitation strengthening")
            analysis_parts.append("- **Welding**: Solidification behavior affects weld microstructure")
            
            return "\n".join(analysis_parts)
            
        except Exception as e:
            return f"Error generating phase diagram analysis: {str(e)}"
    
    def _analyze_composition_temperature(self, composition_data: dict, xB: float, temp_range: Tuple[float, float], A: str = "A", B: str = "B") -> str:
        """Generate detailed deterministic analysis of composition-temperature data."""
        try:
            pct_B = xB * 100
            pct_A = (1 - xB) * 100
            
            analysis_parts = []
            analysis_parts.append(f"## Detailed Phase Stability Analysis: {A}{pct_A:.0f}{B}{pct_B:.0f}\n")
            
            # Find critical transition temperatures  
            temp_array = np.linspace(temp_range[0], temp_range[1], len(list(composition_data.values())[0]))
            
            # Debug: Print what we actually have in composition_data
            print(f"DEBUG: composition_data keys: {list(composition_data.keys())}", flush=True)
            for phase_name, phase_values in composition_data.items():
                max_val = np.max(phase_values)
                indices_above_01 = np.where(np.array(phase_values) > 0.01)[0]
                if len(indices_above_01) > 0:
                    min_temp = temp_array[indices_above_01[0]]  
                    max_temp = temp_array[indices_above_01[-1]]
                    print(f"DEBUG: {phase_name}: max={max_val:.3f}, stable {min_temp:.0f}-{max_temp:.0f}K", flush=True)
                else:
                    print(f"DEBUG: {phase_name}: max={max_val:.3f}, never above 1%", flush=True)
            
            transition_temps = []
            
            # Analyze phase transitions with specific temperatures
            analysis_parts.append("### Critical Phase Transition Temperatures:")
            
            for phase_name, phase_values in composition_data.items():
                phase_array = np.array(phase_values)
                max_fraction = np.max(phase_array)
                
                if max_fraction > 0.01:  # Only analyze significant phases
                    # Find onset and offset temperatures
                    onset_idx = np.where(phase_array > 0.01)[0]
                    if len(onset_idx) > 0:
                        onset_temp = temp_array[onset_idx[0]]
                        offset_temp = temp_array[onset_idx[-1]]
                        
                        phase_label = {
                            'LIQUID': 'Liquid',
                            'FCC_A1': 'FCC (Al-rich)',
                            'HCP_A3': 'HCP (Zn-rich)'
                        }.get(phase_name, phase_name)
                        
                        # Find transition points (where phase fraction changes significantly)
                        diff = np.diff(phase_array)
                        significant_changes = np.where(np.abs(diff) > 0.1)[0]
                        
                        if len(significant_changes) > 0:
                            transition_temp = temp_array[significant_changes[0]]
                            transition_temps.append((transition_temp, phase_label))
                            analysis_parts.append(f"- **{phase_label} Formation**: {onset_temp:.0f}K ({onset_temp-273.15:.0f}°C)")
                            analysis_parts.append(f"  - Peak stability: {max_fraction:.1%} at {temp_array[np.argmax(phase_array)]:.0f}K")
                            if len(significant_changes) > 0:
                                analysis_parts.append(f"  - Major transition at: {transition_temp:.0f}K ({transition_temp-273.15:.0f}°C)")
            
            # Determine melting point more precisely - only if liquid is actually present
            if 'LIQUID' in composition_data and len(composition_data) > 1:
                liquid_array = np.array(composition_data['LIQUID'])
                
                # Check if there's any significant liquid in the temperature range
                max_liquid_fraction = np.max(liquid_array)
                print(f"DEBUG: Maximum liquid fraction in range: {max_liquid_fraction:.6f}", flush=True)
                
                if max_liquid_fraction > 1e-4:  # Only analyze if there's meaningful liquid content
                    analysis_parts.append("\n### Melting Point Analysis:")
                    
                    # Find melting temperatures correctly
                    # SOLIDUS: Find the very first appearance of liquid (even trace amounts)
                    liquid_trace_indices = np.where(liquid_array > 1e-6)[0]  # Any trace of liquid
                    if len(liquid_trace_indices) > 0:
                        # SOLIDUS: First temperature where ANY liquid appears (lowest temp with liquid)
                        solidus_temp = temp_array[liquid_trace_indices[0]] - 1 # First (lowest) temp with any liquid
                        analysis_parts.append(f"**Solidus temperature**: {solidus_temp:.0f}K ({solidus_temp-273.15:.0f}°C)")
                        print(f"DEBUG: Solidus found at {solidus_temp:.0f}K with liquid fraction {liquid_array[liquid_trace_indices[0]]:.6f}", flush=True)
                    
                        # LIQUIDUS: Temperature where material becomes fully liquid
                        mostly_liquid = np.where(liquid_array > 0.9)[0]
                        if len(mostly_liquid) > 0:
                            liquidus_temp = temp_array[mostly_liquid[0]]  # First temp where mostly liquid
                            analysis_parts.append(f"**Liquidus temperature**: {liquidus_temp:.0f}K ({liquidus_temp-273.15:.0f}°C)")
                            print(f"DEBUG: Liquidus found at {liquidus_temp:.0f}K with liquid fraction {liquid_array[mostly_liquid[0]]:.6f}", flush=True)
                        else:
                            # If never becomes mostly liquid, use highest temp with liquid
                            liquidus_temp = temp_array[liquid_trace_indices[-1]]
                            analysis_parts.append(f"**Liquidus temperature**: {liquidus_temp:.0f}K ({liquidus_temp-273.15:.0f}°C)")
                            print(f"DEBUG: Liquidus (max) found at {liquidus_temp:.0f}K with liquid fraction {liquid_array[liquid_trace_indices[-1]]:.6f}", flush=True)
                            
                        # Calculate melting range (liquidus should be higher than solidus)
                        if liquidus_temp > solidus_temp + 10:
                            analysis_parts.append(f"**Melting range**: {liquidus_temp - solidus_temp:.0f}K")
                            analysis_parts.append("**Behavior**: Gradual melting over temperature range (typical alloy behavior)")
                        else:
                            analysis_parts.append("**Behavior**: Sharp melting transition (near-eutectic composition)")
                    else:
                        # Only solidus found
                        analysis_parts.append(f"**Melting begins at**: {solidus_temp:.0f}K ({solidus_temp-273.15:.0f}°C)")
                else:
                    # No significant liquid found - skip melting point analysis
                    print(f"DEBUG: No significant liquid found (max: {max_liquid_fraction:.6f}), skipping melting analysis", flush=True)
            
            # Only include microstructural analysis - no room temperature
            analysis_parts.append("\n### Microstructural Evolution:")
            
            return "\n".join(analysis_parts)
            
        except Exception as e:
            return f"Error generating detailed composition analysis: {str(e)}"

    @ai_function(desc="PREFERRED for phase diagram questions. Generate a binary phase diagram for a chemical system using CALPHAD data. Use for general system queries like 'Al-Zn', 'aluminum-zinc', 'phase diagram', 'liquidus', 'solidus'. Shows full composition range with phase boundaries.")
    async def plot_binary_phase_diagram(
        self,
        system: Annotated[str, AIParam(desc="Chemical system (e.g., 'Al-Zn', 'AlZn', 'aluminum-zinc')")],
        min_temperature: Annotated[Optional[float], AIParam(desc="Minimum temperature in Kelvin. Default: 300")] = None,
        max_temperature: Annotated[Optional[float], AIParam(desc="Maximum temperature in Kelvin. Default: 1000")] = None,
        composition_step: Annotated[Optional[float], AIParam(desc="Composition step size (0-1). Default: 0.02")] = None,
        figure_width: Annotated[Optional[float], AIParam(desc="Figure width in inches. Default: 9")] = None,
        figure_height: Annotated[Optional[float], AIParam(desc="Figure height in inches. Default: 6")] = None
    ) -> str:
        """
        Generate a binary phase diagram using CALPHAD thermodynamic data.
        
        Currently supports:
        - Al-Zn (Aluminum-Zinc) system
        
        Returns phase diagram as base64-encoded PNG image with metadata.
        """
        
        if not PYCALPHAD_AVAILABLE:
            return {
                "success": False,
                "error": "pycalphad library not available. Install with: pip install pycalphad"
            }
        
        try:
            # Clear any previous plot metadata at the start of new generation
            if hasattr(self, '_last_image_metadata'):
                delattr(self, '_last_image_metadata')
            if hasattr(self, '_last_image_data'):
                delattr(self, '_last_image_data')
            
            # Normalize & load DB
            db_path = self._get_database_path(system)
            if not db_path:
                return {"success": False, "error": f"No .tdb found in {self.tdb_dir}."}
            db = Database(str(db_path))

            # Parse system → (A,B); ensure both are in DB
            A, B = self._normalize_system(system, db=db)
            db_elems = _db_elements(db)
            if not (A in db_elems and B in db_elems):
                return {"success": False, "error": f"Elements '{A}' and '{B}' must both exist in the database ({sorted(db_elems)})."}

            # Use A-B; x-axis will be X(B)
            elements = [A, B, 'VA']
            comp_el = B  # x variable
            phases = self._filter_phases_for_system(db, (A, B))
            
            # Set defaults
            temp_range = (min_temperature or 300, max_temperature or 1000)
            comp_step = composition_step or 0.02
            fig_size = (figure_width or 9, figure_height or 6)

            fig = plt.figure(figsize=fig_size)
            axes = fig.gca()

            import pycalphad.variables as v

            temp_points = max(12, min(60, int((temp_range[1] - temp_range[0]) / 20)))

            binplot(
                db, elements, phases,
                { v.X(comp_el): (0, 1, comp_step),
                  v.T: (temp_range[0], temp_range[1], temp_points),
                  v.P: 101325, v.N: 1 },
                plot_kwargs={'ax': axes, 'tielines': False, 'eq_kwargs': {'linewidth': 2}}
            )

            # generic labels / format
            self._add_phase_labels(axes, temp_range, phases, db, elements, v.X(comp_el))
            axes.set_xlabel(f"Mole Fraction {comp_el} (atomic basis)")
            axes.set_ylabel("Temperature (K)")
            axes.set_title(f"{A}-{B} Phase Diagram")
            axes.grid(True, alpha=0.3)
            axes.set_xlim(0, 1)
            axes.set_ylim(temp_range[0], temp_range[1])
            
            # Generate visual analysis before converting to base64
            print("Analyzing visual content...", flush=True)
            visual_analysis = self._analyze_visual_content(fig, axes, f"{A}-{B}", phases, temp_range)
            print(f"Visual analysis complete, length: {len(visual_analysis)}", flush=True)
            
            # Convert to base64
            print("Converting plot to base64...", flush=True)
            img_base64 = self._plot_to_base64(fig)
            print(f"Image conversion complete, size: {len(img_base64)} characters", flush=True)
            plt.close(fig)
            print("Plot closed, generating thermodynamic analysis...", flush=True)
            
            # Generate deterministic analysis
            thermodynamic_analysis = self._analyze_phase_diagram(db, f"{A}-{B}", phases, temp_range)
            print(f"CALPHAD: Generated thermodynamic analysis with length: {len(thermodynamic_analysis)}", flush=True)
            
            # Combine visual and thermodynamic analysis
            combined_analysis = f"{visual_analysis}\n\n{thermodynamic_analysis}"
            print(f"CALPHAD: Combined analysis length: {len(combined_analysis)}", flush=True)
            
            # Store the image data privately and return only a simple success message
            # The image will be handled by the _extract_image_display method
            setattr(self, '_last_image_data', img_base64)
            description = f"Generated binary phase diagram for {A}-{B} system showing stable phases as a function of temperature and composition"
            
            metadata = {
                "system": f"{A}-{B}",
                "database_file": db_path.name,
                "phases": phases,
                "temperature_range_K": temp_range,
                "composition_step": comp_step,
                "description": description,
                "analysis": combined_analysis,
                "visual_analysis": visual_analysis,
                "thermodynamic_analysis": thermodynamic_analysis,
                "image_info": {
                    "format": "png",
                    "size": fig_size,
                    "data_length": len(img_base64)
                }
            }
            setattr(self, '_last_image_metadata', metadata)
            print(f"CALPHAD: Stored metadata, analysis length: {len(metadata['analysis'])}", flush=True)
            
            # Return a simple success message - analysis will be shown in image display
            success_msg = f"Successfully generated {A}-{B} phase diagram showing phases {', '.join(phases)} over temperature range {temp_range[0]:.0f}-{temp_range[1]:.0f}K. The diagram displays phase boundaries and stable regions for this binary system."
            print(f"CALPHAD: Returning success message: {success_msg[:100]}...", flush=True)
            print(f"CALPHAD: SUCCESS MESSAGE LENGTH: {len(success_msg)} characters (should be short text, not base64)", flush=True)
            
            # Safeguard: ensure we're not accidentally returning base64 data
            if "data:image/png;base64," in success_msg or len(success_msg) > 1000:
                print(f"CALPHAD: ERROR - Success message contains base64 data or is too long! Truncating.", flush=True)
                result = "Successfully generated phase diagram. Image will be displayed separately."
            else:
                result = success_msg
            
            # Store the result for tooltip display
            if hasattr(self, 'recent_tool_outputs'):
                self.recent_tool_outputs.append({
                    "tool_name": "plot_binary_phase_diagram",
                    "result": result
                })
            return result
            
        except Exception as e:
            _log.exception(f"Error generating phase diagram for {system}")
            result = f"Failed to generate phase diagram for {system}: {str(e)}"
            if hasattr(self, 'recent_tool_outputs'):
                self.recent_tool_outputs.append({
                    "tool_name": "plot_binary_phase_diagram",
                    "result": result
                })
            return result

    # --- helper: build plotly fig ---
    def _plotly_comp_temp(self, temps, phase_data, labels, colors, special_Ts, title, subtitle) -> go.Figure:
        import numpy as np
        import plotly.graph_objects as go

        fig = go.Figure()
        EPS = 5e-3

        order = list(phase_data.keys())

        # column-stack fractions (nT, nP)
        Y = np.column_stack([np.asarray(phase_data[k], dtype=float) for k in order])
        Y = np.where(Y > EPS, Y, 0.0)  # visually hide tiny slivers

        # row-wise sum for renormalized hover
        row_sum = Y.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        frac_norm = Y / row_sum

        # cumulative for ribbon tops
        cum = np.cumsum(Y, axis=1)

        T = np.asarray(temps, dtype=float)
        Tc = T - 273.15

        for j, k in enumerate(order):
            y_j = Y[:, j]                    # TRUE phase fraction height at each T
            if float(np.nanmax(y_j)) < EPS:  # skip phases never visible
                continue

            upper = cum[:, j]
            lower = cum[:, j-1] if j > 0 else np.zeros_like(upper)

            label = labels.get(k, k)
            base  = k.split('#')[0]
            col   = colors.get(k) or colors.get(base) or "#808080"

            # Build per-point hover text; blank when the phase is absent at that T
            txt = []
            for i in range(len(T)):
                if y_j[i] <= EPS:
                    txt.append("")  # no row for this phase at this T
                else:
                    txt.append(
                        f"<b>{label}</b><br>"
                        f"T = {int(round(T[i]))} K ({int(round(Tc[i]))} °C)<br>"
                        f"fraction = {frac_norm[i, j]:.2f}"
                    )

            fig.add_trace(go.Scatter(
                x=T, y=upper,
                name=label,
                mode="lines",
                line=dict(width=0.5, color=col),
                fill="tozeroy" if j == 0 else "tonexty",
                fillcolor=col,
                text=txt,
                hovertemplate="%{text}<extra></extra>",
                showlegend=True,
                legendgroup=f"phase-{base}"
            ))

        fig.update_layout(
            template="plotly_white",
            hovermode="x unified",
            hoverlabel=dict(namelength=-1),
            title=dict(text=f"<b>{title}</b><br><sup>{subtitle}</sup>", y=0.95),
            xaxis=dict(title="Temperature (K)"),
            yaxis=dict(title="Phase Fraction", range=[0, 1]),
            showlegend=True,
            margin=dict(l=70, r=20, t=80, b=60),
        )
        return fig

    @ai_function(desc="PREFERRED for composition-specific thermodynamic questions. Plot phase stability vs temperature for a specific composition. Use for queries like 'Al20Zn80', 'Al80Zn20', single elements like 'Zn' or 'Al', melting point questions, phase transitions. Shows which phases are stable at different temperatures.")
    async def plot_composition_temperature(
        self,
        composition: Annotated[str, AIParam(desc="Specific composition like 'Al20Zn80', 'Al80Zn20', 'Zn30Al70', or single element like 'Zn' or 'Al'")],
        min_temperature: Annotated[Optional[float], AIParam(desc="Minimum temperature in Kelvin. Default: 300")] = None,
        max_temperature: Annotated[Optional[float], AIParam(desc="Maximum temperature in Kelvin. Default: 1000")] = None,
        composition_type: Annotated[Optional[str], AIParam(desc="Composition type: 'atomic' for at% or 'weight' for wt%. Default: 'atomic'")] = None,
        figure_width: Annotated[Optional[float], AIParam(desc="Figure width in inches. Default: 8")] = None,
        figure_height: Annotated[Optional[float], AIParam(desc="Figure height in inches. Default: 6")] = None,
        interactive: Annotated[Optional[str], AIParam(desc="Interactive output mode: 'html' for interactive Plotly HTML output. Default: 'html'")] = "html"
    ) -> str:
        """
        Generate a temperature vs phase stability plot for a specific composition.
        """
        if not PYCALPHAD_AVAILABLE:
            return "pycalphad library not available. Install with: pip install pycalphad"

        try:
            # reset previous artifacts
            if hasattr(self, '_last_image_metadata'):
                delattr(self, '_last_image_metadata')
            if hasattr(self, '_last_image_data'):
                delattr(self, '_last_image_data')

            # load DB
            db_path = self._get_database_path(composition or "")
            if not db_path:
                return "No .tdb found in tdbs/."
            db = Database(str(db_path))

            # parse composition → ((A,B), x_B, type)
            pair, xB, actual_comp_type = self._parse_composition(composition, composition_type or "atomic", db=db)
            A, B = pair
            db_elems = _db_elements(db)
            if not B:  # e.g., pure 'Al' provided; choose a partner? Here we require two.
                return f"Please specify a binary system like 'Al-Cu' or a composition like 'Al93Cu7'."
            if not (A in db_elems and B in db_elems):
                return f"Elements '{A},{B}' must exist in the database ({sorted(db_elems)})."

            phases = self._filter_phases_for_system(db, (A, B))

            temp_range = (min_temperature or 300, max_temperature or 1000)
            fig_size = (figure_width or 8, figure_height or 6)

            # single-point shortcut
            if abs(temp_range[1] - temp_range[0]) < 1.0:
                return await self._analyze_single_temperature_point(
                    db, f"{A}-{B}", phases, xB, temp_range[0], composition, actual_comp_type, A=A, B=B
                )

            # Generate phase data using pycalphad
            phase_data = {}
            from pycalphad import equilibrium
            import pycalphad.variables as v
            import numpy as np

            # temperature sampling
            span = temp_range[1] - temp_range[0]
            if span <= 100:
                nT = max(50, int(span))
            else:
                nT = max(100, min(350, int(span/2)))
            temps = np.linspace(temp_range[0], temp_range[1], nT)

            # equilibrium
            active = tuple(str(p) for p in phases)
            if self._solver6 is None:
                self._solver6 = Solver(max_phases=6)
            eq = equilibrium(
                db, [A, B, 'VA'], active,
                {v.T: temps, v.P: 101325.0, v.X(B): float(xB if xB is not None else 0.0)},
                calc_opts={'pdens': 2000},
                solver=self._solver6
            )

            # per-T phase moles (preserve miscibility via instance splitting)
            tvals = np.asarray(eq.squeeze().T.values, dtype=float)
            for i,_T in enumerate(tvals):
                eqT = eq.isel(T=i).squeeze()
                names = np.asarray(eqT.Phase.values, dtype=str).ravel()
                np_vals = np.asarray(eqT.NP.values, dtype=float).ravel()

                valid = [p for p in np.unique(names) if p]
                totals = {}
                split_candidates = {"FCC_A1", "HCP_A3", "BCC_A2"}
                for ph in valid:
                    if ph in split_candidates:
                        masks = self._split_by_region(eqT, ph) or \
                                self._split_phase_instances(eqT, ph, element=B, decimals=5, tol=2e-4)
                        if masks:
                            for inst, m in masks.items():
                                mol = float(np_vals[m].sum())
                                if mol > 0: totals[inst] = totals.get(inst, 0.0) + mol
                        else:
                            m = (names == ph)
                            mol = float(np_vals[m].sum())
                            if mol > 0: totals[ph] = totals.get(ph, 0.0) + mol
                    else:
                        m = (names == ph)
                        mol = float(np_vals[m].sum())
                        if mol > 0: totals[ph] = totals.get(ph, 0.0) + mol

                den = float(sum(totals.values())) or 1.0
                for key, mol in totals.items():
                    if key not in phase_data:
                        phase_data[key] = np.zeros_like(tvals, dtype=float)
                    phase_data[key][i] = mol/den

            # --- colors and labels ---
            label_map = {
                'LIQUID': 'Liquid',
                'FCC_A1': 'FCC (A1)',
                'FCC_A1#1': 'FCC (α₁)',
                'FCC_A1#2': 'FCC (α₂)',
                'HCP_A3': 'HCP (A3)',
                'HCP_ZN': 'HCP (A3)',
                'BCC_A2': 'BCC (A2)'
            }

            # Colorblind-friendly, high-contrast overrides
            custom_colors = {
                'LIQUID':  '#7f7f7f',  # grey
                'FCC_A1':  '#1f77b4',  # blue
                'FCC_A1#1':'#1f4e79',  # dark navy for α₁
                'FCC_A1#2':'#8DD1F1',  # lighter blue for α₂
                'HCP_A3':  '#d62728',  # red
                'HCP_ZN':  '#d62728',  # red
                'BCC_A2':  '#2ca02c'   # green
            }

            order    = ['LIQUID','FCC_A1#1','FCC_A1#2','FCC_A1','HCP_A3','HCP_ZN','BCC_A2']
            present  = list(phase_data.keys())
            keys     = [k for k in order if k in present] + [k for k in sorted(present) if k not in order]

            # --------- EVENT DETECTION ---------
            thr_on, thr_off = 0.01, 0.005
            def onset_offset(arr):
                on  = np.where(arr > thr_on)[0]
                off = np.where(arr > thr_off)[0]
                if len(on) == 0: return None, None
                return on[0], (off[-1] if len(off) else None)

            raw = []
            for k, arr in phase_data.items():
                i_on, i_off = onset_offset(np.asarray(arr))
                lbl = label_map.get(k, k)
                if i_on is not None:
                    Ton = float(tvals[i_on])
                    raw.append((Ton, 'appear', lbl))
                if i_off is not None and i_on is not None:
                    Toff = float(tvals[i_off])
                    if Toff - Ton > 20:
                        raw.append((Toff, 'disappear', lbl))

            # cluster nearby same-kind events
            def cluster(events, win=40):
                events = sorted(events, key=lambda x: (x[1], x[0]))
                out, bucket = [], []
                for T, kind, name in events:
                    if not bucket or (kind == bucket[-1][1] and abs(T - bucket[-1][0]) <= win):
                        bucket.append((T, kind, name))
                    else:
                        Tm = float(np.mean([t for t,_,__ in bucket]))
                        kindm = bucket[0][1]
                        names = sorted({n for _,_,n in bucket})
                        out.append((Tm, kindm, names))
                        bucket = [(T, kind, name)]
                if bucket:
                    Tm = float(np.mean([t for t,_,__ in bucket]))
                    kindm = bucket[0][1]
                    names = sorted({n for _,_,n in bucket})
                    out.append((Tm, kindm, names))
                return out

            clustered = cluster(raw, win=40)
            tops = [(T, names) for T,kind,names in clustered if kind=='appear']
            bots = [(T, names) for T,kind,names in clustered if kind=='disappear']

            # Generate both interactive Plotly HTML plot and static PNG
            ordered_phase_data = {k: phase_data[k] for k in keys if k in phase_data}
            
            # 1) Generate interactive Plotly HTML plot
            fig = self._plotly_comp_temp(
                temps=temps,
                phase_data=ordered_phase_data,
                labels=label_map,
                colors=custom_colors,
                special_Ts=[int(round(T)) for T,_ in tops] + [int(round(T)) for T,_ in bots],
                title="Phase Stability vs Temperature",
                subtitle=f"Composition: {A}{(1-float(xB))*100:.0f}{B}{float(xB)*100:.0f} ({'at%' if actual_comp_type=='atomic' else 'wt%'})"
            )
            outdir = Path("/Users/ahmedmuharram/thesis/interactive_plots")
            outdir.mkdir(parents=True, exist_ok=True)
            outfile = outdir / f"phase_stability_{A}{(1-float(xB))*100:.0f}{B}{float(xB)*100:.0f}.html"
            fig.write_html(str(outfile), include_plotlyjs="cdn", full_html=True)
            
            # 2) Generate static PNG plot
            import matplotlib.pyplot as plt
            
            fig_static, axes = plt.subplots(figsize=fig_size)
            
            # Plot the stacked areas
            order = ['LIQUID','FCC_A1#1','FCC_A1#2','FCC_A1','HCP_A3','HCP_ZN','BCC_A2']
            present = list(phase_data.keys())
            keys = [k for k in order if k in present] + [k for k in sorted(present) if k not in order]

            import matplotlib.colors as mcolors
            data, labels, colors = [], [], []
            for k in keys:
                if np.max(phase_data[k]) < 1e-6:
                    continue
                data.append(phase_data[k])
                labels.append(label_map.get(k, k))
                # pick override → fallback to base phase color → final grey
                col = custom_colors.get(
                    k,
                    custom_colors.get(k.split('#')[0],
                    "#808080")
                )
                colors.append(col)

            if data:
                axes.stackplot(temps, *data, labels=labels, colors=colors, alpha=0.8)
                axes.set_xlim(temp_range[0], temp_range[1])
                axes.set_ylim(0, 1)
                axes.set_xlabel("Temperature (K)")
                axes.set_ylabel("Phase Fraction")
                axes.grid(True, alpha=0.3)

                # Simple x-axis formatting - normal temperature ticks
                axes.tick_params(axis='x', labelsize=10)
                
                # Normal subplot adjustment
                fig_static.subplots_adjust(bottom=0.15)

                # Titles / Caption / Legend
                pct_B = float(xB) * 100 if xB is not None else 0
                pct_A = (1 - float(xB)) * 100 if xB is not None else 100
                comp_suffix = "at%" if actual_comp_type == "atomic" else "wt%"

                fig_static.suptitle("Phase Stability vs Temperature", y=0.95, fontsize=14, fontweight='bold')
                fig_static.text(0.5, 0.9125, f"Composition: {A}{pct_A:.0f}{B}{pct_B:.0f} ({comp_suffix})",
                             ha='center', va='top', fontsize=12)

                # De-duped legend centered vertically on right
                handles, lbls = axes.get_legend_handles_labels()
                # map by the plotted key rather than label text
                unique = {}
                for h, l in zip(handles, lbls):
                    unique[id(h)] = (h, l)   # any stable unique key; or carry your 'keys' array and zip keys->(h,l)
                axes.legend([h for h,l in unique.values()],
                            [l for h,l in unique.values()],
                            loc='center left', bbox_to_anchor=(1.02, 0.5),
                            borderaxespad=0., frameon=True)

            # Generate analysis
            thermodynamic_analysis = self._analyze_composition_temperature(phase_data, xB, temp_range, A, B)
            
            # Convert static plot to base64
            img_base64 = self._plot_to_base64(fig_static)
            plt.close(fig_static)
            
            metadata = {
                "system": f"{A}-{B}",
                "database_file": db_path.name,
                "phases": phases,
                "temperature_range_K": temp_range,
                "composition_info": {
                    "target_composition": xB,
                    "B_element": B,
                    "A_element": A,
                    "B_percentage": (float(xB)*100 if xB is not None else None),
                    "A_percentage": ((1-float(xB))*100 if xB is not None else None),
                    "composition_type": actual_comp_type,
                    "composition_suffix": comp_suffix,
                    "original_input": composition
                },
                "description": f"Generated phase stability plot for composition {A}{pct_A:.0f}{B}{pct_B:.0f} ({comp_suffix}) showing phase fractions vs temperature",
                "thermodynamic_analysis": thermodynamic_analysis,
                "analysis": thermodynamic_analysis,
                "image_info": {"format": "png", "size": fig_size, "data_length": len(img_base64)}
            }
            setattr(self, '_last_image_metadata', metadata)
            setattr(self, '_last_image_data', img_base64)

            # Generate clickable link for the interactive plot
            filename = outfile.name
            plot_url = f"http://localhost:8000/static/plots/{filename}"
            
            # Store the result for tooltip display
            result = f"Successfully generated phase stability plot for {A}{pct_A:.0f}{B}{pct_B:.0f} over {temp_range[0]}–{temp_range[1]}K.\n\n[Interactive Plot]({plot_url})"
            if hasattr(self, 'recent_tool_outputs'):
                self.recent_tool_outputs.append({
                    "tool_name": "plot_composition_temperature",
                    "result": result
                })
            return result
        except Exception as e:
            _log.exception(f"Error generating composition-temperature plot for {composition}")
            result = f"Failed to generate composition-temperature plot for {composition}: {str(e)}"
            if hasattr(self, 'recent_tool_outputs'):
                self.recent_tool_outputs.append({
                    "tool_name": "plot_composition_temperature",
                    "result": result
                })
            return result

    @ai_function(desc="Analyze and interpret the most recently generated phase diagram or composition plot. Provides detailed analysis of visual features, phase boundaries, and thermodynamic insights based on the actual generated plot.")
    async def analyze_last_generated_plot(self) -> str:
        """
        Analyze and interpret the most recently generated phase diagram or composition plot.
        
        This function provides detailed analysis of the visual content and thermodynamic features
        of the last generated plot, allowing the model to "see" and interpret its own output.
        """
        
        # Check if we have metadata from a recently generated plot
        metadata = getattr(self, '_last_image_metadata', None)
        if not metadata:
            return "No recently generated phase diagram or composition plot available to analyze. Please generate a plot first using plot_binary_phase_diagram() or plot_composition_temperature()."
        
        # Extract analysis components
        visual_analysis = metadata.get("visual_analysis", "")
        thermodynamic_analysis = metadata.get("thermodynamic_analysis", "")
        combined_analysis = metadata.get("analysis", "")
        
        # Check if image data is available (may be cleared after display to save memory)
        image_data = getattr(self, '_last_image_data', None)
        has_image = image_data is not None and len(image_data) > 1000
        image_was_generated = metadata.get("image_info", {}).get("data_length", 0) > 1000
        
        # Build comprehensive interpretation
        interpretation_parts = []
        
        interpretation_parts.append("# Analysis of Generated Phase Diagram")
        interpretation_parts.append(f"**System**: {metadata.get('system', 'Unknown')}")
        interpretation_parts.append(f"**Database**: {metadata.get('database_file', 'Unknown')}")
        
        # Temperature and composition info
        analysis_type = metadata.get('analysis_type', '')
        if analysis_type == 'single_temperature_point':
            temp_k = metadata.get('temperature_K', 0)
            interpretation_parts.append(f"**Temperature**: {temp_k:.0f}K ({temp_k-273.15:.0f}°C) [Single Point]")
        else:
            temp_range = metadata.get('temperature_range_K', (0, 0))
            interpretation_parts.append(f"**Temperature Range**: {temp_range[0]:.0f}-{temp_range[1]:.0f}K ({temp_range[0]-273.15:.0f} to {temp_range[1]-273.15:.0f}°C)")
        
        # Composition-specific info
        comp_info = metadata.get('composition_info', {}) or metadata.get('composition', {})
        if comp_info:
            zn_pct = comp_info.get('zn_percentage', 0) or comp_info.get('Zn_pct', 0)
            al_pct = comp_info.get('al_percentage', 0) or comp_info.get('Al_pct', 0)
            target_comp = comp_info.get('target_composition', 0) or comp_info.get('mole_fraction_Zn', 0)
            interpretation_parts.append(f"**Specific Composition**: Al{al_pct:.0f}Zn{zn_pct:.0f}")
            interpretation_parts.append(f"**Corresponds to**: x={target_comp:.3f} on binary phase diagram")
        
        # Phases present
        phases = metadata.get('phases', [])
        if phases:
            interpretation_parts.append(f"**Phases Included**: {', '.join(phases)}")
        
        # Image status
        if has_image:
            img_info = metadata.get('image_info', {})
            interpretation_parts.append(f"**Image Status**: Currently available in memory - {img_info.get('format', 'Unknown').upper()} format, {img_info.get('data_length', 0):,} characters")
        elif image_was_generated:
            img_info = metadata.get('image_info', {})
            interpretation_parts.append(f"**Image Status**: Generated and displayed (cleared from memory) - {img_info.get('format', 'Unknown').upper()} format, {img_info.get('data_length', 0):,} characters")
        
        interpretation_parts.append("")
        
        # Include the visual analysis if available
        if visual_analysis:
            interpretation_parts.append(visual_analysis)
            interpretation_parts.append("")
        
        # Include the thermodynamic analysis if available
        if thermodynamic_analysis:
            interpretation_parts.append(thermodynamic_analysis)
            interpretation_parts.append("")
        
        # Add interpretation guidance
        interpretation_parts.append("## How to Interpret This Information:")
        interpretation_parts.append("- **Visual Analysis**: Describes the actual plotted elements, colors, and visual features of the generated diagram")
        interpretation_parts.append("- **Thermodynamic Analysis**: Provides calculated key points, phase transitions, and material properties")
        interpretation_parts.append("- **Phase Boundaries**: Lines/regions where phase stability changes")
        interpretation_parts.append("- **Temperature Transitions**: Key temperatures where phase changes occur")
        
        # Add correlation guidance for binary vs composition plots
        if comp_info:
            interpretation_parts.append("- **Binary Diagram Correlation**: This composition-temperature plot corresponds to a vertical slice through the binary phase diagram")
        else:
            interpretation_parts.append("- **Composition Analysis**: Use plot_composition_temperature() for specific composition analysis")
        
        final_analysis = "\n".join(interpretation_parts)
        
        print(f"CALPHAD: Generated interpretation analysis, length: {len(final_analysis)}", flush=True)
        print(f"CALPHAD: Has visual analysis: {bool(visual_analysis)}, Has thermodynamic analysis: {bool(thermodynamic_analysis)}", flush=True)
        
        return final_analysis

    @ai_function(desc="List available element pairs (binaries) supported by the loaded thermodynamic database.")
    async def list_available_systems(self) -> Dict[str, Any]:
        db_path = self._get_database_path("")
        out = {"pycalphad_available": PYCALPHAD_AVAILABLE, "tdb_directory": str(self.tdb_dir), "systems": []}
        if not db_path or not PYCALPHAD_AVAILABLE:
            return out
        db = Database(str(db_path))
        elems = sorted(_db_elements(db))
        # Prefer Al-X pairs first (DB optimized there), then all pairs (X-Y where both in DB)
        al_pairs = [("AL", e) for e in elems if e != "AL"]
        other_pairs = []
        for i in range(len(elems)):
            for j in range(i+1, len(elems)):
                a, b = elems[i], elems[j]
                if a == "AL" or b == "AL": continue
                other_pairs.append((a, b))
        out["systems"] = [{"system": f"{a}-{b}"} for a,b in (al_pairs + other_pairs)]
        out["total_systems"] = len(out["systems"])
        out["database_file"] = db_path.name
        return out
