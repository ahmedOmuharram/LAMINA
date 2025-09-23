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

# Database mapping for different chemical systems
SYSTEM_DATABASES = {
    "Al-Zn": "COST507.tdb",
    "AlZn": "COST507.tdb", 
    "Al-Zn-*": "COST507.tdb",
    "aluminum-zinc": "COST507.tdb",
}

# Phase mappings for different systems
SYSTEM_PHASES = {
    "Al-Zn": ['LIQUID', 'FCC_A1', 'HCP_A3'],
    "AlZn": ['LIQUID', 'FCC_A1', 'HCP_A3'],
}

# Element mappings (handle common variations)
ELEMENT_ALIASES = {
    "aluminum": "AL",
    "aluminium": "AL", 
    "zinc": "ZN",
    "al": "AL",
    "zn": "ZN",
    "pure zinc": "ZN",
    "pure aluminum": "AL",
    "pure al": "AL",
    "pure zn": "ZN",
    "Zn": "ZN",
    "Al": "AL",
}


class CalPhadHandler(BaseHandler):
    
    def __init__(self):
        if not PYCALPHAD_AVAILABLE:
            _log.warning("pycalphad not available - CALPHAD functionality disabled")
        
        # Find TDB files directory
        self.tdb_dir = Path(__file__).parent.parent.parent.parent / "tdbs"
        if not self.tdb_dir.exists():
            _log.warning(f"TDB directory not found at {self.tdb_dir}")
        
        # Reusable solver instances to avoid hashing issues
        self._solver6 = None
    
    def _filter_phases_for_binary_system(self, all_phases: List[str], system: str) -> List[str]:
        """Filter phases for binary system calculations to improve performance."""
        # Remove invalid phases
        filtered = [phase for phase in all_phases if phase not in ['#']]
        
        # For large databases (like COST507), be more selective
        if len(all_phases) > 50:
            print(f"Large database detected ({len(all_phases)} phases), filtering for binary {system} system", flush=True)
            
            if system == "Al-Zn":
                # Essential phases for Al-Zn binary system
                essential_phases = [
                    'LIQUID',       # Liquid phase
                    'FCC_A1',       # Al-rich FCC phase
                    'HCP_A3',       # Zn-rich HCP phase  
                    'HCP_ZN',       # Pure Zn HCP phase
                    'BCC_A2',       # BCC phase (if present)
                ]
                
                # Add any phases that contain both AL and ZN
                alzn_phases = []
                for phase in all_phases:
                    phase_upper = phase.upper()
                    if ('AL' in phase_upper and 'ZN' in phase_upper) or phase in ['ALCUZN_T']:
                        alzn_phases.append(phase)
                
                # Combine and filter to only those that exist in database
                candidate_phases = essential_phases + alzn_phases
                filtered = [phase for phase in candidate_phases if phase in all_phases]
                
                print(f"Filtered to {len(filtered)} relevant phases for Al-Zn: {filtered}", flush=True)
        
        return filtered
    
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
    
    def _normalize_system(self, system: str) -> str:
        """Normalize system name to standard format."""
        # Remove spaces, hyphens, underscores and convert to title case
        clean = system.replace("-", "").replace("_", "").replace(" ", "").upper()
        
        # Handle common patterns
        if clean in ["ALZN", "ZNAI", "ALUMNIUM"]:
            return "Al-Zn"
        
        # Try direct lookup
        for key in SYSTEM_DATABASES:
            if clean.upper() == key.replace("-", "").upper():
                return key
                
        return system
    
    def _get_database_path(self, system: str) -> Optional[Path]:
        """Get the TDB file path for a given chemical system."""
        normalized = self._normalize_system(system)
        
        if normalized in SYSTEM_DATABASES:
            tdb_file = SYSTEM_DATABASES[normalized]
            tdb_path = self.tdb_dir / tdb_file
            if tdb_path.exists():
                return tdb_path
            _log.warning(f"TDB file not found: {tdb_path}")
        
        return None
    
    def _normalize_elements(self, elements: List[str]) -> List[str]:
        """Normalize element names."""
        normalized = []
        for elem in elements:
            clean = elem.strip().lower()
            if clean in ELEMENT_ALIASES:
                normalized.append(ELEMENT_ALIASES[clean])
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
    
    def _parse_composition(self, system_str: str, composition_type: str = "atomic") -> Tuple[str, Optional[float], str]:
        """Parse composition string like 'Al20Zn80' into system and mole fraction.
        
        Also handles single elements like 'Zn' -> Al0Zn100, 'Al' -> Al100Zn0
        
        Args:
            system_str: Composition string like 'Al20Zn80'
            composition_type: Either "atomic" (at%) or "weight" (wt%) - determines how percentages are interpreted
        
        Returns:
            (normalized_system, zn_mole_fraction, composition_type_used) where zn_mole_fraction is None if no composition specified
        """
        import re
        
        # Clean input
        clean_input = system_str.replace(' ', '').replace('-', '').strip()
        
        # Check for single element queries
        # First try element aliases
        elem_from_alias = ELEMENT_ALIASES.get(clean_input.lower())
        # Then try direct uppercase match
        elem_direct = clean_input.upper() if clean_input.upper() in ["AL", "ZN"] else None
        
        target_elem = elem_from_alias or elem_direct
        _log.info(f"Single element check: '{clean_input}' -> alias: '{elem_from_alias}', direct: '{elem_direct}', target: '{target_elem}'")
        
        if target_elem in ["AL", "ZN"]:
            if target_elem == "ZN":
                _log.info("Detected pure Zn composition")
                return "Al-Zn", 1.0, composition_type  # Pure Zn = Al0Zn100
            elif target_elem == "AL":
                _log.info("Detected pure Al composition")
                return "Al-Zn", 0.0, composition_type  # Pure Al = Al100Zn0
        
        # Check for percentage composition pattern: Al20Zn80, Al80Zn20, etc.
        pattern = r'([A-Za-z]+)(\d+)([A-Za-z]+)(\d+)'
        match = re.match(pattern, clean_input)
        
        if match:
            elem1, pct1, elem2, pct2 = match.groups()
            pct1, pct2 = int(pct1), int(pct2)
            
            print(f"CALPHAD: Parsed input '{system_str}' -> elem1='{elem1}', pct1={pct1}, elem2='{elem2}', pct2={pct2}", flush=True)
            
            # Normalize element names
            elem1_norm = ELEMENT_ALIASES.get(elem1.lower(), elem1.upper())
            elem2_norm = ELEMENT_ALIASES.get(elem2.lower(), elem2.upper())
            
            print(f"CALPHAD: Normalized elements -> elem1_norm='{elem1_norm}', elem2_norm='{elem2_norm}'", flush=True)
            
            # Check if this is Al-Zn system
            if {elem1_norm, elem2_norm} == {"AL", "ZN"}:
                # Determine Zn percentage
                if elem2_norm == "ZN":
                    zn_pct = pct2
                    al_pct = pct1
                    print(f"CALPHAD: elem2 is ZN -> al_pct={al_pct}, zn_pct={zn_pct}", flush=True)
                else:  # elem1_norm == "ZN"
                    zn_pct = pct1
                    al_pct = pct2
                    print(f"CALPHAD: elem1 is ZN -> al_pct={al_pct}, zn_pct={zn_pct}", flush=True)
                
                # Check if percentages add up to 100, if not, warn and normalize
                total_pct = al_pct + zn_pct
                if abs(total_pct - 100) > 0.1:
                    print(f"CALPHAD: WARNING - Input percentages don't add to 100% (Al{al_pct}% + Zn{zn_pct}% = {total_pct}%)", flush=True)
                    print(f"CALPHAD: Normalizing to: Al{al_pct/total_pct*100:.1f}% + Zn{zn_pct/total_pct*100:.1f}% = 100%", flush=True)
                    if composition_type.lower() == "atomic":
                        print(f"CALPHAD: NOTE - If you meant weight percent, specify composition_type='weight'", flush=True)
                        # Show what this would be as weight percent (convert from atomic to weight)
                        al_atomic_mass = 26.98
                        zn_atomic_mass = 65.38
                        # Weight = atomic_fraction * atomic_mass
                        al_weight = (al_pct/100) * al_atomic_mass
                        zn_weight = (zn_pct/100) * zn_atomic_mass
                        total_weight = al_weight + zn_weight
                        al_wt_pct = (al_weight / total_weight) * 100
                        zn_wt_pct = (zn_weight / total_weight) * 100
                        print(f"CALPHAD: As weight%: Al{al_wt_pct:.1f}wt%Zn{zn_wt_pct:.1f}wt% (adds to {al_wt_pct + zn_wt_pct:.1f}%)", flush=True)
                    # Normalize the percentages
                    al_pct_norm = al_pct / total_pct * 100
                    zn_pct_norm = zn_pct / total_pct * 100
                else:
                    al_pct_norm = al_pct
                    zn_pct_norm = zn_pct

                # Convert to atomic fraction based on composition_type
                if composition_type.lower() == "atomic":
                    # Direct conversion for atomic percent
                    zn_fraction = zn_pct_norm / 100.0
                    print(f"CALPHAD: Using atomic percent - Al{al_pct}at%Zn{zn_pct}at% (input) -> Al{al_pct_norm:.1f}at%Zn{zn_pct_norm:.1f}at% (normalized) -> x_Zn={zn_fraction:.3f}", flush=True)
                elif composition_type.lower() == "weight":
                    # Convert weight percent to atomic fraction
                    # Atomic masses: Al = 26.98, Zn = 65.38
                    al_atomic_mass = 26.98
                    zn_atomic_mass = 65.38
                    
                    # Weight fractions (using normalized percentages)
                    al_wt_frac = al_pct_norm / 100.0
                    zn_wt_frac = zn_pct_norm / 100.0
                    
                    # Convert to moles
                    al_moles = al_wt_frac / al_atomic_mass
                    zn_moles = zn_wt_frac / zn_atomic_mass
                    total_moles = al_moles + zn_moles
                    
                    # Atomic fractions
                    zn_fraction = zn_moles / total_moles if total_moles > 0 else 0
                    al_fraction = al_moles / total_moles if total_moles > 0 else 0
                    
                    print(f"CALPHAD: Converting weight percent - Al{al_pct}wt%Zn{zn_pct}wt% (input) -> Al{al_pct_norm:.1f}wt%Zn{zn_pct_norm:.1f}wt% (normalized) -> Al{al_fraction*100:.1f}at%Zn{zn_fraction*100:.1f}at% -> x_Zn={zn_fraction:.3f}", flush=True)
                else:
                    # Default to atomic
                    zn_fraction = zn_pct_norm / 100.0
                    composition_type = "atomic"
                    print(f"CALPHAD: Unknown composition type, defaulting to atomic percent", flush=True)
                
                return "Al-Zn", zn_fraction, composition_type
            else:
                # Other systems - try to construct system name
                system_name = f"{elem1_norm}-{elem2_norm}"
                # For now, assume second element is the one we track
                if elem2_norm == "ZN":
                    return system_name, pct2 / 100.0, composition_type
                else:
                    return system_name, None, composition_type
        
        # No composition found, treat as regular system name
        normalized_system = self._normalize_system(system_str)
        return normalized_system, None, composition_type
    
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
        
        # Add annotations for calculated points with improved visibility
        print(f"Adding annotations for {len(key_points)} key points", flush=True)
        for i, point in enumerate(key_points):
            print(f"Processing point {i+1}: {point['type']} at {point.get('composition_pct', 'N/A')}% Zn, {point.get('temperature', 'N/A')}K", flush=True)
            
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
                
                axes.annotate(f"Eutectic Point\n{point['composition_pct']:.1f}% Zn\n{point['temperature']:.0f}K", 
                             xy=(x_pos, y_pos), 
                             xytext=(x_pos + offset_x, y_pos + offset_y),
                             fontsize=9,
                             fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.8, edgecolor='red'),
                             arrowprops=dict(arrowstyle='->', color='red', lw=2, connectionstyle="arc3,rad=0.1"))
                print(f"Added eutectic annotation at ({x_pos:.3f}, {y_pos:.0f})", flush=True)
            
            elif point['type'] == 'pure_melting':
                element = point['element']
                element_name = 'Al' if element == 'AL' else 'Zn'
                color = 'orange' if element == 'AL' else 'darkgreen'
                
                # Position annotations clearly outside the plot area
                if element == 'AL':  # Pure Al (left side) - place to the left outside
                    text_x = -0.15    # Position text outside left boundary (x=0)
                    text_y = y_pos + (temp_range[1] - temp_range[0]) * 0.1  # Slightly above the point
                else:  # Pure Zn (right side) - place to the right outside  
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
                axes.annotate(f"Phase Transition\n{point['composition_pct']:.1f}% Zn\n{point['temperature']:.0f}K", 
                             xy=(x_pos, y_pos), 
                             xytext=(x_pos + 0.12, y_pos + (temp_range[1] - temp_range[0]) * 0.06),
                             fontsize=7,
                             bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.7, edgecolor='orange'),
                             arrowprops=dict(arrowstyle='->', color='orange', lw=1.5))
                print(f"Added phase transition annotation at ({x_pos:.3f}, {y_pos:.0f})", flush=True)
        
        
        # Let binplot handle the legend - it knows the colors it used        
        print("All phase diagram enhancements complete!", flush=True)

    def _calculate_key_thermodynamic_points(self, db, elements, phases, comp_var, temp_range):
        """Extract key thermodynamic points directly from the existing plot data."""
        import numpy as np
        
        print("Extracting key thermodynamic points from plot data...", flush=True)
        key_points = []
        
        try:
            # Get the plot axes to extract data from the existing phase diagram
            import matplotlib.pyplot as plt
            ax = plt.gca()
            
            # Get all line objects from the plot (these are the phase boundaries)
            lines = ax.get_lines()
            
            if not lines:
                print("No lines found in plot to extract data from", flush=True)
                return key_points
            
            print(f"Found {len(lines)} lines in the plot", flush=True)
            
            # Find the liquidus line (the one with highest average temperature)
            liquidus_line = None
            max_avg_temp = -float('inf')
            
            for line in lines:
                xdata = line.get_xdata()
                ydata = line.get_ydata()
                
                if len(xdata) > 0 and len(ydata) > 0:
                    avg_temp = np.mean(ydata)
                    print(f"Line: x from {np.min(xdata):.3f} to {np.max(xdata):.3f}, y from {np.min(ydata):.1f} to {np.max(ydata):.1f}, avg={avg_temp:.1f}K", flush=True)
                    
                    if avg_temp > max_avg_temp:
                        max_avg_temp = avg_temp
                        liquidus_line = line
            
            if liquidus_line is not None:
                x_liquidus = liquidus_line.get_xdata()
                y_liquidus = liquidus_line.get_ydata()
                
                print(f"Selected liquidus line with {len(x_liquidus)} points, avg temp {max_avg_temp:.1f}K", flush=True)
                
                # Find eutectic (minimum on the liquidus line)
                min_idx = np.argmin(y_liquidus)
                eutectic_comp = x_liquidus[min_idx]
                eutectic_temp = y_liquidus[min_idx]
                
                key_points.append({
                    'type': 'eutectic',
                    'composition': eutectic_comp,
                    'composition_pct': eutectic_comp * 100,
                    'temperature': eutectic_temp
                })
                print(f"Found eutectic on liquidus: {eutectic_temp:.0f}K at {eutectic_comp*100:.1f}% Zn", flush=True)
                
                # Collect all points from all lines for melting points
                all_x_points = []
                all_y_points = []
                for line in lines:
                    all_x_points.extend(line.get_xdata())
                    all_y_points.extend(line.get_ydata())
                
                # Find Al melting point (leftmost high-temperature point)
                left_points = [(x, y) for x, y in zip(all_x_points, all_y_points) if x < 0.1]
                if left_points:
                    al_comp, al_temp = max(left_points, key=lambda item: item[1])
                    key_points.append({
                        'type': 'pure_melting',
                        'element': 'AL',
                        'composition': al_comp,
                        'composition_pct': al_comp * 100,
                        'temperature': al_temp
                    })
                    print(f"Found Al melting point: {al_temp:.0f}K at x={al_comp:.3f}", flush=True)
                
                # Find Zn melting point (rightmost high-temperature point)
                right_points = [(x, y) for x, y in zip(all_x_points, all_y_points) if x > 0.9]
                if right_points:
                    zn_comp, zn_temp = max(right_points, key=lambda item: item[1])
                    key_points.append({
                        'type': 'pure_melting',
                        'element': 'ZN',
                        'composition': zn_comp,
                        'composition_pct': zn_comp * 100,
                        'temperature': zn_temp
                    })
                    print(f"Found Zn melting point: {zn_temp:.0f}K at x={zn_comp:.3f}", flush=True)
            else:
                print("Could not identify liquidus line", flush=True)
            
            print(f"Final result: {len(key_points)} key thermodynamic points extracted from plot", flush=True)
            for i, point in enumerate(key_points):
                print(f"  {i+1}. {point['type']}: {point['temperature']:.0f}K at {point['composition_pct']:.1f}% Zn", flush=True)
            
        except Exception as e:
            print(f"Error extracting key points from plot: {e}", flush=True)
            import traceback
            print(f"Traceback: {traceback.format_exc()}", flush=True)
        
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

    async def _analyze_single_temperature_point(self, db, normalized_system: str, phases: List[str], 
                                              target_composition: float, temperature: float, 
                                              original_composition: str, composition_type: str = "atomic") -> str:
        """Analyze phase equilibrium at a single temperature point instead of generating a plot."""
        try:
            from pycalphad import equilibrium
            import pycalphad.variables as v
            
            print(f"CALPHAD: Analyzing single temperature point: {temperature:.0f}K for composition {original_composition}", flush=True)
            
            # Calculate equilibrium at the single temperature
            eq_result = equilibrium(
                db, 
                ['AL', 'ZN', 'VA'], 
                phases,
                {v.T: temperature, v.P: 101325, v.X('ZN'): target_composition}
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
            pct_zn = target_composition * 100
            pct_al = (1 - target_composition) * 100
            comp_suffix = "at%" if composition_type == "atomic" else "wt%"
            
            analysis_parts = []
            analysis_parts.append(f"# Single Point Analysis: Al{pct_al:.0f}Zn{pct_zn:.0f} ({comp_suffix}) at {temperature:.0f}K ({temperature-273.15:.0f}°C)")
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
                    "Al_pct": pct_al,
                    "Zn_pct": pct_zn,
                    "mole_fraction_Zn": target_composition
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
    
    def _analyze_composition_temperature(self, composition_data: dict, target_composition: float, temp_range: Tuple[float, float]) -> str:
        """Generate detailed deterministic analysis of composition-temperature data."""
        try:
            pct_zn = target_composition * 100
            pct_al = (1 - target_composition) * 100
            
            analysis_parts = []
            analysis_parts.append(f"## Detailed Phase Stability Analysis: Al{pct_al:.0f}Zn{pct_zn:.0f}\n")
            
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
            
            # Normalize system name (no composition parsing for phase diagrams)
            normalized_system = self._normalize_system(system)
            _log.info(f"Plotting phase diagram for system: {system} -> {normalized_system}")
            
            # Get database
            db_path = self._get_database_path(normalized_system)
            if not db_path:
                return {
                    "success": False,
                    "error": f"No thermodynamic database available for system '{system}'. Supported: {list(SYSTEM_DATABASES.keys())}"
                }
            
            # Load database
            db = Database(str(db_path))
            _log.info(f"Loaded database: {db_path}")
            
            # Get all phases from the database automatically
            all_db_phases = list(db.phases.keys())
            print(f"All phases in database: {all_db_phases}", flush=True)
            
            # Filter phases for better performance with large databases
            phases = self._filter_phases_for_binary_system(all_db_phases, normalized_system)
            print(f"Using phases for binary diagram: {phases}", flush=True)
            
            # Set defaults
            temp_range = (min_temperature or 300, max_temperature or 1000)
            comp_step = composition_step or 0.02
            fig_size = (figure_width or 9, figure_height or 6)
            
            # Create figure
            fig = plt.figure(figsize=fig_size)
            axes = fig.gca()
            
            # For Al-Zn system
            if normalized_system == "Al-Zn":
                import pycalphad.variables as v
                elements = ['AL', 'ZN', 'VA']
                comp_var = v.X('ZN')  # Zinc composition
                
                # Plot phase diagram with phase labels
                # Use more temperature points for smoother curves at extreme ranges
                temp_points = max(10, min(50, int((temp_range[1] - temp_range[0]) / 20)))
                
                # Use standard binplot with enhanced shading overlay
                print("Generating standard phase diagram with enhanced shading...", flush=True)
                
                # First, create the standard binplot
                try:
                    binplot(
                        db, 
                        elements, 
                        phases, 
                        {
                            comp_var: (0, 1, comp_step),
                            v.T: (temp_range[0], temp_range[1], temp_points),
                            v.P: 101325,  # 1 atm
                            v.N: 1
                        }, 
                        plot_kwargs={
                            'ax': axes,
                            'tielines': False,  # Cleaner look
                            'eq_kwargs': {'linewidth': 2}  # Slightly thicker boundary lines
                        }
                    )
                    print("Standard binplot completed", flush=True)
                    
                    # Add enhanced phase labels with dynamic key points
                    self._add_phase_labels(axes, temp_range, phases, db, elements, comp_var)
                    print("Enhanced phase labels added successfully", flush=True)
                    
                except Exception as e:
                    print(f"Binplot failed: {e}", flush=True)
                    # Continue with empty plot
                
                # Phase labels are now included in the shading method
                
                axes.set_xlabel('Mole Fraction Zn (atomic basis)')
                axes.set_ylabel('Temperature (K)')
                
                title = f'{normalized_system} Phase Diagram'
                
                axes.set_title(title)
                axes.grid(True, alpha=0.3)
                
                # Improve axis formatting for extreme ranges
                axes.set_xlim(0, 1)
                axes.set_ylim(temp_range[0], temp_range[1])
                
                # Use scientific notation for very large temperature ranges
                if temp_range[1] - temp_range[0] > 2000:
                    axes.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
                    
                # Ensure reasonable number of ticks
                from matplotlib.ticker import MaxNLocator
                axes.yaxis.set_major_locator(MaxNLocator(nbins=8))
            
            # Generate visual analysis before converting to base64
            print("Analyzing visual content...", flush=True)
            visual_analysis = self._analyze_visual_content(fig, axes, normalized_system, phases, temp_range)
            print(f"Visual analysis complete, length: {len(visual_analysis)}", flush=True)
            
            # Convert to base64
            print("Converting plot to base64...", flush=True)
            img_base64 = self._plot_to_base64(fig)
            print(f"Image conversion complete, size: {len(img_base64)} characters", flush=True)
            plt.close(fig)
            print("Plot closed, generating thermodynamic analysis...", flush=True)
            
            # Generate deterministic analysis
            thermodynamic_analysis = self._analyze_phase_diagram(db, normalized_system, phases, temp_range)
            print(f"CALPHAD: Generated thermodynamic analysis with length: {len(thermodynamic_analysis)}", flush=True)
            
            # Combine visual and thermodynamic analysis
            combined_analysis = f"{visual_analysis}\n\n{thermodynamic_analysis}"
            print(f"CALPHAD: Combined analysis length: {len(combined_analysis)}", flush=True)
            
            # Store the image data privately and return only a simple success message
            # The image will be handled by the _extract_image_display method
            setattr(self, '_last_image_data', img_base64)
            description = f"Generated binary phase diagram for {normalized_system} system showing stable phases as a function of temperature and composition"
            
            metadata = {
                "system": normalized_system,
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
            success_msg = f"Successfully generated {normalized_system} phase diagram showing phases {', '.join(phases)} over temperature range {temp_range[0]:.0f}-{temp_range[1]:.0f}K. The diagram displays phase boundaries and stable regions for this binary system."
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

            # parse composition
            comp_type = composition_type or "atomic"
            normalized_system, target_composition, actual_comp_type = self._parse_composition(composition, comp_type)
            if target_composition is None:
                return f"Could not parse composition from '{composition}'. Use format like 'Al20Zn80' or 'Al80Zn20'."

            db_path = self._get_database_path(normalized_system)
            if not db_path:
                return f"No thermodynamic database available for system '{normalized_system}'. Supported: {list(SYSTEM_DATABASES.keys())}"
            db = Database(str(db_path))

            all_db_phases = list(db.phases.keys())
            phases = self._filter_phases_for_binary_system(all_db_phases, normalized_system)

            temp_range = (min_temperature or 300, max_temperature or 1000)
            fig_size = (figure_width or 8, figure_height or 6)

            # single-point shortcut
            if abs(temp_range[1] - temp_range[0]) < 1.0:
                return await self._analyze_single_temperature_point(
                    db, normalized_system, phases, target_composition, temp_range[0], composition, actual_comp_type
                )

            # Generate phase data using pycalphad
            phase_data = {}
            if normalized_system == "Al-Zn":
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
                    db, ['AL','ZN','VA'], active,
                    {v.T: temps, v.P: 101325.0, v.X('ZN'): float(target_composition)},
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
                                    self._split_phase_instances(eqT, ph, element="ZN", decimals=5, tol=2e-4)
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
                    'FCC_A1': 'FCC (Al-rich)',
                    'FCC_A1#1': 'FCC (α₁)',
                    'FCC_A1#2': 'FCC (α₂)',
                    'HCP_A3': 'HCP (Zn-rich)',
                    'HCP_ZN': 'HCP (Zn-rich)',
                    'BCC_A2': 'BCC'
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
                    subtitle=f"Composition: Al{(1-target_composition)*100:.0f}Zn{target_composition*100:.0f} ({'at%' if actual_comp_type=='atomic' else 'wt%'})"
                )
                outdir = Path("/Users/ahmedmuharram/thesis/interactive_plots")
                outdir.mkdir(parents=True, exist_ok=True)
                outfile = outdir / f"phase_stability_Al{(1-target_composition)*100:.0f}Zn{target_composition*100:.0f}.html"
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
                    pct_zn = target_composition * 100
                    pct_al = (1 - target_composition) * 100
                    comp_suffix = "at%" if actual_comp_type == "atomic" else "wt%"

                    fig_static.suptitle("Phase Stability vs Temperature", y=0.95, fontsize=14, fontweight='bold')
                    fig_static.text(0.5, 0.9125, f"Composition: Al{pct_al:.0f}Zn{pct_zn:.0f} ({comp_suffix})",
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
                thermodynamic_analysis = self._analyze_composition_temperature(phase_data, target_composition, temp_range)
                
                pct_zn = target_composition * 100
                pct_al = (1 - target_composition) * 100
                comp_suffix = "at%" if actual_comp_type == "atomic" else "wt%"
                
                # Convert static plot to base64
                img_base64 = self._plot_to_base64(fig_static)
                plt.close(fig_static)
                
                metadata = {
                    "system": normalized_system,
                    "database_file": db_path.name,
                    "phases": phases,
                    "temperature_range_K": temp_range,
                    "composition_info": {
                        "target_composition": target_composition,
                        "zn_percentage": pct_zn,
                        "al_percentage": pct_al,
                        "composition_type": actual_comp_type,
                        "composition_suffix": comp_suffix,
                        "original_input": composition
                    },
                    "description": f"Generated phase stability plot for composition Al{pct_al:.0f}Zn{pct_zn:.0f} ({comp_suffix}) showing phase fractions vs temperature",
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
                result = f"Successfully generated phase stability plot for Al{pct_al:.0f}Zn{pct_zn:.0f} over {temp_range[0]}–{temp_range[1]}K.\n\n[Interactive Plot]({plot_url})"
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

    @ai_function(desc="List available chemical systems and their thermodynamic databases for phase diagram calculation.")
    async def list_available_systems(self) -> Dict[str, Any]:
        """List all available chemical systems for phase diagram calculation."""
        
        available_systems = []
        for system, db_file in SYSTEM_DATABASES.items():
            db_path = self.tdb_dir / db_file
            status = "available" if db_path.exists() else "database_missing"
            phases = SYSTEM_PHASES.get(system, ["unknown"])
            
            available_systems.append({
                "system": system,
                "database_file": db_file,
                "status": status,
                "phases": phases,
                "description": f"Binary system with phases: {', '.join(phases)}"
            })
        
        return {
            "available_systems": available_systems,
            "pycalphad_available": PYCALPHAD_AVAILABLE,
            "tdb_directory": str(self.tdb_dir),
            "total_systems": len(available_systems)
        }
