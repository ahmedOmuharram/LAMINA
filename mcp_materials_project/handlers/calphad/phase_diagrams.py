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

try:
    from pycalphad import Database, binplot
    import pycalphad.variables as v
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
    "Al-Zn": "alzn_mey.tdb",
    "AlZn": "alzn_mey.tdb", 
    "Al-Zn-*": "alzn_mey.tdb",
    "aluminum-zinc": "alzn_mey.tdb",
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
    
    def _parse_composition(self, system_str: str) -> Tuple[str, Optional[float]]:
        """Parse composition string like 'Al20Zn80' into system and mole fraction.
        
        Also handles single elements like 'Zn' -> Al0Zn100, 'Al' -> Al100Zn0
        
        Returns:
            (normalized_system, zn_mole_fraction) where zn_mole_fraction is None if no composition specified
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
                return "Al-Zn", 1.0  # Pure Zn = Al0Zn100
            elif target_elem == "AL":
                _log.info("Detected pure Al composition")
                return "Al-Zn", 0.0  # Pure Al = Al100Zn0
        
        # Check for percentage composition pattern: Al20Zn80, Al80Zn20, etc.
        pattern = r'([A-Za-z]+)(\d+)([A-Za-z]+)(\d+)'
        match = re.match(pattern, clean_input)
        
        if match:
            elem1, pct1, elem2, pct2 = match.groups()
            pct1, pct2 = int(pct1), int(pct2)
            
            # Normalize element names
            elem1_norm = ELEMENT_ALIASES.get(elem1.lower(), elem1.upper())
            elem2_norm = ELEMENT_ALIASES.get(elem2.lower(), elem2.upper())
            
            # Check if this is Al-Zn system
            if {elem1_norm, elem2_norm} == {"AL", "ZN"}:
                # Determine Zn mole fraction
                if elem2_norm == "ZN":
                    zn_fraction = pct2 / 100.0
                else:  # elem1_norm == "ZN"
                    zn_fraction = pct1 / 100.0
                
                return "Al-Zn", zn_fraction
            else:
                # Other systems - try to construct system name
                system_name = f"{elem1_norm}-{elem2_norm}"
                # For now, assume second element is the one we track
                if elem2_norm == "ZN":
                    return system_name, pct2 / 100.0
                else:
                    return system_name, None
        
        # No composition found, treat as regular system name
        normalized_system = self._normalize_system(system_str)
        return normalized_system, None
    
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
            
            # Determine melting point more precisely
            if 'LIQUID' in composition_data:
                liquid_array = np.array(composition_data['LIQUID'])
                melting_indices = np.where(liquid_array > 0.5)[0]  # Where liquid becomes dominant
                if len(melting_indices) > 0:
                    melting_point = temp_array[melting_indices[0]]
                    analysis_parts.append(f"\n### Melting Point Analysis:")
                    analysis_parts.append(f"**Primary melting begins at: {melting_point:.0f}K ({melting_point-273.15:.0f}°C)**")
                    
                    # Complete melting
                    fully_liquid = np.where(liquid_array > 0.99)[0]
                    if len(fully_liquid) > 0:
                        complete_melting = temp_array[fully_liquid[0]]
                        analysis_parts.append(f"**Complete melting achieved at: {complete_melting:.0f}K ({complete_melting-273.15:.0f}°C)**")
                        if complete_melting > melting_point + 10:
                            analysis_parts.append(f"- **Melting range**: {complete_melting - melting_point:.0f}K ({(complete_melting - melting_point):.0f}°C)")
                            analysis_parts.append("- **Behavior**: Gradual melting over temperature range (alloy behavior)")
                        else:
                            analysis_parts.append("- **Behavior**: Sharp melting transition (near-eutectic composition)")
            
            # Microstructural analysis
            analysis_parts.append("\n### Microstructural Evolution:")
            
            # At room temperature
            room_temp_idx = np.argmin(np.abs(temp_array - 298.15))  # ~25°C
            room_temp_phases = []
            total_fraction = 0
            for phase_name, phase_values in composition_data.items():
                fraction = phase_values[room_temp_idx]
                if fraction > 0.01:
                    phase_label = {
                        'LIQUID': 'Liquid',
                        'FCC_A1': 'FCC (Al-rich)',
                        'HCP_A3': 'HCP (Zn-rich)'
                    }.get(phase_name, phase_name)
                    room_temp_phases.append(f"{phase_label} ({fraction:.1%})")
                    total_fraction += fraction
            
            if room_temp_phases:
                analysis_parts.append(f"**At room temperature (25°C)**: {', '.join(room_temp_phases)} (Total: {total_fraction:.1%})")
            else:
                analysis_parts.append("**At room temperature (25°C)**: No significant phases detected")
            
            # Cooling behavior
            analysis_parts.append("\n### Solidification Behavior (upon cooling from liquid):")
            if 'LIQUID' in composition_data:
                liquid_array = np.array(composition_data['LIQUID'])
                # Find where liquid starts to solidify
                solidification_start = np.where(liquid_array < 0.99)[0]
                if len(solidification_start) > 0:
                    solidus_temp = temp_array[solidification_start[0]]
                    analysis_parts.append(f"1. **Solidification begins**: {solidus_temp:.0f}K ({solidus_temp-273.15:.0f}°C)")
                    
                    # Find primary solidifying phase
                    for phase_name, phase_values in composition_data.items():
                        if phase_name != 'LIQUID':
                            phase_array = np.array(phase_values)
                            if phase_array[solidification_start[0]] > 0.01:
                                phase_label = {
                                    'FCC_A1': 'FCC (Al-rich)',
                                    'HCP_A3': 'HCP (Zn-rich)'
                                }.get(phase_name, phase_name)
                                analysis_parts.append(f"2. **Primary phase**: {phase_label} crystallizes first")
                                break
                    
                    # Complete solidification
                    fully_solid = np.where(liquid_array < 0.01)[0]
                    if len(fully_solid) > 0:
                        solidus_end = temp_array[fully_solid[0]]
                        analysis_parts.append(f"3. **Solidification complete**: {solidus_end:.0f}K ({solidus_end-273.15:.0f}°C)")
                        
                        if solidus_temp - solidus_end > 20:
                            analysis_parts.append(f"4. **Solidification range**: {solidus_temp - solidus_end:.0f}K (extended solidification)")
                        else:
                            analysis_parts.append("4. **Solidification range**: Narrow (rapid solidification)")
            
            # Processing recommendations
            analysis_parts.append("\n### Processing Recommendations:")
            
            if pct_zn > 80:
                analysis_parts.append("**Casting temperatures**: 50-100K above liquidus for good fluidity")
                analysis_parts.append("**Cooling rate**: Fast cooling recommended to prevent excessive Zn segregation")
                analysis_parts.append("**Mold preheating**: 473-523K to prevent cold shuts")
            elif pct_zn > 50:
                analysis_parts.append("**Heat treatment potential**: Two-phase region allows for age hardening")
                analysis_parts.append("**Welding considerations**: Moderate solidification range requires controlled cooling")
                analysis_parts.append("**Forming**: Best worked in single-phase temperature ranges")
            else:
                analysis_parts.append("**High-temperature forming**: Utilize liquid + solid regions for thixoforming")
                analysis_parts.append("**Solution treatment**: Heat to single-phase region before quenching")
                analysis_parts.append("**Joining**: Higher melting point suitable for brazing operations")
            
            # Material properties implications
            analysis_parts.append("\n### Expected Material Properties:")
            
            if pct_zn > 70:
                analysis_parts.append("- **Mechanical**: Lower strength, higher ductility, excellent formability")
                analysis_parts.append("- **Corrosion**: Excellent atmospheric corrosion resistance")
                analysis_parts.append("- **Electrical**: Good electrical conductivity (zinc-like)")
                analysis_parts.append("- **Thermal**: Lower melting point enables low-temperature processing")
            elif pct_zn > 30:
                analysis_parts.append("- **Mechanical**: Balanced strength-ductility, work hardenable")
                analysis_parts.append("- **Corrosion**: Good general corrosion resistance")
                analysis_parts.append("- **Electrical**: Moderate electrical conductivity")
                analysis_parts.append("- **Thermal**: Moderate thermal conductivity")
            else:
                analysis_parts.append("- **Mechanical**: Higher strength, age-hardenable, lower ductility")
                analysis_parts.append("- **Corrosion**: Aluminum-like corrosion behavior with passive layer")
                analysis_parts.append("- **Electrical**: High electrical conductivity (aluminum-like)")
                analysis_parts.append("- **Thermal**: High thermal conductivity, lightweight")
            
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
            
            # Use phases from database, but filter out unwanted ones like '#'
            phases = [phase for phase in all_db_phases if phase not in ['#']]
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
                
                axes.set_xlabel('Mole Fraction Zn')
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
            
            # Convert to base64
            print("Converting plot to base64...", flush=True)
            img_base64 = self._plot_to_base64(fig)
            print(f"Image conversion complete, size: {len(img_base64)} characters", flush=True)
            plt.close(fig)
            print("Plot closed, generating analysis...", flush=True)
            
            # Generate deterministic analysis
            analysis = self._analyze_phase_diagram(db, normalized_system, phases, temp_range)
            print(f"CALPHAD: Generated analysis with length: {len(analysis)}", flush=True)
            
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
                "analysis": analysis,
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
            return success_msg
            
        except Exception as e:
            _log.exception(f"Error generating phase diagram for {system}")
            return f"Failed to generate phase diagram for {system}: {str(e)}"

    @ai_function(desc="PREFERRED for composition-specific thermodynamic questions. Plot phase stability vs temperature for a specific composition. Use for queries like 'Al20Zn80', 'Al80Zn20', single elements like 'Zn' or 'Al', melting point questions, phase transitions. Shows which phases are stable at different temperatures.")
    async def plot_composition_temperature(
        self,
        composition: Annotated[str, AIParam(desc="Specific composition like 'Al20Zn80', 'Al80Zn20', 'Zn30Al70', or single element like 'Zn' or 'Al'")],
        min_temperature: Annotated[Optional[float], AIParam(desc="Minimum temperature in Kelvin. Default: 300")] = None,
        max_temperature: Annotated[Optional[float], AIParam(desc="Maximum temperature in Kelvin. Default: 1000")] = None,
        figure_width: Annotated[Optional[float], AIParam(desc="Figure width in inches. Default: 8")] = None,
        figure_height: Annotated[Optional[float], AIParam(desc="Figure height in inches. Default: 6")] = None
    ) -> str:
        """
        Generate a temperature vs phase stability plot for a specific composition.
        
        This shows which phases are stable as temperature changes for a fixed composition.
        Useful for understanding phase transformations during heating/cooling.
        """
        
        if not PYCALPHAD_AVAILABLE:
            return "pycalphad library not available. Install with: pip install pycalphad"
        
        try:
            # Parse composition
            normalized_system, target_composition = self._parse_composition(composition)
            if target_composition is None:
                return f"Could not parse composition from '{composition}'. Use format like 'Al20Zn80' or 'Al80Zn20'."
            
            _log.info(f"Plotting composition-temperature for: {composition} -> {normalized_system}, X_Zn={target_composition}")
            
            # Get database
            db_path = self._get_database_path(normalized_system)
            if not db_path:
                return f"No thermodynamic database available for system '{normalized_system}'. Supported: {list(SYSTEM_DATABASES.keys())}"
            
            # Load database
            db = Database(str(db_path))
            _log.info(f"Loaded database: {db_path}")
            
            # Get all phases from the database automatically
            all_db_phases = list(db.phases.keys())
            print(f"All phases in database: {all_db_phases}", flush=True)
            
            # Use phases from database, but filter out unwanted ones like '#'
            phases = [phase for phase in all_db_phases if phase not in ['#']]
            print(f"Using phases for binary diagram: {phases}", flush=True)
            
            # Set defaults
            temp_range = (min_temperature or 300, max_temperature or 1000)
            fig_size = (figure_width or 8, figure_height or 6)
            
            # Create figure
            fig = plt.figure(figsize=fig_size)
            axes = fig.gca()
            
            # For Al-Zn system, calculate phase fractions vs temperature
            if normalized_system == "Al-Zn":
                from pycalphad import equilibrium
                import numpy as np
                
                # Create temperature array
                temps = np.linspace(temp_range[0], temp_range[1], 50)
                
                # Calculate equilibrium at fixed composition
                eq_result = equilibrium(
                    db, 
                    ['AL', 'ZN', 'VA'], 
                    phases,
                    {v.T: temps, v.P: 101325, v.X('ZN'): target_composition}
                )
                
                # Plot phase fractions
                phase_colors = {
                    'LIQUID': '#1f77b4',    # Blue
                    'FCC_A1': '#ff7f0e',    # Orange  
                    'HCP_A3': '#2ca02c',    # Green
                }
                
                # Use a different approach - calculate equilibrium at each temperature separately
                temp_values = temps
                
                # Initialize arrays for each phase
                phase_data = {phase: np.zeros(len(temp_values)) for phase in phases}
                
                # Calculate equilibrium at each temperature point individually
                for i, temp in enumerate(temp_values):
                    try:
                        eq_single = equilibrium(
                            db, 
                            ['AL', 'ZN', 'VA'], 
                            phases,
                            {v.T: temp, v.P: 101325, v.X('ZN'): target_composition}
                        )
                        
                        
                        # Calculate phase fractions at this temperature
                        total_moles = eq_single.NP.sum()
                        
                        # Debug for a few temperature points
                        if i in [10, 25, 40]:  # Debug a few points
                            print(f"Debug T={temp:.0f}K: total_moles={float(total_moles):.6f}", flush=True)
                            unique_phases = np.unique(eq_single.Phase.values)
                            print(f"  Found phases: {unique_phases}", flush=True)
                        
                        # Get the phase fractions directly from the equilibrium result
                        phase_fractions = {}
                        
                        # Group by phase and sum moles
                        for phase_name in phases:
                            phase_mask = eq_single.Phase == phase_name
                            if phase_mask.any():
                                # Sum moles for this phase
                                phase_moles = float(eq_single.NP.where(phase_mask, eq_single.NP, 0).sum())
                                phase_fractions[phase_name] = phase_moles
                            else:
                                phase_fractions[phase_name] = 0.0
                        
                        # Normalize to get fractions (should sum to 1)
                        total_phase_moles = sum(phase_fractions.values())
                        
                        if i in [10, 25, 40]:  # Debug
                            print(f"  Raw phase moles: {phase_fractions}", flush=True)
                            print(f"  Total phase moles: {total_phase_moles:.6f}", flush=True)
                        
                        for phase in phases:
                            if total_phase_moles > 0:
                                phase_data[phase][i] = phase_fractions[phase] / total_phase_moles
                            else:
                                phase_data[phase][i] = 0.0
                            
                            if i in [10, 25, 40]:
                                print(f"  {phase}: final fraction={phase_data[phase][i]:.6f}", flush=True)
                                
                    except Exception as e:
                        # If calculation fails for this temperature, set all phases to 0
                        for phase in phases:
                            phase_data[phase][i] = 0.0
                        _log.warning(f"Equilibrium calculation failed at T={temp}K: {e}")
                
                # Extract colors from the existing plot
                base_colors = self._extract_phase_colors_from_plot(axes, phases)
                
                
                # Create stacked area plot instead of overlapping lines
                # Prepare data for stacking
                phase_labels = {
                    'LIQUID': 'Liquid',
                    'FCC_A1': 'FCC (Al-rich)', 
                    'HCP_A3': 'HCP (Zn-rich)'
                }
                
                # Only include phases that exist
                plot_phases = []
                plot_data = []
                plot_colors = []
                plot_labels = []
                
                for phase in phases:
                    phase_values = phase_data[phase]
                    if np.max(phase_values) > 1e-9:  # Only significant phases
                        plot_phases.append(phase)
                        plot_data.append(phase_values)
                        plot_colors.append(base_colors.get(phase, '#808080'))
                        plot_labels.append(phase_labels.get(phase, phase))
                
                # Create stacked area plot
                if plot_data:
                    axes.stackplot(temp_values, *plot_data, 
                                 labels=plot_labels,
                                 colors=plot_colors,
                                 alpha=0.8)
                    
                    # Add boundary lines for clarity
                    cumulative = np.zeros_like(temp_values)
                    for i, phase_values in enumerate(plot_data):
                        cumulative += phase_values
                        if i < len(plot_data) - 1:  # Don't draw line at top
                            axes.plot(temp_values, cumulative, 'k-', linewidth=0.5, alpha=0.3)
                
                # Add temperature markers for major visual transitions
                important_temps = []
                
                temp_min, temp_max = temp_values[0], temp_values[-1]
                margin = (temp_max - temp_min) * 0.1  # Stay away from edges
                
                # Look for major changes in phase fractions by examining the actual plot data
                for phase in plot_phases:
                    phase_array = np.array(phase_data[phase])
                    phase_label = phase_labels.get(phase, phase)
                    
                    # Find major jumps in phase fraction (>0.3 change)
                    diff = np.abs(np.diff(phase_array))
                    major_changes = np.where(diff > 0.3)[0]  # Big changes only
                    
                    for change_idx in major_changes:
                        temp = temp_values[change_idx]
                        if temp_min + margin < temp < temp_max - margin:
                            # Determine if phase is appearing or disappearing
                            before_val = phase_array[change_idx]
                            after_val = phase_array[change_idx + 1]
                            
                            if before_val < 0.1 and after_val > 0.4:
                                if phase_label.lower() == 'liquid':
                                    important_temps.append((temp, f"Melting starts"))
                                else:
                                    important_temps.append((temp, f"{phase_label} appears"))
                            elif before_val > 0.2 and after_val < 0.05:
                                # For disappears, use the temperature where it finishes disappearing (after_val position)
                                disappear_temp = temp_values[change_idx + 1]  # Use the end of the transition
                                if phase_label.lower() == 'liquid':
                                    important_temps.append((disappear_temp, f"Solidification starts"))
                                else:
                                    important_temps.append((disappear_temp, f"{phase_label} disappears"))
                
                # Remove duplicates and sort
                unique_temps = []
                for temp, desc in important_temps:
                    if not any(abs(temp - existing_temp) < 40 for existing_temp, _ in unique_temps):
                        unique_temps.append((temp, desc))
                
                unique_temps.sort()
                unique_temps = unique_temps[:3]  # Max 3 transitions
                
                # Add vertical dashed lines and bottom labels with arrows
                for i, (temp, description) in enumerate(unique_temps):
                    # Vertical dashed line
                    axes.axvline(x=temp, color='gray', linestyle='--', alpha=0.7, linewidth=1)
                    
                    # Label at bottom with arrow pointing to line
                    y_label_pos = axes.get_ylim()[0] - 0.15 - (i * 0.08)  # Offset each label
                    
                    axes.annotate(f'{temp:.0f}K\n{description}', 
                                 xy=(temp, axes.get_ylim()[0]), 
                                 xytext=(temp, y_label_pos),
                                 fontsize=7, va='top', ha='center',
                                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9, edgecolor='orange'),
                                 arrowprops=dict(arrowstyle='->', color='orange', lw=1.5))
                
                # Formatting
                pct_zn = target_composition * 100
                pct_al = (1 - target_composition) * 100
                axes.set_xlabel('Temperature (K)')
                axes.set_ylabel('Phase Fraction')
                axes.set_title(f'Phase Stability vs Temperature\nComposition: Al{pct_al:.0f}Zn{pct_zn:.0f}')
                axes.grid(True, alpha=0.3)
                axes.legend()
                
                # Set exact plot limits with no margins
                axes.set_xlim(300, 1000)
                axes.set_ylim(0, 1.0)
            
            # Convert to base64
            img_base64 = self._plot_to_base64(fig)
            plt.close(fig)
            
            # Generate deterministic analysis
            analysis = self._analyze_composition_temperature(phase_data, target_composition, temp_range)
            print(f"CALPHAD: Generated composition analysis with length: {len(analysis)}", flush=True)
            print(f"CALPHAD: Composition analysis preview: {analysis[:200]}...", flush=True)
            
            # Store the image data
            setattr(self, '_last_image_data', img_base64)
            pct_zn = target_composition * 100
            pct_al = (1 - target_composition) * 100
            description = f"Generated phase stability plot for composition Al{pct_al:.0f}Zn{pct_zn:.0f} showing phase fractions vs temperature"
            
            metadata = {
                "system": normalized_system,
                "database_file": db_path.name,
                "phases": phases,
                "temperature_range_K": temp_range,
                "composition_info": {
                    "target_composition": target_composition,
                    "zn_percentage": pct_zn,
                    "al_percentage": pct_al
                },
                "description": description,
                "analysis": analysis,
                "image_info": {
                    "format": "png",
                    "size": fig_size,
                    "data_length": len(img_base64)
                }
            }
            setattr(self, '_last_image_metadata', metadata)
            print(f"CALPHAD: Stored composition metadata with keys: {list(metadata.keys())}", flush=True)
            print(f"CALPHAD: Stored composition analysis length: {len(metadata['analysis'])}", flush=True)
            
            return f"Successfully generated phase stability plot for Al{pct_al:.0f}Zn{pct_zn:.0f} over temperature range {temp_range[0]}-{temp_range[1]}K. Shows phase fractions vs temperature for this specific composition."
            
        except Exception as e:
            _log.exception(f"Error generating composition-temperature plot for {composition}")
            return f"Failed to generate composition-temperature plot for {composition}: {str(e)}"

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
