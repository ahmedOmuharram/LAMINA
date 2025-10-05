"""
Analysis utilities for CALPHAD phase diagrams.

Contains methods for analyzing phase diagrams and composition-temperature data.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging

_log = logging.getLogger(__name__)

class AnalysisMixin:
    """Mixin class containing analysis-related methods for CalPhadHandler."""
    
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
                visual_analysis.append(f"- **Legend**: {len(legend_labels)} phases shown: {', '.join(legend_labels)}")
            
            # Analyze phase information
            if phases:
                visual_analysis.append(f"- **Phases Present**: {len(phases)} phases: {', '.join(phases)}")
            
            # Temperature range analysis
            temp_min, temp_max = temp_range
            visual_analysis.append(f"- **Temperature Coverage**: {temp_min:.0f}K to {temp_max:.0f}K (span: {temp_max-temp_min:.0f}K)")
            
            # System information
            visual_analysis.append(f"- **System**: {normalized_system}")
            
            return "\n".join(visual_analysis)
            
        except Exception as e:
            return f"Error analyzing visual content: {str(e)}"

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
            
            # Find phase transitions (where phase fraction changes significantly)
            for phase_name, phase_values in composition_data.items():
                if phase_name == "LIQUID":
                    continue
                    
                phase_array = np.array(phase_values)
                
                # Find onset and offset temperatures (where phase fraction > 1%)
                indices_above_01 = np.where(phase_array > 0.01)[0]
                if len(indices_above_01) > 0:
                    onset_temp = temp_array[indices_above_01[0]]
                    offset_temp = temp_array[indices_above_01[-1]]
                    
                    # Find peak temperature (maximum phase fraction)
                    peak_idx = np.argmax(phase_array)
                    peak_temp = temp_array[peak_idx]
                    peak_fraction = phase_array[peak_idx]
                    
                    transition_temps.append({
                        'phase': phase_name,
                        'onset': onset_temp,
                        'peak': peak_temp,
                        'offset': offset_temp,
                        'max_fraction': peak_fraction
                    })
            
            # Sort by onset temperature
            transition_temps.sort(key=lambda x: x['onset'])
            
            # Generate analysis
            analysis_parts.append("### Phase Stability Ranges:")
            
            for transition in transition_temps:
                phase = transition['phase']
                onset = transition['onset']
                peak = transition['peak']
                offset = transition['offset']
                max_frac = transition['max_fraction']
                
                analysis_parts.append(f"- **{phase}**: Stable from {onset:.0f}K to {offset:.0f}K")
                analysis_parts.append(f"  - Peak stability: {max_frac:.1%} at {peak:.0f}K")
                analysis_parts.append(f"  - Stability range: {offset-onset:.0f}K")
            
            # Find melting/solidification behavior
            liquid_data = composition_data.get("LIQUID", [])
            if liquid_data:
                liquid_array = np.array(liquid_data)
                
                # Find liquidus temperature (where liquid fraction drops below 95%)
                liquidus_idx = np.where(liquid_array < 0.95)[0]
                if len(liquidus_idx) > 0:
                    liquidus_temp = temp_array[liquidus_idx[0]]
                    analysis_parts.append(f"\n### Melting Behavior:")
                    analysis_parts.append(f"- **Liquidus Temperature**: {liquidus_temp:.0f}K ({liquidus_temp-273.15:.0f}°C)")
                    analysis_parts.append(f"- **Melting Range**: {temp_range[1]-liquidus_temp:.0f}K")
                
                # Find solidus temperature (where liquid fraction drops below 5%)
                solidus_idx = np.where(liquid_array < 0.05)[0]
                if len(solidus_idx) > 0:
                    solidus_temp = temp_array[solidus_idx[0]]
                    analysis_parts.append(f"- **Solidus Temperature**: {solidus_temp:.0f}K ({solidus_temp-273.15:.0f}°C)")
                    analysis_parts.append(f"- **Freezing Range**: {liquidus_temp-solidus_temp:.0f}K")
            
            # Add processing implications
            analysis_parts.append(f"\n### Processing Implications:")
            analysis_parts.append(f"- **Heat Treatment**: Phase transitions can be exploited for precipitation hardening")
            analysis_parts.append(f"- **Casting**: Solidification behavior affects microstructure and properties")
            analysis_parts.append(f"- **Welding**: Thermal cycles must consider phase stability ranges")
            
            return "\n".join(analysis_parts)
            
        except Exception as e:
            return f"Error analyzing composition-temperature data: {str(e)}"
