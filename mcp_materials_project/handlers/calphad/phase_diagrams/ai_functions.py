"""
AI function methods for CALPHAD phase diagrams.

Contains the main AI function methods that are exposed to the AI system.
"""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, Optional
import logging

from pycalphad import Database, binplot
import pycalphad.variables as v
from kani.ai_function import ai_function
from typing_extensions import Annotated
from kani import AIParam

from .utils import _db_elements

_log = logging.getLogger(__name__)

class AIFunctionsMixin:
    """Mixin class containing AI function methods for CalPhadHandler."""
    
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
            comp_step = composition_step or 0.02
            fig_size = (figure_width or 9, figure_height or 6)

            fig = plt.figure(figsize=fig_size)
            axes = fig.gca()

            # --- before binplot: choose AUTO range if user didn't pass T limits ---
            auto_T = (min_temperature is None and max_temperature is None)
            if auto_T:
                # wide bracket so high-melting systems (e.g., Al–Si) are captured
                T_lo, T_hi = 200.0, 2300.0
            else:
                T_lo = min_temperature or 300.0
                T_hi = max_temperature or 1000.0

            # clamp number of temperature points so auto doesn't get heavy
            temp_points = 60 if auto_T else max(12, min(60, int((T_hi - T_lo) / 20)))

            binplot(
                db, elements, phases,
                {
                    v.X(comp_el): (0, 1, comp_step),
                    v.T: (T_lo, T_hi, temp_points),
                    v.P: 101325, v.N: 1
                },
                plot_kwargs={'ax': axes, 'tielines': False},
                eq_kwargs={'linewidth': 2},
                legend=True
            )

            # Get the legend handle that binplot created
            legend = axes.get_legend()
            
            # Give the legend some breathing room by widening the figure
            if legend is not None:
                w, h = fig.get_size_inches()
                fig.set_size_inches(w * 1.2, h)  # widen ~20%

            # generic labels / format
            # self._add_phase_labels(axes, temp_range, phases, db, elements, v.X(comp_el))  # Disabled - use legend instead
            axes.set_xlabel(f"Mole Fraction {comp_el} (atomic basis)")
            axes.set_ylabel("Temperature (K)")
            fig.suptitle(f"{A}-{B} Phase Diagram", x=0.5, fontsize=14, fontweight='bold')
            axes.grid(True, alpha=0.3)
            axes.set_xlim(0, 1)
            
            # ❗ DO NOT force set_ylim when auto_T is True
            # Store the actual displayed temperature range for reporting
            T_display_lo, T_display_hi = T_lo, T_hi
            if not auto_T:
                axes.set_ylim(T_lo, T_hi)
            else:
                # let matplotlib fit to drawn data, then add a tiny pad
                axes.relim(); axes.autoscale_view()
                y0, y1 = axes.get_ylim()
                pad = 0.02 * (y1 - y0)
                axes.set_ylim(y0 - pad, y1 + pad)
                # Store the actual displayed range for reporting
                T_display_lo, T_display_hi = y0 - pad, y1 + pad
                print(f"Auto-scaled temperature range: {T_display_lo:.0f}-{T_display_hi:.0f} K", flush=True)
            
            # Detect and mark eutectic points on the diagram
            # Use the full computation range (T_lo, T_hi) to ensure we capture all features
            print("Detecting eutectic points for visualization...", flush=True)
            eq_coarse = self._coarse_equilibrium_grid(db, A, B, phases, (T_lo, T_hi), nx=101, nT=161)
            detected_eutectics = []
            if eq_coarse is not None:
                ls_data = self._extract_liquidus_solidus(eq_coarse, B)
                if ls_data is not None:
                    # Use sensitive parameters with wider validation spacing
                    detected_eutectics = self._find_eutectic_points(eq_coarse, B, ls_data, delta_T=10.0, min_spacing=0.03, eps_drop=0.1)
                    if detected_eutectics:
                        print(f"Marking {len(detected_eutectics)} eutectic point(s) on the diagram", flush=True)
                        for e in detected_eutectics:
                            print(f"  - {e['temperature']:.0f} K at {e['composition_pct']:.2f} at% {B}: {e['reaction']}", flush=True)
                        self._mark_eutectics_on_axes(axes, detected_eutectics, B_symbol=B)
                        # Store for reuse in analysis to ensure consistency
                        self._cached_eutectics = detected_eutectics
                    else:
                        print("No eutectic points detected to mark", flush=True)
                        self._cached_eutectics = []
            
            # Also cache the equilibrium data for analysis
            self._cached_eq_coarse = eq_coarse
            
            # Generate visual analysis before saving
            print("Analyzing visual content...", flush=True)
            visual_analysis = self._analyze_visual_content(fig, axes, f"{A}-{B}", phases, (T_display_lo, T_display_hi))
            print(f"Visual analysis complete, length: {len(visual_analysis)}", flush=True)
            
            # Save plot to file and get URL - include legend as extra_artist so it won't be cropped
            print("Saving plot to file...", flush=True)
            plot_url = self._save_plot_to_file(
                fig, 
                f"phase_diagram_{A}-{B}",
                extra_artists=[legend] if legend is not None else None
            )
            print(f"Plot saved, URL: {plot_url}", flush=True)
            plt.close(fig)
            print("Plot closed, generating thermodynamic analysis...", flush=True)
            
            # Generate deterministic analysis (use displayed range for reporting)
            thermodynamic_analysis = self._analyze_phase_diagram(db, f"{A}-{B}", phases, (T_display_lo, T_display_hi))
            print(f"CALPHAD: Generated thermodynamic analysis with length: {len(thermodynamic_analysis)}", flush=True)
            
            # Clean up cached data after analysis is complete
            if hasattr(self, '_cached_eq_coarse'):
                delattr(self, '_cached_eq_coarse')
            if hasattr(self, '_cached_eutectics'):
                delattr(self, '_cached_eutectics')
            
            # Combine visual and thermodynamic analysis
            combined_analysis = f"{visual_analysis}\n\n{thermodynamic_analysis}"
            print(f"CALPHAD: Combined analysis length: {len(combined_analysis)}", flush=True)
            
            # Store the image URL privately and return only a simple success message
            # The image will be handled by the stream state
            setattr(self, '_last_image_url', plot_url)
            
            metadata = {
                "system": f"{A}-{B}",
                "database_file": db_path.name,
                "phases": phases,
                "temperature_range_K": (T_display_lo, T_display_hi),
                "composition_step": comp_step,
                "description": f"Phase diagram for {A}-{B} system",
                "analysis": combined_analysis,
                "visual_analysis": visual_analysis,
                "thermodynamic_analysis": thermodynamic_analysis,
                "image_info": {
                    "format": "png",
                    "url": plot_url
                }
            }
            setattr(self, '_last_image_metadata', metadata)
            print(f"CALPHAD: Stored metadata, analysis length: {len(metadata['analysis'])}", flush=True)
            
            # Return success message with key findings so the AI can see them
            success_parts = [
                f"Successfully generated {A}-{B} phase diagram showing phases: {', '.join(phases)}",
                f"Temperature range: {T_display_lo:.0f}-{T_display_hi:.0f} K"
            ]
            
            # Extract and include key points from analysis
            key_points = getattr(self, '_last_key_points', [])
            if key_points:
                # Add melting points
                melting_pts = [kp for kp in key_points if kp.get('type') == 'pure_melting']
                for mp in melting_pts:
                    success_parts.append(f"Pure {mp['element']} melting point: {mp['temperature']:.0f} K")
                
                # Add eutectic points
                eutectic_pts = [kp for kp in key_points if kp.get('type') == 'eutectic']
                if eutectic_pts:
                    for ep in eutectic_pts:
                        success_parts.append(f"Eutectic point: {ep['temperature']:.0f} K at {ep['composition_pct']:.1f} at% {B} ({ep['reaction']})")
                else:
                    success_parts.append("No eutectic points detected in this temperature range")
            
            success_msg = ". ".join(success_parts) + "."
            print(f"CALPHAD: Returning success message: {success_msg[:100]}...", flush=True)
            print(f"CALPHAD: SUCCESS MESSAGE LENGTH: {len(success_msg)} characters", flush=True)
            
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
        try:
            # reset previous artifacts
            if hasattr(self, '_last_image_metadata'):
                delattr(self, '_last_image_metadata')
            if hasattr(self, '_last_image_data'):
                delattr(self, '_last_image_data')
            
            # Parse composition and get system
            (A, B), xB, comp_type = self._parse_composition(composition, composition_type or "atomic")
            
            # Load database
            db_path = self._get_database_path(f"{A}-{B}")
            if not db_path:
                return f"No thermodynamic database found for {A}-{B} system."
            
            db = Database(str(db_path))
            
            # Check elements exist in database
            db_elems = _db_elements(db)
            if not (A in db_elems and B in db_elems):
                return f"Elements '{A}' and '{B}' not found in database. Available: {sorted(db_elems)}"
            
            # Set temperature range
            temp_range = (min_temperature or 300, max_temperature or 1000)
            
            # Get phases for this system
            phases = self._filter_phases_for_system(db, (A, B))
            
            # Calculate phase fractions vs temperature
            elements = [A, B, 'VA']
            comp_var = v.X(B)
            
            # Temperature points
            n_temp = max(50, min(200, int((temp_range[1] - temp_range[0]) / 5)))
            temps = np.linspace(temp_range[0], temp_range[1], n_temp)
            
            # Calculate equilibrium at each temperature
            phase_data = {}
            for phase in phases:
                phase_data[phase] = []
            
            for T in temps:
                try:
                    # Calculate equilibrium at this temperature
                    eq = self._calculate_equilibrium_at_T(db, elements, phases, T, xB, comp_var)
                    
                    # Extract phase fractions
                    for phase in phases:
                        if phase in eq.Phase.values:
                            phase_frac = eq.where(eq.Phase == phase)['NP'].values
                            if len(phase_frac) > 0 and not np.isnan(phase_frac[0]):
                                phase_data[phase].append(float(phase_frac[0]))
                            else:
                                phase_data[phase].append(0.0)
                        else:
                            phase_data[phase].append(0.0)
                            
                except Exception as e:
                    # If calculation fails at this temperature, set all phases to 0
                    for phase in phases:
                        phase_data[phase].append(0.0)
            
            # Create plot
            if interactive == "html":
                # Create interactive Plotly plot
                fig = self._create_interactive_plot(temps, phase_data, A, B, xB, comp_type, temp_range)
                
                # Save as HTML
                html_content = fig.to_html(include_plotlyjs='cdn')
                
                # Store HTML content
                setattr(self, '_last_image_data', html_content)
                
                # Generate analysis
                analysis = self._analyze_composition_temperature(phase_data, xB, temp_range, A, B)
                
                metadata = {
                    "composition": f"{A}{100-xB*100:.0f}{B}{xB*100:.0f}",
                    "system": f"{A}-{B}",
                    "temperature_range_K": temp_range,
                    "composition_type": comp_type,
                    "analysis": analysis,
                    "interactive": True
                }
                setattr(self, '_last_image_metadata', metadata)
                
                return f"Generated interactive phase stability plot for {A}{100-xB*100:.0f}{B}{xB*100:.0f} composition showing phase fractions vs temperature."
            
            else:
                # Create static matplotlib plot
                fig, ax = plt.subplots(figsize=(figure_width or 8, figure_height or 6))
                
                # Plot phase fractions
                colors = plt.cm.Set3(np.linspace(0, 1, len(phases)))
                for i, (phase, fractions) in enumerate(phase_data.items()):
                    if max(fractions) > 0.01:  # Only plot phases with significant fractions
                        ax.plot(temps, fractions, label=phase, linewidth=2, color=colors[i])
                
                ax.set_xlabel("Temperature (K)")
                ax.set_ylabel("Phase Fraction")
                ax.set_title(f"Phase Stability: {A}{100-xB*100:.0f}{B}{xB*100:.0f}")
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 1)
                
                # Create legend for phase fractions
                legend = ax.legend(title="Phases", loc="upper right", frameon=True)
                
                # Save plot to file and get URL
                print("Saving plot to file...", flush=True)
                plot_url = self._save_plot_to_file(fig, f"composition_stability_{A}{100-xB*100:.0f}{B}{xB*100:.0f}", extra_artists=[legend])
                print(f"Plot saved, URL: {plot_url}", flush=True)
                plt.close(fig)
                
                # Store image data
                setattr(self, '_last_image_url', plot_url)
                
                # Generate analysis
                analysis = self._analyze_composition_temperature(phase_data, xB, temp_range, A, B)
                
                metadata = {
                    "composition": f"{A}{100-xB*100:.0f}{B}{xB*100:.0f}",
                    "system": f"{A}-{B}",
                    "temperature_range_K": temp_range,
                    "composition_type": comp_type,
                    "analysis": analysis,
                    "interactive": False,
                    "composition_info": {
                        "target_composition": {A: 1-xB, B: xB},
                        "composition_type": comp_type,
                        "composition_suffix": "at%" if comp_type == "atomic" else "wt%"
                    },
                    "description": f"Phase stability diagram for {A}{int((1-xB)*100)}{B}{int(xB*100)} composition",
                    "phases": phases,
                    "image_info": {
                        "format": "png",
                        "url": plot_url
                    }
                }
                setattr(self, '_last_image_metadata', metadata)
                
                return f"Generated phase stability plot for {A}{100-xB*100:.0f}{B}{xB*100:.0f} composition showing phase fractions vs temperature."
                
        except Exception as e:
            _log.exception(f"Error generating composition-temperature plot for {composition}")
            return f"Failed to generate composition-temperature plot: {str(e)}"

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
        if not image_data:
            return f"Plot analysis (image data cleared to save memory):\n\n{combined_analysis}"
        
        # Return the combined analysis
        return combined_analysis

    @ai_function(desc="List available element pairs (binaries) supported by the loaded thermodynamic database.")
    async def list_available_systems(self) -> Dict[str, Any]:
        db_path = self._get_database_path("")
        out = {"pycalphad_available": True, "tdb_directory": str(self.tdb_dir), "systems": []}
        if not db_path:
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
    
    @ai_function(desc="Calculate equilibrium phase fractions at a specific temperature and composition. Use to verify phase amounts at a single condition. Returns detailed phase information including fractions and compositions.")
    async def calculate_equilibrium_at_point(
        self,
        composition: Annotated[str, AIParam(desc="Composition as element-number pairs (e.g., 'Al30Si55C15', 'Al80Zn20', 'Fe70Cr20Ni10'). Atomic percent by default.")],
        temperature: Annotated[float, AIParam(desc="Temperature in Kelvin")],
        composition_type: Annotated[Optional[str], AIParam(desc="'atomic' for at% or 'weight' for wt%. Default: 'atomic'")] = "atomic"
    ) -> str:
        """
        Calculate thermodynamic equilibrium at a specific point (temperature + composition).
        
        Returns phase fractions, compositions, and stability information.
        Useful for verifying specific phase equilibrium conditions.
        """
        try:
            from pycalphad import Database, equilibrium
            
            # Parse composition string (e.g., "Al30Si55C15" -> {AL: 0.30, SI: 0.55, C: 0.15})
            comp_dict = self._parse_multicomponent_composition(composition)
            if not comp_dict:
                return f"Failed to parse composition: {composition}. Use format like 'Al30Si55C15' or 'Fe70Cr20Ni10'"
            
            elements = list(comp_dict.keys())
            system_str = "-".join(elements)
            
            # Load database (pass elements to select appropriate .tdb)
            db_path = self._get_database_path(system_str, elements=elements)
            if not db_path:
                return f"No thermodynamic database found for {system_str} system."
            
            db = Database(str(db_path))
            
            # Get phases
            phases = self._filter_phases_for_multicomponent(db, elements)
            
            # Build conditions
            elements_with_va = elements + ['VA']
            conditions = {v.T: temperature, v.P: 101325, v.N: 1}
            
            # Set composition conditions (N-1 independent compositions)
            for i, elem in enumerate(elements[1:], 1):
                conditions[v.X(elem)] = comp_dict[elem]
            
            # Calculate equilibrium
            eq = equilibrium(db, elements_with_va, phases, conditions)
            
            # Extract phase fractions and compositions
            phase_info = []
            total_fraction = 0.0
            
            stable_phases = eq.Phase.values.ravel()
            phase_fractions = eq.NP.values.ravel()
            
            for i, phase in enumerate(stable_phases):
                if phase == '' or np.isnan(phase_fractions[i]):
                    continue
                
                frac = float(phase_fractions[i])
                if frac > 1e-6:  # Only report significant phases
                    total_fraction += frac
                    
                    # Get composition of this phase
                    phase_comp = {}
                    for elem in elements:
                        # Extract phase composition for this element
                        x_val = eq.X.sel(component=elem).values.ravel()[i]
                        if not np.isnan(x_val):
                            phase_comp[elem] = float(x_val)
                    
                    phase_info.append({
                        'phase': phase,
                        'fraction': frac,
                        'composition': phase_comp
                    })
            
            # Sort by fraction (descending)
            phase_info.sort(key=lambda x: x['fraction'], reverse=True)
            
            # Format response
            comp_str = " ".join([f"{elem}{comp_dict[elem]*100:.1f}" for elem in elements])
            response_lines = [
                f"**Equilibrium at {temperature:.1f} K for {comp_str}**\n",
                f"**Temperature**: {temperature:.1f} K ({temperature-273.15:.1f} °C)",
                f"**Composition**: {comp_str} (atomic %)\n",
                "**Stable Phases**:"
            ]
            
            for pinfo in phase_info:
                phase_name = pinfo['phase']
                frac = pinfo['fraction']
                comp = pinfo['composition']
                
                comp_str_phase = ", ".join([f"{e}: {comp[e]*100:.2f}%" for e in elements if e in comp])
                response_lines.append(f"  • **{phase_name}**: {frac*100:.2f}% ({comp_str_phase})")
            
            if not phase_info:
                response_lines.append("  • No stable phases found (calculation may have failed)")
            
            response_lines.append(f"\n**Total phase fraction**: {total_fraction*100:.2f}%")
            
            return "\n".join(response_lines)
            
        except Exception as e:
            _log.exception(f"Error calculating equilibrium at {temperature}K for {composition}")
            return f"Failed to calculate equilibrium: {str(e)}"
    
    @ai_function(desc="Calculate how phase fractions change with temperature for a specific composition. Essential for understanding precipitation, dissolution, and phase transformations. Returns phase fraction data across temperature range.")
    async def calculate_phase_fractions_vs_temperature(
        self,
        composition: Annotated[str, AIParam(desc="Composition as element-number pairs (e.g., 'Al30Si55C15', 'Al80Zn20')")],
        min_temperature: Annotated[float, AIParam(desc="Minimum temperature in Kelvin")],
        max_temperature: Annotated[float, AIParam(desc="Maximum temperature in Kelvin")],
        temperature_step: Annotated[Optional[float], AIParam(desc="Temperature step in Kelvin. Default: 10")] = None,
        composition_type: Annotated[Optional[str], AIParam(desc="'atomic' for at% or 'weight' for wt%. Default: 'atomic'")] = "atomic"
    ) -> str:
        """
        Calculate phase fractions as a function of temperature for a fixed composition.
        
        This is the primary tool for understanding:
        - Precipitation behavior (phase fraction increasing with cooling)
        - Dissolution behavior (phase fraction decreasing with heating)
        - Phase transformation temperatures
        - Solvus boundaries
        
        Returns detailed phase fraction data and analysis.
        """
        try:
            from pycalphad import Database, equilibrium
            
            # Parse composition
            comp_dict = self._parse_multicomponent_composition(composition)
            if not comp_dict:
                return f"Failed to parse composition: {composition}"
            
            elements = list(comp_dict.keys())
            system_str = "-".join(elements)
            
            # Load database (pass elements to select appropriate .tdb)
            db_path = self._get_database_path(system_str, elements=elements)
            if not db_path:
                return f"No thermodynamic database found for {system_str} system."
            
            db = Database(str(db_path))
            phases = self._filter_phases_for_multicomponent(db, elements)
            
            # Temperature array
            step = temperature_step or 10.0
            temps = np.arange(min_temperature, max_temperature + step, step)
            
            # Calculate equilibrium at each temperature
            phase_fractions = {}  # {phase_name: [fractions]}
            all_phases_seen = set()
            
            elements_with_va = elements + ['VA']
            
            _log.info(f"Calculating equilibrium for {len(temps)} temperature points...")
            
            for T in temps:
                conditions = {v.T: T, v.P: 101325, v.N: 1}
                for i, elem in enumerate(elements[1:], 1):
                    conditions[v.X(elem)] = comp_dict[elem]
                
                try:
                    eq = equilibrium(db, elements_with_va, phases, conditions)
                    
                    # Extract phases at this temperature
                    stable_phases = eq.Phase.values.ravel()
                    phase_fracs = eq.NP.values.ravel()
                    
                    temp_phases = {}
                    for i, phase in enumerate(stable_phases):
                        if phase and not np.isnan(phase_fracs[i]):
                            frac = float(phase_fracs[i])
                            if frac > 1e-8:
                                temp_phases[phase] = frac
                                all_phases_seen.add(phase)
                    
                    # Store fractions for all seen phases
                    for phase in all_phases_seen:
                        if phase not in phase_fractions:
                            phase_fractions[phase] = []
                        phase_fractions[phase].append(temp_phases.get(phase, 0.0))
                        
                except Exception as e:
                    _log.warning(f"Equilibrium calculation failed at {T}K: {e}")
                    # Append zeros for this temperature
                    for phase in all_phases_seen:
                        if phase not in phase_fractions:
                            phase_fractions[phase] = []
                        phase_fractions[phase].append(0.0)
            
            # Generate analysis
            comp_str = " ".join([f"{elem}{comp_dict[elem]*100:.0f}" for elem in elements])
            
            response_lines = [
                f"**Phase Fractions vs Temperature for {comp_str}**\n",
                f"**Temperature Range**: {min_temperature:.0f} - {max_temperature:.0f} K ({min_temperature-273.15:.0f} - {max_temperature-273.15:.0f} °C)",
                f"**Composition**: {comp_str} (atomic %)",
                f"**Temperature Points**: {len(temps)}\n",
                "**Phase Evolution**:"
            ]
            
            # Analyze each phase
            for phase, fractions in sorted(phase_fractions.items()):
                max_frac = max(fractions)
                min_frac = min(fractions)
                
                if max_frac < 1e-6:
                    continue  # Skip phases that never appear
                
                # Find where phase appears/disappears
                frac_start = fractions[0]
                frac_end = fractions[-1]
                
                # Determine trend
                if frac_end > frac_start + 0.01:
                    trend = "increasing"
                    change = f"+{(frac_end - frac_start)*100:.2f}%"
                elif frac_start > frac_end + 0.01:
                    trend = "decreasing"
                    change = f"{(frac_end - frac_start)*100:.2f}%"
                else:
                    trend = "stable"
                    change = "~0%"
                
                response_lines.append(
                    f"  • **{phase}**: {frac_start*100:.2f}% → {frac_end*100:.2f}% "
                    f"({trend}, {change})"
                )
            
            # Store data for potential plotting
            setattr(self, '_last_phase_fraction_data', {
                'temperatures': temps.tolist(),
                'phase_fractions': {p: f.copy() for p, f in phase_fractions.items()},
                'composition': comp_dict,
                'composition_str': comp_str
            })
            
            return "\n".join(response_lines)
            
        except Exception as e:
            _log.exception(f"Error calculating phase fractions vs temperature")
            return f"Failed to calculate phase fractions vs temperature: {str(e)}"
    
    @ai_function(desc="Analyze whether a specific phase increases or decreases with temperature. Use to verify statements about precipitation or dissolution behavior.")
    async def analyze_phase_fraction_trend(
        self,
        composition: Annotated[str, AIParam(desc="Composition as element-number pairs (e.g., 'Al30Si55C15')")],
        phase_name: Annotated[str, AIParam(desc="Name of the phase to analyze (e.g., 'AL4C3', 'SIC', 'FCC_A1')")],
        min_temperature: Annotated[float, AIParam(desc="Minimum temperature in Kelvin")],
        max_temperature: Annotated[float, AIParam(desc="Maximum temperature in Kelvin")],
        expected_trend: Annotated[Optional[str], AIParam(desc="Expected trend: 'increase', 'decrease', or 'stable'. Optional.")] = None
    ) -> str:
        """
        Analyze the trend of a specific phase fraction with temperature.
        
        This tool is designed to verify statements like:
        - "Phase X increases with decreasing temperature"
        - "Phase Y precipitates upon cooling"
        - "Phase Z dissolves upon heating"
        
        Returns detailed analysis with verification of expected trends.
        """
        try:
            from pycalphad import Database, equilibrium
            
            # Parse composition
            comp_dict = self._parse_multicomponent_composition(composition)
            if not comp_dict:
                return f"Failed to parse composition: {composition}"
            
            elements = list(comp_dict.keys())
            system_str = "-".join(elements)
            
            # Load database (pass elements to select appropriate .tdb)
            db_path = self._get_database_path(system_str, elements=elements)
            if not db_path:
                return f"No thermodynamic database found for {system_str} system."
            
            db = Database(str(db_path))
            phases = self._filter_phases_for_multicomponent(db, elements)
            
            # Normalize phase name
            phase_name_upper = phase_name.upper()
            
            # Check if phase exists in database
            if phase_name_upper not in phases and phase_name not in phases:
                available = ", ".join(sorted(phases))
                return f"Phase '{phase_name}' not found in database. Available phases: {available}"
            
            phase_to_track = phase_name_upper if phase_name_upper in phases else phase_name
            
            # Calculate over temperature range
            temps = np.linspace(min_temperature, max_temperature, 50)
            fractions = []
            
            elements_with_va = elements + ['VA']
            
            for T in temps:
                conditions = {v.T: T, v.P: 101325, v.N: 1}
                for i, elem in enumerate(elements[1:], 1):
                    conditions[v.X(elem)] = comp_dict[elem]
                
                try:
                    eq = equilibrium(db, elements_with_va, phases, conditions)
                    
                    # Find our phase
                    stable_phases = eq.Phase.values.ravel()
                    phase_fracs = eq.NP.values.ravel()
                    
                    phase_frac = 0.0
                    for i, phase in enumerate(stable_phases):
                        if phase == phase_to_track and not np.isnan(phase_fracs[i]):
                            phase_frac = max(phase_frac, float(phase_fracs[i]))
                    
                    fractions.append(phase_frac)
                    
                except Exception as e:
                    _log.warning(f"Calculation failed at {T}K: {e}")
                    fractions.append(0.0)
            
            # Analyze trend
            fractions = np.array(fractions)
            frac_low_T = fractions[0]
            frac_high_T = fractions[-1]
            
            max_frac = np.max(fractions)
            min_frac = np.min(fractions)
            
            # Compute overall trend
            delta = frac_high_T - frac_low_T
            
            if abs(delta) < 0.001:
                trend = "stable"
                trend_desc = "remains relatively stable"
            elif delta > 0.001:
                trend = "increases"
                trend_desc = "increases with increasing temperature (decreases upon cooling)"
            else:
                trend = "decreases"
                trend_desc = "decreases with increasing temperature (increases upon cooling)"
            
            comp_str = "".join([f"{elem}{comp_dict[elem]*100:.0f}" for elem in elements])
            
            response_lines = [
                f"**Phase Fraction Analysis: {phase_to_track} in {comp_str}**\n",
                f"**Temperature Range**: {min_temperature:.0f} - {max_temperature:.0f} K ({min_temperature-273.15:.0f} - {max_temperature-273.15:.0f} °C)",
                f"**Phase**: {phase_to_track}",
                f"**Composition**: {comp_str} (atomic %)\n",
                f"**Results**:",
                f"  • Fraction at {min_temperature:.0f} K: {frac_low_T*100:.3f}%",
                f"  • Fraction at {max_temperature:.0f} K: {frac_high_T*100:.3f}%",
                f"  • Change: {delta*100:.3f}% ({'+' if delta > 0 else ''}{delta*100:.3f}%)",
                f"  • Maximum fraction: {max_frac*100:.3f}%",
                f"  • Minimum fraction: {min_frac*100:.3f}%\n",
                f"**Trend**: The phase fraction **{trend_desc}**."
            ]
            
            # Verify against expected trend if provided
            if expected_trend:
                expected_lower = expected_trend.lower()
                matches = False
                
                # Check for "increasing/decreasing with temperature" patterns
                if "decreasing temperature" in expected_lower or "upon cooling" in expected_lower or "with cooling" in expected_lower:
                    # Phase should be higher at LOW temperature (precipitation upon cooling)
                    if "increase" in expected_lower:
                        matches = (frac_low_T > frac_high_T + 0.001)
                    elif "decrease" in expected_lower:
                        matches = (frac_low_T < frac_high_T - 0.001)
                        
                elif "increasing temperature" in expected_lower or "upon heating" in expected_lower or "with heating" in expected_lower:
                    # Phase should be higher at HIGH temperature
                    if "increase" in expected_lower:
                        matches = (frac_high_T > frac_low_T + 0.001)
                    elif "decrease" in expected_lower:
                        matches = (frac_high_T < frac_low_T - 0.001)
                        
                # Simple increase/decrease without temperature reference
                elif "increase" in expected_lower:
                    matches = (trend == "increases")
                elif "decrease" in expected_lower:
                    matches = (trend == "decreases")
                elif "stable" in expected_lower or "constant" in expected_lower:
                    matches = (trend == "stable")
                
                if matches:
                    response_lines.append(f"\n✅ **Verification**: The expected trend ('{expected_trend}') **matches** the calculated behavior.")
                else:
                    response_lines.append(f"\n❌ **Verification**: The expected trend ('{expected_trend}') **does NOT match** the calculated behavior.")
            
            return "\n".join(response_lines)
            
        except Exception as e:
            _log.exception(f"Error analyzing phase fraction trend")
            return f"Failed to analyze phase fraction trend: {str(e)}"