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
