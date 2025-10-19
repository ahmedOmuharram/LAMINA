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

from .database_utils import get_db_elements, map_phase_name
from .equilibrium_utils import extract_phase_fractions_from_equilibrium

_log = logging.getLogger(__name__)

class AIFunctionsMixin:
    """Mixin class containing AI function methods for CalPhadHandler."""
    
    @ai_function(desc="PREFERRED for phase diagram questions. Generate a binary phase diagram for a chemical system using CALPHAD data. Use for general system queries like 'Al-Zn', 'aluminum-zinc', 'phase diagram', 'liquidus', 'solidus'. Shows full composition range with phase boundaries.")
    async def plot_binary_phase_diagram(
        self,
        system: Annotated[str, AIParam(desc="Chemical system (e.g., 'Al-Zn', 'AlZn', 'aluminum-zinc')")],
        min_temperature: Annotated[Optional[float], AIParam(desc="Minimum temperature in Kelvin. Default: auto")] = None,
        max_temperature: Annotated[Optional[float], AIParam(desc="Maximum temperature in Kelvin. Default: auto")] = None,
        composition_step: Annotated[Optional[float], AIParam(desc="Composition step size (0-1). Default: 0.02")] = None,
        figure_width: Annotated[Optional[float], AIParam(desc="Figure width in inches. Default: 9")] = None,
        figure_height: Annotated[Optional[float], AIParam(desc="Figure height in inches. Default: 6")] = None
    ) -> str:
        """
        Generate a binary phase diagram using CALPHAD thermodynamic data.
        
        Returns:
            Success message with key findings (e.g., phases, eutectic points, melting points).
            
        Side Effects:
            - Saves PNG image to interactive_plots/ directory
            - Stores image URL and metadata in self._last_image_url and self._last_image_metadata
            - Image URL is served at http://localhost:8000/static/plots/[filename]
        
        Currently supports Al-Zn and other systems in available .tdb databases.
        """
        
        try:
            # Clear any previous plot metadata at the start of new generation
            if hasattr(self, '_last_image_metadata'):
                delattr(self, '_last_image_metadata')
            if hasattr(self, '_last_image_data'):
                delattr(self, '_last_image_data')
            
            # Parse system first to get elements for database selection
            A, B = self._normalize_system(system, db=None)
            
            # Select database based on elements
            db_path = self._get_database_path(system, elements=[A, B])
            if not db_path:
                return {"success": False, "error": f"No .tdb found in {self.tdb_dir}.", "citations": ["pycalphad"]}
            db = Database(str(db_path))

            # Ensure both elements are in the selected DB
            db_elems = get_db_elements(db)
            if not (A in db_elems and B in db_elems):
                return {"success": False, "error": f"Elements '{A}' and '{B}' must both exist in the database ({sorted(db_elems)}).", "citations": ["pycalphad"]}
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
            axes.set_xlabel(f"Mole fraction of {comp_el}, $x_{{{comp_el}}}$")
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
                _log.info(f"Auto-scaled temperature range: {T_display_lo:.0f}-{T_display_hi:.0f} K")
            
            # Detect and mark eutectic points on the diagram
            # Use the full computation range (T_lo, T_hi) to ensure we capture all features
            _log.info("Detecting eutectic points for visualization...")
            eq_coarse = self._coarse_equilibrium_grid(db, A, B, phases, (T_lo, T_hi), nx=101, nT=161)
            detected_eutectics = []
            if eq_coarse is not None:
                ls_data = self._extract_liquidus_solidus(eq_coarse, B)
                if ls_data is not None:
                    # Use sensitive parameters with wider validation spacing
                    detected_eutectics = self._find_eutectic_points(eq_coarse, B, ls_data, delta_T=10.0, min_spacing=0.03, eps_drop=0.1)
                    if detected_eutectics:
                        _log.info(f"Marking {len(detected_eutectics)} eutectic point(s) on the diagram")
                        for e in detected_eutectics:
                            _log.info(f"  - {e['temperature']:.0f} K at {e['composition_pct']:.2f} at% {B}: {e['reaction']}")
                        self._mark_eutectics_on_axes(axes, detected_eutectics, B_symbol=B)
                        # Store for reuse in analysis to ensure consistency
                        self._cached_eutectics = detected_eutectics
                    else:
                        _log.info("No eutectic points detected to mark")
                        self._cached_eutectics = []
            
            # Also cache the equilibrium data for analysis
            self._cached_eq_coarse = eq_coarse
            
            # Generate visual analysis before saving
            _log.info("Analyzing visual content...")
            visual_analysis = self._analyze_visual_content(fig, axes, f"{A}-{B}", phases, (T_display_lo, T_display_hi))
            _log.info(f"Visual analysis complete, length: {len(visual_analysis)}")
            
            # Save plot to file and get URL - include legend as extra_artist so it won't be cropped
            _log.info("Saving plot to file...")
            plot_url = self._save_plot_to_file(
                fig, 
                f"phase_diagram_{A}-{B}",
                extra_artists=[legend] if legend is not None else None
            )
            _log.info(f"Plot saved, URL: {plot_url}")
            plt.close(fig)
            _log.info("Plot closed, generating thermodynamic analysis...")
            
            # Generate deterministic analysis (use displayed range for reporting)
            thermodynamic_analysis = self._analyze_phase_diagram(db, f"{A}-{B}", phases, (T_display_lo, T_display_hi))
            _log.info(f"Generated thermodynamic analysis with length: {len(thermodynamic_analysis)}")
            
            # Clean up cached data after analysis is complete
            if hasattr(self, '_cached_eq_coarse'):
                delattr(self, '_cached_eq_coarse')
            if hasattr(self, '_cached_eutectics'):
                delattr(self, '_cached_eutectics')
            
            # Combine visual and thermodynamic analysis
            combined_analysis = f"{visual_analysis}\n\n{thermodynamic_analysis}"
            _log.info(f"Combined analysis length: {len(combined_analysis)}")
            
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
            _log.info(f"Stored metadata, analysis length: {len(metadata['analysis'])}")
            
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
            _log.info(f"Returning success message: {success_msg[:100]}...")
            _log.debug(f"SUCCESS MESSAGE LENGTH: {len(success_msg)} characters")
            
            # Safeguard: ensure we're not accidentally returning base64 data
            if "data:image/png;base64," in success_msg or len(success_msg) > 1000:
                _log.error("ERROR - Success message contains base64 data or is too long! Truncating.")
                result = {"success": True, "message": "Successfully generated phase diagram. Image will be displayed separately.", "citations": ["pycalphad"]}
            else:
                result = {"success": True, "message": success_msg, "citations": ["pycalphad"]}
            
            # Store the result for tooltip display
            if hasattr(self, 'recent_tool_outputs'):
                self.recent_tool_outputs.append({
                    "tool_name": "plot_binary_phase_diagram",
                    "result": result
                })
            return result
            
        except Exception as e:
            _log.exception(f"Error generating phase diagram for {system}")
            result = {"success": False, "error": f"Failed to generate phase diagram for {system}: {str(e)}", "citations": ["pycalphad"]}
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
        min_temperature: Annotated[Optional[float], AIParam(desc="Minimum temperature in Kelvin. Default: auto (200K)")] = None,
        max_temperature: Annotated[Optional[float], AIParam(desc="Maximum temperature in Kelvin. Default: auto (2300K)")] = None,
        composition_type: Annotated[Optional[str], AIParam(desc="Composition type: 'atomic' for at% or 'weight' for wt%. Default: 'atomic'")] = None,
        figure_width: Annotated[Optional[float], AIParam(desc="Figure width in inches. Default: 8")] = None,
        figure_height: Annotated[Optional[float], AIParam(desc="Figure height in inches. Default: 6")] = None,
        interactive: Annotated[Optional[str], AIParam(desc="Interactive output mode: 'html' for interactive Plotly HTML output. Default: 'html'")] = "html"
    ) -> str:
        """
        Generate a temperature vs phase stability plot for a specific composition.
        
        Args:
            composition: Composition string (e.g., 'Al20Zn80'). Numbers are interpreted as percentages.
            composition_type: 'atomic' for at% (default) or 'weight' for wt%. Weight% is converted to mole fractions internally.
            interactive: 'html' for Plotly HTML (default) generates both HTML and static PNG
        
        Returns:
            Success message with composition and temperature range.
            
        Side Effects:
            - Saves PNG and HTML to interactive_plots/ directory
            - Stores URLs and metadata in self._last_image_url, self._last_html_url, and self._last_image_metadata
            - Files are served at http://localhost:8000/static/plots/[filename]
        """
        try:
            # reset previous artifacts
            if hasattr(self, '_last_image_metadata'):
                delattr(self, '_last_image_metadata')
            if hasattr(self, '_last_image_data'):
                delattr(self, '_last_image_data')
            
            # Parse composition and get system
            (A, B), xB, comp_type = self._parse_composition(composition, composition_type or "atomic")
            
            # Load database with element-based selection
            db_path = self._get_database_path(f"{A}-{B}", elements=[A, B])
            if not db_path:
                return f"No thermodynamic database found for {A}-{B} system."
            
            db = Database(str(db_path))
            
            # Check elements exist in database
            db_elems = get_db_elements(db)
            if not (A in db_elems and B in db_elems):
                return f"Elements '{A}' and '{B}' not found in database. Available: {sorted(db_elems)}"
            
            # Set temperature range with AUTO detection (same as plot_binary_phase_diagram)
            auto_T = (min_temperature is None and max_temperature is None)
            if auto_T:
                # Wide bracket so high-melting systems are captured
                T_lo, T_hi = 200.0, 2300.0
                _log.info(f"Using auto temperature range: {T_lo:.0f}-{T_hi:.0f} K")
            else:
                T_lo = min_temperature or 300.0
                T_hi = max_temperature or 1000.0
                _log.info(f"Using specified temperature range: {T_lo:.0f}-{T_hi:.0f} K")
            
            temp_range = (T_lo, T_hi)
            
            # Get phases for this system
            phases = self._filter_phases_for_system(db, (A, B))
            
            # Calculate phase fractions vs temperature
            elements = [A, B, 'VA']
            comp_var = v.X(B)
            
            # Temperature points
            n_temp = max(50, min(200, int((temp_range[1] - temp_range[0]) / 5)))
            temps = np.linspace(temp_range[0], temp_range[1], n_temp)
            _log.info(f"Temperature array: {len(temps)} points from {temps[0]:.1f} to {temps[-1]:.1f} K")
            
            # Calculate equilibrium at each temperature
            phase_data = {}
            for phase in phases:
                phase_data[phase] = []
            
            successful_calcs = 0
            failed_calcs = 0
            
            for T in temps:
                try:
                    # Calculate equilibrium at this temperature
                    eq = self._calculate_equilibrium_at_T(db, elements, phases, T, xB, comp_var)
                    
                    # Extract phase fractions properly (handling multiple vertices in two-phase regions)
                    # Use looser tolerance (1e-4) for better boundary handling
                    temp_fractions = extract_phase_fractions_from_equilibrium(eq, tolerance=1e-4)
                    
                    # Append fractions for all phases (0.0 if not present)
                    for phase in phases:
                        phase_data[phase].append(temp_fractions.get(phase, 0.0))
                    
                    successful_calcs += 1
                            
                except Exception as e:
                    # If calculation fails at this temperature, set all phases to 0
                    if failed_calcs == 0:  # Log first failure
                        _log.warning(f"Equilibrium calculation failed at T={T:.1f}K: {e}")
                    failed_calcs += 1
                    for phase in phases:
                        phase_data[phase].append(0.0)
            
            _log.info(f"Equilibrium calculations: {successful_calcs} successful, {failed_calcs} failed")
            
            # Debug phase data
            _log.debug(f"Phase data collected for {len(phase_data)} phases")
            for phase, fracs in phase_data.items():
                max_frac = max(fracs) if fracs else 0
                if max_frac > 0.01:
                    _log.debug(f"  {phase}: max fraction = {max_frac:.3f}")
            
            # Create plot
            if interactive == "html":
                # Create interactive Plotly plot
                _log.info(f"Creating interactive plot with temps shape: {np.array(temps).shape}, range: {temps[0]:.1f}-{temps[-1]:.1f} K")
                fig = self._create_interactive_plot(temps, phase_data, A, B, xB, comp_type, temp_range)
                
                # Save as HTML
                html_content = fig.to_html(include_plotlyjs='cdn')
                
                # Save HTML to file and get URL
                _log.info("Saving interactive HTML plot to file...")
                html_url = self._save_html_to_file(html_content, f"composition_stability_{A}{100-xB*100:.0f}{B}{xB*100:.0f}")
                _log.info(f"HTML plot saved, URL: {html_url}")
                
                # Also export static PNG for display in OpenWebUI
                _log.info("Exporting static PNG from Plotly figure...")
                try:
                    # Convert Plotly figure to static image
                    png_bytes = fig.to_image(format="png", width=1200, height=700, scale=2)
                    
                    # Save PNG to file
                    from pathlib import Path
                    import time
                    plots_dir = Path(__file__).parent.parent.parent.parent.parent / "interactive_plots"
                    timestamp = int(time.time() * 1000)
                    safe_filename = f"composition_stability_{A}{100-xB*100:.0f}{B}{xB*100:.0f}".replace(" ", "_").replace("/", "_")
                    png_path = plots_dir / f"{safe_filename}_{timestamp}.png"
                    
                    with open(png_path, 'wb') as f:
                        f.write(png_bytes)
                    
                    png_url = f"http://localhost:8000/static/plots/{png_path.name}"
                    _log.info(f"Static PNG saved, URL: {png_url}")
                    
                except Exception as e:
                    _log.warning(f"Could not export static PNG: {e}. Using matplotlib fallback.")
                    # Fallback: create matplotlib version with stacked area
                    fig_mpl, ax = plt.subplots(figsize=(figure_width or 10, figure_height or 6))
                    
                    # Prepare data for stackplot
                    phase_names = []
                    phase_arrays = []
                    colors = plt.cm.tab10(np.linspace(0, 1, len(phases)))
                    
                    for i, (phase, fractions) in enumerate(phase_data.items()):
                        if max(fractions) > 0.01:
                            # Map phase name to readable form (e.g., CSI -> SiC)
                            readable_phase = map_phase_name(phase)
                            phase_names.append(readable_phase)
                            phase_arrays.append(fractions)
                    
                    if phase_arrays:
                        # Create stacked area plot
                        ax.stackplot(temps, *phase_arrays, labels=phase_names, 
                                   colors=colors[:len(phase_arrays)], alpha=0.8, 
                                   edgecolor='white', linewidth=0.5)
                    
                    ax.set_xlabel("Temperature (K)", fontsize=12)
                    ax.set_ylabel("Phase Fraction", fontsize=12)
                    ax.set_title(f"Phase Stability: {A}{100-xB*100:.0f}{B}{xB*100:.0f}", fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3, zorder=0)
                    ax.set_ylim(0, 1)
                    ax.set_xlim(temps[0], temps[-1])
                    
                    # Add legend
                    if phase_names:
                        legend = ax.legend(title="Phases", loc="best", frameon=True, 
                                         fancybox=True, shadow=True, framealpha=0.9)
                    else:
                        legend = None
                    
                    png_url = self._save_plot_to_file(fig_mpl, f"composition_stability_{A}{100-xB*100:.0f}{B}{xB*100:.0f}", 
                                                     extra_artists=[legend] if legend else None)
                    plt.close(fig_mpl)
                
                # Store URLs - PNG for display, HTML for interactive link
                setattr(self, '_last_image_url', png_url)
                setattr(self, '_last_html_url', html_url)
                
                # Generate analysis
                analysis = self._analyze_composition_temperature(phase_data, xB, temp_range, A, B)
                
                metadata = {
                    "composition": f"{A}{100-xB*100:.0f}{B}{xB*100:.0f}",
                    "system": f"{A}-{B}",
                    "temperature_range_K": temp_range,
                    "composition_type": comp_type,
                    "analysis": analysis,
                    "interactive": True,
                    "image_info": {
                        "format": "png",  # Main display format
                        "url": png_url,
                        "interactive_html_url": html_url  # Link to interactive version
                    }
                }
                setattr(self, '_last_image_metadata', metadata)
                
                # Include link to interactive version in the response
                return (f"Generated phase stability plot for {A}{100-xB*100:.0f}{B}{xB*100:.0f} composition showing phase fractions vs temperature.\n\n"
                       f"📊 [View Interactive Plot]({html_url}) - Click to explore the interactive Plotly version with hover details and zoom.")
            
            else:
                # Create static matplotlib plot with stacked area
                fig, ax = plt.subplots(figsize=(figure_width or 8, figure_height or 6))
                
                # Prepare data for stackplot
                phase_names = []
                phase_arrays = []
                colors = plt.cm.tab10(np.linspace(0, 1, len(phases)))
                
                for i, (phase, fractions) in enumerate(phase_data.items()):
                    if max(fractions) > 0.01:  # Only plot phases with significant fractions
                        # Map phase name to readable form (e.g., CSI -> SiC)
                        readable_phase = map_phase_name(phase)
                        phase_names.append(readable_phase)
                        phase_arrays.append(fractions)
                
                if phase_arrays:
                    # Create stacked area plot
                    ax.stackplot(temps, *phase_arrays, labels=phase_names, 
                               colors=colors[:len(phase_arrays)], alpha=0.8,
                               edgecolor='white', linewidth=0.5)
                
                ax.set_xlabel("Temperature (K)", fontsize=12)
                ax.set_ylabel("Phase Fraction", fontsize=12)
                ax.set_title(f"Phase Stability: {A}{100-xB*100:.0f}{B}{xB*100:.0f}", fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3, zorder=0)
                ax.set_ylim(0, 1)
                ax.set_xlim(temps[0], temps[-1])
                
                # Create legend for phase fractions
                if phase_names:
                    legend = ax.legend(title="Phases", loc="best", frameon=True,
                                     fancybox=True, shadow=True, framealpha=0.9)
                else:
                    legend = None
                
                # Save plot to file and get URL
                _log.info("Saving plot to file...")
                plot_url = self._save_plot_to_file(fig, f"composition_stability_{A}{100-xB*100:.0f}{B}{xB*100:.0f}", 
                                                   extra_artists=[legend] if legend else None)
                _log.info(f"Plot saved, URL: {plot_url}")
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
                    "description": f"Phase stability diagram for {A}{round((1-xB)*100)}{B}{round(xB*100)} composition",
                    "phases": phases,
                    "image_info": {
                        "format": "png",
                        "url": plot_url
                    }
                }
                setattr(self, '_last_image_metadata', metadata)
                
                return {"success": True, "message": f"Generated phase stability plot for {A}{100-xB*100:.0f}{B}{xB*100:.0f} composition showing phase fractions vs temperature.", "citations": ["pycalphad"]}
                
        except Exception as e:
            _log.exception(f"Error generating composition-temperature plot for {composition}")
            return {"success": False, "error": f"Failed to generate composition-temperature plot: {str(e)}", "citations": ["pycalphad"]}

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
            return {"success": False, "error": "No recently generated phase diagram or composition plot available to analyze. Please generate a plot first using plot_binary_phase_diagram() or plot_composition_temperature().", "citations": ["pycalphad"]}
        
        # Extract analysis components
        visual_analysis = metadata.get("visual_analysis", "")
        thermodynamic_analysis = metadata.get("thermodynamic_analysis", "")
        combined_analysis = metadata.get("analysis", "")
        
        # Check if image data is available (may be cleared after display to save memory)
        image_data = getattr(self, '_last_image_data', None)
        if not image_data:
            return {"success": True, "message": f"Plot analysis (image data cleared to save memory):\n\n{combined_analysis}", "citations": ["pycalphad"]}
        
        # Return the combined analysis
        return {"success": True, "message": combined_analysis, "citations": ["pycalphad"]}

    @ai_function(desc="Calculate equilibrium phase fractions at a specific temperature and composition. Use to verify phase amounts at a single condition. Returns detailed phase information including fractions and compositions.")
    async def calculate_equilibrium_at_point(
        self,
        composition: Annotated[str, AIParam(desc="Composition as element-number pairs (e.g., 'Al30Si55C15', 'Al80Zn20', 'Fe70Cr20Ni10'). Numbers are percentages.")],
        temperature: Annotated[float, AIParam(desc="Temperature in Kelvin")],
        composition_type: Annotated[Optional[str], AIParam(desc="'atomic' for at% or 'weight' for wt%. Default: 'atomic'")] = "atomic"
    ) -> str:
        """
        Calculate thermodynamic equilibrium at a specific point (temperature + composition).
        
        Args:
            composition: Element-number pairs (e.g., 'Al30Si55C15'). Numbers are percentages.
            composition_type: 'atomic' (default) or 'weight'. Weight% is converted to mole fractions internally.
        
        Returns:
            Formatted text with phase fractions and per-phase compositions.
        """
        try:
            from pycalphad import Database, equilibrium
            
            # Parse composition string (e.g., "Al30Si55C15" -> {AL: 0.30, SI: 0.55, C: 0.15})
            # Note: Always returns atomic (mole) fractions, converting from weight if needed
            comp_dict = self._parse_multicomponent_composition(composition, composition_type)
            if not comp_dict:
                return f"Failed to parse composition: {composition}. Use format like 'Al30Si55C15' or 'Fe70Cr20Ni10'"
            
            elements = list(comp_dict.keys())
            system_str = "-".join(elements)
            
            # Load database (pass elements to select appropriate .tdb)
            db_path = self._get_database_path(system_str, elements=elements)
            if not db_path:
                return f"No thermodynamic database found for {system_str} system."
            
            db = Database(str(db_path))
            
            # Get phases - use correct filter for binary vs multicomponent
            if len(elements) == 2:
                # Binary system - use binary-specific filter with activation pass
                phases = self._filter_phases_for_system(db, tuple(elements))
            else:
                # Multicomponent system (3+ elements)
                phases = self._filter_phases_for_multicomponent(db, elements)
            
            # Build conditions
            elements_with_va = elements + ['VA']
            conditions = {v.T: temperature, v.P: 101325, v.N: 1}
            
            # Set composition conditions (N-1 independent compositions)
            for i, elem in enumerate(elements[1:], 1):
                conditions[v.X(elem)] = comp_dict[elem]
            
            # Calculate equilibrium
            eq = equilibrium(db, elements_with_va, phases, conditions)
            
            # Extract phase fractions properly (handling multiple vertices in two-phase regions)
            # Use looser tolerance (1e-4) for better boundary handling
            phase_fractions_dict = extract_phase_fractions_from_equilibrium(eq, tolerance=1e-4)
            
            # Extract phase fractions and compositions
            phase_info = []
            total_fraction = 0.0
            
            for phase, frac in phase_fractions_dict.items():
                total_fraction += frac
                
                # Get composition of this phase
                phase_comp = {}
                try:
                    # Squeeze and select data for this phase
                    eqp = eq.squeeze()
                    phase_mask = eqp['Phase'] == phase
                    
                    for elem in elements:
                        # Extract phase composition for this element (average over vertices)
                        x_data = eqp['X'].sel(component=elem).where(phase_mask, drop=False)
                        x_val = float(x_data.mean().values)
                        if not np.isnan(x_val):
                            phase_comp[elem] = x_val
                except Exception as e:
                    _log.warning(f"Could not extract composition for phase {phase}: {e}")
                
                # Map phase name to readable form (e.g., CSI -> SiC)
                readable_name = map_phase_name(phase)
                
                phase_info.append({
                    'phase': readable_name,
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
            
            return {"success": True, "message": "\n".join(response_lines), "citations": ["pycalphad"]}
            
        except Exception as e:
            _log.exception(f"Error calculating equilibrium at {temperature}K for {composition}")
            return {"success": False, "error": f"Failed to calculate equilibrium: {str(e)}", "citations": ["pycalphad"]}
    
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
            
            # Parse composition (always returns atomic/mole fractions, converting from weight if needed)
            comp_dict = self._parse_multicomponent_composition(composition, composition_type)
            if not comp_dict:
                return f"Failed to parse composition: {composition}"
            
            elements = list(comp_dict.keys())
            system_str = "-".join(elements)
            
            # Load database (pass elements to select appropriate .tdb)
            db_path = self._get_database_path(system_str, elements=elements)
            if not db_path:
                return f"No thermodynamic database found for {system_str} system."
            
            db = Database(str(db_path))
            
            # Get phases - use correct filter for binary vs multicomponent
            if len(elements) == 2:
                # Binary system - use binary-specific filter with activation pass
                phases = self._filter_phases_for_system(db, tuple(elements))
            else:
                # Multicomponent system (3+ elements)
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
                    
                    # Extract phases at this temperature (properly handling multiple vertices)
                    # Use looser tolerance (1e-4) for better boundary handling
                    temp_phases = extract_phase_fractions_from_equilibrium(eq, tolerance=1e-4)
                    
                    for phase in temp_phases.keys():
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
            
            return {"success": True, "message": "\n".join(response_lines), "citations": ["pycalphad"]}
            
        except Exception as e:
            _log.exception(f"Error calculating phase fractions vs temperature")
            return {"success": False, "error": f"Failed to calculate phase fractions vs temperature: {str(e)}", "citations": ["pycalphad"]}
    
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
            
            # Parse composition (note: expected_trend doesn't have composition_type, default to atomic)
            # Always returns atomic/mole fractions
            comp_dict = self._parse_multicomponent_composition(composition, composition_type="atomic")
            if not comp_dict:
                return f"Failed to parse composition: {composition}"
            
            elements = list(comp_dict.keys())
            system_str = "-".join(elements)
            
            # Load database (pass elements to select appropriate .tdb)
            db_path = self._get_database_path(system_str, elements=elements)
            if not db_path:
                return f"No thermodynamic database found for {system_str} system."
            
            db = Database(str(db_path))
            
            # Get phases - use correct filter for binary vs multicomponent
            if len(elements) == 2:
                # Binary system - use binary-specific filter with activation pass
                phases = self._filter_phases_for_system(db, tuple(elements))
            else:
                # Multicomponent system (3+ elements)
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
                    
                    # Extract phase fractions properly (handling multiple vertices)
                    # Use looser tolerance (1e-4) for better boundary handling
                    temp_phases = extract_phase_fractions_from_equilibrium(eq, tolerance=1e-4)
                    
                    # Find our phase
                    phase_frac = temp_phases.get(phase_to_track, 0.0)
                    
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
            
            return {"success": True, "message": "\n".join(response_lines), "citations": ["pycalphad"]}
            
        except Exception as e:
            _log.exception(f"Error analyzing phase fraction trend")
            return {"success": False, "error": f"Failed to analyze phase fraction trend: {str(e)}", "citations": ["pycalphad"]}