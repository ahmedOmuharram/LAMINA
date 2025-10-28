"""
AI function methods for CALPHAD phase diagrams.

Contains the main AI function methods that are exposed to the AI system.
"""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, Optional, List
import logging

from pycalphad import Database, binplot
import pycalphad.variables as v
from kani.ai_function import ai_function
from typing_extensions import Annotated
from kani import AIParam

from .database_utils import get_db_elements, map_phase_name
from .equilibrium_utils import extract_phase_fractions_from_equilibrium, get_phase_fraction

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

            # Handle degenerate case where min == max
            if T_lo == T_hi:
                # Expand range by ±100K around the point
                T_center = T_lo
                T_lo = max(200.0, T_center - 100.0)  # Don't go below 200K
                T_hi = T_center + 100.0
                _log.info(f"Temperature range was degenerate ({T_center:.0f}K), expanded to {T_lo:.0f}-{T_hi:.0f} K")

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
            
            # Handle degenerate case where min == max
            if T_lo == T_hi:
                # Expand range by ±100K around the point
                T_center = T_lo
                T_lo = max(200.0, T_center - 100.0)  # Don't go below 200K
                T_hi = T_center + 100.0
                _log.info(f"Temperature range was degenerate ({T_center:.0f}K), expanded to {T_lo:.0f}-{T_hi:.0f} K")
            
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
            
            # Debug: List all available phases in the database
            available_phases = {p.name for p in db.phases.values()}
            _log.info(f"Available phases in database: {sorted(available_phases)}")
            
            # Get phases - use correct filter for binary vs multicomponent
            if len(elements) == 2:
                # Binary system - use binary-specific filter with activation pass
                phases = self._filter_phases_for_system(db, tuple(elements))
            else:
                # Multicomponent system (3+ elements)
                phases = self._filter_phases_for_multicomponent(db, elements)
            
            _log.info(f"Filtered phases for calculation: {sorted(phases)}")
            
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
                    
                    # Find our phase, summing all instances (e.g., SIC#1 + SIC#2)
                    phase_frac = get_phase_fraction(temp_phases, phase_to_track)
                    
                    # Debug: Log phase fractions at a few sample temperatures
                    if len(fractions) < 3 or len(fractions) == len(temps) // 2:
                        # Collapse instances to base names for debugging
                        base_phases = {}
                        for k, v in temp_phases.items():
                            base_name = str(k).split('#')[0].upper()
                            base_phases[base_name] = base_phases.get(base_name, 0.0) + v
                        top_phases = sorted(base_phases.items(), key=lambda x: x[1], reverse=True)[:5]
                        _log.info(f"At {T:.0f}K: top phases = {top_phases}, {phase_to_track} fraction = {phase_frac:.4f}")
                    
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
    
    @ai_function(desc="Verify phase formation statements across a composition range. Use to check claims like 'beyond X% of element B, phase Y forms' or 'at compositions greater than X%, phase Z appears'. Analyzes which phases form at different compositions.")
    async def verify_phase_formation_across_composition(
        self,
        system: Annotated[str, AIParam(desc="System (e.g., 'Fe-Al', 'Al-Zn' for binary, 'Al-Mg-Zn' for ternary)")],
        phase_name: Annotated[str, AIParam(desc="Phase name: exact database name (e.g., 'MGZN2', 'TAU', 'FCC_A1') or category (e.g., 'Laves', 'tau', 'fcc', 'gamma')")],
        composition_threshold: Annotated[float, AIParam(desc="Composition threshold to verify (e.g., 50.0 for '50 at.%')")],
        threshold_element: Annotated[str, AIParam(desc="Element being thresholded (e.g., 'Al' in 'beyond 50% Al')")],
        temperature: Annotated[float, AIParam(desc="Temperature in Kelvin to check phase formation. Default: 300K")] = 300.0,
        composition_type: Annotated[Optional[str], AIParam(desc="'atomic' for at% or 'weight' for wt%. Default: 'atomic'")] = "atomic",
        fixed_element: Annotated[Optional[str], AIParam(desc="For ternary: element to keep fixed (e.g., 'Zn')")] = None,
        fixed_composition: Annotated[Optional[float], AIParam(desc="For ternary: fixed element composition in at.% (e.g., 4.0 for 4%)")] = None
    ) -> str:
        """
        Verify whether a specific phase forms above/below a composition threshold.
        
        This tool checks statements like:
        - Binary: "Beyond 50 at.% Al in Fe-Al, the Al2Fe phase forms"
        - Ternary: "In Al-Mg-Zn with 4% Zn, tau phase forms above 8% Mg"
        
        For ternary systems, specify fixed_element and fixed_composition to hold one element constant
        while varying the threshold_element.
        
        Returns detailed analysis showing:
        - Which phases are present below/above the threshold
        - Whether the specified phase forms above/below threshold
        - Phase fractions across the composition range
        """
        try:
            from pycalphad import Database, equilibrium, variables as v
            import numpy as np
            
            # Parse elements from system
            elements_from_system = system.replace('-', ' ').replace('_', ' ').upper().split()
            elements = [self._normalize_element(e) for e in elements_from_system if e]
            elements = [e for e in elements if len(e) <= 2 and e.isalpha()]
            
            if len(elements) < 2:
                return f"System must have at least 2 elements. Got: {system}"
            
            is_ternary = len(elements) >= 3
            system_str = "-".join(elements)
            
            # Validate threshold element
            threshold_elem = threshold_element.strip().upper()
            if threshold_elem not in elements:
                return f"Element '{threshold_element}' not found in system {system_str}. Must be one of: {', '.join(elements)}"
            
            # For ternary systems, validate fixed element
            if is_ternary:
                if not fixed_element or fixed_composition is None:
                    return f"For ternary system {system_str}, must specify fixed_element and fixed_composition"
                
                fixed_elem = fixed_element.strip().upper()
                if fixed_elem not in elements:
                    return f"Fixed element '{fixed_element}' not found in system {system_str}"
                if fixed_elem == threshold_elem:
                    return f"Fixed element and threshold element cannot be the same"
                
                # The third element is the balance
                balance_elem = [e for e in elements if e not in [threshold_elem, fixed_elem]][0]
            else:
                # Binary system: one element varies, the other is balance
                balance_elem = [e for e in elements if e != threshold_elem][0]
                fixed_elem = None
            
            # Load database
            db_path = self._get_database_path(system_str, elements=elements)
            if not db_path:
                return f"No thermodynamic database found for {system_str} system."
            
            db = Database(str(db_path))
            
            # Get phases
            if len(elements) == 2:
                phases = self._filter_phases_for_system(db, tuple(elements[:2]))
            else:
                phases = self._get_phases_for_elements(db, elements)
            
            # Normalize phase name and handle category names
            from .fact_checker import PHASE_CLASSIFICATION
            
            phase_name_upper = phase_name.upper()
            available_phases = sorted(phases)
            
            # Debug: List all available phases
            _log.info(f"Available phases in {system_str} database: {available_phases}")
            
            # Check if input is a category name (e.g., "Laves", "tau", "gamma")
            # Map it to actual database phase names
            phase_to_check = None
            candidate_phases = []
            
            # First, try direct match with database phase names
            if phase_name_upper in phases or phase_name in phases:
                phase_to_check = phase_name_upper if phase_name_upper in phases else phase_name
                candidate_phases = [phase_to_check]
            else:
                # Try category-based matching using PHASE_CLASSIFICATION
                phase_name_lower = phase_name.lower()
                
                for db_phase_name, (readable_name, category, structure) in PHASE_CLASSIFICATION.items():
                    # Check if input matches the category or readable name
                    category_value = category.value if hasattr(category, 'value') else str(category)
                    if (phase_name_lower in category_value.lower() or 
                        phase_name_lower in readable_name.lower() or
                        category_value.lower() in phase_name_lower):
                        # Check if this database phase is available in current system
                        if db_phase_name in phases:
                            candidate_phases.append(db_phase_name)
                
                if candidate_phases:
                    phase_to_check = candidate_phases[0]  # Use first as representative
                    _log.info(f"Mapped category '{phase_name}' to database phases: {candidate_phases}")
            
            # Final check: did we find any matching phase?
            if not phase_to_check:
                return f"Phase '{phase_name}' not found in database. Available phases: {', '.join(available_phases)}. Try using exact database names or categories like: fcc, bcc, hcp, tau, laves, gamma"
            
            # Determine the set of database phase names to aggregate
            target_names = set(candidate_phases) if candidate_phases else {phase_to_check}
            display_name = phase_name if not candidate_phases or len(candidate_phases) == 1 else f"{phase_name} (any)"
            
            # Sample compositions for the threshold element
            threshold_fraction = composition_threshold / 100.0
            n_points = 21
            threshold_compositions = np.linspace(0.0, 1.0, n_points)
            
            # Add specific points around threshold for precision
            threshold_nearby = [
                max(0.0, threshold_fraction - 0.05),
                max(0.0, threshold_fraction - 0.02),
                threshold_fraction,
                min(1.0, threshold_fraction + 0.02),
                min(1.0, threshold_fraction + 0.05)
            ]
            threshold_compositions = np.unique(np.sort(np.concatenate([threshold_compositions, threshold_nearby])))
            
            # Build elements list for pycalphad (include VA)
            pycalphad_elements = elements + ['VA']
            results = []
            
            _log.info(f"Checking phase '{phase_to_check}' formation across {len(threshold_compositions)} compositions at T={temperature}K")
            if is_ternary:
                _log.info(f"  Ternary: varying {threshold_elem}, fixing {fixed_elem}={fixed_composition}%, balance={balance_elem}")
            
            for x_threshold in threshold_compositions:
                try:
                    # Build composition dict
                    comp_dict = {}
                    comp_dict[threshold_elem] = x_threshold
                    
                    if is_ternary:
                        # Fixed element
                        comp_dict[fixed_elem] = fixed_composition / 100.0
                        # Balance element (automatically calculated by pycalphad as 1 - others)
                        comp_dict[balance_elem] = 1.0 - x_threshold - (fixed_composition / 100.0)
                        
                        # Skip if balance is negative or zero
                        if comp_dict[balance_elem] <= 0:
                            continue
                    else:
                        # Binary: balance is just 1 - threshold
                        comp_dict[balance_elem] = 1.0 - x_threshold
                    
                    # For pycalphad equilibrium, only specify N-1 composition constraints
                    # Use the existing helper which handles this correctly
                    from .equilibrium_utils import calculate_equilibrium_at_point
                    eq = calculate_equilibrium_at_point(db, pycalphad_elements, phases, comp_dict, temperature)
                    
                    if eq is None:
                        _log.warning(f"Equilibrium calculation failed at {threshold_elem}={x_threshold:.3f}")
                        continue
                    
                    # Extract phase fractions
                    from .equilibrium_utils import extract_phase_fractions_from_equilibrium
                    phase_fractions = extract_phase_fractions_from_equilibrium(eq, tolerance=1e-4)
                    
                    # Helper to strip instance suffixes (e.g., "FCC_A1#2" -> "FCC_A1")
                    def base_name(n):
                        return str(n).split('#')[0].upper()
                    
                    # Aggregate fraction across all target phase names
                    target_names_upper = {t.upper() for t in target_names}
                    phase_fraction = sum(
                        frac for name, frac in phase_fractions.items()
                        if base_name(name) in target_names_upper
                    )
                    phase_present = phase_fraction > 0.01
                    
                    # Store results with at.% for all elements
                    result = {
                        'phase_present': phase_present,
                        'phase_fraction': phase_fraction,
                        'all_phases': phase_fractions
                    }
                    
                    # Add at.% for each element
                    for el in elements:
                        result[f'at_pct_{el}'] = comp_dict[el] * 100
                    
                    results.append(result)
                    
                except Exception as e:
                    _log.warning(f"Equilibrium calculation failed at {threshold_elem}={x_threshold:.3f}: {e}")
                    continue
            
            if not results:
                return f"Failed to calculate equilibrium for any compositions in {system_str} at {temperature}K"
            
            # Analyze results around threshold
            threshold_pct = composition_threshold
            threshold_key = f'at_pct_{threshold_elem}'
            
            # Find compositions below and above threshold
            below_threshold = [r for r in results if r[threshold_key] < threshold_pct]
            above_threshold = [r for r in results if r[threshold_key] >= threshold_pct]
            
            # Count how many times the phase appears in each region
            phase_count_below = sum(1 for r in below_threshold if r['phase_present'])
            phase_count_above = sum(1 for r in above_threshold if r['phase_present'])
            
            # Build response
            response_lines = [
                f"## Phase Formation Analysis: {system_str} System",
                f"**Phase**: {display_name}",
                f"**Temperature**: {temperature:.1f} K",
                f"**Threshold**: {threshold_pct:.1f} at.% {threshold_elem}",
            ]
            
            if is_ternary:
                response_lines.append(f"**Fixed**: {fixed_composition:.1f} at.% {fixed_elem}")
            
            response_lines.extend([
                "",
                "### Results:",
                ""
            ])
            
            # Summary statistics
            total_below = len(below_threshold)
            total_above = len(above_threshold)
            
            if total_below > 0:
                fraction_below = phase_count_below / total_below
                response_lines.append(f"**Below threshold** (<{threshold_pct:.1f}% {threshold_elem}):")
                response_lines.append(f"  - Phase '{display_name}' present in {phase_count_below}/{total_below} compositions ({fraction_below*100:.1f}%)")
                
                # Show example phases below threshold (pick one where phase is present if possible)
                if below_threshold:
                    # Prefer an example where the phase is actually present
                    present_examples = [r for r in below_threshold if r['phase_present']]
                    example = present_examples[len(present_examples)//2] if present_examples else below_threshold[len(below_threshold)//2]
                    
                    # Format phases with FCC_A1 host labeling
                    phases_str_list = []
                    for p, f in sorted(example['all_phases'].items(), key=lambda x: -x[1]):
                        if f > 0.01:
                            if p == 'FCC_A1':
                                host_el = max(elements, key=lambda el: example[f'at_pct_{el}'])
                                phases_str_list.append(f"{p}({host_el}-rich)({f*100:.1f}%)")
                            else:
                                phases_str_list.append(f"{p}({f*100:.1f}%)")
                    phases_str = ", ".join(phases_str_list)
                    
                    # Build composition string dynamically
                    comp_parts = [f"{example[f'at_pct_{el}']:.1f}% {el}" for el in elements]
                    response_lines.append(f"  - Example at {' / '.join(comp_parts)}: {phases_str}")
            
            response_lines.append("")
            
            if total_above > 0:
                fraction_above = phase_count_above / total_above
                response_lines.append(f"**Above threshold** (≥{threshold_pct:.1f}% {threshold_elem}):")
                response_lines.append(f"  - Phase '{display_name}' present in {phase_count_above}/{total_above} compositions ({fraction_above*100:.1f}%)")
                
                # Show example phases above threshold (pick one where phase is present if possible)
                if above_threshold:
                    # Prefer an example where the phase is actually present
                    present_examples = [r for r in above_threshold if r['phase_present']]
                    example = present_examples[len(present_examples)//2] if present_examples else above_threshold[len(above_threshold)//2]
                    
                    # Format phases with FCC_A1 host labeling
                    phases_str_list = []
                    for p, f in sorted(example['all_phases'].items(), key=lambda x: -x[1]):
                        if f > 0.01:
                            if p == 'FCC_A1':
                                host_el = max(elements, key=lambda el: example[f'at_pct_{el}'])
                                phases_str_list.append(f"{p}({host_el}-rich)({f*100:.1f}%)")
                            else:
                                phases_str_list.append(f"{p}({f*100:.1f}%)")
                    phases_str = ", ".join(phases_str_list)
                    
                    # Build composition string dynamically
                    comp_parts = [f"{example[f'at_pct_{el}']:.1f}% {el}" for el in elements]
                    response_lines.append(f"  - Example at {' / '.join(comp_parts)}: {phases_str}")
            
            response_lines.append("")
            response_lines.append("### Verification:")
            response_lines.append("")
            
            # Compute frequencies (already calculated above, but ensure they're defined)
            fraction_below = (phase_count_below / total_below) if total_below > 0 else 0.0
            fraction_above = (phase_count_above / total_above) if total_above > 0 else 0.0
            
            # Use frequency-based comparison, not raw counts
            eps = 0.05  # 5 percentage points tolerance to avoid false flips from numerical noise
            
            if fraction_above > 0 and fraction_below == 0:
                response_lines.append(
                    f"✅ **VERIFIED**: Phase '{display_name}' forms above {threshold_pct:.1f}% {threshold_elem} "
                    f"and is absent below this threshold."
                )
            
            elif fraction_above >= eps and (fraction_above - fraction_below) > eps:
                response_lines.append(
                    f"⚠️ **PARTIALLY VERIFIED**: Phase '{display_name}' forms more frequently above "
                    f"{threshold_pct:.1f}% {threshold_elem} ({fraction_above*100:.1f}% vs {fraction_below*100:.1f}% of samples), "
                    f"but it can also appear below."
                )
            
            elif fraction_below >= eps and (fraction_below - fraction_above) > eps:
                response_lines.append(
                    f"❌ **CONTRADICTED**: Phase '{display_name}' is actually more frequent below "
                    f"{threshold_pct:.1f}% {threshold_elem} ({fraction_below*100:.1f}% vs {fraction_above*100:.1f}% of samples), "
                    f"opposite to the claim."
                )
            
            elif fraction_above > 0 and fraction_below > 0 and abs(fraction_above - fraction_below) <= eps:
                response_lines.append(
                    f"❌ **NOT VERIFIED**: Phase '{display_name}' appears both above and below "
                    f"{threshold_pct:.1f}% {threshold_elem} with similar frequency "
                    f"({fraction_above*100:.1f}% vs {fraction_below*100:.1f}% of samples)."
                )
            
            elif fraction_above == 0 and fraction_below > 0:
                response_lines.append(
                    f"❌ **CONTRADICTED**: Phase '{display_name}' actually forms below "
                    f"{threshold_pct:.1f}% {threshold_elem}, not above."
                )
            
            else:
                response_lines.append(
                    f"⚠️ **INCONCLUSIVE**: Phase '{display_name}' was not detected at {temperature:.1f} K "
                    f"across the sampled range. It may form at other temperatures."
                )
            
            # Show detailed composition scan
            response_lines.append("")
            response_lines.append("### Detailed Phase Presence:")
            response_lines.append("")
            response_lines.append(f"_Note: Table shows ~10 representative compositions from {len(results)} total sampled points._")
            response_lines.append("")
            
            # Build table header dynamically
            element_headers = " | ".join([f"{el} at.%" for el in elements])
            header = f"| {element_headers} | {phase_to_check} Present | {phase_to_check} Fraction | Other Major Phases |"
            separator = "|" + "|".join(["---------"] * (len(elements) + 3)) + "|"
            
            response_lines.append(header)
            response_lines.append(separator)
            
            for r in results[::max(1, len(results)//10)]:  # Show ~10 representative points
                present = "✓" if r['phase_present'] else "✗"
                fraction = f"{r['phase_fraction']*100:.1f}%" if r['phase_fraction'] > 0 else "-"
                
                # Build other_phases list with FCC_A1 host element labeling
                other_phases_list = []
                for p, f in sorted(r['all_phases'].items(), key=lambda x: -x[1]):
                    if p != phase_to_check and f > 0.05:
                        # For FCC_A1, indicate which element is the host (majority)
                        if p == 'FCC_A1':
                            # Find majority element at this composition
                            host_el = max(elements, key=lambda el: r[f'at_pct_{el}'])
                            other_phases_list.append(f"{p}({host_el}-rich)")
                        else:
                            other_phases_list.append(p)
                    if len(other_phases_list) >= 3:
                        break
                other_phases = ", ".join(other_phases_list)
                
                # Build row with dynamic element columns
                element_values = " | ".join([f"{r[f'at_pct_{el}']:.1f}" for el in elements])
                response_lines.append(f"| {element_values} | {present} | {fraction} | {other_phases} |")
            
            return {"success": True, "message": "\n".join(response_lines), "citations": ["pycalphad"]}
            
        except Exception as e:
            _log.exception(f"Error verifying phase formation across composition")
            return {"success": False, "error": f"Failed to verify phase formation: {str(e)}", "citations": ["pycalphad"]}
    
    @ai_function(desc="Sweep composition space and evaluate whether a microstructure claim holds across an entire region. Use this to test universal claims like 'all Al-Mg-Zn alloys with Mg<8 at.% and Zn<4 at.% form desirable fcc+tau microstructures'. Returns grid of results and overall verdict.")
    async def sweep_microstructure_claim_over_region(
        self,
        system: Annotated[str, AIParam(desc="Chemical system (e.g., 'Al-Mg-Zn')")],
        element_ranges: Annotated[str, AIParam(desc="JSON dict of element ranges, e.g. '{\"MG\": [0, 8], \"ZN\": [0, 4]}'. Units determined by composition_type parameter (default: atomic percent). Remaining composition is balance element (usually first in system).")],
        claim_type: Annotated[str, AIParam(desc="Type of claim: 'two_phase', 'three_phase', 'phase_fraction'")],
        expected_phases: Annotated[Optional[str], AIParam(desc="Expected phases (e.g., 'fcc+tau'). Required for two_phase and three_phase claims.")] = None,
        phase_to_check: Annotated[Optional[str], AIParam(desc="Specific phase to check. Required for phase_fraction claims.")] = None,
        min_fraction: Annotated[Optional[float], AIParam(desc="Minimum phase fraction (0-1)")] = None,
        max_fraction: Annotated[Optional[float], AIParam(desc="Maximum phase fraction (0-1)")] = None,
        grid_points: Annotated[int, AIParam(desc="Number of grid points per element (default: 4 = 16 total points for binary variation)")] = 4,
        composition_type: Annotated[str, AIParam(desc="Composition units: 'atomic' (at.%, DEFAULT) or 'weight' (wt.%)")] = "atomic",
        process_type: Annotated[str, AIParam(desc="Process model: 'as_cast' or 'equilibrium_300K'")] = "as_cast",
        require_mechanical_desirability: Annotated[bool, AIParam(desc="If true, also require positive mechanical desirability score")] = False
    ) -> Dict[str, Any]:
        """
        Sweep over a composition region and test whether a claim holds universally.
        
        This answers: "Does the claim hold for ALL compositions in the stated range?"
        not just "Does it hold for this one specific composition?"
        
        **Composition Units**: By default uses atomic percent (at.%). 
        For example, element_ranges='{"MG": [0, 8], "ZN": [0, 4]}' means:
        - Mg varies from 0 to 8 at.%
        - Zn varies from 0 to 4 at.%
        - Al (balance) = 100 - Mg - Zn at.%
        
        Example: "All lightweight Al-Mg-Zn with Mg<8 at.% and Zn<4 at.% form desirable fcc+tau"
        → Sweep Mg=[0,8), Zn=[0,4) in atomic percent and check if fcc+tau with tau≤20% everywhere.
        
        Note: Weight percent (wt.%) is not yet fully implemented.
        """
        try:
            import json
            import numpy as np
            
            # Validate claim type and required parameters
            claim_type_lower = claim_type.lower()
            
            # --- Validate required args by claim type ---
            if claim_type_lower == "phase_fraction":
                if not phase_to_check:
                    return {
                        "success": False,
                        "error": "phase_to_check is required for phase_fraction claims "
                                 "(e.g. phase_to_check='tau', min_fraction=0.05, max_fraction=0.20)",
                        "citations": ["pycalphad"]
                    }
            
            elif claim_type_lower == "two_phase":
                if not expected_phases:
                    return {
                        "success": False,
                        "error": "expected_phases is required for two_phase claims "
                                 "(e.g. expected_phases='fcc+theta', max_fraction=0.20)",
                        "citations": ["pycalphad"]
                    }
                phase_list = [p.strip().lower() for p in expected_phases.replace('+', ',').split(',')]
                if len(phase_list) != 2:
                    return {
                        "success": False,
                        "error": (
                            "two_phase claim requires exactly 2 phases (matrix+secondary).\n"
                            f"You gave expected_phases={expected_phases!r} which parsed to {len(phase_list)} phase(s).\n"
                            "Examples: expected_phases='fcc+tau' or 'fcc,theta'\n"
                            "If you're trying to say 'phase X exists between 5%-50%', "
                            "use claim_type='phase_fraction', "
                            "phase_to_check='X', min_fraction=0.05, max_fraction=0.5 instead."
                        ),
                        "citations": ["pycalphad"]
                    }
            
            elif claim_type_lower == "three_phase":
                if not expected_phases:
                    return {
                        "success": False,
                        "error": "expected_phases is required for three_phase claims "
                                 "(e.g. expected_phases='fcc+tau+lambda')",
                        "citations": ["pycalphad"]
                    }
                phase_list = [p.strip().lower() for p in expected_phases.replace('+', ',').split(',')]
                if len(phase_list) != 3:
                    return {
                        "success": False,
                        "error": (
                            "three_phase claim requires exactly 3 phases.\n"
                            f"You gave expected_phases={expected_phases!r} which parsed to {len(phase_list)} phase(s).\n"
                            "Example: expected_phases='fcc+tau+gamma'"
                        ),
                        "citations": ["pycalphad"]
                    }
            
            else:
                return {
                    "success": False,
                    "error": f"Unsupported claim_type '{claim_type}'. "
                             "Use 'two_phase', 'three_phase', or 'phase_fraction'.",
                    "citations": ["pycalphad"]
                }
            
            # Validate composition type
            comp_type = composition_type.lower()
            if comp_type not in ["atomic", "weight"]:
                return {"success": False, "error": f"composition_type must be 'atomic' or 'weight', got '{composition_type}'", "citations": ["pycalphad"]}
            
            if comp_type == "weight":
                return {"success": False, "error": "Weight percent (wt.%) is not yet implemented. Please use composition_type='atomic' (at.%)", "citations": ["pycalphad"]}
            
            # Parse system
            elements_from_system = system.replace('-', ' ').replace('_', ' ').upper().split()
            elements = [self._normalize_element(e) for e in elements_from_system if e]
            elements = [e for e in elements if len(e) <= 2 and e.isalpha()]
            
            if len(elements) < 2:
                return {"success": False, "error": f"Need at least 2 elements in system '{system}'", "citations": ["pycalphad"]}
            
            # Parse element ranges
            try:
                ranges_dict = json.loads(element_ranges) if isinstance(element_ranges, str) else element_ranges
            except json.JSONDecodeError as e:
                return {"success": False, "error": f"Invalid element_ranges JSON: {e}", "citations": ["pycalphad"]}
            
            # Determine balance element (first one not in ranges)
            balance_element = None
            for el in elements:
                if el not in ranges_dict:
                    balance_element = el
                    break
            
            if not balance_element:
                return {"success": False, "error": "Need one balance element not specified in ranges", "citations": ["pycalphad"]}
            
            _log.info(f"Sweeping {system} with ranges {ranges_dict}, balance={balance_element}")
            
            # Build grid
            grid_elements = list(ranges_dict.keys())
            if len(grid_elements) == 1:
                # 1D sweep
                el1 = grid_elements[0]
                el1_min, el1_max = ranges_dict[el1]
                el1_vals = np.linspace(el1_min, el1_max - 1e-6, grid_points)
                grid_compositions = []
                
                for v1 in el1_vals:
                    comp = {balance_element: 100 - v1, el1: v1}
                    # Add other elements as 0
                    for el in elements:
                        if el not in comp:
                            comp[el] = 0.0
                    grid_compositions.append(comp)
                    
            elif len(grid_elements) == 2:
                # 2D sweep
                el1, el2 = grid_elements
                el1_min, el1_max = ranges_dict[el1]
                el2_min, el2_max = ranges_dict[el2]
                
                el1_vals = np.linspace(el1_min, el1_max - 1e-6, grid_points)
                el2_vals = np.linspace(el2_min, el2_max - 1e-6, grid_points)
                
                grid_compositions = []
                for v1 in el1_vals:
                    for v2 in el2_vals:
                        balance_val = 100 - v1 - v2
                        if balance_val >= 0:  # Valid composition
                            comp = {balance_element: balance_val, el1: v1, el2: v2}
                            # Add other elements as 0
                            for el in elements:
                                if el not in comp:
                                    comp[el] = 0.0
                            grid_compositions.append(comp)
            else:
                return {"success": False, "error": "Only supports 1D or 2D sweeps currently", "citations": ["pycalphad"]}
            
            _log.info(f"Generated {len(grid_compositions)} grid points")
            
            # Evaluate claim at each grid point
            results_grid = []
            pass_count = 0
            fail_count = 0
            mech_fail_count = 0
            
            for comp_dict in grid_compositions:
                # Format composition string with separators for parser compatibility
                # Use "-" separator when we have decimals to avoid ambiguity
                comp_parts = [f"{el}{comp_dict[el]:.1f}" for el in elements if comp_dict[el] > 0.01]
                comp_str = "-".join(comp_parts)
                
                # Call the single-point checker
                result = await self.fact_check_microstructure_claim(
                    system=system,
                    composition=comp_str,
                    claim_type=claim_type,
                    expected_phases=expected_phases,
                    phase_to_check=phase_to_check,
                    min_fraction=min_fraction,
                    max_fraction=max_fraction,
                    process_type=process_type,
                    temperature=None,
                    composition_constraints=None  # Don't gate - we're generating valid points
                )
                
                point_pass = result.get("verdict", False)
                mech_ok = True
                
                if require_mechanical_desirability and "mechanical_score" in result:
                    mech_ok = result["mechanical_score"] > 0
                    if not mech_ok:
                        mech_fail_count += 1
                
                overall_pass = point_pass and mech_ok
                
                if overall_pass:
                    pass_count += 1
                else:
                    fail_count += 1
                
                # Extract phase information from supporting_data (handles both "phases" and "all_phases" keys)
                supporting_data = result.get("supporting_data", {})
                error_msg = result.get("error")
                
                # Debug: log if we're missing phase data
                if not supporting_data and not result.get("success", True):
                    _log.warning(f"Point {comp_str}: fact_check failed - {error_msg}")
                
                # Check which key exists - don't use 'or' because empty list is falsy
                if "phases" in supporting_data:
                    phase_list = supporting_data["phases"]
                elif "all_phases" in supporting_data:
                    phase_list = supporting_data["all_phases"]
                else:
                    phase_list = []
                    if supporting_data:  # Has data but no phase keys
                        _log.warning(f"Point {comp_str}: supporting_data has keys {list(supporting_data.keys())} but no phases/all_phases")
                
                results_grid.append({
                    "composition": comp_dict,
                    "composition_str": comp_str,
                    "microstructure_verdict": point_pass,
                    "mechanical_ok": mech_ok if require_mechanical_desirability else None,
                    "overall_pass": overall_pass,
                    "score": result.get("score", 0),
                    "phases": phase_list,
                    "error": error_msg  # Include error for debugging
                })
            
            # Aggregate verdict
            total_points = len(results_grid)
            pass_fraction = pass_count / total_points if total_points > 0 else 0.0
            
            if pass_fraction == 1.0:
                overall_verdict = "UNIVERSALLY SUPPORTED"
                overall_score = 2
                confidence = 1.0
            elif pass_fraction >= 0.9:
                overall_verdict = "MOSTLY SUPPORTED"
                overall_score = 1
                confidence = 0.8
            elif pass_fraction >= 0.5:
                overall_verdict = "MIXED"
                overall_score = 0
                confidence = 0.5
            elif pass_fraction > 0:
                overall_verdict = "MOSTLY REJECTED"
                overall_score = -1
                confidence = 0.7
            else:
                overall_verdict = "UNIVERSALLY REJECTED"
                overall_score = -2
                confidence = 1.0
            
            # Format response
            comp_unit = "at.%" if comp_type == "atomic" else "wt.%"
            message_lines = [
                f"## Region Sweep Fact-Check Result",
                f"",
                f"**System**: {system}",
                f"**Composition Region**: {', '.join([f'{el} ∈ [{r[0]}, {r[1]}) {comp_unit}' for el, r in ranges_dict.items()])}",
                f"**Grid Points**: {total_points} ({grid_points} per element)",
                f"**Process**: {process_type}",
                f"**Claim**: {claim_type} - {expected_phases or phase_to_check}",
                f"",
                f"### Verdict: **{overall_verdict}**",
                f"- **Score**: {overall_score:+d}/2 (confidence: {confidence:.0%})",
                f"- **Pass Rate**: {pass_count}/{total_points} compositions ({pass_fraction:.1%})",
            ]
            
            if require_mechanical_desirability:
                microstructure_pass = pass_count + mech_fail_count  # Points that passed microstructure check
                message_lines.append(f"- **Mechanical Desirability**: Required (score > 0)")
                message_lines.append(f"  - Microstructure match: {microstructure_pass}/{total_points}")
                message_lines.append(f"  - Mechanical failures: {mech_fail_count}/{total_points} (brittle/undesirable phases)")
                message_lines.append(f"  - Overall pass: {pass_count}/{total_points}")
            
            message_lines.extend([
                f"",
                f"### Interpretation:",
            ])
            
            if overall_score >= 1:
                message_lines.append(f"The claim holds across {'all' if overall_score == 2 else 'most of'} the stated composition region. The microstructure prediction is {'universally' if overall_score == 2 else 'generally'} valid.")
                if require_mechanical_desirability and mech_fail_count > 0:
                    message_lines.append(f"Note: Some compositions matched the claimed phases but were rejected due to poor mechanical properties (e.g., brittle intermetallics dominating).")
            elif overall_score == 0:
                message_lines.append(f"The claim holds in some parts of the region but fails in others. The statement is over-generalized.")
                if require_mechanical_desirability and mech_fail_count > 0:
                    message_lines.append(f"Note: {mech_fail_count} compositions matched the phases but had undesirable mechanical properties.")
            else:
                message_lines.append(f"The claim fails across {'all' if overall_score == -2 else 'most of'} the composition region. The statement is incorrect.")
                if require_mechanical_desirability and mech_fail_count > 0:
                    message_lines.append(f"Note: Some failures were due to poor mechanical desirability rather than wrong phase mix.")
            
            # Sample failures with detailed reasons
            failures = [r for r in results_grid if not r["overall_pass"]]
            if failures and len(failures) <= 5:
                message_lines.append(f"")
                message_lines.append(f"**Failed Compositions:**")
                for fail in failures:
                    reason = ""
                    if require_mechanical_desirability and fail.get("mechanical_ok") is False:
                        reason = " (mechanical failure)"
                    elif not fail.get("microstructure_verdict"):
                        reason = " (microstructure mismatch)"
                    message_lines.append(f"- {fail['composition_str']}: Score {fail['score']:+d}{reason}")
            elif failures:
                message_lines.append(f"")
                message_lines.append(f"**Sample Failures** (showing 5 of {len(failures)}):")
                for fail in failures[:5]:
                    reason = ""
                    if require_mechanical_desirability and fail.get("mechanical_ok") is False:
                        reason = " (mechanical failure)"
                    elif not fail.get("microstructure_verdict"):
                        reason = " (microstructure mismatch)"
                    message_lines.append(f"- {fail['composition_str']}: Score {fail['score']:+d}{reason}")
            
            result_dict = {
                "success": True,
                "message": "\n".join(message_lines),
                "overall_verdict": overall_verdict,
                "overall_score": overall_score,
                "confidence": confidence,
                "pass_count": pass_count,
                "fail_count": fail_count,
                "total_points": total_points,
                "pass_fraction": pass_fraction,
                "grid_results": results_grid[:20],  # Limit to first 20 for response size
                "citations": ["pycalphad"]
            }
            
            if require_mechanical_desirability:
                result_dict["mechanical_fail_count"] = mech_fail_count
                result_dict["microstructure_pass_count"] = pass_count + mech_fail_count
            
            return result_dict
            
        except Exception as e:
            _log.exception("Error in region sweep fact-check")
            return {"success": False, "error": f"Region sweep failed: {str(e)}", "citations": ["pycalphad"]}
    
    @ai_function(desc="Evaluate microstructure claims for multicomponent alloys. Use this to fact-check metallurgical assertions like 'Al-8Mg-4Zn forms fcc + tau phase with tau < 20%' or 'eutectic composition shows >20% intermetallics'. Returns detailed verdict with score (-2 to +2) and supporting thermodynamic data.")
    async def fact_check_microstructure_claim(
        self,
        system: Annotated[str, AIParam(desc="Chemical system (e.g., 'Al-Mg-Zn', 'Fe-Cr-Ni')")],
        composition: Annotated[str, AIParam(desc="Composition in at.% (e.g., 'Al88Mg8Zn4', '88Al-8Mg-4Zn', or 'Al-8Mg-4Zn')")],
        claim_type: Annotated[str, AIParam(desc="Type of claim: 'two_phase', 'three_phase', 'phase_fraction', or 'custom'")],
        expected_phases: Annotated[Optional[str], AIParam(desc="Expected phases (e.g., 'fcc+tau', 'fcc+laves+gamma'). Comma or + separated.")] = None,
        phase_to_check: Annotated[Optional[str], AIParam(desc="Specific phase to check fraction (e.g., 'tau', 'gamma', 'laves')")] = None,
        min_fraction: Annotated[Optional[float], AIParam(desc="Minimum required phase fraction (0-1, e.g., 0.20 for 20%)")] = None,
        max_fraction: Annotated[Optional[float], AIParam(desc="Maximum allowed phase fraction (0-1, e.g., 0.20 for 20%)")] = None,
        process_type: Annotated[str, AIParam(desc="Process model: 'as_cast' (after solidification, DEFAULT) or 'equilibrium_300K' (infinite-time room-temp equilibrium)")] = "as_cast",
        temperature: Annotated[Optional[float], AIParam(desc="Temperature in K for evaluation (only used if process_type='equilibrium_300K')")] = None,
        composition_constraints: Annotated[Optional[str], AIParam(desc="Composition constraints as JSON string, For example, if the claim is about an Al-Mg-Zn alloy, and the claim is that the alloy has less than 8% Mg and less than 4% Zn, the composition_constraints would be '{\"MG\": {\"lt\": 8.0}, \"ZN\": {\"lt\": 4.0}}'. Supports: lt, lte, gt, gte, between:[min,max]")] = None
    ) -> Dict[str, Any]:
        """
        Evaluate microstructure claims using CALPHAD thermodynamic calculations.
        
        This function acts as an automated "materials expert witness" that can verify
        metallurgical assertions about multicomponent alloys by:
        1. Simulating the processing path (as-cast solidification or full equilibrium)
        2. Calculating resulting phases and phase fractions
        3. Interpreting CALPHAD phases into metallurgical categories (fcc, tau, gamma, etc.)
        4. Evaluating mechanical desirability based on phase distribution
        5. Returning a verdict with score and detailed reasoning
        
        **IMPORTANT**: By default uses 'as_cast' process_type which simulates
        slow solidification from the melt (typical casting). This answers:
        "What do I get after the alloy freezes?" NOT "What do I get after infinite
        time at room temperature?" (which would require process_type='equilibrium_300K').
        
        Returns detailed verdict with:
        - verdict: True/False (claim supported or rejected)
        - score: -2 to +2 (-2 = completely wrong, +2 = fully correct)
        - confidence: 0-1 (confidence in the verdict)
        - reasoning: Explanation of why claim passed/failed
        - mechanical_score: -1/0/+1 (brittleness/desirability assessment)
        - supporting_data: Phase fractions and thermodynamic details
        
        Example usage:
        - Claim: "Slowly solidified Al-8Mg-4Zn forms fcc + tau with tau < 20%"
          → claim_type='two_phase', expected_phases='fcc+tau', max_fraction=0.20, process_type='as_cast'
        
        - Claim: "After equilibration at 300K, Al-8Mg-4Zn has fcc + Laves + gamma"
          → claim_type='three_phase', expected_phases='fcc+laves+gamma', process_type='equilibrium_300K'
        
        - Claim: "Eutectic Al-Mg-Zn (~34.5% Mg, 5% Zn) has >20% tau after casting"
          → claim_type='phase_fraction', phase_to_check='tau', min_fraction=0.20, process_type='as_cast'
        """
        try:
            from .fact_checker import (
                AlloyFactChecker, TwoPhaseChecker, ThreePhaseChecker, 
                PhaseFractionChecker, atpct_to_molefrac, interpret_microstructure
            )
            from .solidification_utils import (
                simulate_as_cast_microstructure_simple,
                mechanical_desirability_score
            )
            
            # Parse system to get elements
            parsed = self._normalize_system(system, db=None)
            if len(parsed) < 2:
                return {"success": False, "error": "System must have at least 2 elements", "citations": ["pycalphad"]}
            
            # For ternary systems, extract all 3 elements from system string
            elements_from_system = system.replace('-', ' ').replace('_', ' ').upper().split()
            elements = [self._normalize_element(e) for e in elements_from_system if e]
            elements = [e for e in elements if len(e) <= 2 and e.isalpha()]  # Valid element symbols
            
            if len(elements) < 2:
                return {"success": False, "error": f"Could not parse elements from system '{system}'", "citations": ["pycalphad"]}
            
            # Parse composition (e.g., "Al88Mg8Zn4" or "88Al-8Mg-4Zn" or "Al-8Mg-4Zn")
            comp_dict = self._parse_composition_string(composition, elements)
            if not comp_dict:
                return {"success": False, "error": f"Could not parse composition '{composition}'", "citations": ["pycalphad"]}
            
            # Check composition constraints if provided
            violations = []
            composition_within_bounds = True
            
            if composition_constraints:
                import json
                try:
                    constraints_dict = json.loads(composition_constraints) if isinstance(composition_constraints, str) else composition_constraints
                    
                    for el, rules in constraints_dict.items():
                        el_upper = el.upper()
                        if el_upper not in comp_dict:
                            continue  # Element not in alloy, skip
                        
                        val = comp_dict[el_upper]  # at.%
                        
                        # Check less than
                        if "lt" in rules:
                            if not (val < rules["lt"]):
                                violations.append(f"{el_upper}={val:.2f} at.% is not < {rules['lt']:.2f} at.%")
                        
                        # Check less than or equal
                        if "lte" in rules:
                            if not (val <= rules["lte"]):
                                violations.append(f"{el_upper}={val:.2f} at.% is not ≤ {rules['lte']:.2f} at.%")
                        
                        # Check greater than
                        if "gt" in rules:
                            if not (val > rules["gt"]):
                                violations.append(f"{el_upper}={val:.2f} at.% is not > {rules['gt']:.2f} at.%")
                        
                        # Check greater than or equal
                        if "gte" in rules:
                            if not (val >= rules["gte"]):
                                violations.append(f"{el_upper}={val:.2f} at.% is not ≥ {rules['gte']:.2f} at.%")
                        
                        # Check between range
                        if "between" in rules:
                            lo, hi = rules["between"]
                            if not (lo <= val <= hi):
                                violations.append(f"{el_upper}={val:.2f} at.% not in [{lo:.2f}, {hi:.2f}] at.%")
                    
                    composition_within_bounds = (len(violations) == 0)
                    
                    if violations:
                        _log.warning(f"Composition constraint violations: {'; '.join(violations)}")
                    
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    _log.error(f"Failed to parse composition_constraints: {e}")
                    return {"success": False, "error": f"Invalid composition_constraints format: {str(e)}", "citations": ["pycalphad"]}
            
            # Convert to mole fractions
            comp_molefrac = atpct_to_molefrac(comp_dict)
            
            # Select database
            db_path = self._get_database_path(system, elements=elements)
            if not db_path:
                return {"success": False, "error": f"No .tdb found for system {system}", "citations": ["pycalphad"]}
            db = Database(str(db_path))
            
            # Get phases
            phases = self._get_phases_for_elements(db, elements)
            _log.debug(f"Selected {len(phases)} phases for {'-'.join(elements)} system")
            
            # Determine processing path and phase fractions
            precalc_fractions = None
            process_description = ""
            
            if process_type.lower() == "as_cast":
                # Simulate as-cast microstructure (after slow solidification)
                _log.info("Using as-cast solidification simulation")
                
                # Warn if user passed temperature but it will be ignored
                if temperature is not None and temperature != 300.0:
                    _log.warning(f"Temperature {temperature}K was provided but will be ignored in as_cast mode. "
                               f"As-cast uses the freezing range temperature from solidification simulation. "
                               f"Use process_type='equilibrium_300K' to honor the specified temperature.")
                
                precalc_fractions, T_ascast = simulate_as_cast_microstructure_simple(
                    db, elements, phases, comp_molefrac
                )
                
                if not precalc_fractions or T_ascast is None:
                    return {
                        "success": False,
                        "error": "As-cast solidification simulation failed. Try process_type='equilibrium_300K' instead.",
                        "citations": ["pycalphad"]
                    }
                
                # Use the actual as-cast temperature from simulation
                T_ref = T_ascast
                process_description = f"as-cast after slow solidification from melt (T={T_ascast:.0f}K, ~{T_ascast-273.15:.0f}°C)"
                
                # Add note to description if temperature was overridden
                if temperature is not None and temperature != 300.0:
                    process_description += f" [Note: provided temperature {temperature}K was overridden by solidification model]"
                
            elif process_type.lower() == "equilibrium_300k":
                # Traditional equilibrium at specified temperature
                T_ref = temperature or 300.0
                _log.info(f"Using equilibrium calculation at {T_ref}K")
                
                # For low temperatures (<500K), exclude LIQUID as it's metastable
                if T_ref < 500.0 and "LIQUID" in phases:
                    phases = [p for p in phases if p != "LIQUID"]
                    _log.info(f"Excluded LIQUID phase at low temperature ({T_ref}K)")
                
                process_description = f"equilibrium at {T_ref:.0f}K after infinite diffusion time"
                # precalc_fractions stays None, will be calculated during check()
                
            else:
                return {
                    "success": False,
                    "error": f"Unknown process_type '{process_type}'. Use 'as_cast' or 'equilibrium_300K'.",
                    "citations": ["pycalphad"]
                }
            
            # Create fact-checker (T_ref used only for equilibrium calculations if needed)
            fact_checker = AlloyFactChecker(db, elements, phases, T_ref)
            
            # Add appropriate checker based on claim type
            if claim_type.lower() == "two_phase":
                # Parse expected phases
                if not expected_phases:
                    return {"success": False, "error": "expected_phases required for two_phase claim", "citations": ["pycalphad"]}
                
                phase_list = [p.strip().lower() for p in expected_phases.replace('+', ',').split(',')]
                if len(phase_list) != 2:
                    return {"success": False, "error": "two_phase claim requires exactly 2 phases", "citations": ["pycalphad"]}
                
                # Map phase names to categories
                primary_cat = self._map_phase_to_category(phase_list[0])
                secondary_cat = self._map_phase_to_category(phase_list[1])
                
                checker = TwoPhaseChecker(
                    db, elements, phases,
                    primary_category=primary_cat,
                    secondary_category=secondary_cat,
                    secondary_max_fraction=max_fraction or 0.20,
                    temperature=T_ref
                )
                fact_checker.add_checker(checker)
                
            elif claim_type.lower() == "three_phase":
                # Parse expected phases
                if not expected_phases:
                    return {"success": False, "error": "expected_phases required for three_phase claim", "citations": ["pycalphad"]}
                
                phase_list = [p.strip().lower() for p in expected_phases.replace('+', ',').split(',')]
                if len(phase_list) != 3:
                    return {"success": False, "error": "three_phase claim requires exactly 3 phases", "citations": ["pycalphad"]}
                
                # Map to categories
                categories = [self._map_phase_to_category(p) for p in phase_list]
                
                checker = ThreePhaseChecker(
                    db, elements, phases,
                    expected_categories=categories,
                    temperature=T_ref
                )
                fact_checker.add_checker(checker)
                
            elif claim_type.lower() == "phase_fraction":
                if not phase_to_check:
                    return {"success": False, "error": "phase_to_check required for phase_fraction claim", "citations": ["pycalphad"]}
                
                target_cat = self._map_phase_to_category(phase_to_check.lower())
                
                checker = PhaseFractionChecker(
                    db, elements, phases,
                    target_category=target_cat,
                    min_fraction=min_fraction,
                    max_fraction=max_fraction,
                    temperature=T_ref
                )
                fact_checker.add_checker(checker)
            else:
                return {"success": False, "error": f"Unknown claim_type: {claim_type}. Use 'two_phase', 'three_phase', or 'phase_fraction'", "citations": ["pycalphad"]}
            
            # Evaluate claims (with precalculated fractions if using as_cast)
            results = fact_checker.evaluate_all(comp_molefrac, precalculated_fractions=precalc_fractions)
            report = fact_checker.generate_report(
                comp_molefrac, 
                precalculated_fractions=precalc_fractions,
                process_description=process_description
            )
            
            # Calculate mechanical desirability score (only for as_cast)
            mech_score = 0.0
            mech_interpretation = "Not evaluated"
            
            if process_type.lower() == "as_cast" and precalc_fractions:
                # For as-cast, evaluate mechanical properties
                # (Only meaningful for as-cast microstructures, not infinite-time equilibrium)
                microstructure = interpret_microstructure(precalc_fractions)
                phase_categories = {p.base_name: p.category.value for p in microstructure}
                mech_score, mech_interpretation = mechanical_desirability_score(
                    precalc_fractions, phase_categories
                )
                _log.info(f"Mechanical desirability: {mech_score} - {mech_interpretation}")
            
            # Format response
            if results:
                result = results[0]  # Get first (usually only) result
                
                # Apply composition constraint violations
                if composition_constraints and not composition_within_bounds:
                    # Composition is outside the claim's stated bounds
                    # Even if microstructure matched, the claim doesn't apply here
                    original_verdict = result.verdict
                    original_score = result.score
                    
                    result.verdict = False
                    
                    # If microstructure was good but chemistry is wrong, mild fail (-1)
                    # If both were wrong, keep the harsh score
                    if original_score > 0:
                        result.score = -1
                    
                    # Append violation info to reasoning
                    result.reasoning += f" | COMPOSITION OUT OF BOUNDS: {'; '.join(violations)}"
                    
                    # Store in supporting data
                    result.supporting_data["composition_within_bounds"] = False
                    result.supporting_data["composition_violations"] = violations
                    
                    _log.warning(f"Verdict adjusted due to composition constraints: {original_verdict} → {result.verdict}, score: {original_score} → {result.score}")
                else:
                    result.supporting_data["composition_within_bounds"] = True
                    result.supporting_data["composition_violations"] = []
                
                # Format supporting data for display
                phases_info = []
                _log.info(f"Supporting data keys: {list(result.supporting_data.keys())}")
                
                if "phases" in result.supporting_data or "all_phases" in result.supporting_data:
                    phase_data = result.supporting_data.get("phases") or result.supporting_data.get("all_phases")
                    _log.info(f"Phase data type: {type(phase_data)}, length: {len(phase_data) if phase_data else 0}")
                    
                    if phase_data:
                        for phase_info in phase_data[:10]:  # Top 10 phases
                            if len(phase_info) >= 3:
                                phases_info.append(f"{phase_info[0]}: {phase_info[1]*100:.1f}% ({phase_info[2]})")
                else:
                    _log.warning("No 'phases' or 'all_phases' key in supporting_data")
                
                verdict_emoji = "✓" if result.verdict else "✗"
                score_text = f"{result.score:+d}/2"
                
                message_lines = [
                    f"## Microstructure Fact-Check Result",
                    f"",
                    f"**Composition**: {composition} ({'-'.join(elements)} system)",
                    f"**Process**: {process_description}",
                    f"**Claim**: {result.claim_text}",
                    f"",
                    f"### {verdict_emoji} Verdict: **{'SUPPORTED' if result.verdict else 'REJECTED'}**",
                    f"- **Score**: {score_text} (confidence: {result.confidence:.0%})",
                    f"- **Reasoning**: {result.reasoning}",
                ]
                
                # Add composition constraint status if checked
                if composition_constraints:
                    if composition_within_bounds:
                        message_lines.append(f"- **Composition Bounds**: ✓ Within stated constraints")
                    else:
                        message_lines.append(f"- **Composition Bounds**: ✗ VIOLATED - {'; '.join(violations)}")
                
                # Add mechanical desirability if evaluated (only for as_cast)
                if process_type.lower() == "as_cast" and (mech_score != 0.0 or mech_interpretation != "Not evaluated"):
                    mech_emoji = "✓" if mech_score > 0 else ("✗" if mech_score < 0 else "○")
                    message_lines.append(f"- **Mechanical Desirability**: {mech_emoji} {mech_interpretation} (score: {mech_score:+.1f})")
                
                if phases_info:
                    message_lines.append(f"\n### Calculated Phase Fractions:")
                    for phase_line in phases_info:
                        message_lines.append(f"- {phase_line}")
                
                message_lines.append(f"\n---")
                message_lines.append(f"\n**Full Report:**\n```\n{report}\n```")
                
                return {
                    "success": True,
                    "message": "\n".join(message_lines),
                    "verdict": result.verdict,
                    "score": result.score,
                    "confidence": result.confidence,
                    "mechanical_score": mech_score,
                    "mechanical_interpretation": mech_interpretation,
                    "process_type": process_type,
                    "supporting_data": result.supporting_data,
                    "citations": ["pycalphad"]
                }
            else:
                return {"success": False, "error": "No results from fact-checker", "citations": ["pycalphad"]}
                
        except Exception as e:
            _log.exception(f"Error in fact-check microstructure claim")
            return {"success": False, "error": f"Fact-check failed: {str(e)}", "citations": ["pycalphad"]}
    
    # Helper methods for fact-checker integration
    
    def _normalize_element(self, element: str) -> str:
        """Normalize element symbol to uppercase (e.g., 'al' -> 'AL', 'Mg' -> 'MG')."""
        element = element.strip()
        if len(element) == 1:
            return element.upper()
        elif len(element) == 2:
            return element[0].upper() + element[1].upper()
        else:
            return element.upper()
    
    def _parse_composition_string(self, comp_str: str, elements: List[str]) -> Optional[Dict[str, float]]:
        """
        Parse composition string into {element: at%} dict.
        
        Supports formats:
        - "Al88Mg8Zn4" -> {"AL": 88, "MG": 8, "ZN": 4}
        - "88Al-8Mg-4Zn" -> {"AL": 88, "MG": 8, "ZN": 4}
        - "Al-8Mg-4Zn" -> {"AL": remaining, "MG": 8, "ZN": 4}
        """
        import re
        
        _log.info(f"Parsing composition: '{comp_str}' with elements {elements}")
        comp_dict = {}
        comp_str = comp_str.replace(' ', '').replace('_', '').upper()
        _log.info(f"Cleaned composition string: '{comp_str}'")
        
        # Try pattern like "AL88MG8ZN4" or "88AL8MG4ZN"
        for element in elements:
            el_upper = element.upper()
            # Look for patterns like "AL88" or "88AL"
            pattern1 = rf"{el_upper}(\d+\.?\d*)"  # AL88
            pattern2 = rf"(\d+\.?\d*){el_upper}"  # 88AL
            
            match = re.search(pattern1, comp_str)
            if not match:
                match = re.search(pattern2, comp_str)
            
            if match:
                comp_dict[el_upper] = float(match.group(1))
        
        # If we found some but not all elements, it might be hyphen-separated format
        # Clear and try the split method instead
        if comp_dict and len(comp_dict) < len(elements):
            _log.info(f"Found {len(comp_dict)}/{len(elements)} elements in first pass, trying split method")
            comp_dict = {}
        
        # If we didn't find explicit percentages, try parsing "Al-8Mg-4Zn" format
        if not comp_dict:
            parts = re.split(r'[-,]', comp_str)
            _log.info(f"Split parts: {parts}")
            for part in parts:
                _log.info(f"Processing part: '{part}'")
                # Try two patterns: "AL8" or "8AL"
                # Pattern 1: Element followed by number (AL8)
                match = re.match(r'^([A-Z][A-Z]?)(\d+\.?\d*)$', part)
                if match:
                    el = match.group(1)
                    pct = float(match.group(2))
                    _log.info(f"  Pattern 1 matched: el={el}, pct={pct}, in_elements={el in [e.upper() for e in elements]}")
                    if el in [e.upper() for e in elements]:
                        comp_dict[el] = pct
                    continue
                
                # Pattern 2: Number followed by element (8AL)
                match = re.match(r'^(\d+\.?\d*)([A-Z][A-Z]?)$', part)
                if match:
                    pct = float(match.group(1))
                    el = match.group(2)
                    _log.info(f"  Pattern 2 matched: pct={pct}, el={el}, in_elements={el in [e.upper() for e in elements]}")
                    if el in [e.upper() for e in elements]:
                        comp_dict[el] = pct
                    continue
                
                # Pattern 3: Element only (no number - will be balance)
                match = re.match(r'^([A-Z][A-Z]?)$', part)
                if match:
                    el = match.group(1)
                    _log.info(f"  Pattern 3 matched: el={el}, in_elements={el in [e.upper() for e in elements]}")
                    if el in [e.upper() for e in elements]:
                        comp_dict[el] = None
                    continue
                
                _log.warning(f"  No pattern matched for part '{part}'")
        
        # Handle case where one element doesn't have a number (it's the balance)
        if None in comp_dict.values():
            specified_total = sum(v for v in comp_dict.values() if v is not None)
            _log.info(f"Calculating balance: specified_total={specified_total}%, balance={100.0-specified_total}%")
            for el, val in comp_dict.items():
                if val is None:
                    comp_dict[el] = 100.0 - specified_total
        
        _log.info(f"Parsed composition dict (before normalization): {comp_dict}")
        
        # Validate
        if not comp_dict:
            return None
        
        total = sum(comp_dict.values())
        
        # Normalize composition to sum to 100%
        if abs(total - 100.0) > 0.1:
            # Try normalizing if close to 1.0 (mole fractions given)
            if 0.9 < total < 1.1:
                comp_dict = {el: val * 100 for el, val in comp_dict.items()}
            elif total > 0:
                # Normalize to 100% if it's off (e.g., 96% -> scale to 100%)
                _log.info(f"Normalizing composition from {total}% to 100%")
                comp_dict = {el: (val / total) * 100.0 for el, val in comp_dict.items()}
            else:
                _log.error(f"Invalid composition: total = {total}%")
                return None
        
        _log.info(f"Final parsed composition: {comp_dict} (total: {sum(comp_dict.values()):.1f}%)")
        return comp_dict if comp_dict else None
    
    def _map_phase_to_category(self, phase_name: str):
        """Map phase name to PhaseCategory enum."""
        from .fact_checker import PhaseCategory
        
        phase_lower = phase_name.lower().strip()
        
        # Map common names to categories
        mapping = {
            'fcc': PhaseCategory.PRIMARY_FCC,
            'fcc_a1': PhaseCategory.PRIMARY_FCC,
            'bcc': PhaseCategory.PRIMARY_BCC,
            'bcc_a2': PhaseCategory.PRIMARY_BCC,
            'hcp': PhaseCategory.PRIMARY_HCP,
            'hcp_a3': PhaseCategory.PRIMARY_HCP,
            'tau': PhaseCategory.TAU_PHASE,
            'tau_phase': PhaseCategory.TAU_PHASE,
            't_phase': PhaseCategory.TAU_PHASE,
            't': PhaseCategory.TAU_PHASE,
            'mgalzn_t': PhaseCategory.TAU_PHASE,
            'al2mg3zn3': PhaseCategory.TAU_PHASE,
            'gamma': PhaseCategory.GAMMA,
            'gamma_prime': PhaseCategory.GAMMA,
            'laves': PhaseCategory.LAVES,
            'c14': PhaseCategory.LAVES,
            'c15': PhaseCategory.LAVES,
            'mgzn2': PhaseCategory.LAVES,
            'sigma': PhaseCategory.SIGMA,
            'liquid': PhaseCategory.LIQUID,
        }
        
        return mapping.get(phase_lower, PhaseCategory.OTHER)
    
    def _get_phases_for_elements(self, db: Database, elements: List[str]) -> List[str]:
        """Get relevant phases for a set of elements - only phases with EXACTLY our elements (no extras)."""
        # For ternary+, use simplified filtering
        if len(elements) == 2:
            return self._filter_phases_for_system(db, tuple(elements[:2]))
        else:
            # For ternary/higher, include ONLY phases that contain exclusively our elements (+ VA)
            all_phases = list(db.phases.keys())
            relevant = []
            allowed_elements = set(el.upper() for el in elements) | {"VA"}  # Our elements + vacancy
            
            _log.info(f"Filtering phases for elements: {elements}, allowed: {allowed_elements}")
            
            for phase_name in all_phases:
                # Always include LIQUID (will be excluded at low T in caller)
                if phase_name == "LIQUID":
                    relevant.append(phase_name)
                    continue
                
                # Get elements in this phase
                phase_els = self._phase_elements(db, phase_name)
                
                # Only include if phase contains ONLY our elements (no extras like Cr, Cu)
                # Phase must have at least one of our elements AND no forbidden elements
                has_our_elements = any(el in allowed_elements for el in phase_els)
                has_forbidden_elements = any(el not in allowed_elements for el in phase_els)
                
                # Special case: always include common phases and known ternary phases
                is_common_phase = phase_name in ["FCC_A1", "BCC_A2", "BCC_B2", "HCP_A3", "HCP_ZN"]
                is_ternary_phase = phase_name in ["TAU", "TAU_PHASE", "T_PHASE", "PHI", "VPHASE"]
                
                if (has_our_elements and not has_forbidden_elements) or is_common_phase or is_ternary_phase:
                    relevant.append(phase_name)
                    _log.debug(f"  ✓ Including phase {phase_name}: elements={phase_els}")
                else:
                    _log.debug(f"  ✗ Excluding phase {phase_name}: elements={phase_els} (forbidden elements present)")
            
            _log.info(f"Selected {len(relevant)} phases for {'-'.join(elements)} system")
            return relevant if relevant else all_phases