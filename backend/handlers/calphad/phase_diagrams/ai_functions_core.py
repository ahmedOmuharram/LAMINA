"""
Core AI function methods for CALPHAD phase diagram visualization.

Contains the primary visualization functions:
- plot_binary_phase_diagram: Generate binary phase diagrams
- plot_composition_temperature: Plot phase stability vs temperature for specific compositions
- analyze_last_generated_plot: Analyze recently generated plots
"""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, Optional
import logging
import time

from pycalphad import Database, binplot
import pycalphad.variables as v
from kani.ai_function import ai_function
from typing_extensions import Annotated
from kani import AIParam

from .database_utils import get_db_elements, map_phase_name
from ...shared.calphad_utils import (
    extract_phase_fractions_from_equilibrium,
    load_tdb_database,
    compute_equilibrium
)
from ...shared.result_wrappers import success_result, error_result, Confidence, ErrorType

_log = logging.getLogger(__name__)


class CoreVisualizationMixin:
    """Mixin class containing core visualization AI functions for CalPhadHandler."""
    
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
        start_time = time.time()
        
        try:
            # Clear any previous plot metadata at the start of new generation
            if hasattr(self, '_last_image_metadata'):
                delattr(self, '_last_image_metadata')
            if hasattr(self, '_last_image_data'):
                delattr(self, '_last_image_data')
            
            # Parse system first to get elements for database selection
            A, B = self._normalize_system(system, db=None)
            
            # Select database based on elements
            db = load_tdb_database([A, B])
            if db is None:
                duration_ms = (time.time() - start_time) * 1000
                return error_result(
                    handler="calphad",
                    function="plot_binary_phase_diagram",
                    error="No .tdb database found for this system.",
                    error_type=ErrorType.NOT_FOUND,
                    citations=["pycalphad"],
                    duration_ms=duration_ms
                )

            # Ensure both elements are in the selected DB
            db_elems = get_db_elements(db)
            if not (A in db_elems and B in db_elems):
                duration_ms = (time.time() - start_time) * 1000
                return error_result(
                    handler="calphad",
                    function="plot_binary_phase_diagram",
                    error=f"Elements '{A}' and '{B}' must both exist in the database ({sorted(db_elems)}).",
                    error_type=ErrorType.INVALID_INPUT,
                    citations=["pycalphad"],
                    duration_ms=duration_ms
                )
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
            axes.set_xlabel(f"Mole fraction of {comp_el}, $x_{{{comp_el}}}$")
            axes.set_ylabel("Temperature (K)")
            fig.suptitle(f"{A}-{B} Phase Diagram", x=0.5, fontsize=14, fontweight='bold')
            axes.grid(True, alpha=0.3)
            axes.set_xlim(0, 1)
            
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
            duration_ms = (time.time() - start_time) * 1000
            
            if "data:image/png;base64," in success_msg or len(success_msg) > 1000:
                _log.error("ERROR - Success message contains base64 data or is too long! Truncating.")
                success_msg = "Successfully generated phase diagram. Image will be displayed separately."
            
            result = success_result(
                handler="calphad",
                function="plot_binary_phase_diagram",
                data={
                    "message": success_msg,
                    "system": f"{A}-{B}",
                    "phases": phases,
                    "temperature_range_K": [T_display_lo, T_display_hi],
                    "key_points": key_points if key_points else []
                },
                citations=["pycalphad"],
                has_image=True,
                image_url=plot_url,
                confidence=Confidence.HIGH,
                duration_ms=duration_ms
            )
            
            # Store the result for tooltip display
            self._track_tool_output("plot_binary_phase_diagram", result)
            return result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            _log.exception(f"Error generating phase diagram for {system}")
            result = error_result(
                handler="calphad",
                function="plot_binary_phase_diagram",
                error=f"Failed to generate phase diagram for {system}: {str(e)}",
                error_type=ErrorType.COMPUTATION_ERROR,
                citations=["pycalphad"],
                duration_ms=duration_ms
            )
            self._track_tool_output("plot_binary_phase_diagram", result)
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
        start_time = time.time()
        
        try:
            # reset previous artifacts
            if hasattr(self, '_last_image_metadata'):
                delattr(self, '_last_image_metadata')
            if hasattr(self, '_last_image_data'):
                delattr(self, '_last_image_data')
            
            # Parse composition and get system
            (A, B), xB, comp_type = self._parse_composition(composition, composition_type or "atomic")
            
            # Load database with element-based selection
            db = load_tdb_database([A, B])
            if db is None:
                duration_ms = (time.time() - start_time) * 1000
                return error_result(
                    handler="calphad",
                    function="plot_composition_temperature",
                    error=f"No thermodynamic database found for {A}-{B} system.",
                    error_type=ErrorType.NOT_FOUND,
                    citations=["pycalphad"],
                    duration_ms=duration_ms
                )
            
            # Check elements exist in database
            db_elems = get_db_elements(db)
            if not (A in db_elems and B in db_elems):
                duration_ms = (time.time() - start_time) * 1000
                return error_result(
                    handler="calphad",
                    function="plot_composition_temperature",
                    error=f"Elements '{A}' and '{B}' not found in database. Available: {sorted(db_elems)}",
                    error_type=ErrorType.INVALID_INPUT,
                    citations=["pycalphad"],
                    duration_ms=duration_ms
                )
            
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
                    # Calculate equilibrium at this temperature using shared utility
                    composition_dict = {A: 1-xB, B: xB}
                    eq = compute_equilibrium(db, [A, B], phases, composition_dict, T)
                    
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
            
            # Generate composition label
            comp_label = f"{A}{100-xB*100:.0f}{B}{xB*100:.0f}"
            filename = f"composition_stability_{comp_label}"
            
            # Create plot (interactive HTML with PNG export, or static matplotlib)
            if interactive == "html":
                # Create interactive Plotly plot
                _log.info(f"Creating interactive Plotly plot")
                fig = self._create_interactive_plot(temps, phase_data, A, B, xB)
                
                # Save as HTML
                html_content = fig.to_html(include_plotlyjs='cdn')
                html_url = self._save_html_to_file(html_content, filename)
                
                # Export static PNG (with matplotlib fallback)
                try:
                    png_url = self._save_plotly_figure_as_png(fig, filename)
                except Exception as e:
                    _log.warning(f"Plotly PNG export failed: {e}. Using matplotlib fallback.")
                    fig_mpl, ax, legend = self._create_matplotlib_stackplot(
                        temps, phase_data, comp_label, 
                        figure_size=(figure_width or 10, figure_height or 6)
                    )
                    png_url = self._save_plot_to_file(fig_mpl, filename, 
                                                     extra_artists=[legend] if legend else None)
                    plt.close(fig_mpl)
                
                # Store URLs and generate analysis
                setattr(self, '_last_image_url', png_url)
                setattr(self, '_last_html_url', html_url)
                analysis = self._analyze_composition_temperature(phase_data, xB, temp_range, A, B)
                
                metadata = {
                    "composition": comp_label,
                    "system": f"{A}-{B}",
                    "temperature_range_K": temp_range,
                    "composition_type": comp_type,
                    "analysis": analysis,
                    "interactive": True,
                    "image_info": {
                        "format": "png",
                        "url": png_url,
                        "interactive_html_url": html_url
                    }
                }
                setattr(self, '_last_image_metadata', metadata)
                
                duration_ms = (time.time() - start_time) * 1000
                result = success_result(
                    handler="calphad",
                    function="plot_composition_temperature",
                    data={
                        "message": f"Generated phase stability plot for {comp_label} composition showing phase fractions vs temperature.",
                        "composition": comp_label,
                        "system": f"{A}-{B}",
                        "temperature_range_K": list(temp_range),
                        "composition_type": comp_type,
                        "interactive_html_url": html_url
                    },
                    citations=["pycalphad"],
                    has_image=True,
                    image_url=png_url,
                    has_html=True,
                    html_url=html_url,
                    confidence=Confidence.HIGH,
                    notes=[f"📊 View interactive plot at {html_url} for hover details and zoom"],
                    duration_ms=duration_ms
                )
                return result
            
            else:
                # Create static matplotlib plot
                fig, ax, legend = self._create_matplotlib_stackplot(
                    temps, phase_data, comp_label,
                    figure_size=(figure_width or 8, figure_height or 6)
                )
                
                # Save plot and generate analysis
                plot_url = self._save_plot_to_file(fig, filename, 
                                                   extra_artists=[legend] if legend else None)
                plt.close(fig)
                
                setattr(self, '_last_image_url', plot_url)
                analysis = self._analyze_composition_temperature(phase_data, xB, temp_range, A, B)
                
                metadata = {
                    "composition": comp_label,
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
                    "description": f"Phase stability diagram for {comp_label} composition",
                    "phases": phases,
                    "image_info": {
                        "format": "png",
                        "url": plot_url
                    }
                }
                setattr(self, '_last_image_metadata', metadata)
                
                duration_ms = (time.time() - start_time) * 1000
                return success_result(
                    handler="calphad",
                    function="plot_composition_temperature",
                    data={
                        "message": f"Generated phase stability plot for {comp_label} composition showing phase fractions vs temperature.",
                        "composition": comp_label,
                        "system": f"{A}-{B}",
                        "temperature_range_K": list(temp_range),
                        "composition_type": comp_type,
                        "phases": phases
                    },
                    citations=["pycalphad"],
                    has_image=True,
                    image_url=plot_url,
                    confidence=Confidence.HIGH,
                    duration_ms=duration_ms
                )
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            _log.exception(f"Error generating composition-temperature plot for {composition}")
            return error_result(
                handler="calphad",
                function="plot_composition_temperature",
                error=f"Failed to generate composition-temperature plot: {str(e)}",
                error_type=ErrorType.COMPUTATION_ERROR,
                citations=["pycalphad"],
                duration_ms=duration_ms
            )

    @ai_function(desc="Analyze and interpret the most recently generated phase diagram or composition plot. Provides detailed analysis of visual features, phase boundaries, and thermodynamic insights based on the actual generated plot.")
    async def analyze_last_generated_plot(self) -> str:
        """
        Analyze and interpret the most recently generated phase diagram or composition plot.
        
        This function provides detailed analysis of the visual content and thermodynamic features
        of the last generated plot, allowing the model to "see" and interpret its own output.
        """
        start_time = time.time()
        
        # Check if we have metadata from a recently generated plot
        metadata = getattr(self, '_last_image_metadata', None)
        if not metadata:
            duration_ms = (time.time() - start_time) * 1000
            return error_result(
                handler="calphad",
                function="analyze_last_generated_plot",
                error="No recently generated phase diagram or composition plot available to analyze. Please generate a plot first using plot_binary_phase_diagram() or plot_composition_temperature().",
                error_type=ErrorType.NOT_FOUND,
                citations=["pycalphad"],
                duration_ms=duration_ms
            )
        
        # Extract analysis components
        visual_analysis = metadata.get("visual_analysis", "")
        thermodynamic_analysis = metadata.get("thermodynamic_analysis", "")
        combined_analysis = metadata.get("analysis", "")
        
        # Check if image data is available (may be cleared after display to save memory)
        image_data = getattr(self, '_last_image_data', None)
        duration_ms = (time.time() - start_time) * 1000
        
        if not image_data:
            return success_result(
                handler="calphad",
                function="analyze_last_generated_plot",
                data={
                    "message": combined_analysis,
                    "analysis": combined_analysis,
                    "image_data_available": False
                },
                citations=["pycalphad"],
                notes=["Image data cleared to save memory"],
                confidence=Confidence.HIGH,
                duration_ms=duration_ms
            )
        
        # Return the combined analysis
        return success_result(
            handler="calphad",
            function="analyze_last_generated_plot",
            data={
                "message": combined_analysis,
                "analysis": combined_analysis,
                "visual_analysis": visual_analysis,
                "thermodynamic_analysis": thermodynamic_analysis,
                "image_data_available": True
            },
            citations=["pycalphad"],
            confidence=Confidence.HIGH,
            duration_ms=duration_ms
        )

