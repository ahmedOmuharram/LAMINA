"""
Phase diagram generation utilities for CALPHAD calculations.

Functions for generating and formatting phase diagrams.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import logging
from typing import Tuple, Optional, List, Dict, Any
from pycalphad import Database, binplot
import pycalphad.variables as v

_log = logging.getLogger(__name__)


def generate_binary_phase_diagram(
    db: Database,
    A: str,
    B: str,
    phases: List[str],
    min_temperature: Optional[float] = None,
    max_temperature: Optional[float] = None,
    composition_step: float = 0.02,
    figure_size: Tuple[float, float] = (9, 6)
) -> Tuple[plt.Figure, plt.Axes, Tuple[float, float]]:
    """
    Generate a binary phase diagram using pycalphad.
    
    Args:
        db: PyCalphad Database instance
        A: First element symbol
        B: Second element symbol (x-axis variable)
        phases: List of phase names to include
        min_temperature: Minimum temperature in K (None for auto)
        max_temperature: Maximum temperature in K (None for auto)
        composition_step: Composition step size (0-1)
        figure_size: Figure size as (width, height) in inches
        
    Returns:
        Tuple of (figure, axes, (T_display_lo, T_display_hi))
    """
    # Create figure
    fig = plt.figure(figsize=figure_size)
    axes = fig.gca()
    
    # Set up elements and composition variable
    elements = [A, B, 'VA']
    comp_el = B  # x variable
    
    # Determine temperature range
    auto_T = (min_temperature is None and max_temperature is None)
    if auto_T:
        # Wide bracket for high-melting systems (e.g., Al–Si)
        T_lo, T_hi = 200.0, 2300.0
    else:
        T_lo = min_temperature or 300.0
        T_hi = max_temperature or 1000.0
    
    # Clamp number of temperature points
    temp_points = 60 if auto_T else max(12, min(60, int((T_hi - T_lo) / 20)))
    
    # Generate phase diagram using binplot
    binplot(
        db, elements, phases,
        {
            v.X(comp_el): (0, 1, composition_step),
            v.T: (T_lo, T_hi, temp_points),
            v.P: 101325,
            v.N: 1
        },
        plot_kwargs={'ax': axes, 'tielines': False},
        eq_kwargs={'linewidth': 2},
        legend=True
    )
    
    # Adjust figure size for legend
    legend = axes.get_legend()
    if legend is not None:
        w, h = fig.get_size_inches()
        fig.set_size_inches(w * 1.2, h)  # Widen ~20%
    
    # Set labels and formatting
    axes.set_xlabel(f"Mole Fraction {comp_el} (atomic basis)")
    axes.set_ylabel("Temperature (K)")
    fig.suptitle(f"{A}-{B} Phase Diagram", x=0.5, fontsize=14, fontweight='bold')
    axes.grid(True, alpha=0.3)
    axes.set_xlim(0, 1)
    
    # Handle y-axis limits
    if not auto_T:
        axes.set_ylim(T_lo, T_hi)
        T_display_lo, T_display_hi = T_lo, T_hi
    else:
        # Auto-scale to fit data
        axes.relim()
        axes.autoscale_view()
        y0, y1 = axes.get_ylim()
        pad = 0.02 * (y1 - y0)
        axes.set_ylim(y0 - pad, y1 + pad)
        T_display_lo, T_display_hi = y0 - pad, y1 + pad
        _log.info(f"Auto-scaled temperature range: {T_display_lo:.0f}-{T_display_hi:.0f} K")
    
    return fig, axes, (T_display_lo, T_display_hi)


def format_success_message(
    A: str,
    B: str,
    phases: List[str],
    T_range: Tuple[float, float],
    key_points: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Format a success message for phase diagram generation.
    
    Args:
        A: First element symbol
        B: Second element symbol
        phases: List of phase names
        T_range: Temperature range as (min, max) in K
        key_points: Optional list of key points (melting, eutectic, etc.)
        
    Returns:
        Formatted success message string
    """
    T_lo, T_hi = T_range
    
    success_parts = [
        f"Successfully generated {A}-{B} phase diagram showing phases: {', '.join(phases)}",
        f"Temperature range: {T_lo:.0f}-{T_hi:.0f} K"
    ]
    
    if key_points:
        # Add melting points
        melting_pts = [kp for kp in key_points if kp.get('type') == 'pure_melting']
        for mp in melting_pts:
            success_parts.append(f"Pure {mp['element']} melting point: {mp['temperature']:.0f} K")
        
        # Add eutectic points
        eutectic_pts = [kp for kp in key_points if kp.get('type') == 'eutectic']
        if eutectic_pts:
            for ep in eutectic_pts:
                success_parts.append(
                    f"Eutectic point: {ep['temperature']:.0f} K at {ep['composition_pct']:.1f} at% {B} ({ep['reaction']})"
                )
        else:
            success_parts.append("No eutectic points detected in this temperature range")
    
    success_msg = ". ".join(success_parts) + "."
    
    # Safeguard: ensure we're not accidentally including large data
    if "data:image/png;base64," in success_msg or len(success_msg) > 1000:
        _log.warning("Success message contains base64 data or is too long. Truncating.")
        return "Successfully generated phase diagram. Image will be displayed separately."
    
    return success_msg


def validate_elements_in_database(db: Database, elements: List[str]) -> Tuple[bool, Optional[str]]:
    """
    Validate that elements exist in database.
    
    Args:
        db: PyCalphad Database instance
        elements: List of element symbols to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    from .database_utils import get_db_elements
    
    db_elems = get_db_elements(db)
    missing = [el for el in elements if el not in db_elems]
    
    if missing:
        return False, f"Elements {missing} not found in database (available: {sorted(db_elems)})"
    
    return True, None

