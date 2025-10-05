"""
Plotting utilities for CALPHAD phase diagrams.

Contains methods for creating and formatting phase diagrams and composition-temperature plots.
"""

import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional, Tuple
import logging

# Import pycalphad legend utilities
try:
    from pycalphad.plot.utils import phase_legend, phase_colormap
except ImportError:
    # Fallback if pycalphad version doesn't have these
    phase_legend = None
    phase_colormap = None

_log = logging.getLogger(__name__)

class PlottingMixin:
    """Mixin class containing plotting-related methods for CalPhadHandler."""
    
    

    def _add_phase_labels(self, axes, temp_range: Tuple[float, float], phases: List[str], db=None, elements=None, comp_var=None) -> None:
        """Add enhanced phase labels with colored backgrounds to the plot."""
        if not phases:
            return
            
        # Get phase colors from the plot
        phase_colors = self._extract_phase_colors_from_plot(axes, phases)
        
        # Position labels near bottom in axes-fraction coordinates
        label_y = 0.05  # 5% from bottom
        
        # Add labels for each phase
        for i, phase in enumerate(phases):
            # Get color for this phase
            color = phase_colors.get(phase, '#888888')
            
            # Calculate x position (spread across composition range)
            x_pos = 0.1 + (i * 0.8) / max(1, len(phases) - 1)
            x_pos = min(0.9, max(0.1, x_pos))  # Clamp to [0.1, 0.9]
            
            # Create label text
            label_text = phase
            
            # Add special annotations for key phases
            if phase in ["FCC_A1", "HCP_A3", "BCC_A2"]:
                label_text = f"{phase}\n(crystal)"
            elif "AL" in phase.upper() and "ZN" in phase.upper():
                label_text = f"{phase}\n(intermetallic)"
            
            # Add the label with colored background using axes-fraction coordinates
            axes.text(x_pos, label_y, label_text, 
                     transform=axes.transAxes,
                     fontsize=10, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", 
                              facecolor=color, 
                              alpha=0.7,
                              edgecolor='black',
                              linewidth=0.5),
                     ha='center', va='bottom',
                     zorder=10)
 
    def _save_plot_to_file(self, fig, filename: str, extra_artists=None) -> str:
        """Save matplotlib figure to file and return URL."""
        from pathlib import Path
        import time
        
        # Create directory if it doesn't exist
        plots_dir = Path(__file__).parent.parent.parent.parent.parent / "interactive_plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Generate unique filename with timestamp
        timestamp = int(time.time() * 1000)
        safe_filename = filename.replace(" ", "_").replace("/", "_")
        file_path = plots_dir / f"{safe_filename}_{timestamp}.png"
        
        # Use tight_layout for automatic spacing, then save with bbox_inches='tight' to crop to content
        fig.tight_layout()
        fig.savefig(file_path, format='png', dpi=150, 
                   bbox_inches='tight',
                   bbox_extra_artists=(extra_artists or []))
        print(f"Plotting: Saved plot to {file_path}", flush=True)
        
        # Return URL
        url = f"http://localhost:8000/static/plots/{file_path.name}"
        print(f"Plotting: Plot URL: {url}", flush=True)
        return url

    def _plotly_comp_temp(self, temps, phase_data, labels, colors, special_Ts, title, subtitle) -> go.Figure:
        """Create a Plotly figure for composition-temperature plot."""
        fig = go.Figure()
        
        # Add phase regions
        for i, (phase, data) in enumerate(phase_data.items()):
            if len(data) == 0:
                continue
                
            # Convert to numpy arrays for easier handling
            temps_array = np.array(data['temperature'])
            comps_array = np.array(data['composition'])
            
            # Create filled area
            fig.add_trace(go.Scatter(
                x=comps_array,
                y=temps_array,
                mode='lines',
                fill='tonexty' if i > 0 else 'tozeroy',
                name=phase,
                line=dict(color=colors.get(phase, '#888888'), width=2),
                fillcolor=colors.get(phase, '#888888'),
                opacity=0.7,
                hovertemplate=f'<b>{phase}</b><br>' +
                             'Temperature: %{y:.1f} K<br>' +
                             'Composition: %{x:.1f}%<br>' +
                             '<extra></extra>'
            ))
        
        # Add special temperature markers
        for temp_info in special_Ts:
            fig.add_hline(
                y=temp_info['temperature'],
                line_dash="dash",
                line_color="red",
                opacity=0.8,
                annotation_text=temp_info['label'],
                annotation_position="top right"
            )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b><br><sub>{subtitle}</sub>",
                x=0.5,
                font=dict(size=16)
            ),
            xaxis_title="Composition (at% B)",
            yaxis_title="Temperature (K)",
            hovermode='closest',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(t=100, b=60, l=60, r=60),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
