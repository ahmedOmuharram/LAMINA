"""
Plotting utilities for CALPHAD phase diagrams.

Contains methods for creating and formatting phase diagrams and composition-temperature plots.
"""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
import numpy as np
import plotly.graph_objects as go
import logging
from typing import Dict, List, Tuple, Optional
from .database_utils import map_phase_name

_log = logging.getLogger(__name__)

class PlottingMixin:
    """Mixin class containing plotting-related methods for CalPhadHandler."""
    
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
        _log.info(f"Saved plot to {file_path}")
        
        # Return URL
        url = f"http://localhost:8000/static/plots/{file_path.name}"
        _log.info(f"Plot URL: {url}")
        return url
    
    def _save_html_to_file(self, html_content: str, filename: str) -> str:
        """Save HTML content to file and return URL."""
        from pathlib import Path
        import time
        
        # Create directory if it doesn't exist
        plots_dir = Path(__file__).parent.parent.parent.parent.parent / "interactive_plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Generate unique filename with timestamp
        timestamp = int(time.time() * 1000)
        safe_filename = filename.replace(" ", "_").replace("/", "_")
        file_path = plots_dir / f"{safe_filename}_{timestamp}.html"
        
        # Write HTML content to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        _log.info(f"Saved HTML plot to {file_path}")
        
        # Return URL
        url = f"http://localhost:8000/static/plots/{file_path.name}"
        _log.info(f"HTML plot URL: {url}")
        return url
    
    def _save_plotly_figure_as_png(self, fig, filename: str) -> str:
        """Export Plotly figure to PNG and save it."""
        from pathlib import Path
        import time
        
        plots_dir = Path(__file__).parent.parent.parent.parent.parent / "interactive_plots"
        plots_dir.mkdir(exist_ok=True)
        
        timestamp = int(time.time() * 1000)
        safe_filename = filename.replace(" ", "_").replace("/", "_")
        png_path = plots_dir / f"{safe_filename}_{timestamp}.png"
        
        # Convert Plotly figure to static image
        png_bytes = fig.to_image(format="png", width=1200, height=700, scale=2)
        
        with open(png_path, 'wb') as f:
            f.write(png_bytes)
        
        url = f"http://localhost:8000/static/plots/{png_path.name}"
        _log.info(f"Static PNG saved, URL: {url}")
        return url
    
    def _create_matplotlib_stackplot(
        self,
        temps: np.ndarray,
        phase_data: Dict[str, List[float]],
        composition_label: str,
        figure_size: Tuple[float, float] = (10, 6)
    ) -> Tuple[plt.Figure, plt.Axes, Optional[Legend]]:
        """
        Create matplotlib stacked area plot for phase fractions vs temperature.
        
        Args:
            temps: Temperature array
            phase_data: Dictionary of phase_name: [fractions]
            composition_label: Label for composition (e.g., "Al20Zn80")
            figure_size: Figure dimensions (width, height)
            
        Returns:
            Tuple of (figure, axes, legend)
        """
        fig, ax = plt.subplots(figsize=figure_size)
        
        # Prepare data for stackplot
        phase_names = []
        phase_arrays = []
        colors = plt.cm.tab10(np.linspace(0, 1, len(phase_data)))
        
        for phase, fractions in phase_data.items():
            if max(fractions) > 0.01:  # Only plot significant phases
                readable_phase = map_phase_name(phase)
                phase_names.append(readable_phase)
                phase_arrays.append(fractions)
        
        legend = None
        if phase_arrays:
            # Create stacked area plot
            ax.stackplot(temps, *phase_arrays, labels=phase_names, 
                       colors=colors[:len(phase_arrays)], alpha=0.8,
                       edgecolor='white', linewidth=0.5)
            
            # Add legend
            legend = ax.legend(title="Phases", loc="best", frameon=True,
                             fancybox=True, shadow=True, framealpha=0.9)
        
        ax.set_xlabel("Temperature (K)", fontsize=12)
        ax.set_ylabel("Phase Fraction", fontsize=12)
        ax.set_title(f"Phase Stability: {composition_label}", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, zorder=0)
        ax.set_ylim(0, 1)
        ax.set_xlim(temps[0], temps[-1])
        
        return fig, ax, legend
    
    def _create_interactive_plot(
        self,
        temps: np.ndarray,
        phase_data: Dict[str, List[float]],
        A: str,
        B: str,
        xB: float
    ) -> go.Figure:
        """
        Create interactive Plotly plot for composition-temperature data with filled regions.
        
        Args:
            temps: Temperature array
            phase_data: Dictionary of phase_name: [fractions]
            A: First element
            B: Second element
            xB: Mole fraction of B
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
        
        temps_array = np.array(temps)
        _log.info(f"Creating Plotly figure with temps range: {temps_array[0]:.1f}-{temps_array[-1]:.1f} K")
        
        # Color palette
        colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        
        # Add traces for each phase with stacked area
        traces_added = 0
        for idx, (phase, fractions) in enumerate(phase_data.items()):
            fractions_array = np.array(fractions)
            max_frac = np.max(fractions_array) if len(fractions_array) > 0 else 0
            
            if max_frac > 0.01:
                color = colors[traces_added % len(colors)]
                readable_phase = map_phase_name(phase)
                
                fig.add_trace(go.Scatter(
                    x=temps_array,
                    y=fractions_array,
                    mode='lines',
                    name=readable_phase,
                    line=dict(width=0.5, color=color),
                    fillcolor=color,
                    fill='tonexty' if traces_added > 0 else 'tozeroy',
                    stackgroup='one',
                    groupnorm='',
                    hovertemplate=f'<b>{readable_phase}</b><br>T: %{{x:.1f}} K<br>Fraction: %{{y:.3f}}<extra></extra>'
                ))
                traces_added += 1
                _log.info(f"  Added trace for {phase} -> {readable_phase} (max fraction: {max_frac:.3f})")
        
        if traces_added == 0:
            _log.warning("No phases with significant fractions to plot!")
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"Phase Stability: {A}{round((1-xB)*100)}{B}{round(xB*100)}",
                x=0.5,
                xanchor='center',
                font=dict(size=18)
            ),
            xaxis=dict(
                title="Temperature (K)",
                range=[temps_array[0], temps_array[-1]],
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray'
            ),
            yaxis=dict(
                title="Phase Fraction",
                range=[0, 1],
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray'
            ),
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='black',
                borderwidth=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=600,
            width=900
        )
        
        return fig