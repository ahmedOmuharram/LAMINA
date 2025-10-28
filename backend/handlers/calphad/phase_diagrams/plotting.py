"""
Plotting utilities for CALPHAD phase diagrams.

Contains methods for creating and formatting phase diagrams and composition-temperature plots.
"""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np
import plotly.graph_objects as go
import logging

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