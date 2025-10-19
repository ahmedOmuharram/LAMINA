"""
CALPHAD phase diagram generation using pycalphad.

Supports binary phase diagram calculation and plotting for various chemical systems.
"""
from typing import List, Optional, Tuple
import logging
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import pycalphad.variables as v
from ...base import BaseHandler

# Import from utility modules
from .database_utils import is_excluded_phase, upper_symbol, compose_alias_map, pick_tdb_path, get_db_elements, map_phase_name
from .consts import weight_to_mole_fraction
from .plotting import PlottingMixin
from .analysis import AnalysisMixin
from .ai_functions import AIFunctionsMixin

_log = logging.getLogger(__name__)

class CalPhadHandler(PlottingMixin, AnalysisMixin, AIFunctionsMixin, BaseHandler):
    """
    CALPHAD thermodynamic calculation handler.
    
    Provides phase diagram generation and thermodynamic property calculation
    using pycalphad and thermodynamic databases.
    """
    
    def __init__(self):
        self.tdb_dir = Path(__file__).parent.parent.parent.parent.parent / "tdbs"
        if not self.tdb_dir.exists():
            _log.warning(f"TDB directory not found: {self.tdb_dir}")
        self._last_image_metadata = None
        self._last_image_data = None
        self._last_key_points = []

    def _phase_constituent_elements(self, db, phase: str) -> set[str]:
        """Get the constituent elements of a phase from the database."""
        try:
            phase_obj = db.phases.get(phase)
            if phase_obj is None:
                return set()
            
            # Extract constituent elements from phase definition
            constituents = set()
            if hasattr(phase_obj, 'constituents'):
                for constituent in phase_obj.constituents:
                    if hasattr(constituent, 'elements'):
                        constituents.update(constituent.elements)
            
            return constituents
        except Exception as e:
            _log.warning(f"Could not extract elements for phase {phase}: {e}")
            return set()

    def _phase_elements(self, db, phase: str) -> set[str]:
        """Extract elements from phase constituents."""
        try:
            ph = db.phases.get(phase)
            if ph is None or not hasattr(ph, "constituents"):
                return set()
            elems = set()
            for subl in ph.constituents:
                # subl is a sequence of species; pick bare elements (strip endmembers like 'AL' vs 'AL+')
                elems.update(str(x).upper() for x in subl)
            return elems
        except Exception:
            return set()

    def _filter_phases_for_system(self, db, pair: tuple[str, str], include_metastable: bool=False) -> list[str]:
        """Filter phases to only include those relevant for the given element pair."""
        A, B = pair
        all_names = list(db.phases.keys())

        # 1) exclude helpers/metastables unless explicitly included
        candidates = []
        for name in all_names:
            if not include_metastable and is_excluded_phase(name):
                continue
            candidates.append(name)

        # 2) keep phases that are plausibly relevant for A–B
        kept = []
        for name in candidates:
            if name == "LIQUID":
                kept.append(name); continue
            elems = self._phase_elements(db, name)
            # needs both A and B, OR a terminal structure (element + VA)
            has_both = (A in elems and B in elems)
            terminal_A = (A in elems and "VA" in elems and B not in elems)
            terminal_B = (B in elems and "VA" in elems and A not in elems)
            if has_both or terminal_A or terminal_B:
                kept.append(name)

        # 3) optional: quick activation pass to prune dead phases
        # (coarse grid → keep only phases that ever appear)
        try:
            from pycalphad import equilibrium
            elements = [A, B, "VA"]
            cond = {
                v.X(B): np.linspace(0, 1, 5),
                v.T: np.linspace(300, 1600, 25),  # coarse and cheap
                v.P: 101325, v.N: 1
            }
            eq = equilibrium(db, elements, kept, cond)
            active = {str(p) for p in np.unique(eq["Phase"].values) if p is not None}
            kept = [p for p in kept if p in active or p == "LIQUID"]
        except Exception:
            pass  # fall back to heuristic list if pre-scan fails

        # Nice ordering: LIQUID first, then terminals, then intermetallics
        def key(n):
            if n == "LIQUID": return (0, n)
            e = self._phase_elements(db, n)
            if (A in e and "VA" in e) or (B in e and "VA" in e):
                return (1, n)
            return (2, n)

        return sorted(dict.fromkeys(kept), key=key)

    def _split_phase_instances(self, eq, phase_name, element="ZN", decimals=5, tol=2e-4):
        """Split phase instances that may have multiple regions."""
        try:
            # Get all instances of this phase
            phase_data = eq.where(eq["Phase"] == phase_name)
            
            if len(phase_data) == 0:
                return []
            
            # Group by composition (rounded to avoid floating point issues)
            comp_var = f"X({element})"
            if comp_var not in phase_data.coords:
                return [phase_data]
            
            compositions = phase_data[comp_var].values
            rounded_comps = np.round(compositions, decimals)
            
            # Find unique composition regions
            unique_comps = np.unique(rounded_comps)
            regions = []
            
            for comp in unique_comps:
                # Find all points with this composition (within tolerance)
                mask = np.abs(compositions - comp) < tol
                region_data = phase_data.where(mask, drop=True)
                if len(region_data) > 0:
                    regions.append(region_data)
            
            return regions
            
        except Exception as e:
            _log.warning(f"Error splitting phase instances for {phase_name}: {e}")
            return [eq.where(eq["Phase"] == phase_name)]

    def _split_by_region(self, eq, phase_name):
        """Split equilibrium data by phase regions."""
        try:
            phase_data = eq.where(eq["Phase"] == phase_name)
            if len(phase_data) == 0:
                return []
            
            # Simple splitting by temperature ranges
            temps = phase_data.T.values
            if len(temps) <= 1:
                return [phase_data]
            
            # Find temperature gaps (regions where phase is not present)
            temp_diff = np.diff(temps)
            gap_threshold = np.mean(temp_diff) * 2
            
            gap_indices = np.where(temp_diff > gap_threshold)[0]
            
            if len(gap_indices) == 0:
                return [phase_data]
            
            # Split at gap indices
            regions = []
            start_idx = 0
            
            for gap_idx in gap_indices:
                end_idx = gap_idx + 1
                region = phase_data.isel(T=slice(start_idx, end_idx))
                if len(region) > 0:
                    regions.append(region)
                start_idx = end_idx
            
            # Add final region
            if start_idx < len(phase_data):
                final_region = phase_data.isel(T=slice(start_idx, None))
                if len(final_region) > 0:
                    regions.append(final_region)
            
            return regions
            
        except Exception as e:
            _log.warning(f"Error splitting by region for {phase_name}: {e}")
            return [eq.where(eq["Phase"] == phase_name)]

    def _normalize_system(self, system: str, db=None) -> Tuple[str, str]:
        """Normalize system string to (A, B) tuple of element symbols."""
        system = system.strip()
        
        # Handle common separators
        for sep in ['-', '_', ' ']:
            if sep in system:
                parts = system.split(sep)
                if len(parts) == 2:
                    A, B = parts[0].strip(), parts[1].strip()
                    return upper_symbol(A), upper_symbol(B)
        
        # Handle concatenated format like "ALZN" or "AlZn"
        # Normalize to uppercase first
        sysu = system.upper()
        if len(sysu) >= 4:
            # Try to split at common element boundaries
            for i in range(2, len(sysu) - 1):
                A, B = sysu[:i], sysu[i:]
                # Check if both substrings are valid normalized elements
                if upper_symbol(A) == A and upper_symbol(B) == B:
                    return A, B
        
        # Fallback: try to parse as single element
        if len(system) <= 3:
            return upper_symbol(system), "AL"  # Default to Al-based system
        
        raise ValueError(f"Could not parse system: {system}")

    def _get_database_path(self, _system_ignored: str = "", elements: Optional[List[str]] = None) -> Optional[Path]:
        """
        Get the path to the thermodynamic database file.
        
        Args:
            _system_ignored: System string (legacy parameter)
            elements: List of element symbols to help select appropriate database
        
        Returns:
            Path to appropriate .tdb file
        """
        return pick_tdb_path(self.tdb_dir, elements=elements)

    def _normalize_elements(self, elements: List[str], db=None) -> List[str]:
        """Normalize element symbols to database format."""
        if db is None:
            return [upper_symbol(el) for el in elements]
        
        # Use database-specific aliases
        aliases = compose_alias_map(db)
        normalized = []
        for el in elements:
            normalized_el = aliases.get(el.lower(), upper_symbol(el))
            normalized.append(normalized_el)
        
        return normalized

    def _parse_composition(self, system_or_comp: str, composition_type: str = "atomic", db=None) -> Tuple[Tuple[str,str], Optional[float], str]:
        """
        Parse composition string like 'Al20Zn80' or 'Al80Zn20'.
        
        Args:
            system_or_comp: Composition string
            composition_type: 'atomic' for at% or 'weight' for wt%
            db: Optional database for element validation
            
        Returns:
            Tuple of ((element_A, element_B), mole_fraction_B, 'atomic')
            Note: Always returns atomic (mole) fractions, converting from weight if needed
        """
        system_or_comp = system_or_comp.strip()
        
        # Handle single element
        if len(system_or_comp) <= 3 and system_or_comp.isalpha():
            element = upper_symbol(system_or_comp)
            return (element, "AL"), 0.0 if element == "AL" else 1.0, "atomic"
        
        # Parse composition like "Al20Zn80"
        def norm_token(tok):
            tok = tok.strip().upper()
            if tok.isalpha():
                return upper_symbol(tok)
            return tok
        
        # Try to extract numbers and elements
        import re
        pattern = r'([A-Za-z]+)(\d+(?:\.\d+)?)'
        matches = re.findall(pattern, system_or_comp)
        
        if len(matches) >= 2:
            # Parse as composition
            elem1, num1 = matches[0]
            elem2, num2 = matches[1]
            
            elem1 = norm_token(elem1)
            elem2 = norm_token(elem2)
            num1 = float(num1)
            num2 = float(num2)
            
            # Normalize to fractions
            total = num1 + num2
            frac1 = num1 / total
            frac2 = num2 / total
            
            # Convert weight% to mole fractions if needed
            if composition_type.lower() in ('weight', 'wt', 'wt%', 'weight%'):
                comp_dict = weight_to_mole_fraction({elem1: frac1, elem2: frac2})
                x1 = comp_dict[elem1]
                x2 = comp_dict[elem2]
            else:
                x1 = frac1
                x2 = frac2
            
            # Return in order (A, B) where B is the second element, always as mole fractions
            return (elem1, elem2), x2, "atomic"
        
        # Fallback: try to parse as system
        try:
            A, B = self._normalize_system(system_or_comp, db)
            return (A, B), 0.5, "atomic"  # Default to 50-50 atomic
        except:
            raise ValueError(f"Could not parse composition: {system_or_comp}")
    
    def _parse_multicomponent_composition(self, comp_str: str, composition_type: str = "atomic") -> Optional[dict]:
        """
        Parse multicomponent composition string like 'Al30Si55C15' into {AL: 0.30, SI: 0.55, C: 0.15}.
        
        Args:
            comp_str: Composition string with format ElementNumber pairs (e.g., 'Al30Si55C15')
            composition_type: 'atomic' for at% or 'weight' for wt%
            
        Returns:
            Dictionary mapping element symbols to mole fractions (always atomic), or None if parsing fails
        """
        import re
        
        # Pattern to match element symbol followed by number
        pattern = r'([A-Z][a-z]?)(\d+(?:\.\d+)?)'
        matches = re.findall(pattern, comp_str.strip())
        
        if not matches:
            return None
        
        # Extract elements and amounts
        comp_dict = {}
        total = 0.0
        
        for elem, amount_str in matches:
            elem_upper = upper_symbol(elem)
            amount = float(amount_str)
            comp_dict[elem_upper] = amount
            total += amount
        
        if total == 0.0:
            return None
        
        # Normalize to sum to 1.0
        for elem in comp_dict:
            comp_dict[elem] /= total
        
        # Convert weight% to mole fractions if needed
        if composition_type.lower() in ('weight', 'wt', 'wt%', 'weight%'):
            comp_dict = weight_to_mole_fraction(comp_dict)
        
        return comp_dict
    
    def _filter_phases_for_multicomponent(self, db, elements: List[str], include_metastable: bool = False) -> List[str]:
        """
        Filter phases relevant for a multicomponent system (3+ elements).
        
        Args:
            db: pycalphad Database
            elements: List of element symbols
            include_metastable: Whether to include metastable phases
            
        Returns:
            List of relevant phase names
        """
        all_phases = list(db.phases.keys())
        candidates = []
        
        # Filter out excluded phases unless metastable requested
        for phase in all_phases:
            if not include_metastable and is_excluded_phase(phase):
                continue
            candidates.append(phase)
        
        # Keep phases that contain any of our elements
        kept = []
        for phase in candidates:
            if phase == "LIQUID":
                kept.append(phase)
                continue
            
            phase_elems = self._phase_elements(db, phase)
            
            # Include phase if it contains any combination of our elements
            if any(elem in phase_elems for elem in elements):
                kept.append(phase)
        
        # Optional: activation pass to prune dead phases
        # For multicomponent systems, we'll skip this for now as it's expensive
        
        return kept

    def _calculate_equilibrium_at_T(self, db, elements, phases, T, xB, comp_var):
        """Calculate equilibrium at a specific temperature."""
        try:
            from pycalphad import equilibrium
            
            # Set up calculation conditions
            conditions = {
                comp_var: xB,
                v.T: T,
                v.P: 101325,
                v.N: 1
            }
            
            # Calculate equilibrium (this will give us phase fractions)
            result = equilibrium(db, elements, phases, conditions)
            
            return result
            
        except Exception as e:
            _log.warning(f"Error calculating equilibrium at T={T}: {e}")
            # Return empty result
            import xarray as xr
            return xr.Dataset()

    def _create_interactive_plot(self, temps, phase_data, A, B, xB, comp_type, temp_range):
        """Create interactive Plotly plot for composition-temperature data with filled regions."""
        import logging
        _log = logging.getLogger(__name__)
        
        fig = go.Figure()
        
        # Convert temps to numpy array for safety
        temps_array = np.array(temps)
        _log.info(f"Creating Plotly figure with temps range: {temps_array[0]:.1f}-{temps_array[-1]:.1f} K")
        
        # Color palette for phases
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
                
                # Map phase name to readable form (e.g., CSI -> SiC)
                readable_phase = map_phase_name(phase)
                
                fig.add_trace(go.Scatter(
                    x=temps_array,
                    y=fractions_array,
                    mode='lines',
                    name=readable_phase,
                    line=dict(width=0.5, color=color),
                    fillcolor=color,
                    fill='tonexty' if traces_added > 0 else 'tozeroy',
                    stackgroup='one',  # This creates a stacked area chart
                    groupnorm='',  # Don't normalize (we want actual fractions)
                    hovertemplate=f'<b>{readable_phase}</b><br>T: %{{x:.1f}} K<br>Fraction: %{{y:.3f}}<extra></extra>'
                ))
                traces_added += 1
                _log.info(f"  Added trace for {phase} -> {readable_phase} (max fraction: {max_frac:.3f})")
        
        if traces_added == 0:
            _log.warning("No phases with significant fractions to plot!")
        
        # Update layout with explicit axis ranges
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