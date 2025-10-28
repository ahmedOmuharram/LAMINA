"""
Analysis utilities for CALPHAD phase diagrams.

Contains methods for analyzing phase diagrams and composition-temperature data.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging

_log = logging.getLogger(__name__)

class AnalysisMixin:
    """Robust, system-agnostic analysis utilities for CALPHAD phase diagrams."""

    # ---------------------------
    # Small internal helpers
    # ---------------------------
    def _safe_coord_name(self, eq, B: Optional[str] = None) -> Optional[str]:
        """
        Return the composition coord name used by pycalphad.
        Prefers exact matches for element B (e.g., 'X_ZN' or 'X(ZN)'),
        then falls back to the first coord that looks like an X-variable.
        """
        try:
            coords = [str(c) for c in eq.coords]
            # Prefer exact B matches
            if B:
                cand = [f"X_{B}", f"X_{B.upper()}", f"X({B})", f"X({B.upper()})"]
                for name in cand:
                    if name in eq.coords:
                        return name
            # Generic X-variables
            for c in coords:
                if c.startswith("X_") or (c.startswith("X(") and c.endswith(")")):
                    return c
        except Exception:
            pass
        return None

    def _reduce_phase_fraction(self, eq, phase_name: str):
        """
        Return phase fraction for `phase_name`, summed over vertex if needed.
        Handles shape/dim mismatches more forgivingly.
        """
        try:
            da_phase = eq["Phase"]
            da_np = eq["NP"]
            mask = (da_phase == phase_name)
            arr = da_np.where(mask).fillna(0)
            if "vertex" in arr.dims:
                arr = arr.sum(dim="vertex", skipna=True)
            return arr
        except Exception as e:
            _log.debug(f"_reduce_phase_fraction failed for {phase_name}: {e}")
            return None

    def _dominant_solids(self, eq, exclude=("LIQUID",), top_k=2):
        """
        Return a list of solid phase names sorted by their global presence,
        excluding any in `exclude`.
        """
        try:
            phases = np.unique(eq["Phase"].values).tolist()
        except Exception:
            return []
        solids = [p for p in phases if p and p not in exclude]
        # Score each solid by total fraction across grid (coarse but effective)
        scores = []
        for p in solids:
            arr = self._reduce_phase_fraction(eq, p)
            if arr is None:
                continue
            try:
                score = float(np.nan_to_num(arr.values).sum())
            except Exception:
                score = 0.0
            scores.append((p, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in scores[:top_k]]

    def _coarse_equilibrium_grid(self, db, A: str, B: str, phases: List[str],
                                 temp_range: Tuple[float, float],
                                 nx: int = 51, nT: int = 81):
        """Compute a coarse equilibrium grid for robust, cheap analysis."""
        try:
            from pycalphad import equilibrium
            import pycalphad.variables as v
            elements = [A, B, "VA"]
            T_lo, T_hi = temp_range
            cond = {
                v.X(B): np.linspace(0.0, 1.0, max(5, nx)),
                v.T: np.linspace(T_lo, T_hi, max(10, nT)),
                v.P: 101325, v.N: 1
            }
            eq = equilibrium(db, elements, phases, cond)
            return eq
        except Exception as e:
            _log.warning(f"Coarse equilibrium failed ({A}-{B}): {e}")
            return None

    def _extract_liquidus_solidus(self, eq, B: str, liquid_name="LIQUID",
                                  liq_hi=0.99, liq_lo=0.01):
        """
        Estimate liquidus/solidus vs composition from a coarse equilibrium grid.
        Works with X_ZN or X(ZN) coords and any dim order; tolerates LIQUID# variants.
        """
        if eq is None:
            return None

        comp_name = self._safe_coord_name(eq, B)
        if comp_name is None or "T" not in eq.coords:
            return None

        # Resolve actual liquid phase name present in the dataset
        try:
            phs = {str(p) for p in np.unique(eq["Phase"].values) if p is not None}
        except Exception:
            phs = set()
        if liquid_name not in phs:
            liq_candidates = [p for p in phs if p.upper().startswith("LIQUID")]
            if not liq_candidates:
                return None
            liquid_name = liq_candidates[0]

        liq_frac = self._reduce_phase_fraction(eq, liquid_name)
        if liq_frac is None:
            return None

        # Identify dimension names and align as (comp, T, ...)
        dims = list(liq_frac.dims)
        t_dim = "T" if "T" in dims else None
        if t_dim is None:
            return None
        comp_dim = comp_name if comp_name in dims else next(
            (d for d in dims if str(d).startswith("X_") or (str(d).startswith("X(") and str(d).endswith(")"))),
            None
        )
        if comp_dim is None:
            return None

        # Reorder for consistent indexing
        try:
            liq_da = liq_frac.transpose(comp_dim, t_dim, ...)
        except Exception:
            liq_da = liq_frac  # best effort

        # Coordinates
        if comp_dim in eq.coords:
            xs = eq.coords[comp_dim].values
        else:
            xs = np.arange(liq_da.sizes.get(comp_dim, 0), dtype=float)

        Ts = eq.coords[t_dim].values
        # Make Ts decreasing for "cooling" interpretation and align data
        try:
            if Ts[0] < Ts[-1]:
                liq_da = liq_da.sortby(t_dim, ascending=False)
                Ts = liq_da.coords[t_dim].values
        except Exception:
            pass

        liq_vals = np.nan_to_num(liq_da.values)
        # If data came in (T, comp, …), flip axes
        if liq_vals.shape[0] == len(Ts) and liq_vals.shape[1] == len(xs):
            liq_vals = np.swapaxes(liq_vals, 0, 1)

        T_liq = np.full_like(xs, np.nan, dtype=float)
        T_sol = np.full_like(xs, np.nan, dtype=float)

        for i in range(len(xs)):
            col = liq_vals[i, :]
            # Liquidus: first T (cooling) where liquid < liq_hi
            idx_liq = np.where(col < liq_hi)[0]
            if idx_liq.size > 0:
                T_liq[i] = Ts[idx_liq[0]]
            # Solidus: first T where liquid < liq_lo
            idx_sol = np.where(col < liq_lo)[0]
            if idx_sol.size > 0:
                T_sol[i] = Ts[idx_sol[0]]

        return {"x": xs, "T_liquidus": T_liq, "T_solidus": T_sol}

    def _parabolic_min_vertex(self, x0, y0, x1, y1, x2, y2):
        """Parabolic (quadratic) fit through 3 points → vertex (x*, y*).
        Returns (xv, yv) or (None, None) if fit is degenerate or not convex."""
        try:
            coef = np.polyfit([x0, x1, x2], [y0, y1, y2], 2)  # a x^2 + b x + c
            a, b, c = coef
            if not np.isfinite(a) or abs(a) < 1e-12:
                return None, None
            xv = -b / (2.0 * a)
            yv = a * xv * xv + b * xv + c
            # eutectic is a minimum → require convex parabola
            if a <= 0:
                return None, None
            return float(xv), float(yv)
        except Exception:
            return None, None

    def _dominant_solid_at(self, eq, B, x, T, thr=1e-3):
        """Return dominant solid phase name at (xB≈x, T≈T) or None if no solid dominates."""
        try:
            comp_name = self._safe_coord_name(eq, B)
            if comp_name is None or "T" not in eq.coords:
                return None
            phases = [p for p in np.unique(eq["Phase"].values).tolist() if p and p.upper().startswith("LIQUID") is False]
            T_coord = eq.coords["T"].values
            t_idx = int(np.argmin(np.abs(T_coord - T)))
            prefs = []
            for p in phases:
                arr = self._reduce_phase_fraction(eq, p)
                if arr is None:
                    continue
                try:
                    val = float(arr.sel({comp_name: x, "T": T_coord[t_idx]}, method="nearest").values)
                except Exception:
                    val = 0.0
                prefs.append((p, val))
            prefs.sort(key=lambda z: z[1], reverse=True)
            return prefs[0][0] if prefs and prefs[0][1] > thr else None
        except Exception:
            return None

    def _find_eutectic_points(self, eq, B: str, ls: dict,
                              delta_T: float = 10.0,
                              min_spacing: float = 0.01,
                              eps_drop: float = 0.2):
        """
        Detect eutectics from the liquidus:
          • find interior local minima (derivative sign change or small-drop),
          • refine by 3-point parabola,
          • validate by finding two different solids around xv using adaptive (Δx, ΔT).
        """
        xs = np.asarray(ls["x"], float)
        Tliq = np.asarray(ls["T_liquidus"], float)

        # finite interior
        mask = np.isfinite(Tliq)
        xs_f = xs[mask]; Tliq_f = Tliq[mask]
        
        print(f"DEBUG eutectic: Total points: {len(xs)}, Finite points: {len(xs_f)}")
        if xs_f.size > 0:
            print(f"DEBUG eutectic: x range: {xs_f.min():.3f} to {xs_f.max():.3f}")
            print(f"DEBUG eutectic: T range: {Tliq_f.min():.1f} to {Tliq_f.max():.1f} K")
            # Find the minimum temperature and its location
            min_idx = np.argmin(Tliq_f)
            print(f"DEBUG eutectic: Minimum T = {Tliq_f[min_idx]:.1f} K at x = {xs_f[min_idx]:.3f}")
        
        if xs_f.size < 3:
            print(f"DEBUG eutectic: Not enough finite points ({xs_f.size}) for analysis")
            return []

        # smooth slightly to suppress single-step noise (3-pt moving average)
        Tsm = Tliq_f.copy()
        if len(Tsm) >= 3:
            Tsm[1:-1] = (Tliq_f[:-2] + Tliq_f[1:-1] + Tliq_f[2:]) / 3.0

        # candidate minima by derivative sign change OR small-drop test
        cand_idx = []
        d1 = np.diff(Tsm)
        print(f"DEBUG eutectic: Computing derivatives, eps_drop={eps_drop}")
        for i in range(1, len(xs_f) - 1):
            sign_change = (d1[i-1] < 0) and (d1[i] > 0)
            small_drop  = (Tsm[i] <= Tsm[i-1] - eps_drop) and (Tsm[i] <= Tsm[i+1] - eps_drop)
            if sign_change or small_drop:
                cand_idx.append(i)
                print(f"DEBUG eutectic: Candidate at i={i}, x={xs_f[i]:.3f}, T={Tsm[i]:.1f} K (sign_change={sign_change}, small_drop={small_drop})")
        
        print(f"DEBUG eutectic: Found {len(cand_idx)} candidate minima")
        
        # Fallback: if no candidates found, look for global minimum away from endpoints
        if len(cand_idx) == 0:
            print(f"DEBUG eutectic: No candidates from derivative test, trying global minimum fallback")
            # Exclude points too close to boundaries (first/last 10%)
            n_edge = max(2, int(len(Tsm) * 0.1))
            interior_slice = slice(n_edge, len(Tsm) - n_edge)
            interior_T = Tsm[interior_slice]
            if len(interior_T) > 0:
                min_interior_idx = np.argmin(interior_T) + n_edge
                # Check if it's significantly lower than the endpoints
                T_endpoints_min = min(Tsm[0], Tsm[-1])
                if Tsm[min_interior_idx] < T_endpoints_min - 5.0:  # At least 5K lower
                    cand_idx.append(min_interior_idx)
                    print(f"DEBUG eutectic: Global minimum fallback: i={min_interior_idx}, x={xs_f[min_interior_idx]:.3f}, T={Tsm[min_interior_idx]:.1f} K")
        
        print(f"DEBUG eutectic: Total candidates (after fallback): {len(cand_idx)}")

        eutectics = []
        for i in cand_idx:
            x0, y0 = xs_f[i-1], Tsm[i-1]
            x1, y1 = xs_f[i],   Tsm[i]
            x2, y2 = xs_f[i+1], Tsm[i+1]

            xv, yv = self._parabolic_min_vertex(x0, y0, x1, y1, x2, y2)
            if xv is None or not (min(x0, x2) <= xv <= max(x0, x2)):
                print(f"DEBUG eutectic: Parabolic fit failed/out-of-bounds at i={i}, using grid point")
                xv, yv = float(x1), float(y1)
            else:
                print(f"DEBUG eutectic: Parabolic refinement: x={xv:.3f}, T={yv:.1f} K")

            # Robust validation around xv
            # Use wider spacing to probe on opposite sides of the eutectic
            print(f"DEBUG eutectic: Validating candidate at x={xv:.3f}, T={yv:.1f} K")
            left_phase, right_phase, used_dT, used_dx = self._resolve_two_solids_around(
                eq, B, xv, yv,
                dT_seq=(delta_T, 2*delta_T, 3*delta_T, 5*delta_T),
                dx_seq=(0.03, 0.05, 0.08, 0.10, 0.15, 0.20)
            )
            
            print(f"DEBUG eutectic: Validation result: left={left_phase}, right={right_phase}, dT={used_dT}, dx={used_dx}")

            if left_phase and right_phase and left_phase != right_phase and np.isfinite(yv):
                print(f"DEBUG eutectic: ✓ Accepted eutectic: {yv:.1f} K at x={xv:.3f}")
                eutectics.append({
                    "type": "eutectic",
                    "xB": float(xv),
                    "temperature": float(yv),
                    "composition_pct": float(xv * 100.0),
                    "solids": (left_phase, right_phase),
                    "reaction": f"L → {left_phase} + {right_phase}",
                    "validation": {"dT": used_dT, "dx": used_dx}
                })
            else:
                print(f"DEBUG eutectic: ✗ Rejected candidate (validation failed)")

        # Deduplicate nearby candidates
        eutectics.sort(key=lambda d: d["xB"])
        deduped = []
        for e in eutectics:
            if not deduped or abs(e["xB"] - deduped[-1]["xB"]) > min_spacing/2:
                deduped.append(e)
            else:
                if e["temperature"] < deduped[-1]["temperature"]:
                    deduped[-1] = e
        return deduped

    def _top_solids(self, eq, B, x, T, k=2, thr=1e-4):
        """Return up to k solid phases at (x≈,T≈), ordered by fraction."""
        try:
            comp_name = self._safe_coord_name(eq, B)
            if comp_name is None or "T" not in eq.coords:
                return []
            phases = [p for p in np.unique(eq["Phase"].values).tolist()
                      if p and not str(p).upper().startswith("LIQUID")]
            T_coord = eq.coords["T"].values
            t_idx = int(np.argmin(np.abs(T_coord - T)))
            prefs = []
            for p in phases:
                arr = self._reduce_phase_fraction(eq, p)
                if arr is None:
                    continue
                try:
                    val = float(arr.sel({comp_name: x, "T": T_coord[t_idx]}, method="nearest").values)
                except Exception:
                    val = 0.0
                if val > thr:
                    prefs.append((p, val))
            prefs.sort(key=lambda z: z[1], reverse=True)
            return [p for p, _ in prefs[:k]]
        except Exception:
            return []

    def _resolve_two_solids_around(self, eq, B, xv, Tv,
                                   dT_seq=(5.0, 10.0, 20.0, 30.0),
                                   dx_seq=(0.005, 0.01, 0.02, 0.04, 0.08)):
        """
        Try multiple ΔT and Δx to find two *different* solids on either side of x_eut.
        Returns (left_phase, right_phase, used_dT, used_dx) or (None, None, None, None).
        
        For a eutectic, we need to verify that two distinct solid phases coexist in the
        vicinity of the candidate point. This can happen in two ways:
        1. Different dominant phases on left vs right (e.g., Al-Zn)
        2. Two phases present on both sides (e.g., Al-Si near terminal eutectic)
        """
        for dT in dT_seq:
            T_probe = Tv - dT
            for dx in dx_seq:
                xl = max(0.0, xv - dx)
                xr = min(1.0, xv + dx)
                left_list  = self._top_solids(eq, B, xl, T_probe, k=2)
                right_list = self._top_solids(eq, B, xr, T_probe, k=2)
                print(f"  DEBUG validation: dT={dT:.1f}, dx={dx:.3f} -> xl={xl:.3f}, xr={xr:.3f}, T={T_probe:.1f} K")
                print(f"    left_list={left_list}, right_list={right_list}")
                if not left_list or not right_list:
                    continue
                
                # Collect unique solid phases from both sides
                union = []
                for p in left_list + right_list:
                    if p not in union:
                        union.append(p)
                
                print(f"    union={union}, num_distinct={len(union)}")
                
                # Accept if we have at least 2 distinct solid phases
                if len(union) >= 2:
                    # Return the first two distinct phases found
                    phase1, phase2 = union[0], union[1]
                    print(f"  DEBUG validation: ✓ Success! Two distinct solid phases found: {phase1} + {phase2} with dT={dT}, dx={dx}")
                    return phase1, phase2, dT, dx
                else:
                    print(f"    Only {len(union)} distinct phase(s), need 2")
                    
        print(f"  DEBUG validation: ✗ Failed to find two different solid phases")
        return None, None, None, None

    def _mark_eutectics_on_axes(self, ax, eutectics, B_symbol="B"):
        """Mark eutectic points on a phase diagram axes."""
        for e in eutectics:
            ax.plot(e["xB"], e["temperature"], marker="o", markersize=8, 
                   color='red', markeredgecolor='darkred', markeredgewidth=2, 
                   zorder=10)
            ax.annotate(
                f"Eutectic\n{e['temperature']:.0f} K\n{e['composition_pct']:.1f} at% {B_symbol}",
                xy=(e["xB"], e["temperature"]),
                xytext=(10, -20),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->", lw=1.0, color='red'),
                fontsize=9, ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                         edgecolor='red', alpha=0.8)
            )

    # ---------------------------
    # 1) Visual analysis (safe, plot-only)
    # ---------------------------
    def _analyze_visual_content(self, fig, axes, normalized_system: str,
                                phases: List[str], temp_range: Tuple[float, float]) -> str:
        """Analyze the visible content of a phase diagram / composition plot robustly."""
        try:
            lines = axes.get_lines() or []
            patches = getattr(axes, "patches", []) or []
            collections = getattr(axes, "collections", []) or []

            xlim = axes.get_xlim()
            ylim = axes.get_ylim()
            xlabel = axes.get_xlabel() or "x"
            ylabel = axes.get_ylabel() or "y"
            title = axes.get_title() or ""

            analysis = []
            analysis.append("### Visual Analysis of Generated Plot")
            # Axes interpretation
            if "Temperature" in xlabel and ("Phase Fraction" in ylabel or "fraction" in ylabel.lower()):
                analysis.append(f"- **Type**: Composition-specific stability vs temperature.")
                analysis.append(f"- **Temperature**: {xlim[0]:.0f}–{xlim[1]:.0f} K (x-axis).")
                analysis.append(f"- **Phase fraction range**: {ylim[0]:.2f}–{ylim[1]:.2f}.")
            elif ("Mole Fraction" in xlabel or "X(" in xlabel) and "Temperature" in ylabel:
                analysis.append(f"- **Type**: Binary phase diagram (x = composition, y = temperature).")
                analysis.append(f"- **Composition range**: {xlim[0]:.2f}–{xlim[1]:.2f} (fraction).")
                analysis.append(f"- **Temperature**: {ylim[0]:.0f}–{ylim[1]:.0f} K.")
            else:
                analysis.append(f"- **Axes**: X={xlabel} [{xlim[0]:.2f},{xlim[1]:.2f}] / Y={ylabel} [{ylim[0]:.2f},{ylim[1]:.2f}].")

            if title:
                analysis.append(f"- **Title**: {title}")

            # Plot elements
            elem_bits = []
            if len(lines): elem_bits.append(f"{len(lines)} line(s)")
            if len(patches): elem_bits.append(f"{len(patches)} patch(es)")
            if len(collections): elem_bits.append(f"{len(collections)} collection(s)")
            if elem_bits:
                analysis.append(f"- **Drawn elements**: {', '.join(elem_bits)}")
            else:
                analysis.append("- **Drawn elements**: none detected (empty or rendering failed).")

            # Legend/phases
            legend = axes.get_legend()
            if legend:
                labels = [t.get_text() for t in legend.get_texts()]
                if labels:
                    analysis.append(f"- **Legend**: {len(labels)} entries → {', '.join(labels)}")
            if phases:
                analysis.append(f"- **Phases considered**: {len(phases)} → {', '.join(phases)}")

            # Coverage
            Tmin, Tmax = temp_range
            analysis.append(f"- **System**: {normalized_system}")

            return "\n".join(analysis)
        except Exception as e:
            return f"Error analyzing visual content: {e}"

    # ---------------------------
    # 2) Phase diagram analysis (system-agnostic, uses coarse equilibrium)
    # ---------------------------
    def _analyze_phase_diagram(self, db, normalized_system: str,
                               phases: List[str],
                               temp_range: Tuple[float, float]) -> str:
        """
        Analyze a binary phase diagram generically:
        - Terminal melting points (pure A and pure B)
        - Liquidus/solidus trends
        - Dominant solid phases
        - Invariant candidates (eutectic/peritectic heuristics)
        Stores key points in self._last_key_points.
        """
        try:
            if not phases:
                return "No phases available for analysis."

            # Parse system
            try:
                A, B = [t.strip().upper() for t in normalized_system.split("-")]
            except Exception:
                return f"Unrecognized system string: {normalized_system}"

            # Check if we have cached equilibrium data from visualization (for consistency)
            eq = getattr(self, '_cached_eq_coarse', None)
            if eq is None:
                # Fall back to computing a new coarse grid
                eq = self._coarse_equilibrium_grid(db, A, B, phases, temp_range)
            else:
                print(f"DEBUG: Using cached equilibrium data from visualization for analysis")
                
            if eq is None:
                return f"Could not compute coarse equilibrium for {normalized_system}; analysis unavailable."

            # Liquidus/solidus
            ls = self._extract_liquidus_solidus(eq, B)
            if ls is None:
                return f"Could not extract liquidus/solidus information for {normalized_system}."

            xs = ls["x"]; Tliq = ls["T_liquidus"]; Tsol = ls["T_solidus"]

            key_points = []
            report = [f"## Phase Diagram Analysis: {normalized_system}"]

            # Terminal (pure) melting points (approximate)
            def finite_near(array, idx):
                val = array[idx]
                if np.isfinite(val): return float(val)
                # Try nearest finite
                for k in range(1, 5):
                    i1 = max(0, idx - k); i2 = min(len(array) - 1, idx + k)
                    if np.isfinite(array[i1]): return float(array[i1])
                    if np.isfinite(array[i2]): return float(array[i2])
                return np.nan

            Tm_A = finite_near(Tliq, 0)
            Tm_B = finite_near(Tliq, len(Tliq) - 1)

            if np.isfinite(Tm_A):
                key_points.append({"type": "pure_melting", "element": A, "temperature": Tm_A})
            if np.isfinite(Tm_B):
                key_points.append({"type": "pure_melting", "element": B, "temperature": Tm_B})

            # Dominant solids for context
            dom_solids = self._dominant_solids(eq, exclude=("LIQUID",), top_k=3)
            if dom_solids:
                report.append("### Dominant solid phases (coarse scan):")
                report.append("- " + ", ".join(dom_solids))

            # --- Eutectic points ---
            # Use cached eutectics if available (from visualization pass) for consistency
            eutectics = getattr(self, '_cached_eutectics', None)
            if eutectics is None:
                # Fall back to computing eutectics
                print(f"DEBUG: Computing eutectics for analysis (no cache)")
                eutectics = self._find_eutectic_points(eq, B, ls, delta_T=10.0, min_spacing=0.03, eps_drop=0.1)
            else:
                print(f"DEBUG: Using cached eutectics from visualization for analysis")
                
            for e in eutectics:
                key_points.append(e)  # keep for later labeling/metadata
            
            # Print eutectic detection results
            if eutectics:
                print(f"Found {len(eutectics)} eutectic point(s) in {normalized_system}:")
                for i, e in enumerate(eutectics, 1):
                    print(f"  {i}. {e['temperature']:.0f} K at {e['composition_pct']:.2f} at% {B}: {e['reaction']}")
            else:
                print(f"No eutectic points detected in {normalized_system} system")

            # Peritectic candidate (very simple heuristic):
            # Look for a local maximum in T_liquidus with a single dominant solid just below.
            try:
                with np.errstate(invalid="ignore"):
                    # discrete second derivative sign: max when diff changes from + to -
                    d1 = np.diff(Tliq)
                    d2 = np.diff(np.sign(d1))
                candidates = np.where(d2 == -2)[0]  # rough "peaks"
                for c in candidates:
                    i_peak = c + 1
                    if i_peak <= 0 or i_peak >= len(xs) - 1: 
                        continue
                    T_peak = Tliq[i_peak]
                    if not np.isfinite(T_peak): 
                        continue
                    # probe just below
                    T_probe = float(T_peak) - 5.0
                    # dominant solid at the peak composition
                    import xarray as xr
                    comp_name = self._safe_coord_name(eq, B) or f"X({B})"
                    T_coord = eq.coords["T"].values
                    t_idx = int(np.argmin(np.abs(T_coord - T_probe)))
                    dom = None
                    prefs = []
                    for p in np.unique(eq["Phase"].values).tolist():
                        if p in (None, "LIQUID"): 
                            continue
                        arr = self._reduce_phase_fraction(eq, p)
                        if arr is None:
                            continue
                        try:
                            val = float(arr.sel({comp_name: xs[i_peak], "T": T_coord[t_idx]}, method="nearest").values)
                        except Exception:
                            val = 0.0
                        prefs.append((p, val))
                    prefs.sort(key=lambda z: z[1], reverse=True)
                    dom = prefs[0][0] if prefs and prefs[0][1] > 1e-3 else None
                    if dom:
                        key_points.append({
                            "type": "peritectic_candidate",
                            "temperature": float(T_peak),
                            "xB": float(xs[i_peak]),
                            "composition_pct": float(xs[i_peak] * 100.0),
                            "dominant_solid_below": dom,
                            "reaction_like": f"L + {dom} → (other solid)"
                        })
                        # only a couple to avoid spam
                        if len([k for k in key_points if k["type"] == "peritectic_candidate"]) >= 2:
                            break
            except Exception:
                pass

            # Persist key points for other tools
            self._last_key_points = key_points

            # Build report
            report.append("\n### Key Thermodynamic Points (estimated, coarse grid):")
            if any(k["type"] == "pure_melting" for k in key_points):
                for kp in key_points:
                    if kp["type"] == "pure_melting":
                        el = kp["element"]; Tm = kp["temperature"]
                        report.append(f"- **Pure {el}** melting: {Tm:.0f} K ({Tm - 273.15:.0f} °C)")
            else:
                report.append("- Pure-element melting points could not be resolved in the requested range.")

            # Eutectic points
            if eutectics:
                report.append("\n### Eutectic point(s):")
                for e in eutectics:
                    report.append(f"- {e['temperature']:.0f} K ({e['temperature']-273.15:.0f} °C) "
                                  f"at {e['composition_pct']:.2f} at% {B}: {e['reaction']}")
            # Peritectic candidates
            peris = [k for k in key_points if k["type"] == "peritectic_candidate"]
            for kp in peris:
                T = kp["temperature"]; comp = kp["composition_pct"]; dom = kp["dominant_solid_below"]
                report.append(f"- **Peritectic (candidate)**: ~{T:.0f} K near {comp:.1f} at% {B} (dominant solid below: {dom})")

            # Liquidus/solidus ranges
            valid_liq = np.isfinite(Tliq)
            valid_sol = np.isfinite(Tsol)
            if valid_liq.any():
                report.append("\n### Liquidus/solidus trends:")
                report.append(f"- Liquidus available for {valid_liq.sum()}/{len(Tliq)} compositions; min {np.nanmin(Tliq):.0f} K, max {np.nanmax(Tliq):.0f} K.")
            if valid_sol.any():
                report.append(f"- Solidus available for {valid_sol.sum()}/{len(Tsol)} compositions; min {np.nanmin(Tsol):.0f} K, max {np.nanmax(Tsol):.0f} K.")

            # Practical implications (generic but useful)
            report.append("\n### Processing implications:")
            report.append("- **Casting/solidification**: Use liquidus for superheat setpoints; freezing range (liquidus–solidus) indicates susceptibility to segregation.")
            report.append("- **Heat treatment**: Two-phase fields enable precipitation/ageing; dominant solids suggest matrix structure.")
            report.append("- **Welding**: Wide freezing ranges imply larger mushy zones → hot cracking risk.")

            return "\n".join(report)
        except Exception as e:
            return f"Error generating phase diagram analysis: {e}"

    # ---------------------------
    # 3) Composition–temperature analysis (fixed & robust)
    # ---------------------------
    def _analyze_composition_temperature(self, composition_data: Dict[str, List[float]],
                                         xB: float,
                                         temp_range: Tuple[float, float],
                                         A: str = "A", B: str = "B") -> str:
        """
        Analyze phase fractions vs temperature at a fixed composition.
        - Robust to missing phases and unequal array lengths.
        - Reports stability windows, peak fractions, and (liquidus/solidus) from liquid curve if present.
        """
        try:
            Tmin, Tmax = temp_range
            # Determine series length robustly
            lengths = [len(v) for v in composition_data.values() if isinstance(v, (list, tuple, np.ndarray))]
            if not lengths:
                return "No phase-fraction data available to analyze."
            n = int(np.median(lengths))
            if n < 3:
                return "Insufficient temperature sampling for meaningful analysis."

            # Construct temperature grid (assumes upstream used linspace)
            temps = np.linspace(Tmin, Tmax, n)

            # Normalize/trim each series to length n
            series = {}
            for phase, vals in composition_data.items():
                arr = np.array(vals, dtype=float).ravel()
                if len(arr) == n:
                    series[phase] = arr
                elif len(arr) > n:
                    series[phase] = arr[:n]
                else:
                    # pad with zeros if shorter
                    pad = np.zeros(n - len(arr), dtype=float)
                    series[phase] = np.concatenate([arr, pad])

            # Build analysis
            pct_B = xB * 100.0
            pct_A = 100.0 - pct_B
            out = [f"## Detailed Phase Stability: {A}{pct_A:.0f}{B}{pct_B:.0f}"]

            # Threshold for "present"
            present_thr = 0.01

            # Stability windows for non-liquid phases
            transitions = []
            for phase, y in series.items():
                if phase == "LIQUID":
                    continue
                y = np.nan_to_num(y)
                above = np.where(y > present_thr)[0]
                if above.size == 0:
                    continue
                onset = temps[above[0]]
                offset = temps[above[-1]]
                peak_idx = int(np.argmax(y))
                transitions.append({
                    "phase": phase,
                    "onset": float(onset),
                    "offset": float(offset),
                    "peak": float(temps[peak_idx]),
                    "max_fraction": float(y[peak_idx])
                })

            transitions.sort(key=lambda d: d["onset"])
            if transitions:
                out.append("### Phase stability ranges (≥1% fraction):")
                for t in transitions:
                    out.append(f"- **{t['phase']}**: {t['onset']:.0f}–{t['offset']:.0f} K "
                               f"(span {t['offset']-t['onset']:.0f} K); peak {t['max_fraction']:.1%} at {t['peak']:.0f} K.")
            else:
                out.append("### Phase stability ranges: No solid phases exceed 1% in the scanned window.")

            # Melting/solidification behavior from LIQUID curve if available
            liq = series.get("LIQUID", None)
            if liq is not None and np.any(liq > 0):
                y = np.clip(np.nan_to_num(liq), 0, 1)
                
                # Search from LOW to HIGH temperature (heating direction)
                # Liquidus: temperature where material becomes fully liquid (liquid ≥ 95%)
                # Solidus: temperature where liquid first appears (liquid ≥ 5%)
                
                # Liquidus: FIRST temperature (from low to high) where liquid >= 95%
                hi95 = np.where(y >= 0.95)[0]
                if hi95.size:
                    idx_liq = hi95[0]  # FIRST (lowest temp) point where liquid >= 95%
                    T_liq = float(temps[idx_liq])
                    liq_note = ""
                else:
                    # Never reaches 95% - probably a mixture composition
                    # Find where liquid is maximum
                    hi50 = np.where(y >= 0.5)[0]
                    if hi50.size:
                        T_liq = float(temps[hi50[0]])
                        liq_note = " (mixture; liquid never reaches 95%)"
                    else:
                        T_liq = None
                        liq_note = ""
                
                # Solidus: FIRST temperature (from low to high) where liquid >= 5%
                hi05 = np.where(y >= 0.05)[0]
                if hi05.size:
                    idx_sol = hi05[0]  # FIRST point where liquid appears
                    T_sol = float(temps[idx_sol])
                    sol_note = ""
                    
                    # Sanity check: solidus should be below liquidus
                    if T_liq is not None and T_sol > T_liq:
                        # Something wrong, swap them or use midpoint
                        T_sol = None
                        sol_note = ""
                else:
                    # Liquid is always above 5% in range (unlikely)
                    T_sol = None
                    sol_note = ""

                out.append("\n### Melting/solidification (estimated):")
                if T_liq is not None:
                    out.append(f"- **Liquidus**: {T_liq:.0f} K ({T_liq - 273.15:.0f} °C){liq_note}")
                else:
                    out.append("- **Liquidus**: not reached in this range.")
                if T_sol is not None:
                    out.append(f"- **Solidus**: {T_sol:.0f} K ({T_sol - 273.15:.0f} °C){sol_note}")
                else:
                    out.append("- **Solidus**: not reached in this range.")
                if T_liq is not None and T_sol is not None and T_liq >= T_sol:
                    out.append(f"- **Freezing range**: {T_liq - T_sol:.0f} K")
            else:
                out.append("\n### Melting/solidification: No liquid phase present in this window.")

            # Practical notes
            out.append("\n### Processing implications:")
            if transitions:
                out.append("- Temperatures spanning multi-solid regions are suitable for precipitation/ageing studies.")
            if series.get("LIQUID", None) is not None:
                out.append("- Use liquidus as a minimum casting superheat reference; larger (liquidus–solidus) suggests greater segregation risk.")
            out.append("- Rapid transitions (narrow stability windows) imply fast microstructural changes across small ΔT.")

            return "\n".join(out)
        except Exception as e:
            return f"Error analyzing composition-temperature data: {e}"
