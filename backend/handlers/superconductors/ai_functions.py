"""
AI Functions for Superconductor Analysis

This module contains all AI-accessible functions for analyzing superconducting
materials, particularly cuprates and their structural properties.
"""
import logging
import time
from typing import Any, Dict, Optional, Annotated

from kani import ai_function, AIParam
from ..shared import success_result, error_result, ErrorType, Confidence

from . import utils

_log = logging.getLogger(__name__)


class SuperconductorAIFunctionsMixin:
    """Mixin class containing AI function methods for Superconductor handlers."""
    
    @ai_function(
        desc=(
            "Analyze how c-axis spacing affects Cu-O octahedral stability in cuprate superconductors "
            "(e.g., La2CuO4, YBCO). FAMILY-AWARE: returns different analysis for 214-type (La2CuO4) vs "
            "123-type (YBCO) vs T′/infinite-layer. Use for questions about structural effects on octahedral "
            "coordination, Jahn-Teller distortions, or claims about c-axis expansion stabilizing/destabilizing octahedra. "
            "Default mode answers the general trend question (does increasing c stabilize or destabilize?)."
        ),
        auto_truncate=128000,
    )
    async def analyze_cuprate_octahedral_stability(
        self,
        material_formula: Annotated[str, AIParam(desc="Cuprate formula, e.g., 'La2CuO4', 'YBa2Cu3O7', 'La1.85Sr0.15CuO4'.")],
        c_axis_spacing: Annotated[Optional[float], AIParam(desc="c-axis lattice parameter in Å (optional).")] = None,
        scenario: Annotated[str, AIParam(desc="Use 'trend_increase' (default), 'trend_decrease', or 'observed'.")] = "trend_increase",
        trend_probe: Annotated[float, AIParam(desc="Hypothetical ±fractional change used in trend mode (default 0.01 = 1%).")] = 0.01,
        data_source: Annotated[str, AIParam(desc="Data source: 'MP' | 'experiment' | 'high_precision_xrd' | 'unknown'. Affects classification threshold.")] = "unknown",
    ) -> Dict[str, Any]:
        """
        Analyze cuprate octahedral stability as a function of c-axis spacing.
        
        Returns:
        - Whether increasing c-axis stabilizes or destabilizes octahedral coordination
        - Mechanism (apical Cu-O bond length changes)
        - Analysis of stability effects
        - Structural details (typical bond distances, coordination)
        """
        start_time = time.time()
        
        try:
            # Get structure from Materials Project if available
            structure_data = None
            if hasattr(self, "mpr") and self.mpr:
                try:
                    docs = self.mpr.materials.summary.search(
                        formula=material_formula,
                        fields=["material_id", "formula_pretty", "structure", "symmetry"]
                    )
                    if docs:
                        doc = docs[0] if isinstance(docs, list) else docs
                        structure = doc.structure if hasattr(doc, "structure") else None
                        if structure:
                            c_from_mp_raw = float(structure.lattice.c)
                            
                            # Use pymatgen standardization to get conventional cell
                            try:
                                from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
                                sga = SpacegroupAnalyzer(structure)
                                conv_structure = sga.get_conventional_standard_structure()
                                c_from_mp = float(conv_structure.lattice.c)
                                cell_type = "conventionalized via spglib"
                                conventionalized = True
                            except Exception as e:
                                # Fallback: heuristic for La2CuO4-like if standardization fails
                                _log.debug(f"Standardization failed, using heuristic: {e}")
                                if "la2cuo4" in material_formula.lower().replace(" ", "") and c_from_mp_raw < 10.0:
                                    c_from_mp = c_from_mp_raw * 2.0
                                    cell_type = "primitive (doubled to conventional, heuristic)"
                                    conventionalized = False
                                else:
                                    c_from_mp = c_from_mp_raw
                                    cell_type = "as returned by MP"
                                    conventionalized = False
                            
                            if c_axis_spacing is None:
                                c_axis_spacing = c_from_mp
                            structure_data = {
                                "material_id": str(doc.material_id) if hasattr(doc, "material_id") else None,
                                "c_axis_mp_raw": c_from_mp_raw,
                                "c_axis_mp_conventional": c_from_mp,
                                "cell_type": cell_type,
                                "conventionalized": conventionalized,
                                "note": "MP data for reference; trend baseline uses typical_c from literature",
                            }
                except Exception as e:
                    _log.warning(f"Could not fetch structure from MP: {e}")
            
            # Override data_source if MP was fetched
            if structure_data and data_source == "unknown":
                data_source = "MP"
            
            util_result = utils.analyze_cuprate_octahedral_stability(
                material_formula, c_axis_spacing, structure_data,
                scenario=scenario, trend_probe=trend_probe,
                data_source=data_source
            )
            
            if structure_data:
                util_result["materials_project_data"] = structure_data
            
            # Filter out success, citations, notes, caveats (handled separately)
            data = {k: v for k, v in util_result.items() 
                    if k not in {"success", "citations", "notes", "caveats"}}
            
            citations = util_result.get("citations", [
                "Avella/Guarino et al., Phys. Rev. B 105, 014512 (2022): Oxygen reduction decreases c via removal of apical O.",
                "Yamamoto et al., Physica C (2010): T′-La2CuO4 has c ≈ 12.55 Å (no apical O), smaller than T-LCO.",
                "Ueda et al., Physica C (2010): T′-La2CuO4+δ c ≈ 12.568 Å (consistent with apical-O removal)."
            ])
            
            duration_ms = (time.time() - start_time) * 1000
            
            result = success_result(
                handler="superconductors",
                function="analyze_cuprate_octahedral_stability",
                data=data,
                citations=citations,
                confidence=Confidence.MEDIUM,
                notes=["Analysis based on cuprate crystal chemistry and structural trends"],
                caveats=["Simplified model based on apical Cu-O bond length changes", "Does not include electronic structure effects"],
                duration_ms=duration_ms
            )
            
            return result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            _log.error(f"Error in cuprate analysis for {material_formula}: {e}", exc_info=True)
            return error_result(
                handler="superconductors",
                function="analyze_cuprate_octahedral_stability",
                error=str(e),
                error_type=ErrorType.COMPUTATION_ERROR,
                duration_ms=duration_ms
            )

