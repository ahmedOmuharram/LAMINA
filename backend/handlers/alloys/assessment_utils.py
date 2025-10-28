"""
Mechanical strengthening and embrittlement assessment utilities.

This module provides functions to assess the likelihood of precipitation strengthening
and embrittlement risk in alloys based on microstructure and phase properties.
"""
from __future__ import annotations
import logging
from typing import Dict, Any

_log = logging.getLogger(__name__)


def assess_mechanical_effects(
    matrix_desc: Dict[str, Any],
    sec_descs: Dict[str, Dict[str, Any]],
    microstructure: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Assess strengthening and embrittlement based on microstructure and mechanical properties.
    
    This function implements heuristics for precipitation strengthening and embrittlement:
    - Precipitation strengthening: 5-30% hard phase in ductile matrix (optimal range)
    - Embrittlement: Large brittle secondary phases or brittle matrix
    
    Args:
        matrix_desc: Mechanical descriptors for matrix phase from mechanical_utils
        sec_descs: Dict mapping phase name to mechanical descriptors for secondary phases
        microstructure: Equilibrium microstructure data from equilibrium_utils
        
    Returns:
        {
            "strengthening_likelihood": "high"/"moderate"/"low"/"mixed",
            "embrittlement_risk": "high"/"moderate"/"low",
            "explanations": {
                "matrix": {...},
                "secondary": {...},
                "rationale": {
                    "strength": str,
                    "embrittlement": str
                }
            }
        }
    """
    try:
        matrix_name = microstructure["matrix_phase"]
        matrix_frac = microstructure["phase_fractions"].get(matrix_name, 0.0)
        
        total_secondary_frac = 1.0 - matrix_frac
        
        # Check if we have hard intermetallic secondaries
        hard_secondary_present = any(
            desc.get("is_intermetallic", False) 
            for desc in sec_descs.values()
        )
        
        # Strengthening assessment
        # Classic precipitation strengthening: 5-30% hard phase in ductile matrix
        if hard_secondary_present and 0.05 <= total_secondary_frac <= 0.30:
            if not matrix_desc.get("brittle_flag", True):
                strengthening = "high"
            else:
                strengthening = "moderate"
        elif hard_secondary_present and 0.01 <= total_secondary_frac < 0.05:
            strengthening = "moderate"
        elif hard_secondary_present and total_secondary_frac > 0.30:
            strengthening = "mixed"  # Too much precipitate
        else:
            strengthening = "low"
        
        # Embrittlement assessment
        brittle_secondaries = []
        for phase_info in microstructure["secondary_phases"]:
            phase_name = phase_info["name"]
            phase_frac = phase_info["fraction"]
            desc = sec_descs.get(phase_name, {})
            
            # Large brittle secondary phases are concerning
            if desc.get("brittle_flag") and phase_frac >= 0.15:
                brittle_secondaries.append({
                    "name": phase_name,
                    "fraction": phase_frac,
                    "pugh_ratio": desc.get("pugh_ratio")
                })
        
        # Embrittlement logic
        if matrix_desc.get("brittle_flag"):
            embrittle = "high"
            embrittle_reason = "Matrix phase itself is predicted to be brittle"
        elif brittle_secondaries:
            embrittle = "high"
            embrittle_reason = f"Large fraction of brittle secondary phases: {', '.join(b['name'] for b in brittle_secondaries)}"
        elif hard_secondary_present and total_secondary_frac > 0.40:
            embrittle = "moderate"
            embrittle_reason = "Very high secondary phase fraction may reduce ductility"
        else:
            embrittle = "low"
            embrittle_reason = "Matrix dominates and is ductile; secondary fraction is limited"
        
        # Build explanations
        explanations = {
            "matrix": {
                "phase": matrix_name,
                "fraction": round(matrix_frac, 3),
                "brittle_flag": matrix_desc.get("brittle_flag"),
                "pugh_ratio": matrix_desc.get("pugh_ratio"),
                "type": "BCC" if matrix_desc.get("is_bcc_like") else "FCC" if matrix_desc.get("is_fcc_like") else "other",
                "source": matrix_desc.get("source")
            },
            "secondary": {
                "total_fraction": round(total_secondary_frac, 3),
                "hard_intermetallic_present": hard_secondary_present,
                "brittle_secondaries": brittle_secondaries,
                "phases": [
                    {
                        "name": p["name"],
                        "fraction": round(p["fraction"], 3),
                        "is_intermetallic": sec_descs.get(p["name"], {}).get("is_intermetallic"),
                        "brittle_flag": sec_descs.get(p["name"], {}).get("brittle_flag"),
                        "pugh_ratio": sec_descs.get(p["name"], {}).get("pugh_ratio"),
                        "source": sec_descs.get(p["name"], {}).get("source")
                    }
                    for p in microstructure["secondary_phases"]
                ]
            },
            "rationale": {
                "strength": get_strengthening_rationale(
                    strengthening, hard_secondary_present, total_secondary_frac, matrix_desc
                ),
                "embrittlement": embrittle_reason
            }
        }
        
        return {
            "strengthening_likelihood": strengthening,
            "embrittlement_risk": embrittle,
            "explanations": explanations
        }
        
    except Exception as e:
        _log.error(f"Error assessing mechanical effect: {e}", exc_info=True)
        return {
            "strengthening_likelihood": "unknown",
            "embrittlement_risk": "unknown",
            "explanations": {"error": str(e)}
        }


def get_strengthening_rationale(
    level: str, 
    has_hard_phase: bool, 
    sec_frac: float,
    matrix_desc: Dict[str, Any]
) -> str:
    """
    Generate human-readable strengthening rationale.
    
    Args:
        level: Strengthening likelihood level ("high", "moderate", "mixed", "low")
        has_hard_phase: Whether hard intermetallic phases are present
        sec_frac: Total secondary phase fraction
        matrix_desc: Matrix phase mechanical descriptors
        
    Returns:
        Human-readable explanation string
    """
    if level == "high":
        return (
            f"Secondary hard intermetallic precipitates ({sec_frac*100:.1f}% volume fraction) "
            "in a ductile matrix are expected to impede dislocation motion via "
            "Orowan strengthening, increasing yield strength significantly."
        )
    elif level == "moderate":
        if sec_frac < 0.05:
            return (
                f"Small volume fraction ({sec_frac*100:.1f}%) of hard phase present. "
                "Some strengthening expected but limited by low precipitate density."
            )
        else:
            return (
                "Moderate strengthening expected, though matrix brittleness may limit effectiveness."
            )
    elif level == "mixed":
        return (
            f"Very high secondary phase fraction ({sec_frac*100:.1f}%) exceeds typical "
            "precipitation strengthening regime. The alloy is essentially a multi-phase "
            "material rather than a precipitate-strengthened alloy."
        )
    else:
        return (
            "No significant hard secondary phase fraction detected. "
            "Limited precipitation strengthening expected. Strength primarily from "
            "solid solution strengthening (if present) or base matrix properties."
        )

