"""
Physics-based mechanical strengthening and embrittlement assessment utilities.

This module provides research-grade models for:
- Precipitation strengthening (Ashby-Orowan, coherency, modulus mismatch)
- Hall-Petch grain boundary strengthening
- Embrittlement risk assessment (Pugh ratio, Cauchy pressure, topology)
- Yield strength prediction with mechanistic superposition

References:
    - Orowan (1948); Ashby (1970s); Ardell, Metall. Trans. (1985)
    - Nembach, "Particle Strengthening of Metals"
    - Pugh, Phil. Mag. 45 (1954) — G/B ratio for ductility
    - Pettifor — Cauchy pressure indicator
"""
from __future__ import annotations
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List, Optional

_log = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Strength models (Ashby-Orowan, coherency, Hall-Petch, modulus mismatch)
# --------------------------------------------------------------------------------------

@dataclass
class StrengthInputs:
    """Input parameters for strength prediction models."""
    matrix_B_GPa: float
    matrix_G_GPa: float
    matrix_nu: float
    burgers_b_nm: float   # Burgers vector magnitude in nm (e.g., Al fcc ~ 0.286)
    # Particle (precipitate) stats per hard secondary phase
    particles: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Example per phase: {"volume_fraction": 0.05, "radius_nm": 10.0, "shearable": 0, "misfit_strain": 0.01}
    # Grain size (μm) for Hall-Petch
    grain_size_um: Optional[float] = None
    # Base (solution) yield of matrix at T (MPa)
    sigma0_MPa: float = 25.0  # default Al-like value; override per system
    # Taylor factor for polycrystal fcc/bcc
    M_taylor: float = 3.06  # fcc ~3.06, bcc ~2.75


def orowan_delta_tau_MPa(G_GPa: float, b_nm: float, nu: float, f: float, r_nm: float) -> float:
    """Ashby-Orowan strengthening for non-shearable particles.
    
    Δτ ≈ (G b / (2π(1-ν) λ)) ln( r / b )
    Where obstacle spacing on slip plane λ ~ k r / sqrt(f). We use k≈1.62 for random spheres.
    
    Args:
        G_GPa: Shear modulus in GPa
        b_nm: Burgers vector in nm
        nu: Poisson ratio
        f: Volume fraction of particles
        r_nm: Particle radius in nm
    
    Returns:
        Strengthening contribution in MPa
    
    References:
        • Orowan (1948); Ashby (1970s); Ardell, Metall. Trans. (1985) review.
    """
    if f <= 0.0 or r_nm <= 0.0:
        return 0.0
    b = b_nm * 1e-9
    r = r_nm * 1e-9
    G = G_GPa * 1e9
    k = 1.62
    lam = max(1e-12, k * r / math.sqrt(max(1e-12, f)))
    ln_term = max(1.0, r/b)
    d_tau = (G * b) / (2.0*math.pi*(1.0 - nu) * lam) * math.log(ln_term)
    return d_tau/1e6  # Pa→MPa


def coherency_delta_tau_MPa(G_GPa: float, misfit: float, f: float) -> float:
    """Coherency strengthening (shearable precipitates).
    
    Δτ ≈ A · G · |ε|^{3/2} · f^{1/2}; A≈0.7–1.0 for fcc Al systems (fitted constant).
    
    Args:
        G_GPa: Shear modulus in GPa
        misfit: Lattice misfit strain (dimensionless)
        f: Volume fraction of particles
    
    Returns:
        Strengthening contribution in MPa
    
    Source: Nembach, "Particle Strengthening of Metals"; Ardell reviews.
    """
    if f <= 0.0 or abs(misfit) <= 0.0:
        return 0.0
    A = 0.85
    return A * (G_GPa) * (abs(misfit)**1.5) * (math.sqrt(max(0.0, f)))


def hall_petch_delta_sigma_MPa(d_um: Optional[float], k_y_MPa_sqrtm: float = 0.2) -> float:
    """Hall-Petch strengthening: Δσ = k_y / sqrt(d).
    
    Args:
        d_um: Grain size in micrometers
        k_y_MPa_sqrtm: Hall-Petch coefficient (typical: 0.2 for Al, 0.6-1.0 for steels)
    
    Returns:
        Strengthening contribution in MPa
    
    d in meters; typical k_y for high-purity Al ~0.2 MPa·m^{1/2}; steels larger (0.6–1.0+).
    """
    if not d_um or d_um <= 0:
        return 0.0
    d_m = d_um * 1e-6
    return k_y_MPa_sqrtm / math.sqrt(d_m)


def modulus_mismatch_factor(Gm: float, Gp: float) -> float:
    """Simple scalar to represent load-transfer/mismatch contribution.
    
    Δσ/σ0 ~ c · (Gp/Gm − 1) · f (small for coherent shearable; larger for strong particles)
    We keep c small (0.2) to avoid overstating; see shear-lag models.
    
    Args:
        Gm: Matrix shear modulus in GPa
        Gp: Particle shear modulus in GPa
    
    Returns:
        Mismatch factor (dimensionless)
    """
    if (Gm <= 0) or (Gp <= 0):
        return 0.0
    return (Gp/Gm - 1.0)


def predict_yield_strength_MPa(
    inp: StrengthInputs, 
    secondary_elastic: Dict[str, Dict[str, float]]
) -> Tuple[float, Dict[str, float]]:
    """Predict macroscopic yield strength using mechanistic superposition (square-sum).
    
    σ_y ≈ σ0 + M·sqrt(Δτ_orowan^2 + Δτ_coh^2) + Δσ_HP + Δσ_mismatch
    
    Args:
        inp: StrengthInputs with matrix properties and particle statistics
        secondary_elastic: Dict mapping phase name to {"B_GPa": float, "G_GPa": float}
    
    Returns:
        (sigma_y_MPa, components_dict)
    """
    nu = inp.matrix_nu
    d_orowan2 = 0.0
    d_coh2 = 0.0
    d_mismatch = 0.0
    
    for ph, stats in inp.particles.items():
        f = float(stats.get("volume_fraction", 0.0))
        r = float(stats.get("radius_nm", 0.0))
        shearable = int(stats.get("shearable", 0))
        misfit = float(stats.get("misfit_strain", 0.0))
        
        if shearable:
            d_tau = coherency_delta_tau_MPa(inp.matrix_G_GPa, misfit, f)
            d_coh2 += d_tau**2
        else:
            d_tau = orowan_delta_tau_MPa(inp.matrix_G_GPa, inp.burgers_b_nm, nu, f, r)
            d_orowan2 += d_tau**2
        
        # modest mismatch term
        pel = secondary_elastic.get(ph, {})
        Gp = pel.get("G_GPa")
        if Gp and inp.matrix_G_GPa:
            d_mismatch += 0.2 * abs(modulus_mismatch_factor(inp.matrix_G_GPa, Gp)) * f * inp.sigma0_MPa
    
    d_hp = hall_petch_delta_sigma_MPa(inp.grain_size_um)
    sigma = inp.sigma0_MPa + inp.M_taylor*math.sqrt(d_orowan2 + d_coh2) + d_hp + d_mismatch
    
    return sigma, {
        "sigma0_MPa": inp.sigma0_MPa,
        "Taylor_M": inp.M_taylor,
        "Delta_tau_orowan_MPa": math.sqrt(d_orowan2),
        "Delta_tau_coherency_MPa": math.sqrt(d_coh2),
        "Delta_sigma_HP_MPa": d_hp,
        "Delta_sigma_mismatch_MPa": d_mismatch,
    }


# --------------------------------------------------------------------------------------
# Embrittlement risk (Pugh, Cauchy, Poisson) + topology penalty
# --------------------------------------------------------------------------------------

@dataclass
class EmbrittlementInputs:
    """Input parameters for embrittlement risk assessment."""
    matrix_B_GPa: float
    matrix_G_GPa: float
    matrix_nu: Optional[float]
    matrix_C: Optional[List[List[float]]]  # 6x6 elastic tensor for Cauchy pressure
    phase_fractions: Dict[str, float]
    phase_categories: Dict[str, str]
    brittle_categories: Tuple[str, ...] = ("laves", "tau", "sigma", "chi", "mu")
    percolation_threshold: float = 0.6  # matrix fraction below → network risk


def embrittlement_score(inp: EmbrittlementInputs) -> Tuple[float, Dict[str, Any]]:
    """Return risk score in [0,1] and components.
    
    0 = ductile, 1 = brittle; map to labels via thresholds.
    
    Components:
        • Ductility metrics: Pugh margin (0.57 cut), Poisson (0.26 cut), Cauchy sign
        • Brittle-phase volume fraction and percolation penalty
    
    Args:
        inp: EmbrittlementInputs with matrix properties and phase data
    
    Returns:
        (risk_score, components_dict) where risk_score in [0, 1]
    """
    from .mechanical_utils import pugh_ratio as calc_pugh, cauchy_pressure_indicator
    
    # Ductility metrics (matrix only)
    B, G, nu = inp.matrix_B_GPa, inp.matrix_G_GPa, inp.matrix_nu
    pr = calc_pugh(B, G)
    duct = 0.0
    comp = {}
    
    if pr is not None:
        # 0 if far ductile (G/B≪0.57), →1 if far brittle (G/B≫0.57)
        duct += 0.5 * (1.0/(1.0 + math.exp(-20.0*(float(pr) - 0.57))))
        comp["pugh_ratio"] = pr
    
    if nu is not None:
        # 0 if ν≫0.26; →1 if ν≪0.26
        duct += 0.3 * (1.0/(1.0 + math.exp(-40.0*(0.26 - float(nu)))))
        comp["poisson"] = nu
    
    cp = cauchy_pressure_indicator(inp.matrix_C)
    if cp is not None:
        # Negative Cauchy → more brittle; scale smoothly
        duct += 0.2 * (1.0/(1.0 + math.exp(-0.03*(-float(cp)))))
        comp["cauchy_pressure_C12_minus_C44_GPa"] = cp
    
    # Topology: brittle secondary volume and percolation risk
    f_mat = 0.0
    f_brittle = 0.0
    
    # Identify matrix by max fraction
    mat_name = max(inp.phase_fractions, key=inp.phase_fractions.get) if inp.phase_fractions else None
    
    for phase_name, frac in inp.phase_fractions.items():
        if phase_name == mat_name:
            f_mat += frac
        elif (inp.phase_categories.get(phase_name, "")).lower() in inp.brittle_categories:
            f_brittle += frac
    
    percolation = 1.0/(1.0 + math.exp(-20.0*(0.6 - f_mat)))  # low matrix fraction → penalty →1
    
    # Aggregate: weight ductility + brittle volume + percolation
    risk = min(1.0, max(0.0, 0.6*duct + 0.25*f_brittle + 0.15*percolation))
    
    comp.update({
        "matrix_fraction": f_mat,
        "brittle_secondaries_fraction": f_brittle,
        "percolation_penalty": percolation,
    })
    return risk, comp


# --------------------------------------------------------------------------------------
# Assessment wrapper for integration with existing code
# --------------------------------------------------------------------------------------

def assess_mechanical_effects(
    matrix_desc: Dict[str, Any],
    sec_descs: Dict[str, Dict[str, Any]],
    microstructure: Dict[str, Any],
    phase_categories: Dict[str, str],
    # Optional: parameters for strength prediction
    grain_size_um: Optional[float] = None,
    particle_model: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """
    Assess strengthening and embrittlement using physics-based models.
    
    This function:
    1. Computes embrittlement risk using Pugh ratio, Cauchy pressure, and topology
    2. Estimates yield strength using Orowan, coherency, Hall-Petch models
    3. Maps continuous physics to discrete labels for interpretability
    
    Args:
        matrix_desc: Mechanical descriptors for matrix phase from mechanical_utils
        sec_descs: Dict mapping phase name to mechanical descriptors for secondary phases
        microstructure: Equilibrium microstructure data from equilibrium_utils
        phase_categories: Dict mapping phase name to category ("laves", "tau", "gamma")
        grain_size_um: Grain size in micrometers (optional, for Hall-Petch)
        particle_model: Dict mapping phase name to particle parameters (optional)
        
    Returns:
        {
            "strengthening_likelihood": "high"/"moderate"/"low",
            "embrittlement_risk": "high"/"moderate"/"low",
            "yield_strength_MPa": float,
            "embrittlement_score": float in [0, 1],
            "explanations": {
                "matrix": {...},
                "secondary": {...},
                "strength_components": {...},
                "embrittlement_components": {...}
            }
        }
    """
    try:
        phase_fractions = microstructure["phase_fractions"]
        matrix_name = microstructure["matrix_phase"]
        
        # Extract matrix elastic properties
        B_mat = matrix_desc.get("bulk_modulus_GPa")
        G_mat = matrix_desc.get("shear_modulus_GPa")
        nu_mat = matrix_desc.get("nu")
        
        if B_mat is None or G_mat is None:
            _log.warning(f"Matrix {matrix_name} missing elastic data, using defaults")
            # Defaults for FCC Al-like
            if matrix_desc.get("is_fcc_like"):
                B_mat, G_mat, nu_mat = 76.0, 26.0, 0.33
            elif matrix_desc.get("is_bcc_like"):
                B_mat, G_mat, nu_mat = 170.0, 82.0, 0.29
            else:
                B_mat, G_mat, nu_mat = 120.0, 60.0, 0.30
        
        # 1) Embrittlement assessment
        emb_inp = EmbrittlementInputs(
            matrix_B_GPa=B_mat,
            matrix_G_GPa=G_mat,
            matrix_nu=nu_mat,
            matrix_C=None,  # Could extract from matrix_desc if available
            phase_fractions=phase_fractions,
            phase_categories=phase_categories
        )
        risk_score, emb_comp = embrittlement_score(emb_inp)
        
        # Map score to label
        if risk_score > 0.65:
            embrittle_label = "high"
        elif risk_score > 0.40:
            embrittle_label = "moderate"
        else:
            embrittle_label = "low"
        
        # 2) Strength assessment
        # Determine defaults based on matrix type
        if matrix_desc.get("is_bcc_like"):
            burgers_b_nm = 0.248  # Fe-like
            sigma0_MPa = 60.0
            M_taylor = 2.75
        else:  # FCC or other
            burgers_b_nm = 0.286  # Al-like
            sigma0_MPa = 25.0
            M_taylor = 3.06
        
        # Build particle model if not provided
        if particle_model is None:
            particle_model = {}
            for p in microstructure.get("secondary_phases", []):
                pname = p["name"]
                pfrac = p["fraction"]
                pcat = phase_categories.get(pname, "")
                
                # Conservative defaults
                if pcat.lower() in ("gamma",):
                    # Shearable coherent precipitate
                    particle_model[pname] = {
                        "volume_fraction": pfrac,
                        "radius_nm": 7.0,
                        "shearable": 1,
                        "misfit_strain": 0.01
                    }
                else:
                    # Non-shearable intermetallic
                    particle_model[pname] = {
                        "volume_fraction": pfrac,
                        "radius_nm": 15.0,
                        "shearable": 0,
                        "misfit_strain": 0.0
                    }
        
        # Prepare secondary elastic data
        sec_elastic = {}
        for pname, desc in sec_descs.items():
            sec_elastic[pname] = {
                "B_GPa": desc.get("bulk_modulus_GPa"),
                "G_GPa": desc.get("shear_modulus_GPa"),
            }
        
        strength_inp = StrengthInputs(
            matrix_B_GPa=B_mat,
            matrix_G_GPa=G_mat,
            matrix_nu=nu_mat if nu_mat is not None else 0.33,
            burgers_b_nm=burgers_b_nm,
            particles=particle_model,
            grain_size_um=grain_size_um,
            sigma0_MPa=sigma0_MPa,
            M_taylor=M_taylor,
        )
        
        sigma_y, strength_comps = predict_yield_strength_MPa(strength_inp, sec_elastic)
        
        # Assess strengthening based on predicted increase
        delta_sigma = sigma_y - sigma0_MPa
        if delta_sigma > 50.0:
            strength_label = "high"
        elif delta_sigma > 20.0:
            strength_label = "moderate"
        else:
            strength_label = "low"
        
        # Build explanations
        total_secondary_frac = 1.0 - phase_fractions.get(matrix_name, 0.0)
        
        explanations = {
            "matrix": {
                "phase": matrix_name,
                "fraction": round(phase_fractions.get(matrix_name, 0.0), 3),
                "brittle_flag": matrix_desc.get("brittle_flag"),
                "pugh_ratio": emb_comp.get("pugh_ratio"),
                "type": "BCC" if matrix_desc.get("is_bcc_like") else "FCC" if matrix_desc.get("is_fcc_like") else "other",
                "source": matrix_desc.get("source"),
            },
            "secondary": {
                "total_fraction": round(total_secondary_frac, 3),
                "brittle_secondaries_fraction": round(emb_comp.get("brittle_secondaries_fraction", 0.0), 3),
                "phases": [
                    {
                        "name": p["name"],
                        "fraction": round(p["fraction"], 3),
                        "is_intermetallic": sec_descs.get(p["name"], {}).get("is_intermetallic"),
                        "brittle_flag": sec_descs.get(p["name"], {}).get("brittle_flag"),
                        "pugh_ratio": sec_descs.get(p["name"], {}).get("pugh_ratio"),
                        "source": sec_descs.get(p["name"], {}).get("source"),
                    }
                    for p in microstructure.get("secondary_phases", [])
                ],
            },
            "strength_components": strength_comps,
            "embrittlement_components": emb_comp,
        }
        
        return {
            "strengthening_likelihood": strength_label,
            "embrittlement_risk": embrittle_label,
            "yield_strength_MPa": round(sigma_y, 1),
            "embrittlement_score": round(risk_score, 3),
            "explanations": explanations,
        }
        
    except Exception as e:
        _log.error(f"Error assessing mechanical effects: {e}", exc_info=True)
        return {
            "strengthening_likelihood": "unknown",
            "embrittlement_risk": "unknown",
            "yield_strength_MPa": None,
            "embrittlement_score": None,
            "explanations": {"error": str(e)},
        }
