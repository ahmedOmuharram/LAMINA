"""
Alloys and surfaces handlers.

Provides tools to estimate diffusion barriers of adatoms on metal surfaces
and assess alloy microstructure, strengthening, and stiffness using CALPHAD
and Materials Project data.

Modular structure:
- alloy_handler.py: Handler class with AI-exposed kani functions
- atomic_utils.py: Element data utilities (cohesive energy, radii)
- surface_utils.py: Surface diffusion barrier calculations  
- composition_utils.py: Composition string parsing
- mechanical_utils.py: Phase mechanical property assessment via Materials Project
- stiffness_utils.py: Elastic modulus (stiffness) estimation
- assessment_utils.py: Strengthening/embrittlement assessment logic
- verification_utils.py: Claim verification and interpretation

Uses shared utilities:
- handlers/shared/calphad_utils.py: compute_equilibrium_microstructure()
- handlers/calphad/phase_diagrams/equilibrium_utils.py: extract_phase_fractions_from_equilibrium()
"""

from .alloy_handler import AlloyHandler

__all__ = ["AlloyHandler"]
