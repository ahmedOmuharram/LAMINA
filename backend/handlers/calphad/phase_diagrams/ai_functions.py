"""
AI function methods aggregator for CALPHAD phase diagrams.

This module combines all AI function mixins into a single AIFunctionsMixin class.
The actual implementations are split across:
- ai_functions_core.py: Core visualization functions
- ai_functions_calculations.py: Calculation and analysis functions  
- ai_functions_verification.py: Advanced verification and fact-checking functions
"""

from .ai_functions_core import CoreVisualizationMixin
from .ai_functions_calculations import CalculationsMixin
from .ai_functions_verification import VerificationMixin


class AIFunctionsMixin(CoreVisualizationMixin, CalculationsMixin, VerificationMixin):
    """
    Aggregated mixin class containing all AI function methods for CalPhadHandler.
    
    This class combines:
    - CoreVisualizationMixin: plot_binary_phase_diagram, plot_composition_temperature, analyze_last_generated_plot
    - CalculationsMixin: calculate_equilibrium_at_point (AI function), calculate_phase_fractions_vs_temperature, analyze_phase_fraction_trend
    - VerificationMixin: verify_phase_formation_across_composition, sweep_microstructure_claim_over_region, fact_check_microstructure_claim
    
    All methods are available through this single mixin.
    """
    pass
