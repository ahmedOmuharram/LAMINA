"""
Solutes Handler

Handler for analyzing lattice parameter effects of substitutional solute atoms
in fcc matrices using Vegard's law and Hume-Rothery principles.
"""

from typing import Optional
from mp_api.client import MPRester
from .ai_functions import SolutesAIFunctionsMixin


class SolutesHandler(SolutesAIFunctionsMixin):
    """
    Handler for solute lattice effects analysis.
    
    Provides tools to:
    - Calculate lattice parameter changes due to substitutional solutes
    - Rank multiple solutes by their expansion/contraction effects
    - Validate claims using CALPHAD equilibrium data
    - Apply Vegard's law and Hume-Rothery size mismatch criteria
    
    Inherits AI functions from SolutesAIFunctionsMixin.
    """
    
    def __init__(self, mpr: Optional[MPRester] = None):
        """
        Initialize the SolutesHandler.
        
        Args:
            mpr: Optional Materials Project MPRester instance for future extensions
        """
        self.mpr = mpr


__all__ = ["SolutesHandler"]

