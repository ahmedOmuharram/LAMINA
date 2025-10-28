"""
Solutes Handler

Handler for analyzing lattice parameter effects of substitutional solute atoms
in fcc matrices using Vegard's law and Hume-Rothery principles.
"""

import logging
from typing import Optional
from mp_api.client import MPRester
from ..base import BaseHandler
from .ai_functions import SolutesAIFunctionsMixin

_log = logging.getLogger(__name__)


class SolutesHandler(BaseHandler, SolutesAIFunctionsMixin):
    """
    Handler for solute lattice effects analysis.
    
    Provides tools to:
    - Calculate lattice parameter changes due to substitutional solutes
    - Rank multiple solutes by their expansion/contraction effects
    - Validate claims using CALPHAD equilibrium data
    - Apply Vegard's law and Hume-Rothery size mismatch criteria
    
    Inherits AI functions from SolutesAIFunctionsMixin.
    """
    
    def __init__(self, mpr: Optional[MPRester] = None, **kwargs):
        """
        Initialize the SolutesHandler.
        
        Args:
            mpr: Optional Materials Project MPRester instance for future extensions
        """
        if mpr is not None and 'mpr' not in kwargs:
            kwargs['mpr'] = mpr
        super().__init__(**kwargs)
        if mpr is not None:
            self.mpr = mpr
        _log.info("SolutesHandler initialized")

