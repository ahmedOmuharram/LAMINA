"""
Semiconductor and Defect Analysis Handler

This handler provides functionality for:
- Structural analysis (octahedral distortions, phase transitions)
- Magnetic property analysis and comparison
- Defect formation energy calculations
- Doping site preference analysis
"""

import logging
from typing import Dict, Any
from mp_api.client import MPRester

from ..base.base import BaseHandler
from .ai_functions import SemiconductorAIFunctionsMixin

_log = logging.getLogger(__name__)


class SemiconductorHandler(BaseHandler, SemiconductorAIFunctionsMixin):
    """
    Handler for semiconductor and defect-related queries.
    
    Combines BaseHandler functionality with semiconductor-specific AI functions
    for analyzing crystal structures, defects, doping, and magnetic properties.
    """
    
    def __init__(self, mpr: MPRester = None, **kwargs):
        """
        Initialize the semiconductor handler.
        
        Args:
            mpr: MPRester client instance for Materials Project API access
        """
        if mpr is not None and 'mpr' not in kwargs:
            kwargs['mpr'] = mpr
        super().__init__(**kwargs)
        if mpr is not None:
            self.mpr = mpr
        _log.info("SemiconductorHandler initialized")

