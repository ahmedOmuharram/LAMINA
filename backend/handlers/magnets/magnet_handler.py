"""
Magnet Strength Assessment Handler

This handler provides functionality for:
- Assessing whether doping improves permanent magnet strength
- Evaluating pull force capabilities
- Analyzing magnetic properties (Br, Hc, (BH)max)
- Comparing baseline vs doped magnetic materials
"""

import logging
from typing import Dict, Any
from mp_api.client import MPRester

from ..base.base import BaseHandler
from .ai_functions import MagnetAIFunctionsMixin

_log = logging.getLogger(__name__)


class MagnetHandler(BaseHandler, MagnetAIFunctionsMixin):
    """
    Handler for magnet strength assessment and magnetic material queries.
    
    Combines BaseHandler functionality with magnet-specific AI functions
    for analyzing permanent magnet properties, pull forces, and the effects
    of doping on magnetic performance.
    
    Focus: Permanent magnet applications where "strength" = pull force capability.
    """
    
    def __init__(self, mpr: MPRester = None, **kwargs):
        """
        Initialize the magnet handler.
        
        Args:
            mpr: MPRester client instance for Materials Project API access
        """
        if mpr is not None and 'mpr' not in kwargs:
            kwargs['mpr'] = mpr
        super().__init__(**kwargs)
        if mpr is not None:
            self.mpr = mpr
        _log.info("MagnetHandler initialized")

