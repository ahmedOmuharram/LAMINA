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

from ..base import BaseHandler
from .ai_functions import SemiconductorAIFunctionsMixin

_log = logging.getLogger(__name__)


class SemiconductorHandler(BaseHandler, SemiconductorAIFunctionsMixin):
    """
    Handler for semiconductor and defect-related queries.
    
    Combines BaseHandler functionality with semiconductor-specific AI functions
    for analyzing crystal structures, defects, doping, and magnetic properties.
    """
    
    def __init__(self, mpr: MPRester):
        """
        Initialize the semiconductor handler.
        
        Args:
            mpr: MPRester client instance for Materials Project API access
        """
        super().__init__(mpr)
        self.recent_tool_outputs = []
        _log.info("SemiconductorHandler initialized")


def create_semiconductor_handler(mpr: MPRester) -> SemiconductorHandler:
    """
    Factory function to create a SemiconductorHandler instance.
    
    Args:
        mpr: MPRester client instance
        
    Returns:
        Configured SemiconductorHandler instance
    """
    return SemiconductorHandler(mpr)

