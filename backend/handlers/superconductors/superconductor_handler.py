"""
Superconductor analysis handler for Materials Project API.

Provides AI functions for analyzing superconducting materials,
particularly cuprates and structural-property relationships.
"""
import logging
from typing import Optional
from mp_api.client import MPRester

from ..base import BaseHandler
from .ai_functions import SuperconductorAIFunctionsMixin

_log = logging.getLogger(__name__)


class SuperconductorHandler(BaseHandler, SuperconductorAIFunctionsMixin):
    """
    Handler for superconductor materials analysis using Materials Project data.
    
    Focuses on cuprate superconductors and structural effects on
    electronic/superconducting properties.
    """
    
    def __init__(self, mpr: Optional[MPRester] = None, **kwargs):
        """
        Initialize the superconductor handler.
        
        Args:
            mpr: MPRester client instance (optional)
        """
        if mpr is not None and 'mpr' not in kwargs:
            kwargs['mpr'] = mpr
        super().__init__(**kwargs)
        if mpr is not None:
            self.mpr = mpr
        _log.info("SuperconductorHandler initialized")

