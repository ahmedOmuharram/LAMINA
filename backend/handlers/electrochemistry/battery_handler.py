"""
Battery and electrochemistry handler for Materials Project API.

Provides AI functions for:
- Electrode voltage calculations
- Battery material search
- Electrochemical potential analysis
- Capacity calculations
"""
import logging
from typing import Optional, List, Dict, Any
from mp_api.client import MPRester

from ..base import BaseHandler
from .ai_functions import BatteryAIFunctionsMixin
from . import utils

_log = logging.getLogger(__name__)


class BatteryHandler(BaseHandler, BatteryAIFunctionsMixin):
    """
    Handler for battery and electrochemistry calculations using Materials Project data.
    
    Uses the MP electrodes endpoint to get computed voltage profiles and the
    summary endpoint for formation energy calculations.
    """
    
    def __init__(self, mpr: Optional[MPRester] = None, **kwargs):
        """
        Initialize the battery handler.
        
        Args:
            mpr: MPRester client instance (optional, will use existing if available)
        """
        if mpr is not None and 'mpr' not in kwargs:
            kwargs['mpr'] = mpr
        super().__init__(**kwargs)
        if mpr is not None:
            self.mpr = mpr
        _log.info("BatteryHandler initialized")
    
    async def _compute_alloy_voltage_via_hull(
        self,
        formula: str,
        working_ion: str = "Li",
        x_max: float = 3.0,
        dx: float = 0.05,
        ebh_tol: float = 0.03,
        min_voltage: float = 0.02
    ) -> Dict[str, Any]:
        """Delegate to utils.compute_alloy_voltage_via_hull for implementation."""
        return utils.compute_alloy_voltage_via_hull(
            self.mpr, formula, working_ion, x_max, dx, min_voltage
        )
    
    async def _fallback_voltage_calculation(
        self,
        formula: Optional[str] = None,
        elements: Optional[str] = None,
        working_ion: str = "Li"
    ) -> Dict[str, Any]:
        """
        Fallback method when insertion_electrodes endpoint is not available.
        Tries to calculate from DFT energies of lithiated phases.
        """
        _log.info("insertion_electrodes endpoint not available, trying formation energy calculation")
        
        if formula:
            return await self.calculate_voltage_from_formation_energy(
                electrode_formula=formula,
                working_ion=working_ion
            )
        elif elements:
            # Try to construct a formula from elements
            element_list = [e.strip() for e in elements.split(',')]
            # Simple formula: just concatenate elements
            formula_attempt = ''.join(element_list)
            return await self.calculate_voltage_from_formation_energy(
                electrode_formula=formula_attempt,
                working_ion=working_ion
            )
        else:
            return {
                "success": False,
                "error": "insertion_electrodes endpoint not available and no formula/elements provided",
                "suggestions": [
                    "Provide a specific chemical formula to search for lithiated phases",
                    "The insertion_electrodes database may not be accessible"
                ]
            }
    
