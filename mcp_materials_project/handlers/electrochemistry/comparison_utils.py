"""
Comparison utilities for electrochemistry handlers.

Functions for comparing electrode materials and generating summaries.
"""
import logging
from typing import List, Dict, Any

_log = logging.getLogger(__name__)


def generate_comparison_summary(comparison_results: List[Dict[str, Any]], working_ion: str) -> Dict[str, Any]:
    """
    Generate a clear summary for electrode comparison results.
    
    Args:
        comparison_results: List of comparison result dictionaries
        working_ion: Working ion symbol
        
    Returns:
        Dictionary with comparison summary
    """
    summary = {
        "voltages": {},
        "capacities": {},
        "energy_densities": {}
    }
    
    # Extract key metrics
    for result in comparison_results:
        formula = result.get("formula")
        if result.get("source") == "failed":
            continue
            
        data = result.get("data", {})
        
        # Extract voltage
        voltage = data.get("average_voltage") or data.get("voltage")
        if voltage is not None:
            summary["voltages"][formula] = float(voltage)
        
        # Extract capacity
        if "capacity_grav" in data:
            summary["capacities"][formula] = float(data["capacity_grav"])
        
        # Extract energy density
        if "energy_grav" in data:
            summary["energy_densities"][formula] = float(data["energy_grav"])
    
    # Add rankings
    if len(summary["voltages"]) >= 2:
        sorted_by_voltage = sorted(summary["voltages"].items(), key=lambda x: x[1], reverse=True)
        summary["highest_voltage"] = sorted_by_voltage[0][0]
        summary["lowest_voltage"] = sorted_by_voltage[-1][0]
        
        # Generate comparison statement
        formulas = list(summary["voltages"].keys())
        if len(formulas) == 2:
            v1, v2 = summary["voltages"][formulas[0]], summary["voltages"][formulas[1]]
            diff = abs(v1 - v2)
            if v1 > v2:
                summary["voltage_comparison"] = f"{formulas[0]} has {diff:.4f} V higher voltage than {formulas[1]}"
            else:
                summary["voltage_comparison"] = f"{formulas[1]} has {diff:.4f} V higher voltage than {formulas[0]}"
    
    if len(summary["capacities"]) >= 2:
        sorted_by_capacity = sorted(summary["capacities"].items(), key=lambda x: x[1], reverse=True)
        summary["highest_capacity"] = sorted_by_capacity[0][0]
    
    if len(summary["energy_densities"]) >= 2:
        sorted_by_energy = sorted(summary["energy_densities"].items(), key=lambda x: x[1], reverse=True)
        summary["highest_energy_density"] = sorted_by_energy[0][0]
    
    return summary

