"""
CALPHAD thermodynamic calculation handlers.

Provides phase diagram generation and thermodynamic property calculation
using pycalphad and thermodynamic databases.
"""

from .phase_diagrams import CalPhadHandler

__all__ = ['CalPhadHandler']
