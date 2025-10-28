"""
Automated alloy phase fact-checker for evaluating microstructure claims.

This module provides tools to verify metallurgical assertions about multicomponent
alloys using CALPHAD thermodynamic calculations. It evaluates claims like:
  - "Al-8Mg-4Zn forms a two-phase microstructure with fcc + tau phase"
  - "Tau phase fraction does not exceed 20%"
  - "Eutectic Al-Mg-Zn exhibits >20% intermetallic phases"

The fact-checker uses three layers:
  Layer A: Thermodynamic equilibrium calculations (uses existing equilibrium_utils)
  Layer B: Phase interpretation (maps CALPHAD names to metallurgical categories)
  Layer C: Claim evaluation (scores natural-language assertions)
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import numpy as np
from pycalphad import Database

from ...shared.calphad_utils import (
    compute_equilibrium,
    extract_phase_fractions_from_equilibrium
)
from .database_utils import map_phase_name
from ...base.constants import PhaseCategory, PHASE_CLASSIFICATION

_log = logging.getLogger(__name__)


# ============================================================================
# LAYER B: PHASE INTERPRETER
# ============================================================================

@dataclass
class PhaseInfo:
    """Interpreted phase information."""
    raw_name: str                         # CALPHAD phase name (e.g., "FCC_A1#1")
    base_name: str                        # Base name without instance suffix (e.g., "FCC_A1")
    readable_name: str                    # Human-readable name (e.g., "fcc Al-rich")
    category: PhaseCategory               # Metallurgical category
    fraction: float                       # Phase fraction (mole or mass)
    crystal_structure: Optional[str] = None  # Crystal structure (fcc, bcc, hcp, etc.)


def interpret_phase(phase_name: str, fraction: float) -> PhaseInfo:
    """
    Interpret a CALPHAD phase name into metallurgical context.
    
    Args:
        phase_name: Raw phase name from pycalphad (e.g., "FCC_A1#1", "TAU_PHASE")
        fraction: Phase fraction (0-1)
        
    Returns:
        PhaseInfo with interpreted metadata
    """
    # Strip instance suffix (#1, #2, etc.)
    base_name = phase_name.split('#')[0].upper().strip()
    
    # Look up in classification table
    if base_name in PHASE_CLASSIFICATION:
        readable, category, structure = PHASE_CLASSIFICATION[base_name]
    else:
        # Fallback: try to infer from name
        readable = map_phase_name(base_name)  # Use existing mapping
        category = _infer_category(base_name)
        structure = _infer_structure(base_name)
    
    return PhaseInfo(
        raw_name=phase_name,
        base_name=base_name,
        readable_name=readable,
        category=category,
        fraction=fraction,
        crystal_structure=structure
    )


def _infer_category(name: str) -> PhaseCategory:
    """Infer category from phase name using heuristics."""
    name_upper = name.upper()
    
    if "LIQUID" in name_upper:
        return PhaseCategory.LIQUID
    elif "FCC" in name_upper:
        return PhaseCategory.PRIMARY_FCC
    elif "BCC" in name_upper:
        return PhaseCategory.PRIMARY_BCC
    elif "HCP" in name_upper:
        return PhaseCategory.PRIMARY_HCP
    elif "TAU" in name_upper or "T_PHASE" in name_upper:
        return PhaseCategory.TAU_PHASE
    elif "GAMMA" in name_upper:
        return PhaseCategory.GAMMA
    elif "LAVES" in name_upper or "C14" in name_upper or "C15" in name_upper:
        return PhaseCategory.LAVES
    elif "SIGMA" in name_upper:
        return PhaseCategory.SIGMA
    elif any(x in name_upper for x in ["CARBIDE", "M23C6", "M7C3", "M6C", "CEMENTITE"]):
        return PhaseCategory.CARBIDE
    elif "NITRIDE" in name_upper or "TIN" in name_upper or "ALN" in name_upper:
        return PhaseCategory.NITRIDE
    else:
        return PhaseCategory.OTHER


def _infer_structure(name: str) -> Optional[str]:
    """Infer crystal structure from phase name."""
    name_upper = name.upper()
    
    if "FCC" in name_upper:
        return "fcc"
    elif "BCC" in name_upper:
        return "bcc"
    elif "HCP" in name_upper:
        return "hcp"
    elif "C14" in name_upper or "C36" in name_upper:
        return "hexagonal"
    elif "C15" in name_upper:
        return "cubic"
    else:
        return None


def interpret_microstructure(phase_fractions: Dict[str, float]) -> List[PhaseInfo]:
    """
    Interpret a complete equilibrium microstructure.
    
    Args:
        phase_fractions: Dict of {phase_name: fraction} from equilibrium calculation
        
    Returns:
        List of PhaseInfo objects, sorted by fraction (descending)
    """
    phases = [interpret_phase(name, frac) for name, frac in phase_fractions.items()]
    phases.sort(key=lambda p: p.fraction, reverse=True)
    return phases


# ============================================================================
# LAYER C: CLAIM CHECKERS
# ============================================================================

@dataclass
class ClaimResult:
    """Result of a microstructure claim evaluation."""
    claim_id: str                        # Identifier for the claim
    claim_text: str                      # Natural language claim
    verdict: bool                        # True if claim is supported
    score: int                           # Score (-2 to +2, where +2 = fully correct)
    confidence: float                    # Confidence in verdict (0-1)
    reasoning: str                       # Explanation of verdict
    supporting_data: Dict[str, Any]      # Numerical evidence


class ClaimChecker:
    """Base class for claim evaluation."""
    
    def __init__(self, db: Database, elements: List[str], phases: List[str], 
                 temperature: float = 300.0):
        """
        Initialize claim checker.
        
        Args:
            db: PyCalphad Database instance
            elements: List of element symbols (e.g., ["AL", "MG", "ZN"])
            phases: List of phase names to consider
            temperature: Reference temperature in K (default: 300K room temp)
        """
        self.db = db
        self.elements = elements
        self.phases = phases
        self.temperature = temperature
    
    def calculate_microstructure(self, composition: Dict[str, float], 
                                 precalculated_fractions: Optional[Dict[str, float]] = None) -> List[PhaseInfo]:
        """
        Calculate or use pre-calculated equilibrium microstructure.
        
        Args:
            composition: Dict of {element: mole_fraction}, e.g., {"AL": 0.88, "MG": 0.08, "ZN": 0.04}
            precalculated_fractions: Optional pre-calculated phase fractions (for as-cast simulations)
            
        Returns:
            List of PhaseInfo objects
        """
        if precalculated_fractions is not None:
            # Use provided phase fractions (e.g., from solidification simulation)
            phase_fractions = precalculated_fractions
        else:
            # Calculate equilibrium
            eq = compute_equilibrium(
                self.db, self.elements, self.phases,
                composition, self.temperature
            )
            
            if eq is None:
                _log.error("Equilibrium calculation failed")
                return []
            
            # Extract phase fractions
            phase_fractions = extract_phase_fractions_from_equilibrium(eq)
        
        # Interpret phases
        return interpret_microstructure(phase_fractions)
    
    def check(self, composition: Dict[str, float], 
             precalculated_fractions: Optional[Dict[str, float]] = None) -> ClaimResult:
        """
        Evaluate claim for given composition.
        Must be implemented by subclasses.
        
        Args:
            composition: Element mole fractions
            precalculated_fractions: Optional pre-calculated phase fractions
        """
        raise NotImplementedError


class TwoPhaseChecker(ClaimChecker):
    """
    Checks claim: "Composition forms a two-phase microstructure with primary phase + secondary phase"
    
    Example: "Al-8Mg-4Zn forms fcc + tau phase with tau < 20%"
    """
    
    def __init__(self, db: Database, elements: List[str], phases: List[str],
                 primary_category: PhaseCategory,
                 secondary_category: PhaseCategory,
                 secondary_max_fraction: float = 0.20,
                 temperature: float = 300.0,
                 tolerance: float = 0.05):
        """
        Args:
            primary_category: Expected dominant phase category
            secondary_category: Expected secondary phase category
            secondary_max_fraction: Maximum allowed fraction of secondary phase
            tolerance: Tolerance for "other" phases (default 5%)
        """
        super().__init__(db, elements, phases, temperature)
        self.primary_category = primary_category
        self.secondary_category = secondary_category
        self.secondary_max_fraction = secondary_max_fraction
        self.tolerance = tolerance
    
    def check(self, composition: Dict[str, float],
             precalculated_fractions: Optional[Dict[str, float]] = None) -> ClaimResult:
        """Evaluate two-phase claim."""
        microstructure = self.calculate_microstructure(composition, precalculated_fractions)
        
        if not microstructure:
            return ClaimResult(
                claim_id="two_phase",
                claim_text=f"Two-phase {self.primary_category.value} + {self.secondary_category.value}",
                verdict=False,
                score=-2,
                confidence=0.0,
                reasoning="Equilibrium calculation failed",
                supporting_data={}
            )
        
        # Extract phase fractions by category
        primary_frac = sum(p.fraction for p in microstructure 
                          if p.category == self.primary_category)
        secondary_frac = sum(p.fraction for p in microstructure 
                            if p.category == self.secondary_category)
        other_frac = max(0.0, 1.0 - (primary_frac + secondary_frac))  # Clamp negative zeros
        
        supporting_data = {
            f"{self.primary_category.value}_fraction": primary_frac,
            f"{self.secondary_category.value}_fraction": secondary_frac,
            "other_fraction": other_frac,
            "phases": [(p.readable_name, p.fraction, p.category.value) 
                      for p in microstructure]
        }
        
        # Evaluate criteria
        has_primary = primary_frac > 0.5
        has_secondary = secondary_frac > 1e-3
        secondary_within_limit = secondary_frac <= self.secondary_max_fraction + 1e-3
        minimal_other = other_frac < self.tolerance
        
        verdict = all([has_primary, has_secondary, secondary_within_limit, minimal_other])
        
        # Score based on how well it matches
        if verdict:
            score = 2  # Perfect match
        elif has_primary and has_secondary:
            if not secondary_within_limit:
                score = -1  # Right phases, wrong fraction
                reasoning = f"{self.secondary_category.value} exceeds limit: {secondary_frac:.1%} > {self.secondary_max_fraction:.1%}"
            else:
                score = 0   # Right phases, but too many others
                reasoning = f"Has {self.primary_category.value} + {self.secondary_category.value}, but {other_frac:.1%} other phases"
        elif has_primary or has_secondary:
            score = -1  # Only one of the two phases
            if not has_primary:
                reasoning = f"{self.primary_category.value} is not dominant (>50%); observed {primary_frac:.1%}"
            else:
                reasoning = f"{self.secondary_category.value} not found; observed {secondary_frac:.1%}"
        else:
            score = -2  # Neither phase present
            reasoning = "Neither expected phase is present"
        
        if verdict:
            reasoning = (f"✓ Dominant {self.primary_category.value} ({primary_frac:.1%}) + "
                        f"{self.secondary_category.value} ({secondary_frac:.1%}) within limits")
        
        confidence = 1.0 if (has_primary or has_secondary) else 0.3
        
        return ClaimResult(
            claim_id="two_phase",
            claim_text=f"Two-phase: dominant {self.primary_category.value} + {self.secondary_category.value} <{self.secondary_max_fraction:.0%}",
            verdict=verdict,
            score=score,
            confidence=confidence,
            reasoning=reasoning,
            supporting_data=supporting_data
        )


class ThreePhaseChecker(ClaimChecker):
    """
    Checks claim: "Composition forms a three-phase microstructure"
    
    Example: "Al-8Mg-4Zn forms fcc + Laves + gamma"
    """
    
    def __init__(self, db: Database, elements: List[str], phases: List[str],
                 expected_categories: List[PhaseCategory],
                 min_fraction: float = 0.01,
                 temperature: float = 300.0):
        """
        Args:
            expected_categories: List of 3 expected phase categories
            min_fraction: Minimum fraction to consider a phase "present"
        """
        super().__init__(db, elements, phases, temperature)
        self.expected_categories = expected_categories
        self.min_fraction = min_fraction
    
    def check(self, composition: Dict[str, float],
             precalculated_fractions: Optional[Dict[str, float]] = None) -> ClaimResult:
        """Evaluate three-phase claim."""
        from collections import Counter
        
        microstructure = self.calculate_microstructure(composition, precalculated_fractions)
        
        if not microstructure:
            return ClaimResult(
                claim_id="three_phase",
                claim_text=f"Three-phase: {', '.join(c.value for c in self.expected_categories)}",
                verdict=False,
                score=-2,
                confidence=0.0,
                reasoning="Equilibrium calculation failed",
                supporting_data={}
            )
        
        # Count present categories with distinct phases above threshold
        present_categories = [p.category for p in microstructure if p.fraction > self.min_fraction]
        present_counts = Counter(present_categories)
        
        expected_counts = Counter(self.expected_categories)
        
        # Check if all expected categories are present with sufficient counts
        verdict = all(present_counts[cat] >= cnt for cat, cnt in expected_counts.items())
        
        supporting_data = {
            "present_category_counts": {c.value: present_counts.get(c, 0) for c in expected_counts},
            "all_phases": [(p.readable_name, p.fraction, p.base_name, p.category.value) 
                          for p in microstructure]
        }
        
        # Calculate how many category requirements are satisfied
        satisfied = sum(min(present_counts.get(cat, 0), cnt) for cat, cnt in expected_counts.items())
        total_needed = sum(expected_counts.values())
        
        if verdict:
            score = 2
            reasoning = " | ".join(
                f"{cat.value}: need ≥{cnt}, have {present_counts.get(cat, 0)}"
                for cat, cnt in expected_counts.items()
            )
        else:
            # Grade by how many categories satisfied
            if satisfied == total_needed - 1:
                score = 0
            elif satisfied >= 1:
                score = -1
            else:
                score = -2
            reasoning = " | ".join(
                f"{cat.value}: need ≥{cnt}, have {present_counts.get(cat, 0)}"
                for cat, cnt in expected_counts.items()
            )
        
        confidence = satisfied / total_needed if total_needed > 0 else 0.0
        
        return ClaimResult(
            claim_id="three_phase",
            claim_text=f"Three-phase: {', '.join(c.value for c in self.expected_categories)}",
            verdict=verdict,
            score=score,
            confidence=confidence,
            reasoning=reasoning,
            supporting_data=supporting_data
        )


class PhaseFractionChecker(ClaimChecker):
    """
    Checks claim: "Phase X has fraction greater/less than Y%"
    
    Example: "Tau phase exceeds 20% in Al-34.5Mg-5Zn"
    """
    
    def __init__(self, db: Database, elements: List[str], phases: List[str],
                 target_category: PhaseCategory,
                 min_fraction: Optional[float] = None,
                 max_fraction: Optional[float] = None,
                 temperature: float = 300.0):
        """
        Args:
            target_category: Phase category to check
            min_fraction: Minimum required fraction (None = no min)
            max_fraction: Maximum allowed fraction (None = no max)
        """
        super().__init__(db, elements, phases, temperature)
        self.target_category = target_category
        self.min_fraction = min_fraction
        self.max_fraction = max_fraction
    
    def check(self, composition: Dict[str, float],
             precalculated_fractions: Optional[Dict[str, float]] = None) -> ClaimResult:
        """Evaluate phase fraction claim."""
        microstructure = self.calculate_microstructure(composition, precalculated_fractions)
        
        if not microstructure:
            return ClaimResult(
                claim_id="phase_fraction",
                claim_text=f"{self.target_category.value} fraction check",
                verdict=False,
                score=-2,
                confidence=0.0,
                reasoning="Equilibrium calculation failed",
                supporting_data={}
            )
        
        # Get target phase fraction
        target_frac = sum(p.fraction for p in microstructure 
                         if p.category == self.target_category)
        
        supporting_data = {
            f"{self.target_category.value}_fraction": target_frac,
            "all_phases": [(p.readable_name, p.fraction, p.category.value) 
                          for p in microstructure]
        }
        
        # Evaluate criteria
        passes = True
        reasons = []
        
        if self.min_fraction is not None:
            if target_frac < self.min_fraction:
                passes = False
                reasons.append(f"{target_frac:.1%} < {self.min_fraction:.1%} (minimum)")
            else:
                reasons.append(f"{target_frac:.1%} ≥ {self.min_fraction:.1%} (minimum)")
        
        if self.max_fraction is not None:
            if target_frac > self.max_fraction:
                passes = False
                reasons.append(f"{target_frac:.1%} > {self.max_fraction:.1%} (maximum)")
            else:
                reasons.append(f"{target_frac:.1%} ≤ {self.max_fraction:.1%} (maximum)")
        
        # Score
        if passes:
            score = 2
            verdict = True
            reasoning = f"✓ {self.target_category.value} fraction {target_frac:.1%} passes: " + ", ".join(reasons)
        else:
            # How far off?
            if self.min_fraction and target_frac < self.min_fraction * 0.5:
                score = -2
            elif self.max_fraction and target_frac > self.max_fraction * 1.5:
                score = -2
            else:
                score = -1
            verdict = False
            reasoning = f"✗ {self.target_category.value} fraction {target_frac:.1%} fails: " + ", ".join(reasons)
        
        confidence = 1.0 if target_frac > 1e-3 else 0.3
        
        return ClaimResult(
            claim_id="phase_fraction",
            claim_text=f"{self.target_category.value} fraction check",
            verdict=verdict,
            score=score,
            confidence=confidence,
            reasoning=reasoning,
            supporting_data=supporting_data
        )


# ============================================================================
# ORCHESTRATION: Fact-Checker
# ============================================================================

class AlloyFactChecker:
    """
    Automated alloy phase fact-checker that evaluates multiple microstructure claims.
    """
    
    def __init__(self, db: Database, elements: List[str], phases: List[str],
                 temperature: float = 300.0):
        """
        Initialize fact-checker.
        
        Args:
            db: PyCalphad Database instance
            elements: List of element symbols
            phases: List of phase names to consider
            temperature: Reference temperature in K (default: 300K for "after slow solidification")
        """
        self.db = db
        self.elements = elements
        self.phases = phases
        self.temperature = temperature
        self.checkers: List[ClaimChecker] = []
    
    def add_checker(self, checker: ClaimChecker):
        """Add a claim checker to the evaluation pipeline."""
        self.checkers.append(checker)
    
    def evaluate_all(self, composition: Dict[str, float],
                     precalculated_fractions: Optional[Dict[str, float]] = None) -> List[ClaimResult]:
        """
        Evaluate all registered claims for a given composition.
        
        Args:
            composition: Dict of {element: mole_fraction}
            precalculated_fractions: Optional pre-calculated phase fractions (for as-cast simulations)
            
        Returns:
            List of ClaimResult objects
        """
        results = []
        for checker in self.checkers:
            try:
                result = checker.check(composition, precalculated_fractions)
                results.append(result)
            except Exception as e:
                _log.error(f"Checker {checker.__class__.__name__} failed: {e}")
                results.append(ClaimResult(
                    claim_id=checker.__class__.__name__,
                    claim_text="Unknown",
                    verdict=False,
                    score=-2,
                    confidence=0.0,
                    reasoning=f"Error: {e}",
                    supporting_data={}
                ))
        
        return results
    
    def generate_report(self, 
                        composition: Dict[str, float],
                        precalculated_fractions: Optional[Dict[str, float]] = None,
                        process_description: str = "") -> str:
        """
        Generate a comprehensive fact-check report.
        
        Args:
            composition: Dict of {element: mole_fraction}
            precalculated_fractions: Optional pre-calculated phase fractions (for as-cast simulations)
            process_description: Description of the process model used
            
        Returns:
            Formatted report string
        """
        results = self.evaluate_all(composition, precalculated_fractions=precalculated_fractions)
        
        # Format composition
        comp_str = " + ".join(f"{frac*100:.1f}% {el}" for el, frac in composition.items())
        
        lines = [
            "=" * 80,
            f"ALLOY FACT-CHECK REPORT",
            "=" * 80,
            f"Composition: {comp_str}",
            f"Temperature: {self.temperature:.0f} K ({self.temperature - 273.15:.0f} °C)",
            f"System: {'-'.join(self.elements)}",
        ]
        
        if process_description:
            lines.append(f"Process model: {process_description}")
        
        lines.extend([
            "",
            "CLAIMS EVALUATED:",
            "-" * 80,
        ])
        
        for i, result in enumerate(results, 1):
            verdict_symbol = "✓" if result.verdict else "✗"
            score_text = f"{result.score:+d}/2"
            
            lines.append(f"\n{i}. {result.claim_text}")
            lines.append(f"   {verdict_symbol} Verdict: {'SUPPORTED' if result.verdict else 'REJECTED'}")
            lines.append(f"   Score: {score_text} | Confidence: {result.confidence:.0%}")
            lines.append(f"   Reasoning: {result.reasoning}")
            
            # Show supporting data
            if result.supporting_data:
                lines.append(f"   Supporting data:")
                for key, value in result.supporting_data.items():
                    if isinstance(value, float):
                        lines.append(f"     - {key}: {value:.3f}")
                    elif isinstance(value, list) and key in ["phases", "all_phases"]:
                        lines.append(f"     - {key}:")
                        for phase_data in value[:5]:  # Show top 5
                            if len(phase_data) >= 2:
                                lines.append(f"         {phase_data[0]}: {phase_data[1]:.1%}")
        
        lines.extend([
            "",
            "-" * 80,
            f"SUMMARY: {sum(1 for r in results if r.verdict)}/{len(results)} claims supported",
            f"Average score: {np.mean([r.score for r in results]):.2f}/2.0",
            "=" * 80,
        ])
        
        return "\n".join(lines)
