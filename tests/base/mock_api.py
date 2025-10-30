"""
Mock Materials Project API responses for testing.

This module provides mock objects and data to simulate Materials Project API
responses without making actual API calls.
"""

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock


class MockMaterialDoc:
    """Mock Material Document object mimicking Materials Project API response."""
    
    def __init__(self, **kwargs):
        """Initialize mock document with provided attributes."""
        self.material_id = kwargs.get("material_id", "mp-000")
        self.formula_pretty = kwargs.get("formula_pretty", "X")
        self.composition = MockComposition(kwargs.get("composition", {"X": 1.0}))
        self.energy_above_hull = kwargs.get("energy_above_hull", 0.0)
        self.is_stable = kwargs.get("is_stable", True)
        self.chemsys = kwargs.get("chemsys", "X")
        self.elements = kwargs.get("elements", ["X"])
        self.num_elements = kwargs.get("num_elements", 1)
        self.nsites = kwargs.get("nsites", 1)
        self.volume = kwargs.get("volume", 10.0)
        self.density = kwargs.get("density", 1.0)
        
        # Elastic properties
        self.bulk_modulus = kwargs.get("bulk_modulus")
        self.shear_modulus = kwargs.get("shear_modulus")
        self.universal_anisotropy = kwargs.get("universal_anisotropy")
        self.homogeneous_poisson = kwargs.get("homogeneous_poisson")
        
        # Additional attributes
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)
    
    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary (compatibility method)."""
        return self._to_dict()
    
    def model_dump(self) -> Dict[str, Any]:
        """Convert to dictionary (Pydantic v2 compatibility)."""
        return self._to_dict()
    
    def _to_dict(self) -> Dict[str, Any]:
        """Internal method to convert all attributes to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if hasattr(value, 'dict'):
                result[key] = value.dict()
            elif hasattr(value, 'model_dump'):
                result[key] = value.model_dump()
            elif hasattr(value, 'as_dict'):
                result[key] = value.as_dict()
            elif isinstance(value, (list, tuple)):
                result[key] = [
                    item.dict() if hasattr(item, 'dict') else
                    item.model_dump() if hasattr(item, 'model_dump') else
                    item.as_dict() if hasattr(item, 'as_dict') else
                    item
                    for item in value
                ]
            else:
                result[key] = value
        return result


class MockComposition:
    """Mock Composition object mimicking pymatgen Composition."""
    
    def __init__(self, comp_dict: Dict[str, float]):
        """Initialize with composition dictionary."""
        self._comp_dict = comp_dict
        self.reduced_formula = self._compute_reduced_formula()
    
    def as_dict(self) -> Dict[str, float]:
        """Return composition as dictionary."""
        return self._comp_dict.copy()
    
    def _compute_reduced_formula(self) -> str:
        """Compute reduced formula (simplified)."""
        return "".join(f"{el}{int(amt)}" if amt != 1 else el 
                      for el, amt in sorted(self._comp_dict.items()))
    
    def __str__(self):
        return self.reduced_formula


class MockModulus:
    """Mock Modulus object with VRH, Voigt, and Reuss values."""
    
    def __init__(self, vrh: float, voigt: Optional[float] = None, reuss: Optional[float] = None, 
                 modulus_type: str = "bulk"):
        """
        Initialize modulus with VRH and optionally Voigt/Reuss bounds.
        
        Args:
            vrh: VRH average value
            voigt: Voigt bound (upper)
            reuss: Reuss bound (lower)
            modulus_type: Either "bulk" (uses k_vrh) or "shear" (uses g_vrh)
        """
        self.vrh = vrh
        self.voigt = voigt if voigt is not None else vrh * 1.05
        self.reuss = reuss if reuss is not None else vrh * 0.95
        self._type = modulus_type
    
    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary matching Materials Project API format."""
        # Real API uses k_vrh/k_voigt/k_reuss for bulk modulus
        # and g_vrh/g_voigt/g_reuss for shear modulus
        if self._type == "bulk":
            return {
                "k_vrh": self.vrh,
                "k_voigt": self.voigt,
                "k_reuss": self.reuss
            }
        else:  # shear
            return {
                "g_vrh": self.vrh,
                "g_voigt": self.voigt,
                "g_reuss": self.reuss
            }
    
    def model_dump(self) -> Dict[str, Any]:
        """Convert to dictionary (Pydantic v2 compatibility)."""
        return self.dict()


class MockElasticityDoc:
    """Mock Elasticity Document from elasticity endpoint."""
    
    def __init__(self, material_id: str, K_VRH: Optional[float] = None, 
                 G_VRH: Optional[float] = None, elastic_tensor: Optional[List] = None,
                 warnings: Optional[List[str]] = None):
        """Initialize elasticity document."""
        self.material_id = material_id
        self.K_VRH = K_VRH
        self.G_VRH = G_VRH
        self.elastic_tensor = elastic_tensor
        self.warnings = warnings or []
    
    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "material_id": self.material_id,
            "K_VRH": self.K_VRH,
            "G_VRH": self.G_VRH,
            "elastic_tensor": self.elastic_tensor,
            "warnings": self.warnings
        }
    
    def model_dump(self) -> Dict[str, Any]:
        """Convert to dictionary (Pydantic v2 compatibility)."""
        return self.dict()


class MockMPRester:
    """Mock MPRester client for testing."""
    
    def __init__(self):
        """Initialize mock client with materials and elasticity endpoints."""
        self.materials = MagicMock()
        self.materials.summary = MagicMock()
        self.materials.elasticity = MagicMock()
        
        # Storage for test data
        self._material_docs = {}
        self._elasticity_docs = {}
    
    def add_material(self, doc: MockMaterialDoc):
        """Add a material document to the mock database."""
        self._material_docs[doc.material_id] = doc
    
    def add_elasticity(self, doc: MockElasticityDoc):
        """Add elasticity data to the mock database."""
        self._elasticity_docs[doc.material_id] = doc
    
    def setup_search_response(self, docs: List[MockMaterialDoc]):
        """Configure summary.search to return specific documents."""
        self.materials.summary.search.return_value = docs
    
    def setup_elasticity_response(self, docs: List[MockElasticityDoc]):
        """Configure elasticity.search to return specific documents."""
        self.materials.elasticity.search.return_value = docs


def create_sample_material(
    material_id: str = "mp-149",
    formula: str = "Si",
    composition: Optional[Dict[str, float]] = None,
    is_stable: bool = True,
    energy_above_hull: float = 0.0,
    bulk_modulus_vrh: Optional[float] = None,
    shear_modulus_vrh: Optional[float] = None,
) -> MockMaterialDoc:
    """
    Factory function to create sample material documents.
    
    Args:
        material_id: Material ID
        formula: Chemical formula
        composition: Composition dictionary (defaults to single element)
        is_stable: Stability flag
        energy_above_hull: Energy above hull in eV/atom
        bulk_modulus_vrh: Bulk modulus VRH value in GPa
        shear_modulus_vrh: Shear modulus VRH value in GPa
        
    Returns:
        MockMaterialDoc instance
    """
    if composition is None:
        # Extract element from formula (simplified)
        import re
        match = re.match(r"([A-Z][a-z]?)", formula)
        element = match.group(1) if match else "X"
        composition = {element: 1.0}
    
    kwargs = {
        "material_id": material_id,
        "formula_pretty": formula,
        "composition": composition,
        "is_stable": is_stable,
        "energy_above_hull": energy_above_hull,
        "chemsys": "-".join(sorted(composition.keys())),
        "elements": list(composition.keys()),
        "num_elements": len(composition),
    }
    
    if bulk_modulus_vrh is not None:
        kwargs["bulk_modulus"] = MockModulus(bulk_modulus_vrh, modulus_type="bulk")
    
    if shear_modulus_vrh is not None:
        kwargs["shear_modulus"] = MockModulus(shear_modulus_vrh, modulus_type="shear")
    
    return MockMaterialDoc(**kwargs)


def create_sample_alloy(
    material_id: str,
    elements: List[str],
    fractions: List[float],
    is_stable: bool = True,
    energy_above_hull: float = 0.0,
    bulk_modulus_vrh: Optional[float] = None,
) -> MockMaterialDoc:
    """
    Factory function to create sample alloy documents.
    
    Args:
        material_id: Material ID
        elements: List of element symbols
        fractions: List of atomic fractions (must sum to 1.0)
        is_stable: Stability flag
        energy_above_hull: Energy above hull in eV/atom
        bulk_modulus_vrh: Bulk modulus VRH value in GPa
        
    Returns:
        MockMaterialDoc instance
    """
    assert len(elements) == len(fractions), "Elements and fractions must have same length"
    assert abs(sum(fractions) - 1.0) < 1e-6, "Fractions must sum to 1.0"
    
    composition = {el: frac for el, frac in zip(elements, fractions)}
    
    # Create simplified formula
    formula_parts = []
    for el, frac in zip(elements, fractions):
        if frac > 0:
            # Convert fraction to approximate integer ratio
            ratio = int(round(frac * 100))
            if ratio > 0:
                formula_parts.append(f"{el}{ratio}" if ratio != 1 else el)
    formula = "".join(formula_parts)
    
    return create_sample_material(
        material_id=material_id,
        formula=formula,
        composition=composition,
        is_stable=is_stable,
        energy_above_hull=energy_above_hull,
        bulk_modulus_vrh=bulk_modulus_vrh,
    )


# Sample material database for common test cases
SAMPLE_MATERIALS = {
    "silicon": create_sample_material(
        material_id="mp-149",
        formula="Si",
        composition={"Si": 1.0},
        bulk_modulus_vrh=97.9,
        shear_modulus_vrh=65.8,
    ),
    "copper": create_sample_material(
        material_id="mp-30",
        formula="Cu",
        composition={"Cu": 1.0},
        bulk_modulus_vrh=137.0,
        shear_modulus_vrh=48.0,
    ),
    "silver": create_sample_material(
        material_id="mp-124",
        formula="Ag",
        composition={"Ag": 1.0},
        bulk_modulus_vrh=100.0,
        shear_modulus_vrh=30.0,
    ),
    "gold_bad_data": create_sample_material(
        material_id="mp-81",
        formula="Au",
        composition={"Au": 1.0},
        bulk_modulus_vrh=180.0,
        shear_modulus_vrh=-5.74,  # Problematic negative shear modulus
    ),
}

