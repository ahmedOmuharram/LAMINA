Alloys Handler
==============

The Alloys handler provides AI functions for analyzing metallic alloys, surface diffusion barriers, and phase-specific mechanical properties using DFT calculations and thermodynamic modeling.

Overview
--------

This handler enables:

1. **Surface Diffusion**: Estimate surface diffusion barriers for adatoms on alloy surfaces
2. **Phase Strength Analysis**: Assess mechanical strength and stiffness claims for specific phases
3. **Structure-Property Relationships**: Connect phase structure to mechanical performance

Functions
---------

.. _estimate_surface_diffusion_barrier:

estimate_surface_diffusion_barrier
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Estimate the surface diffusion barrier for an adatom species on an alloy surface. Uses structure-property relationships and reference data to predict activation energies.

**When to Use:**

- Understanding surface mobility of atoms on alloys
- Predicting sintering or grain growth behavior
- Analyzing surface reactions and catalysis
- Comparing diffusion barriers between different surfaces

**Parameters:**

- ``alloy_composition`` (str, required): Alloy composition (e.g., ``'Al95Cu5'``, ``'FeCrNi'``)
- ``adatom_species`` (str, required): Diffusing species (e.g., ``'Al'``, ``'Cu'``, ``'Fe'``)
- ``surface_facet`` (str, optional): Crystallographic surface (e.g., ``'111'``, ``'100'``, ``'110'``). Default: ``'111'``
- ``temperature`` (float, optional): Temperature in K for reference. Default: 300

**Returns:**

Dictionary containing:

- ``alloy_composition``: Alloy composition analyzed
- ``adatom_species``: Diffusing species
- ``surface_facet``: Surface crystallography
- ``estimated_barrier_eV``: Estimated diffusion barrier (eV)
- ``barrier_range_eV``: Uncertainty range [min, max] (eV)
- ``confidence``: Confidence level (HIGH, MEDIUM, LOW)
- ``method``: Estimation method used
- ``reference_systems``: Comparable reference systems
- ``factors``: Factors affecting the barrier:
  
  - ``binding_energy``: Adatom-surface binding
  - ``coordination``: Surface coordination number
  - ``lattice_mismatch``: Size mismatch effects

- ``notes``: Additional context and caveats

**Example:**

.. code-block:: python

   # Estimate Al diffusion barrier on Al-Cu alloy (111) surface
   result = await handler.estimate_surface_diffusion_barrier(
       alloy_composition="Al95Cu5",
       adatom_species="Al",
       surface_facet="111",
       temperature=600
   )

**Technical Details:**

- Uses Brønsted-Evans-Polanyi (BEP) relationships
- Incorporates binding energy correlations
- Considers surface coordination effects
- Accounts for lattice mismatch
- References experimental and DFT benchmark data

**Physical Factors:**

- **Binding Energy**: Stronger binding → higher barrier
- **Surface Coordination**: Lower coordination → lower barrier
- **Lattice Mismatch**: Larger mismatch → higher barrier
- **Temperature**: Affects kinetic prefactor, not barrier height

**Typical Barriers:**

- Close-packed surfaces (111): 0.05-0.30 eV
- Open surfaces (100): 0.30-0.60 eV
- Step edges: 0.15-0.40 eV
- Rough surfaces: 0.40-0.80 eV

.. _assess_phase_strength_and_stiffness_claims:

assess_phase_strength_and_stiffness_claims
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Assess claims about mechanical strength and stiffness of specific phases in alloy systems. Validates statements like "Phase X provides strength" or "Phase Y increases stiffness" using DFT elastic constants.

**When to Use:**

- Verifying metallurgical claims about phase properties
- Understanding phase contributions to alloy strength
- Evaluating microstructure-property relationships
- Fact-checking statements from literature

**Parameters:**

- ``alloy_system`` (str, required): Alloy system (e.g., ``'Al-Cu'``, ``'Fe-Cr-Ni'``)
- ``phase_name`` (str, required): Phase name (e.g., ``'theta'``, ``'sigma'``, ``'gamma_prime'``)
- ``property_claim`` (str, required): Claimed property (e.g., ``'increases strength'``, ``'provides stiffness'``, ``'hard and brittle'``)
- ``reference_phase`` (str, optional): Comparison phase (e.g., ``'matrix'``, ``'fcc'``). If provided, performs comparative assessment.

**Returns:**

Dictionary containing:

- ``alloy_system``: Alloy system analyzed
- ``phase_name``: Phase evaluated
- ``property_claim``: Claim assessed
- ``verdict``: Verdict on claim (``'SUPPORTED'``, ``'PARTIALLY SUPPORTED'``, ``'CONTRADICTED'``, ``'INSUFFICIENT DATA'``)
- ``confidence``: Confidence in verdict
- ``supporting_evidence``: Evidence supporting verdict:
  
  - ``bulk_modulus``: Bulk modulus (GPa)
  - ``shear_modulus``: Shear modulus (GPa)
  - ``youngs_modulus``: Young's modulus (GPa)
  - ``hardness_estimate``: Estimated hardness
  - ``ductility_indicator``: Pugh ratio or similar

- ``comparison``: If reference_phase provided:
  
  - ``phase_property_value``: Property value for claimed phase
  - ``reference_property_value``: Property value for reference
  - ``relative_change``: Percent difference
  - ``interpretation``: Textual interpretation

- ``reasoning``: Detailed reasoning for verdict
- ``caveats``: Important caveats and limitations

**Example:**

.. code-block:: python

   # Verify that theta phase increases strength in Al-Cu
   result = await handler.assess_phase_strength_and_stiffness_claims(
       alloy_system="Al-Cu",
       phase_name="theta",
       property_claim="increases strength",
       reference_phase="fcc_Al"
   )

**Technical Details:**

- Retrieves DFT elastic constants from Materials Project
- Calculates mechanical property indicators:
  
  - Bulk modulus: resistance to volume change
  - Shear modulus: resistance to shape change
  - Young's modulus: tensile stiffness
  - Pugh ratio (G/B): ductility indicator

- Compares properties against reference phases
- Interprets results in metallurgical context

**Property Indicators:**

- **High Bulk Modulus**: Resists compression
- **High Shear Modulus**: Resists shear deformation, indicates strength
- **High G/B Ratio** (>0.57): Brittle behavior
- **Low G/B Ratio** (<0.57): Ductile behavior
- **High Young's Modulus**: Stiff material

**Verdict Criteria:**

- **SUPPORTED**: Properties strongly support claim
- **PARTIALLY SUPPORTED**: Some support but with caveats
- **CONTRADICTED**: Properties contradict claim
- **INSUFFICIENT DATA**: Inadequate data for assessment

**Example Claims Assessed:**

- "θ phase provides strengthening in Al-Cu alloys"
- "σ phase increases stiffness but reduces ductility"
- "γ′ precipitates harden Ni-base superalloys"
- "Laves phases are hard and brittle"

Citations
---------

All Alloys functions cite:

- **Materials Project**: DFT-calculated elastic constants and formation energies
- **PyMatGen**: Structure analysis and property calculations
- **Surface Science Literature**: Diffusion barrier references

Technical Notes
---------------

**Surface Diffusion:**

- Barriers in electron volts (eV)
- 1 eV ≈ 96.5 kJ/mol ≈ 23 kcal/mol
- Typical barriers: 0.05-0.80 eV depending on surface
- Prefactor A ≈ 10^12 to 10^13 s^-1 (typical)
- Diffusion coefficient: D = A * exp(-Ea/kT)

**Mechanical Properties:**

- Moduli in GPa (GigaPascals)
- Pugh ratio G/B indicates ductility
- Hardness estimated from elastic moduli
- DFT values at 0 K, temperature effects not included

**Limitations:**

- Surface barriers estimated from correlations, not explicit DFT
- Mechanical properties from 0 K elastic constants
- Actual performance depends on microstructure, not just phase properties
- Grain size, defects, and processing affect real-world behavior

Notes
-----

- Diffusion barriers increase with binding energy
- Close-packed surfaces have lowest barriers
- Mechanical property comparisons at same temperature
- Phase identification from Materials Project database
- Elastic constants from DFT perturbation calculations

