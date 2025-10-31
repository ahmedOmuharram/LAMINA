Superconductors Handler
=======================

The Superconductors handler provides AI functions for analyzing superconducting materials, with a focus on cuprate high-temperature superconductors and the structural factors that influence superconducting properties.

All functions query the Materials Project API for structural data and apply physics-based analysis of crystallographic features relevant to superconductivity.

Overview
--------

The Superconductors handler provides specialized analysis for:

1. **Cuprate Structural Analysis**: Analyze how c-axis spacing affects Cu-O octahedral stability in cuprate superconductors
2. **Octahedral Distortion Effects**: Evaluate the impact of Jahn-Teller distortions on electronic structure
3. **Structure-Property Relationships**: Understand how apical oxygen coordination influences superconducting properties
4. **Trend Analysis**: Predict stability changes from hypothetical structural modifications

Core Analysis Function
----------------------

.. _analyze_cuprate_octahedral_stability:

analyze_cuprate_octahedral_stability
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Function Definition:**

.. code-block:: python

   async def analyze_cuprate_octahedral_stability(
       self,
       material_formula: str,
       c_axis_spacing: Optional[float] = None,
       scenario: str = "trend_increase",
       trend_probe: float = 0.01
   ) -> Dict[str, Any]

**Description:**

Analyze how c-axis lattice spacing affects copper-oxygen octahedral stability in cuprate superconductors. This function evaluates the correlation between c-axis parameter and apical oxygen retention, which is critical for understanding octahedral coordination stability and its impact on superconducting properties.

**When to Use:**

- Analyzing cuprate superconductor structures (La₂CuO₄, YBa₂Cu₃O₇, etc.)
- Understanding how lattice parameters affect octahedral coordination
- Evaluating claims about c-axis expansion effects on stability
- Predicting structural stability trends from lattice parameter changes
- Understanding Jahn-Teller distortions in CuO₆ octahedra

**How It Fetches Data:**

1. **Materials Project Structure Lookup:**
   
   If Materials Project API access is available, fetches structural data for the cuprate:
   
   .. code-block:: python
   
      docs = mpr.materials.summary.search(
          formula=material_formula,
          fields=["material_id", "formula_pretty", "structure", "symmetry"]
      )
      
      # Extract lattice parameters from structure
      c_from_mp = structure.lattice.c  # in Ångströms
   
   **Cell Type Detection:**
   
   For La₂CuO₄-like cuprates, automatically detects primitive vs. conventional cells:
   
   - If :math:`c < 10` Å → likely primitive cell, doubles to conventional (:math:`c_{\text{conv}} = 2 \times c_{\text{prim}}`)
   - Otherwise → uses as conventional cell directly
   
   This is necessary because Materials Project may return primitive cells (:math:`c \approx 6.6` Å) while cuprate literature uses conventional cells (:math:`c \approx 13.15` Å).

2. **Reference Data Selection:**
   
   Retrieves known cuprate structural parameters from ``CUPRATE_DATA`` constants:
   
   .. code-block:: python
   
      CUPRATE_DATA = {
          "La2CuO4": {
              "typical_c": 13.15,  # Å, tetragonal T-phase with apical O
              "coordination": "elongated octahedral",
              "apical_distance": 2.4,  # Å, Cu-O apical bond
              "planar_distance": 1.9,  # Å, Cu-O in-plane bond
              "t_c_max": 40.0,  # K, maximum Tc
              ...
          },
          "YBa2Cu3O7": {...},
          "Bi2Sr2CaCu2O8": {...},
          ...
      }
   
   Supports:
   
   - **La₂CuO₄**: 214-type, elongated octahedral CuO₆
   - **YBa₂Cu₃O₇**: 123-type (YBCO), square pyramidal CuO₅
   - **Bi₂Sr₂CaCu₂O₈**: Bi-2212, square pyramidal
   - **Bi₂Sr₂Ca₂Cu₃O₁₀**: Bi-2223, three CuO₂ planes
   - **Tl₂Ba₂Ca₂Cu₃O₁₀**: Tl-2223, :math:`T_c \approx 125` K
   - **HgBa₂Ca₂Cu₃O₈**: Hg-1223, highest :math:`T_c \approx 135` K

3. **Analysis Mode Selection:**
   
   The function operates in three modes based on the ``scenario`` parameter:
   
   **Mode 1: Trend Increase (default):**
   
   Analyzes the effect of hypothetically increasing c-axis by a small fraction (default 1%):
   
   .. math::
      
      c_{\text{projected}} &= c_{\text{baseline}} \times (1 + p) \\
      \Delta c_{\text{abs}} &= p \times c_{\text{baseline}} \\
      p &= \text{trend\_probe} \quad \text{(default 0.01)}
   
   where :math:`c_{\text{baseline}}` is the typical literature value from ``CUPRATE_DATA``.
   
   **Mode 2: Trend Decrease:**
   
   Same as Mode 1 but with negative probe (:math:`p = -0.01` by default).
   
   **Mode 3: Observed:**
   
   Analyzes the actual difference between provided/MP c-axis and typical reference:
   
   .. math::
      
      \Delta c_{\text{abs}} &= c_{\text{used}} - c_{\text{typical}} \\
      \Delta c_{\text{rel}} &= \frac{\Delta c_{\text{abs}}}{c_{\text{typical}}}
   
   Classifies change magnitude using threshold (default 1%):
   
   - :math:`|\Delta c_{\text{rel}}| < 0.01` → minimal change
   - :math:`\Delta c_{\text{rel}} > 0.01` → stabilized (larger c)
   - :math:`\Delta c_{\text{rel}} < -0.01` → destabilized (smaller c)

4. **Stability Assessment:**
   
   Applies the empirical correlation from cuprate literature:
   
   **Literature Consensus:**
   
   - Oxygen reduction removes apical oxygen → decreases c-axis
   - Larger c-axis correlates with retained apical oxygen → stabilizes octahedral coordination
   - Smaller c-axis correlates with apical loss → destabilizes octahedra (toward square planar)
   
   **Mechanism:**
   
   .. math::
      
      \text{Larger } c &\to \text{Apical O retained} \to \text{CuO}_6 \text{ octahedral} \\
      \text{Smaller } c &\to \text{Apical O removed} \to \text{CuO}_4 \text{ square planar}
   
   Example: La₂CuO₄ phases
   
   - **T-phase** (with apical O): :math:`c \approx 13.15` Å, octahedral coordination
   - **T′-phase** (no apical O): :math:`c \approx 12.55` Å, square planar coordination
   - :math:`\Delta c \approx -0.6` Å (:math:`-4.6\%`) upon apical oxygen removal

5. **Verdict Generation:**
   
   Returns stability assessment:
   
   - **Trend mode**: Answer general question "Does increasing c stabilize octahedra?" → ``TRUE``
   - **Observed mode**: Classify actual structural change based on threshold
   
   Includes:
   
   - Stability effect: ``"stabilized"``, ``"destabilized"``, or ``"minimal_change"``
   - Mechanism description with quantitative changes
   - Claim verdict: ``"TRUE"``, ``"FALSE"``, or ``"AMBIGUOUS"``
   - Structural details (bond distances, coordination)
   - Literature citations

**Parameters:**

- ``material_formula`` (str, required): Cuprate chemical formula
  
  - Format: ``'La2CuO4'``, ``'YBa2Cu3O7'``, ``'Bi2Sr2CaCu2O8'``
  - Case-insensitive matching
  - If not in known database, provides generic cuprate analysis

- ``c_axis_spacing`` (float, optional): c-axis lattice parameter in Ångströms
  
  - If ``None``: uses Materials Project value if available, otherwise uses typical literature value
  - If provided: overrides MP lookup and uses given value
  - Should be conventional cell c-axis (not primitive)

- ``scenario`` (str, optional): Analysis mode. Default: ``"trend_increase"``
  
  - ``"trend_increase"``: Analyze effect of hypothetical c-axis increase
  - ``"trend_decrease"``: Analyze effect of hypothetical c-axis decrease
  - ``"observed"``: Analyze actual c-axis deviation from typical value

- ``trend_probe`` (float, optional): Fractional change for trend modes. Default: ``0.01`` (1%)
  
  - Used only in trend modes (``"trend_increase"`` or ``"trend_decrease"``)
  - Defines hypothetical :math:`\Delta c / c` for stability assessment
  - Example: ``0.02`` = 2% change

**Returns:**

Dictionary containing:

.. code-block:: python

   {
       "success": bool,
       "metadata": {
           "handler": str,
           "function": str,
           "timestamp": str,
           "version": str,
           "duration_ms": float
       },
       "data": {
           # Common fields
           "scenario": str,  # "trend_increase", "trend_decrease", or "observed"
           "material": str,  # Matched cuprate formula
           "coordination": str,  # e.g., "elongated octahedral"
           "stability_effect": str,  # "stabilized", "destabilized", or "minimal_change"
           "mechanism": str,  # Detailed explanation of mechanism
           "claim_increasing_c_stabilizes": str,  # "TRUE", "FALSE", or "AMBIGUOUS"
           
           # Trend mode fields (scenario = "trend_increase" or "trend_decrease")
           "baseline_c_axis": float,  # Å, typical literature value
           "projected_c_axis": float,  # Å, hypothetical value after change
           "hypothetical_delta_angstrom": float,  # Å, absolute change
           "hypothetical_delta_percent": float,  # %, relative change
           
           # Observed mode fields (scenario = "observed")
           "c_axis_analyzed": float,  # Å, actual c-axis used
           "c_axis_typical": float,  # Å, reference literature value
           "observed_change_angstrom": float,  # Å, deviation from typical
           "observed_change_percent": float,  # %, relative deviation
           
           # Structural details (all modes)
           "structural_details": {
               "typical_apical_distance_A": float,  # Å, Cu-O apical bond
               "typical_planar_distance_A": float,  # Å, Cu-O in-plane bond
               "note": str  # Structural description
           }
       },
       # Optional: if Materials Project data fetched
       "materials_project_data": {
           "material_id": str,  # e.g., "mp-1234"
           "c_axis_mp_raw": float,  # Å, as returned by MP
           "c_axis_mp_conventional": float,  # Å, converted to conventional if needed
           "cell_type": str,  # e.g., "primitive (doubled to conventional)"
           "note": str  # MP data usage note
       },
       "confidence": str,  # "MEDIUM" (based on literature trends, not DFT)
       "citations": List[str],  # Literature references
       "notes": List[str],  # Analysis assumptions
       "caveats": List[str]  # Limitations
   }

**Side Effects:**

- None (read-only API query if Materials Project accessed)
- No state modification or caching

**Example:**

.. code-block:: python

   # Analyze general trend for La2CuO4
   result = await handler.analyze_cuprate_octahedral_stability(
       material_formula="La2CuO4",
       scenario="trend_increase",
       trend_probe=0.01  # 1% increase
   )
   # Returns: stability_effect = "stabilized", claim_increasing_c_stabilizes = "TRUE"
   
   # Analyze actual structure from Materials Project
   result = await handler.analyze_cuprate_octahedral_stability(
       material_formula="La2CuO4",
       scenario="observed"
       # c_axis_spacing omitted → fetches from MP
   )
   
   # Analyze custom c-axis value
   result = await handler.analyze_cuprate_octahedral_stability(
       material_formula="YBa2Cu3O7",
       c_axis_spacing=11.82,  # Å, oxygen-depleted phase
       scenario="observed"
   )

Physical Background
-------------------

Cuprate Superconductor Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

High-temperature cuprate superconductors are characterized by CuO₂ planes, which are the essential structural element for superconductivity. The coordination environment of copper atoms critically affects electronic structure and :math:`T_c`.

**CuO₆ Octahedral Coordination:**

In La₂CuO₄-type cuprates, Cu²⁺ ions are coordinated by six oxygen atoms in an octahedral geometry. However, due to the Jahn-Teller effect (Cu²⁺ has :math:`d^9` electronic configuration), the octahedra are strongly distorted:

.. math::
   
   r_{\text{apical}} &\approx 2.4 \text{ Å} \\
   r_{\text{planar}} &\approx 1.9 \text{ Å} \\
   \frac{r_{\text{apical}}}{r_{\text{planar}}} &\approx 1.26

This elongation along the c-axis places apical oxygen atoms at significantly larger distances than the four in-plane oxygen atoms.

**Electronic Structure:**

The Jahn-Teller distortion lifts the degeneracy of Cu :math:`d` orbitals:

- :math:`d_{x^2-y^2}` orbital (in CuO₂ plane): half-filled, strong in-plane Cu-O :math:`\sigma`-bonding
- :math:`d_{z^2}` orbital (apical direction): higher energy, weaker apical Cu-O overlap
- :math:`d_{xy}`, :math:`d_{xz}`, :math:`d_{yz}` orbitals: non-bonding, lower energy

The electronic behavior near the Fermi level is dominated by the half-filled :math:`d_{x^2-y^2}` band hybridized with O :math:`2p` orbitals in the CuO₂ planes.

C-Axis and Apical Oxygen Correlation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Empirical Relationship:**

Multiple experimental studies on cuprates demonstrate that:

.. math::
   
   \text{Oxygen reduction} \to \text{Apical O removal} \to \text{Decrease in } c

**Evidence:**

1. **Phase Transformation in La₂CuO₄:**
   
   - **T-phase** (tetragonal, with apical O): :math:`c = 13.15` Å, elongated octahedral CuO₆
   - **T′-phase** (tetragonal, no apical O): :math:`c = 12.55` Å, square planar CuO₄
   - Oxygen reduction converts T → T′ with :math:`\Delta c \approx -0.6` Å

2. **Oxygen Content Variation:**
   
   In YBa₂Cu₃O\ :sub:`7-δ`:
   
   - Fully oxygenated (:math:`\delta = 0`): :math:`c = 11.68` Å, :math:`T_c = 92` K
   - Oxygen-depleted (:math:`\delta = 1`): :math:`c = 11.82` Å, :math:`T_c = 0` K (insulator)
   
   Note: In YBCO, oxygen depletion affects chain oxygen, but general trend of oxygen-c correlation holds.

**Physical Mechanism:**

- Apical oxygen presence extends the unit cell along the c-axis
- Removal of apical oxygen allows collapse of Cu-O-Cu stacking
- Shorter c-axis → transition from octahedral to square planar coordination

**Implication for Stability:**

.. math::
   
   \uparrow c &\implies \text{Apical O retained} \implies \text{Octahedral stable} \\
   \downarrow c &\implies \text{Apical O lost} \implies \text{Octahedral unstable}

This correlation is used by the handler to assess octahedral stability from c-axis changes.

Jahn-Teller Distortion
^^^^^^^^^^^^^^^^^^^^^^^

**Origin:**

Cu²⁺ (:math:`d^9`) in octahedral coordination is Jahn-Teller active. The electronic configuration has an unpaired electron in the :math:`e_g` manifold:

.. math::
   
   t_{2g}^6 e_g^3

**Distortion Modes:**

Two primary distortion modes can occur:

1. **Elongation (observed in cuprates):**
   
   .. math::
      
      Q_{3z^2-r^2} > 0 \quad \to \quad r_{\text{axial}} > r_{\text{equatorial}}
   
   Lowers :math:`d_{x^2-y^2}` orbital energy, stabilizes in-plane Cu-O bonding.

2. **Compression (rare in cuprates):**
   
   .. math::
      
      Q_{3z^2-r^2} < 0 \quad \to \quad r_{\text{axial}} < r_{\text{equatorial}}

**Energy Gain:**

Jahn-Teller distortion lowers total energy by approximately 0.5–1.0 eV per Cu site. The stabilization is essential for cuprate crystal chemistry and directly influences the electronic structure relevant to superconductivity.

**Impact on Superconductivity:**

- Elongation enhances in-plane Cu-O :math:`\sigma^*` antibonding character of the :math:`d_{x^2-y^2}` band
- Optimizes orbital overlap for Zhang-Rice singlet formation
- Weak apical Cu-O bonding provides quasi-2D electronic structure (critical for high :math:`T_c`)

Doping and T\ :sub:`c` Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Cuprate superconductivity requires **hole doping** of the CuO₂ planes. Pure La₂CuO₄ is an antiferromagnetic Mott insulator. Doping (e.g., La₂₋\ :sub:`x`\ Sr\ :sub:`x`\ CuO₄) introduces charge carriers:

.. math::
   
   x &= \text{doping level} \\
   p &= \text{hole concentration per Cu} \approx x

**Doping Regimes:**

1. **Underdoped** (:math:`p < 0.16`):
   
   - Pseudogap phase
   - Low :math:`T_c`
   - Antiferromagnetic correlations

2. **Optimal doping** (:math:`p \approx 0.16`):
   
   - Maximum :math:`T_c`
   - La₂₋\ :sub:`x`\ Sr\ :sub:`x`\ CuO₄: :math:`T_c^{\max} \approx 40` K at :math:`x \approx 0.15`

3. **Overdoped** (:math:`p > 0.19`):
   
   - Fermi liquid behavior
   - Decreasing :math:`T_c`
   - Loss of d-wave pairing strength

**Structural Effects:**

Doping affects lattice parameters:

- Sr²⁺ substitution for La³⁺ introduces holes and increases in-plane Cu-O bond length
- Optimal :math:`T_c` correlates with specific bond lengths and octahedral distortion magnitude

Database Coverage
-----------------

**Cuprate Superconductors:**

The handler has reference data for the following cuprate families:

1. **La₂CuO₄ (214-type):**
   
   - :math:`c = 13.15` Å (T-phase), :math:`T_c^{\max} = 40` K (with Sr doping)
   - Prototype cuprate, elongated octahedral CuO₆
   - Structure type: K₂NiF₄

2. **YBa₂Cu₃O₇ (123-type, YBCO):**
   
   - :math:`c = 11.68` Å, :math:`T_c = 92` K
   - Square pyramidal CuO₅ (plane sites) + CuO₄ chains
   - First cuprate above liquid nitrogen temperature

3. **Bi₂Sr₂CaCu₂O₈ (Bi-2212):**
   
   - :math:`c = 30.89` Å, :math:`T_c^{\max} = 95` K
   - Two CuO₂ planes per unit cell
   - BiO layers cause large c-axis

4. **Bi₂Sr₂Ca₂Cu₃O₁₀ (Bi-2223):**
   
   - :math:`c = 37.1` Å, :math:`T_c^{\max} = 110` K
   - Three CuO₂ planes per unit cell

5. **Tl₂Ba₂Ca₂Cu₃O₁₀ (Tl-2223):**
   
   - :math:`c = 35.9` Å, :math:`T_c^{\max} = 125` K
   - Three CuO₂ planes, less disorder than Bi-2223

6. **HgBa₂Ca₂Cu₃O₈ (Hg-1223):**
   
   - :math:`c = 15.78` Å, :math:`T_c^{\max} = 135` K (ambient pressure)
   - Highest :math:`T_c` at ambient pressure
   - Under pressure: :math:`T_c > 160` K

**Generic Cuprate Fallback:**

For cuprates not in the reference database, the handler provides a generic analysis based on the general principle that c-axis spacing correlates with apical oxygen retention.

Methodology and Data Sources
-----------------------------

**Materials Project DFT Structures:**

When available, the handler fetches crystal structures from the Materials Project database:

- **Method**: DFT with PBE functional, PAW pseudopotentials
- **Structural relaxation**: Full cell and atomic position optimization
- **Lattice parameters**: Extracted from optimized structures
- **Limitations**: DFT may over- or under-estimate lattice parameters by ~1–2%

**Literature Reference Data:**

Reference c-axis values and structural parameters are from experimental measurements:

- X-ray diffraction (XRD) or neutron diffraction on single crystals or powders
- Typical accuracy: ±0.01 Å for c-axis
- Temperature: Usually room temperature or specific reported conditions

**Analysis Method:**

The handler uses **empirical correlations** from cuprate literature rather than first-principles calculations:

- Correlation: :math:`\Delta c \propto` apical oxygen content
- Does not perform DFT or quantum chemistry calculations
- Does not compute electronic structure or superconducting properties
- Provides qualitative stability trends based on structural chemistry

**Confidence Level:**

Results are assigned **MEDIUM** confidence because:

- Analysis is based on well-established empirical trends in cuprate literature
- No first-principles calculation of octahedral stability
- Does not account for electronic structure effects or doping
- Structural trends are robust but not quantitative predictors of :math:`T_c`

Citations
---------

All Superconductor functions cite:

**Primary References:**

- **Avella, A. & Guarino, A.** (2022). Oxygen reduction effects in cuprates. *Physical Review B*, 105(1), 014512. DOI: 10.1103/PhysRevB.105.014512
  
  *Key finding*: "Oxygen reduction produces a decrease of the c-axis parameter associated with the removal of apical oxygen."

- **Yamamoto, A., Takeshita, N., Terakura, C., & Tokura, Y.** (2010). T and T′ phase cuprate structures. *Physica C: Superconductivity*, 470(20), 1383–1389. DOI: 10.1016/j.physc.2010.05.086
  
  *Key data*: T-phase La₂CuO₄ :math:`c \approx 13.15` Å (with apical O); T′-phase :math:`c \approx 12.55` Å (no apical O).

- **Singh, D. K., et al.** (2017). Annealing effects on c-axis and apical oxygen. arXiv:1710.09028
  
  *Key finding*: Decrease of c-axis upon annealing attributed to removal of apical oxygen.

**Cuprate Physics Reviews:**

- **Keimer, B., Kivelson, S. A., Norman, M. R., Uchida, S., & Zaanen, J.** (2015). From quantum matter to high-temperature superconductivity in copper oxides. *Nature*, 518(7538), 179–186. DOI: 10.1038/nature14165
  
  *Comprehensive review of cuprate physics, electronic structure, and superconducting mechanism.*

**Jahn-Teller Effect:**

- **Bersuker, I. B.** (2006). *The Jahn-Teller Effect*. Cambridge University Press. ISBN: 9780521822121
  
  *Theory and applications of Jahn-Teller distortions in transition metal complexes.*

**Materials Project:**

- **Jain, A., et al.** (2013). The Materials Project: A materials genome approach to accelerating materials innovation. *APL Materials*, 1(1), 011002. DOI: 10.1063/1.4812323

- **Ong, S. P., et al.** (2013). Python Materials Genomics (pymatgen). *Computational Materials Science*, 68, 314–319. DOI: 10.1016/j.commatsci.2012.10.028

Notes and Best Practices
-------------------------

**Units:**

- **Lattice parameters**: Ångströms (Å)
- **Bond lengths**: Ångströms (Å)
- **Transition temperatures**: Kelvin (K)
- **Relative changes**: Percent (%) or fractional (dimensionless)

**Cuprate Formula Conventions:**

- Use conventional formulas: ``La2CuO4``, not ``La2Cu1O4``
- Oxygen stoichiometry notation: ``YBa2Cu3O7-δ`` (δ = oxygen deficiency)
- Doping notation: ``La2-xSrxCuO4`` (x = Sr concentration)

**C-Axis Convention:**

- Always use **conventional cell** c-axis (not primitive)
- For La₂CuO₄: conventional :math:`c \approx 13.15` Å, primitive :math:`c \approx 6.6` Å
- Handler auto-detects and converts primitive → conventional when necessary

**Interpretation Guidelines:**

1. **Trend Analysis (default mode):**
   
   - Use for general questions: "Does increasing c stabilize octahedra?"
   - Answers conceptual question independent of specific material instance
   - Returns: ``"TRUE"`` (increasing c stabilizes), based on literature consensus

2. **Observed Analysis:**
   
   - Use when specific c-axis value is known or fetched from Materials Project
   - Compares actual structure to literature reference
   - Classifies deviation: stabilized, destabilized, or minimal change

3. **Threshold Sensitivity:**
   
   - Default threshold: 1% (0.01)
   - Changes below threshold classified as "minimal"
   - Adjust threshold for more/less sensitive classification

**Limitations:**

1. **Empirical Correlation:**
   
   - Analysis based on observed trends, not first-principles calculations
   - Does not compute electronic structure or superconducting properties
   - Cannot predict :math:`T_c` quantitatively

2. **Simplified Model:**
   
   - Focuses on apical Cu-O bond length changes
   - Does not include:
     
     - Electronic structure effects (band structure, Fermi surface)
     - Doping effects
     - Strain effects
     - Temperature effects
   
   - Assumes correlation from literature holds universally

3. **DFT Lattice Parameter Errors:**
   
   - Materials Project DFT may have ~1–2% errors in lattice parameters
   - PBE functional tends to overestimate lattice constants slightly
   - Room-temperature experimental values may differ from 0 K DFT

4. **Superconducting Property Prediction:**
   
   - Handler does **not** predict or analyze superconducting transition temperatures
   - Does **not** compute pairing mechanisms or critical currents
   - Focuses purely on structural stability of octahedral coordination

**When to Use Each Analysis Mode:**

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Mode
     - Use Case
     - Example Question
   * - ``trend_increase``
     - General conceptual question
     - "Does increasing c stabilize octahedra in cuprates?"
   * - ``trend_decrease``
     - General conceptual question (opposite)
     - "Does decreasing c destabilize octahedra?"
   * - ``observed``
     - Specific material analysis
     - "Is this La₂CuO₄ structure with c = 13.20 Å more stable than typical?"

**Error Handling:**

- **Unknown cuprate**: Returns generic analysis with general principle
- **Materials Project lookup failure**: Uses literature reference values
- **Invalid parameters**: Returns error with ``ErrorType.INVALID_INPUT``
- **Computation errors**: Returns error with ``ErrorType.COMPUTATION_ERROR``

**Future Extensions:**

The handler framework supports addition of:

- Iron-based superconductor analysis (FeSe, Ba₁₋\ :sub:`x`\ K\ :sub:`x`\ Fe₂As₂)
- MgB₂ two-gap superconductor analysis
- Conventional superconductor BCS parameter calculations
- Electronic structure analysis from DFT band structures
- Quantitative :math:`T_c` estimation models (empirical or ML-based)

**Performance:**

- Materials Project API query: ~100–500 ms (network dependent)
- Structure analysis computation: ~10–50 ms
- Total typical response time: ~150–600 ms

**Caching:**

- No caching implemented (each call queries Materials Project afresh)
- For repeated queries, consider external caching of MP structure data

**Comparison with Materials Handler:**

The Superconductors handler differs from the Materials handler in scope:

- **Materials handler**: General-purpose Materials Project queries (any property, any material)
- **Superconductors handler**: Specialized analysis of structural factors affecting superconductivity
- **Complementary use**: Use Materials handler to find cuprate materials, then Superconductors handler to analyze octahedral stability

