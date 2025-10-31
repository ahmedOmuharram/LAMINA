Superconductors Handler
=======================

The Superconductors handler provides AI functions for analyzing superconducting materials, with a focus on cuprate high-temperature superconductors and the structural factors that influence superconducting properties.

All functions query the Materials Project API for structural data and apply **family-aware** physics-based analysis of crystallographic features relevant to superconductivity.

.. important::

   **C-axis/apical-oxygen correlation is FAMILY-SPECIFIC:**
   
   - **214 (La₂CuO₄)**: ↑c ↔ apical O retained; ↓c ↔ apical O lost (T→T′ transition)
   - **123 (YBCO)**: c tracks **CHAIN** oxygen; deoxygenation **INCREASES** c; apical O on planes retained
   - **Bi/Tl/Hg families**: Large c from block layers; apical present; c doesn't cleanly track apical stability
   - **T′ (Nd₂CuO₄)**: NO apical O by design; c tracks interstitial oxygen reorganization
   - **Infinite-layer**: NO apical O; octahedral stability logic not applicable

Overview
--------

The Superconductors handler provides specialized, family-aware analysis for:

1. **Cuprate Structural Analysis**: Analyze how c-axis spacing affects Cu-O octahedral stability **(family-specific)**
2. **Octahedral Distortion Effects**: Evaluate the impact of Jahn-Teller distortions on electronic structure
3. **Structure-Property Relationships**: Understand how apical oxygen coordination influences superconducting properties
4. **Trend Analysis**: Predict stability changes from hypothetical structural modifications **(with family-specific verdicts)**

Core Analysis Functions
-----------------------

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
       trend_probe: float = 0.01,
       data_source: str = "unknown"
   ) -> Dict[str, Any]

**Description:**

Analyze how c-axis lattice spacing affects copper-oxygen octahedral stability in cuprate superconductors. This function evaluates the correlation between c-axis parameter and apical oxygen retention.

**CRITICAL:** Returns **family-specific verdicts**. The c-axis/apical-oxygen correlation varies dramatically between cuprate families:

- **214-type (La₂CuO₄)**: Increasing c → apical O retained → octahedra stabilized [**TRUE**]
- **123-type (YBCO)**: Increasing c → chain O depleted → worse SC properties [**FALSE** for octahedral stability claim; apical O on planes retained]
- **T′-type (Nd₂CuO₄)**: No apical O by design → octahedral stability not applicable [**AMBIGUOUS**]
- **Infinite-layer**: No apical O → analysis not applicable [**NOT_APPLICABLE**]
- **Bi/Tl/Hg families**: c dominated by block layer stacking → ambiguous apical correlation [**AMBIGUOUS**]

**When to Use:**

- Analyzing cuprate superconductor structures (La₂CuO₄, YBa₂Cu₃O₇, etc.)
- Understanding **family-specific** c-axis effects on octahedral coordination
- Evaluating claims about c-axis expansion effects on stability **(requires knowing the family)**
- Predicting structural stability trends from lattice parameter changes
- Understanding Jahn-Teller distortions in CuO₆ octahedra (214-type)

**When NOT to Use:**

- Predicting superconducting :math:`T_c` (function focuses on structural stability only)
- Analyzing non-cuprate superconductors (Fe-based, MgB₂, etc.) — data available but analysis not implemented
- Quantitative electronic structure calculations (function provides qualitative trends)

**How It Fetches Data:**

1. **Family Detection:**
   
   First, the function automatically detects the cuprate family from the formula:
   
   .. code-block:: python
   
      family = _detect_cuprate_family(material_formula)
      # Returns: "214", "123", "Bi-22n", "Tl-22n", "Hg-12n", 
      #          "T_prime", "infinite_layer", or None
   
   Family detection is critical because different families have **completely different** c-axis behavior.

2. **Materials Project Structure Lookup:**
   
   If Materials Project API access is available, fetches structural data for the cuprate:
   
   .. code-block:: python
   
      docs = mpr.materials.summary.search(
          formula=material_formula,
          fields=["material_id", "formula_pretty", "structure", "symmetry"]
      )
      
      # Extract lattice parameters from structure
      c_from_mp = structure.lattice.c  # in Ångströms
      
      # Detect primitive vs conventional (for 214-type)
      if "la2cuo4" in formula and c_from_mp < 10.0:
          c_conventional = c_from_mp * 2.0  # Primitive → conventional
   
   **Note:** Materials Project DFT values may have ~1–2% systematic error in lattice parameters (PBE functional tends to overestimate slightly). The function adjusts the classification threshold to 2% when using MP data.

3. **Reference Data Selection:**
   
   Retrieves known cuprate structural parameters from ``CUPRATE_DATA`` constants:
   
   .. code-block:: python
   
      CUPRATE_DATA = {
          "La2CuO4": {
              "typical_c": 13.15,  # Å, T-phase with apical O
              "typical_c_tprime": 12.55,  # Å, T′-phase no apical O
              "coordination": "elongated octahedral",
              "family": "214",
              "c_axis_driver": "apical_oxygen",
              "c_increase_means": "apical_retention",
              ...
          },
          "YBa2Cu3O7": {
              "typical_c": 11.68,  # Å, O₇ superconducting
              "typical_c_tetra": 11.82,  # Å, O₆ insulating
              "family": "123",
              "c_axis_driver": "chain_oxygen",  # NOT apical!
              "c_increase_means": "chain_depletion",
              ...
          },
          # Nd₂CuO₄ (T′), CaCuO₂ (infinite-layer), Bi/Tl/Hg families, etc.
      }
   
   Supported cuprate families:
   
   - **214-type**: La₂CuO₄, elongated octahedral CuO₆
   - **123-type**: YBa₂Cu₃O₇, square pyramidal CuO₅ + chains
   - **Bi-22n**: Bi₂Sr₂Ca(n₋₁)CuₙO(2n+4+δ) (n=2,3)
   - **Tl-22n**: Tl₂Ba₂Ca₂Cu₃O₁₀
   - **Hg-12n**: HgBa₂Ca(n₋₁)CuₙO(2n+2+δ) (n=1,2,3; highest :math:`T_c`)
   - **T′-type**: Nd₂CuO₄ (electron-doped, no apical O)
   - **Infinite-layer**: CaCuO₂, Sr₀.₉La₀.₁CuO₂ (no apical O, minimal c)

4. **Family-Specific Stability Rules:**
   
   The function retrieves family-specific c-axis interpretation rules from ``FAMILY_C_AXIS_RULES``:
   
   .. code-block:: python
   
      FAMILY_C_AXIS_RULES = {
          "214": {
              "c_axis_meaning": "Apical oxygen presence",
              "trend": "larger_c_means_apical_retention",
              "verdict_increasing_c": "TRUE",  # Stabilizes octahedra
              "applicable_to_octahedral_stability": True,
          },
          "123": {
              "c_axis_meaning": "Chain oxygen content (NOT apical)",
              "trend": "larger_c_means_chain_depletion",
              "verdict_increasing_c": "FALSE",  # Chain loss, NOT apical
              "applicable_to_octahedral_stability": False,
          },
          "T_prime": {
              "c_axis_meaning": "Interstitial oxygen reorganization",
              "verdict_increasing_c": "AMBIGUOUS",
              "applicable_to_octahedral_stability": False,
          },
          "infinite_layer": {
              "verdict_increasing_c": "NOT_APPLICABLE",
              "applicable_to_octahedral_stability": False,
          },
          # Bi-22n, Tl-22n, Hg-12n: all "AMBIGUOUS"
      }

5. **Analysis Mode Selection:**
   
   The function operates in three modes based on the ``scenario`` parameter:
   
   **Mode 1: Trend Increase (default):**
   
   Analyzes the effect of hypothetically increasing c-axis by a small fraction (default 1%):
   
   .. math::
      
      c_{\text{projected}} &= c_{\text{baseline}} \times (1 + p) \\
      \Delta c_{\text{abs}} &= p \times c_{\text{baseline}} \\
      p &= \text{trend_probe} \quad \text{(default 0.01)}
   
   Returns **family-specific verdict**:
   
   - **214**: ``"TRUE"`` (apical O retained → stabilizes octahedra)
   - **123**: ``"FALSE"`` (chain O depleted → worse SC; apical O on planes retained)
   - **T′/Bi/Tl/Hg**: ``"AMBIGUOUS"`` (c doesn't cleanly track apical O)
   - **Infinite-layer**: ``"NOT_APPLICABLE"`` (no apical O)
   
   **Mode 2: Trend Decrease:**
   
   Same as Mode 1 but with negative probe (:math:`p = -0.01` by default). Verdicts inverted for 214-type.
   
   **Mode 3: Observed:**
   
   Analyzes the actual difference between provided/MP c-axis and typical reference:
   
   .. math::
      
      \Delta c_{\text{abs}} &= c_{\text{used}} - c_{\text{typical}} \\
      \Delta c_{\text{rel}} &= \frac{\Delta c_{\text{abs}}}{c_{\text{typical}}}
   
   Classifies change magnitude using **data-source-aware threshold**:
   
   - **Materials Project data**: threshold ≥ 2% (to account for DFT errors)
   - **Experimental data**: threshold ≥ 1% (default)
   - :math:`|\Delta c_{\text{rel}}| <` threshold → ``"minimal_change"``
   - :math:`\Delta c_{\text{rel}} >` threshold → ``"stabilized"`` (for 214-type) or family-specific
   - :math:`\Delta c_{\text{rel}} <` -threshold → ``"destabilized"`` (for 214-type) or family-specific

6. **Special Family Handling:**
   
   **YBCO (123-type):**
   
   Returns special explanation:
   
      "In YBa₂Cu₃O₇₋δ (123-type), increasing c correlates with CHAIN oxygen depletion 
      (O₇ → O₆), which REDUCES superconducting properties. IMPORTANT: Apical oxygen 
      on the CuO₂ planes is RETAINED; c-axis changes track chain oxygen, not apical 
      oxygen removal. Octahedral stability on planes is NOT affected."
   
   **T′-type (Nd₂CuO₄):**
   
   Returns:
   
      "Nd₂CuO₄ is a T′-type cuprate with NO apical oxygen by design (square planar 
      CuO₄ coordination). C-axis changes reflect interstitial oxygen reorganization 
      during annealing, which can move c in EITHER direction depending on starting 
      oxygen configuration. The octahedral stability question is not applicable."
   
   **Infinite-layer:**
   
   Returns immediate ``NOT_APPLICABLE`` verdict:
   
      "CaCuO₂ is an infinite-layer cuprate with NO apical oxygen by design. 
      Octahedral stability analysis is not applicable. Structure has square planar 
      CuO₄ coordination with minimal c-axis (≈3.2–3.4 Å)."

**Parameters:**

- ``material_formula`` (str, required): Cuprate chemical formula
  
  - Format: ``'La2CuO4'``, ``'YBa2Cu3O7'``, ``'La1.85Sr0.15CuO4'``, ``'Nd2CuO4'``, ``'CaCuO2'``
  - Handles doped formulas: ``'La2-xSrxCuO4'``, ``'Nd1.85Ce0.15CuO4'``, ``'YBa2Cu3O7-δ'``
  - Case-insensitive matching with composition-aware detection
  - If not in known database, provides generic analysis with family-detection attempt

- ``c_axis_spacing`` (float, optional): c-axis lattice parameter in Ångströms
  
  - If ``None``: uses Materials Project value if available, otherwise uses typical literature value
  - If provided: overrides MP lookup and uses given value
  - Should be **conventional cell** c-axis (not primitive)

- ``scenario`` (str, optional): Analysis mode. Default: ``"trend_increase"``
  
  - ``"trend_increase"``: Analyze effect of hypothetical c-axis increase (family-specific verdict)
  - ``"trend_decrease"``: Analyze effect of hypothetical c-axis decrease (family-specific verdict)
  - ``"observed"``: Analyze actual c-axis deviation from typical value

- ``trend_probe`` (float, optional): Fractional change for trend modes. Default: ``0.01`` (1%)
  
  - Used only in trend modes (``"trend_increase"`` or ``"trend_decrease"``)
  - Defines hypothetical :math:`\Delta c / c` for stability assessment
  - Example: ``0.02`` = 2% change

- ``data_source`` (str, optional): Data source type. Default: ``"unknown"``
  
  - ``"MP"``: Materials Project DFT data (auto-set if structure fetched from MP)
  - ``"experiment"``: Experimental XRD data (typical precision)
  - ``"high_precision_xrd"``: High-quality single-crystal XRD
  - ``"unknown"``: Unknown source (conservative threshold)
  - Affects classification threshold: MP/unknown=2%, experiment=1%, high_precision=0.5%

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
           # Common fields (all modes, all families)
           "scenario": str,  # "trend_increase", "trend_decrease", or "observed"
           "material": str,  # Matched cuprate formula
           "family": str,  # "214", "123", "T_prime", "infinite_layer", etc.
           "claim": str,  # The question: "Does increasing c-axis stabilize Cu–O octahedral coordination?"
           "verdict": str,  # "TRUE", "FALSE", "AMBIGUOUS", "NOT_APPLICABLE"
           "coordination": str,  # e.g., "elongated octahedral", "square pyramidal"
           "stability_effect": str,  # "stabilized", "destabilized", "minimal_change", 
                                     # "chain_depleted", "chain_retained", "ambiguous"
           "mechanism": str,  # Detailed explanation of mechanism
           "threshold_used_percent": float,  # Classification threshold used (%)
           "data_source": str,  # "MP", "experiment", "high_precision_xrd", "unknown"
           "c_axis_driver": str,  # "apical_oxygen", "chain_oxygen", "interstitial_oxygen", 
                                  # "BiO_stacking", "not_applicable"
           "applicable_to_octahedral_stability": bool,  # True only for 214-type
           
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
           "threshold_used_percent": float,  # %, threshold applied (2% for MP, 1% for experiment)
           "data_source": str,  # "MP", "experiment", or "unknown"
           
           # Structural details (all modes, if available)
           "structural_details": {
               "typical_apical_distance_A": float,  # Å, Cu-O apical bond (if applicable)
               "typical_planar_distance_A": float,  # Å, Cu-O in-plane bond
               "note": str  # Structural description
           },
           
           # Family-specific warnings (YBCO, T′, infinite-layer)
           "warning": str,  # e.g., "YBCO c-axis tracks CHAIN oxygen, NOT apical oxygen on planes"
           "explanation": str,  # Full family-specific explanation (for special cases)
       },
       # Optional: if Materials Project data fetched
       "materials_project_data": {
           "material_id": str,  # e.g., "mp-1234"
           "c_axis_mp_raw": float,  # Å, as returned by MP
           "c_axis_mp_conventional": float,  # Å, converted to conventional if needed
           "cell_type": str,  # e.g., "primitive (doubled to conventional)"
           "note": str  # MP data usage note
       },
       "confidence": str,  # "MEDIUM" (based on literature trends, not first-principles)
       "citations": List[str],  # Family-specific literature references
       "notes": List[str],  # Analysis assumptions
       "caveats": List[str]  # Limitations
   }

**Side Effects:**

- None (read-only API query if Materials Project accessed)
- No state modification or caching

**Example:**

.. code-block:: python

   # 214-type: Analyze general trend for La₂CuO₄
   result = await handler.analyze_cuprate_octahedral_stability(
       material_formula="La2CuO4",
       scenario="trend_increase",
       trend_probe=0.01  # 1% increase
   )
   # Returns: 
   # family = "214"
   # claim = "Does increasing c-axis stabilize Cu–O octahedral coordination?"
   # verdict = "TRUE"
   # stability_effect = "stabilized"
   # mechanism: "Hypothetical c-axis increase... correlates with retention of apical oxygen..."
   
   # 123-type: Analyze YBCO (gets DIFFERENT verdict)
   result = await handler.analyze_cuprate_octahedral_stability(
       material_formula="YBa2Cu3O7",
       scenario="trend_increase"
   )
   # Returns:
   # family = "123"
   # claim = "Does increasing c-axis stabilize Cu–O octahedral coordination?"
   # verdict = "FALSE"
   # stability_effect = "chain_depleted"
   # warning: "YBCO c-axis tracks CHAIN oxygen on Cu1 sites, NOT apical oxygen on Cu2 plane sites"
   # mechanism: "In YBa₂Cu₃O₇₋δ (123-type), increasing c correlates with CHAIN oxygen depletion..."
   
   # T′-type: Analyze electron-doped cuprate
   result = await handler.analyze_cuprate_octahedral_stability(
       material_formula="Nd2CuO4",
       scenario="trend_increase"
   )
   # Returns:
   # family = "T_prime"
   # verdict = "AMBIGUOUS"
   # applicable_to_octahedral_stability = False
   # explanation: "Nd₂CuO₄ is a T′-type cuprate with NO apical oxygen by design..."
   
   # Infinite-layer: Analysis not applicable
   result = await handler.analyze_cuprate_octahedral_stability(
       material_formula="CaCuO2",
       scenario="trend_increase"
   )
   # Returns:
   # family = "infinite_layer"
   # verdict = "NOT_APPLICABLE"
   # explanation: "CaCuO₂ is an infinite-layer cuprate with NO apical oxygen..."
   
   # Analyze actual structure from Materials Project
   result = await handler.analyze_cuprate_octahedral_stability(
       material_formula="La2CuO4",
       scenario="observed"
       # c_axis_spacing omitted → fetches from MP
   )
   # Returns: data_source = "MP", threshold_used_percent = 2.0 (adjusted for DFT errors)

Physical Background
-------------------

Cuprate Superconductor Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

High-temperature cuprate superconductors are characterized by CuO₂ planes, which are the essential structural element for superconductivity. The coordination environment of copper atoms critically affects electronic structure and :math:`T_c`.

**Family-Specific C-Axis Behavior:**

The c-axis parameter has **completely different physical meaning** across cuprate families:

.. list-table:: C-Axis Behavior by Family
   :header-rows: 1
   :widths: 15 20 30 35

   * - Family
     - c-axis driver
     - Increasing c means
     - Octahedral stability?
   * - **214** (La₂CuO₄)
     - Apical O
     - Apical O retained → stabilizes CuO₆ octahedra
     - **Yes** (T→T′: 13.15→12.55 Å)
   * - **123** (YBCO)
     - **Chain O**
     - Chain O depleted → worse SC (apical on planes retained!)
     - **No** (O₇→O₆: 11.66→11.84 Å; opposite!)
   * - **Bi-22n**
     - BiO stacking
     - Intercalation/stacking faults
     - **No** (large c from BiO layers)
   * - **Tl-22n**
     - TlO stacking
     - Similar to Bi-22n
     - **No**
   * - **Hg-12n**
     - HgO stacking
     - Minimal blocking; cleanest
     - **No**
   * - **T′** (Nd₂CuO₄)
     - Interstitial O
     - Ambiguous (annealing reorganizes O)
     - **No** (no apical O by design)
   * - **Infinite-layer**
     - Not applicable
     - Not applicable
     - **No** (no apical O; c~3.2 Å)

**CuO₆ Octahedral Coordination (214-type only):**

In La₂CuO₄-type cuprates, Cu²⁺ ions are coordinated by six oxygen atoms in an octahedral geometry. However, due to the Jahn-Teller effect (Cu²⁺ has :math:`d^9` electronic configuration), the octahedra are strongly distorted:

.. math::
   
   r_{\text{apical}} &\approx 2.4 \text{ Å} \\
   r_{\text{planar}} &\approx 1.9 \text{ Å} \\
   \frac{r_{\text{apical}}}{r_{\text{planar}}} &\approx 1.25

This elongation along the c-axis places apical oxygen atoms at significantly larger distances than the four in-plane oxygen atoms.

**Electronic Structure:**

The Jahn-Teller distortion lifts the degeneracy of Cu :math:`d` orbitals:

- :math:`d_{x^2-y^2}` orbital (in CuO₂ plane): half-filled, strong in-plane Cu-O :math:`\sigma`-bonding
- :math:`d_{z^2}` orbital (apical direction): higher energy, weaker apical Cu-O overlap
- :math:`d_{xy}`, :math:`d_{xz}`, :math:`d_{yz}` orbitals: non-bonding, lower energy

The electronic behavior near the Fermi level is dominated by the half-filled :math:`d_{x^2-y^2}` band hybridized with O :math:`2p` orbitals in the CuO₂ planes.

C-Axis and Apical Oxygen Correlation (214-Type Only)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Empirical Relationship (La₂CuO₄ Family):**

For **214-type cuprates** (K₂NiF₄ structure), experimental studies demonstrate:

.. math::
   
   \text{Oxygen reduction} \to \text{Apical O removal} \to \text{Decrease in } c

**Evidence (214-type):**

1. **Phase Transformation in La₂CuO₄:**
   
   - **T-phase** (tetragonal, with apical O): :math:`c = 13.15` Å, elongated octahedral CuO₆
   - **T′-phase** (tetragonal, no apical O): :math:`c = 12.55` Å, square planar CuO₄
   - Oxygen reduction converts T → T′ with :math:`\Delta c \approx -0.6` Å (:math:`-4.6\%`)
   
   *Citation*: Yamamoto et al., Physica C 470, 1383 (2010)

2. **Physical Mechanism:**
   
   - Apical oxygen presence extends the unit cell along the c-axis
   - Removal of apical oxygen allows collapse of Cu-O-Cu stacking
   - Shorter c-axis → transition from octahedral to square planar coordination

**Implication for 214-Type:**

.. math::
   
   \uparrow c &\implies \text{Apical O retained} \implies \text{Octahedral stable} \\
   \downarrow c &\implies \text{Apical O lost} \implies \text{Octahedral unstable}

This correlation is used by the handler to assess octahedral stability from c-axis changes **for 214-type cuprates only**.

YBCO (123-Type) Exception
^^^^^^^^^^^^^^^^^^^^^^^^^^

**CRITICAL DIFFERENCE:**

In YBa₂Cu₃O₇₋δ, the c-axis tracks **chain oxygen content**, **NOT** apical oxygen on the CuO₂ planes.

**Oxygen Content Variation:**

.. math::
   
   \text{O}_7 \text{ (fully oxygenated)} &: c = 11.66 \text{ Å}, \, T_c = 92 \text{ K} \\
   \text{O}_6 \text{ (oxygen-depleted)} &: c = 11.84 \text{ Å}, \, T_c = 0 \text{ K (insulator)}

**Increasing c with deoxygenation: OPPOSITE to 214-type!**

*Citation*: web.njit.edu/~tyson/supercon_papers/Oxygen_Content_vs_c-axis.pdf

**Physical Mechanism:**

- Chain oxygen (in Cu-O chains between planes) provides charge carriers
- Removing chain oxygen → loss of hole doping → insulating behavior
- Chain oxygen removal → c-axis **expands** (less interlayer binding)
- **Apical oxygen on the CuO₂ planes is RETAINED** regardless of chain oxygenation

**Implication:**

For YBCO, **increasing c does NOT mean apical oxygen retention**; it means **chain oxygen depletion** and **worse superconducting properties**. The handler returns ``verdict = "FALSE"`` for YBCO when asked if increasing c stabilizes octahedra.

T′-Type and Infinite-Layer
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**T′-type (Nd₂CuO₄, NCCO):**

- **No apical oxygen by design** (square planar CuO₄)
- C-axis changes reflect **interstitial oxygen reorganization** during annealing
- Direction of c-axis change depends on starting oxygen configuration
- Can move **either direction** with annealing
- Octahedral stability question not applicable

*Citation*: Avella & Guarino, Phys. Rev. B 105, 014512 (2022)

**Infinite-layer (CaCuO₂):**

- **No apical oxygen** (single CuO₂ planes with no intervening layers)
- Minimal c-axis (:math:`c \approx 3.2` Å)
- Square planar CuO₄ coordination
- Octahedral stability analysis not applicable

Jahn-Teller Distortion
^^^^^^^^^^^^^^^^^^^^^^^

**Origin:**

Cu²⁺ (:math:`d^9`) in octahedral coordination is Jahn-Teller active. The electronic configuration has an unpaired electron in the :math:`e_g` manifold:

.. math::
   
   t_{2g}^6 e_g^3

**Distortion Modes:**

1. **Elongation (observed in 214-type cuprates):**
   
   .. math::
      
      Q_{3z^2-r^2} > 0 \quad \to \quad r_{\text{axial}} > r_{\text{equatorial}}
   
   Lowers :math:`d_{x^2-y^2}` orbital energy, stabilizes in-plane Cu-O bonding.

2. **Compression (rare in cuprates):**
   
   .. math::
      
      Q_{3z^2-r^2} < 0 \quad \to \quad r_{\text{axial}} < r_{\text{equatorial}}

**Energy Gain:**

Jahn-Teller distortion lowers total energy by **order-of-magnitude tenths of eV** per Cu site, depending on compound and calculation method. The stabilization is essential for cuprate crystal chemistry and directly influences the electronic structure relevant to superconductivity.

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
   - Rising :math:`T_c`
   - Antiferromagnetic correlations

2. **Optimal doping** (:math:`p \approx 0.16`):
   
   - Maximum :math:`T_c` (family-dependent)
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
   
   - :math:`c = 13.15` Å (T-phase), :math:`c = 12.55` Å (T′-phase)
   - :math:`T_c^{\max} = 40` K (with Sr doping)
   - Prototype cuprate, elongated octahedral CuO₆
   - **Family**: 214, **c-axis driver**: apical oxygen

2. **YBa₂Cu₃O₇ (123-type, YBCO):**
   
   - :math:`c = 11.66` Å (O₇), :math:`c = 11.84` Å (O₆)
   - :math:`T_c = 92` K (O₇)
   - Square pyramidal CuO₅ (plane sites) + CuO₄ chains
   - **Family**: 123, **c-axis driver**: **CHAIN oxygen** (NOT apical!)

3. **Bi₂Sr₂CaCu₂O₈ (Bi-2212):**
   
   - :math:`c = 30.6{-}30.9` Å, :math:`T_c^{\max} = 95` K
   - Two CuO₂ planes per unit cell
   - **Family**: Bi-22n, **c-axis driver**: BiO stacking

4. **Bi₂Sr₂Ca₂Cu₃O₁₀ (Bi-2223):**
   
   - :math:`c \approx 37.1` Å, :math:`T_c^{\max} = 110` K
   - Three CuO₂ planes per unit cell

5. **Tl₂Ba₂Ca₂Cu₃O₁₀ (Tl-2223):**
   
   - :math:`c = 35.9` Å, :math:`T_c^{\max} = 125` K
   - Three CuO₂ planes, less disorder than Bi-2223

6. **HgBa₂Ca₂Cu₃O₈ (Hg-1223):**
   
   - :math:`c = 15.76{-}15.82` Å
   - :math:`T_c^{\max} = 134{-}135` K (ambient pressure, **record holder**)
   - :math:`T_c \approx 160{-}164` K under pressure

7. **HgBa₂CuO₄₊δ (Hg-1201):**
   
   - :math:`c = 9.5` Å, :math:`T_c^{\max} = 97` K
   - Single CuO₂ layer, cleanest tetragonal structure

8. **HgBa₂CaCu₂O₆₊δ (Hg-1212):**
   
   - :math:`c = 12.5{-}12.7` Å, :math:`T_c^{\max} = 127` K
   - Double-layer Hg cuprate

9. **Nd₂CuO₄ (T′-type, electron-doped):**
   
   - :math:`c = 12.07` Å, :math:`T_c^{\max} = 24` K (with Ce doping)
   - **NO apical oxygen by design** (square planar CuO₄)
   - **Family**: T_prime, **c-axis driver**: interstitial oxygen

10. **CaCuO₂ (infinite-layer):**
    
    - :math:`c = 3.2` Å
    - **NO apical oxygen** (minimal stacking)
    - **Family**: infinite_layer, **not superconducting at ambient**

11. **Sr₀.₉La₀.₁CuO₂ (infinite-layer, doped):**
    
    - :math:`c = 3.4` Å, :math:`T_c^{\max} = 43` K (thin films)
    - **NO apical oxygen**

**Generic Cuprate Fallback:**

For cuprates not in the reference database, the handler attempts family detection from formula patterns and provides appropriate warnings about family-specific behavior.

Methodology and Data Sources
-----------------------------

**Materials Project DFT Structures:**

When available, the handler fetches crystal structures from the Materials Project database:

- **Method**: DFT with PBE functional, PAW pseudopotentials
- **Structural relaxation**: Full cell and atomic position optimization
- **Lattice parameters**: Extracted from optimized structures
- **Limitations**: DFT may over- or under-estimate lattice parameters by ~1–2%; PBE tends to overestimate slightly

**Threshold Adjustment for DFT Errors:**

The handler automatically adjusts the classification threshold based on the ``data_source`` parameter:

- **Materials Project (DFT)**: 2% threshold (to account for systematic PBE errors)
- **Experimental XRD data**: 1% threshold (typical precision)
- **High-precision XRD**: 0.5% threshold (for high-quality single-crystal data)
- **Unknown source**: 2% threshold (conservative)

**Literature Reference Data:**

Reference c-axis values and structural parameters are from experimental measurements:

- X-ray diffraction (XRD) or neutron diffraction on single crystals or powders
- Typical accuracy: ±0.01 Å for c-axis
- Temperature: Usually room temperature or specific reported conditions

**Analysis Method:**

The handler uses **family-specific empirical correlations** from cuprate literature rather than first-principles calculations:

- **214-type**: :math:`\Delta c \propto` apical oxygen content
- **123-type**: :math:`\Delta c \propto` chain oxygen content (opposite trend)
- **T′-type**: :math:`\Delta c \propto` interstitial oxygen (ambiguous direction)
- **Other families**: c dominated by block layer stacking (ambiguous apical correlation)

Does not perform:

- DFT or quantum chemistry calculations
- Electronic structure computations
- Superconducting property predictions

Provides qualitative stability trends based on structural chemistry.

**Confidence Level:**

Results are assigned **MEDIUM** confidence because:

- Analysis is based on well-established empirical trends in cuprate literature
- No first-principles calculation of octahedral stability
- Does not account for electronic structure effects or doping (beyond structural trends)
- Structural trends are robust but not quantitative predictors of :math:`T_c`
- Family-specific rules are well-documented but have exceptions within families

Citations
---------

All Superconductor functions cite family-specific literature:

**Primary References:**

- **Keimer, B., Kivelson, S. A., Norman, M. R., Uchida, S., & Zaanen, J.** (2015). From quantum matter to high-temperature superconductivity in copper oxides. *Nature*, 518(7538), 179–186. DOI: 10.1038/nature14165
  
  *Comprehensive review of cuprate physics, electronic structure, and superconducting mechanisms.*

- **Pavarini, E., Dasgupta, I., Saha-Dasgupta, T., Jepsen, O., & Andersen, O. K.** (2001). Band-structure trend in hole-doped cuprates and correlation with :math:`T_c^{\max}`. *Physical Review Letters*, 87(4), 047003. DOI: 10.1103/PhysRevLett.87.047003
  
  *Key finding*: Apical oxygen distance tunes axial orbital character and long-range hopping (t′), correlating with :math:`T_c` across families.

- **Ohta, Y., Tohyama, T., & Maekawa, S.** (1991). Apex oxygen and critical temperature in copper oxide superconductors. *Physical Review B*, 43(4), 2968. DOI: 10.1103/PhysRevB.43.2968
  
  *Establishes apical Cu-O distance correlation with :math:`T_c` across cuprate families.*

**214-type (La₂CuO₄) Specific:**

- **Yamamoto, A., Takeshita, N., Terakura, C., & Tokura, Y.** (2010). High pressure effects revisited for the cuprate superconductor family with highest critical temperature. *Physica C: Superconductivity*, 470(20), 1383–1389. DOI: 10.1016/j.physc.2010.05.086
  
  *Key data*: T-phase La₂CuO₄ :math:`c \approx 13.15` Å (with apical O); T′-phase :math:`c \approx 12.55` Å (no apical O).

- **Matsumoto, K., et al.** (2009). Synthesis and characterization of T′-La₂CuO₄. *Physica C*, 469(15–20), 940–943.
  
  *C-axis vs apical oxygen during reduction in La₂CuO₄.*

**123-type (YBCO) Specific:**

- **Jorgensen, J. D., et al.** (1990). Structural properties of oxygen-deficient YBa₂Cu₃O₇₋δ. *Physical Review B*, 41(4), 1863. DOI: 10.1103/PhysRevB.41.1863
  
  *Primary reference for YBCO structure vs oxygen content; establishes c-axis expansion with deoxygenation.*

- **Oxygen determination from cell dimensions in YBCO.** Available at: web.njit.edu/~tyson/supercon_papers/Oxygen_Content_vs_c-axis.pdf
  
  *Establishes O₇ (c≈11.66 Å, superconducting) vs O₆ (c≈11.84 Å, insulating) relationship.*

**T′-type (Electron-Doped) Specific:**

- **Avella, A. & Guarino, A.** (2022). Superconductivity induced by structural reorganization in the electron-doped cuprate Nd₂₋ₓCeₓCuO₄. *Physical Review B*, 105(1), 014512. DOI: 10.1103/PhysRevB.105.014512
  
  *Key finding*: In electron-doped T′ cuprates (NCCO), annealing changes c-axis by reorganizing apical/interstitial oxygen; direction depends on starting oxygen state. Not a universal "oxygen reduction → smaller c" law.

- **Matsumoto, K., et al.** (2009). Reduction dependence of superconductivity in the electron-doped T′-La₂₋ₓCeₓCuO₄₋δ. *Physica C*, 469(15–20), 940–943.
  
  *T′-type reduction dependence; c-axis behavior during oxygen manipulation.*

**Bi/Tl/Hg Families:**

- **RSC Advances** 2, 239 (2012): Bi-2212/Bi-2223 c-axis and stacking.
- **AIP Advances** (2018): Bi₂Sr₂CaCu₂O₈₊ₓ thin films, DOI: 10.1063/1.5009330
- **arXiv:2301.08313**: Tl-2223 c-axis.
- **RSC Advances** 12, 32700 (2022): Hg-1223 structure.
- **arXiv:2401.17079**: Hg-1223 under pressure (Tc ≈ 160–164 K).

**Jahn-Teller Effect:**

- **Bersuker, I. B.** (2006). *The Jahn-Teller Effect*. Cambridge University Press. ISBN: 9780521822121
  
  *Comprehensive theory and applications of Jahn-Teller distortions.*

- **Pavarini, E., et al.** (2004). Jahn-Teller physics and high-:math:`T_c` superconductivity. *Journal of Physics: Condensed Matter*, 16(40), S4313.

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
- Handler auto-detects and converts primitive → conventional when necessary (for :math:`c < 10` Å)

**Family-Aware Analysis:**

Always check the ``family`` field in results to understand which physical mechanism is being analyzed:

.. code-block:: python

   result = await handler.analyze_cuprate_octahedral_stability(material_formula="YBa2Cu3O7")
   
   if result["data"]["family"] == "123":
       # YBCO: c-axis tracks CHAIN oxygen, not apical
       # verdict = "FALSE" for "increasing c stabilizes octahedra"
       pass
   elif result["data"]["family"] == "214":
       # La₂CuO₄: c-axis tracks APICAL oxygen
       # verdict = "TRUE" for "increasing c stabilizes octahedra"
       pass

**Interpretation Guidelines:**

1. **Trend Analysis (default mode):**
   
   - Use for general questions: "Does increasing c stabilize octahedra?"
   - Returns **family-specific verdict** (always about increasing c):
     
     - **214**: ``"TRUE"`` (apical O retained)
     - **123**: ``"FALSE"`` (chain O depleted; apical on planes retained)
     - **T′/Bi/Tl/Hg**: ``"AMBIGUOUS"``
     - **Infinite-layer**: ``"NOT_APPLICABLE"``

2. **Observed Analysis:**
   
   - Use when specific c-axis value is known or fetched from Materials Project
   - Compares actual structure to literature reference
   - Classifies deviation: stabilized, destabilized, or minimal change
   - **Threshold auto-adjusted by data_source**: 2% for MP, 1% for experiment, 0.5% for high-precision

3. **Data Source Selection:**
   
   - **MP**: Auto-set when structure fetched from Materials Project
   - **experiment**: Use when you have typical lab XRD data
   - **high_precision_xrd**: Use for high-quality single-crystal XRD (enables 0.5% threshold for 214-type, where T→T′ shift is ~4–5%)
   - **unknown**: Conservative 2% threshold

**Limitations:**

1. **Empirical Correlation (Not First-Principles):**
   
   - Analysis based on observed trends, not DFT or quantum chemistry calculations
   - Does not compute electronic structure or superconducting properties
   - Cannot predict :math:`T_c` quantitatively

2. **Family-Specific (Not Universal):**
   
   - The "increasing c stabilizes octahedra" claim is **TRUE only for 214-type**
   - **FALSE for 123-type** (YBCO): c tracks chain oxygen, not apical
   - **AMBIGUOUS for Bi/Tl/Hg**: c dominated by block layer stacking
   - **NOT_APPLICABLE for T′ and infinite-layer**: no apical oxygen by design

3. **Simplified Model:**
   
   - Focuses on c-axis as a proxy for oxygen content
   - Does not include:
     
     - Electronic structure effects (band structure, Fermi surface)
     - Doping effects (beyond structural trends)
     - Strain effects
     - Temperature effects (DFT at 0 K, experiments at room temperature)
   
   - Assumes empirical correlations from literature hold within families

4. **DFT Lattice Parameter Errors:**
   
   - Materials Project PBE DFT may have ~1–2% systematic bias in lattice parameters
   - PBE functional tends to overestimate lattice constants slightly
   - Room-temperature experimental values may differ from 0 K DFT
   - Handler adjusts threshold to 2% for MP data to account for this

5. **Superconducting Property Prediction:**
   
   - Handler does **not** predict or analyze superconducting transition temperatures
   - Does **not** compute pairing mechanisms or critical currents
   - Focuses purely on structural stability of octahedral coordination (214-type) or family-specific c-axis behavior

**When to Use Each Analysis Mode:**

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Mode
     - Use Case
     - Example Question
   * - ``trend_increase``
     - General conceptual question (family-aware)
     - "Does increasing c stabilize octahedra in La₂CuO₄?" → **TRUE** (214-type)
   * - ``trend_increase``
     - General conceptual question (YBCO)
     - "Does increasing c stabilize octahedra in YBCO?" → **FALSE** (chain oxygen, not apical)
   * - ``trend_decrease``
     - General conceptual question (opposite)
     - "Does decreasing c destabilize octahedra in La₂CuO₄?" → **TRUE** (toward T′-phase)
   * - ``observed``
     - Specific material analysis
     - "Is this La₂CuO₄ structure with c = 13.20 Å more stable than typical?" → Compare to 13.15 Å baseline

**Error Handling:**

- **Unknown cuprate**: Attempts family detection; if unsuccessful, returns generic analysis with family warnings
- **Materials Project lookup failure**: Uses literature reference values
- **Invalid parameters**: Returns error with ``ErrorType.INVALID_INPUT``
- **Computation errors**: Returns error with ``ErrorType.COMPUTATION_ERROR``

**Future Extensions:**

The handler framework supports addition of:

- Iron-based superconductor analysis (FeSe, Ba₁₋\ :sub:`x`\ K\ :sub:`x`\ Fe₂As₂) — data available in constants
- MgB₂ two-gap superconductor analysis — data available
- Conventional superconductor BCS parameter calculations
- Electronic structure analysis from DFT band structures
- Quantitative :math:`T_c` estimation models (empirical or ML-based)

**Performance:**

- Materials Project API query: ~100–500 ms (network dependent)
- Structure analysis computation: ~10–50 ms
- Total typical response time: ~150–600 ms
- Family detection: < 1 ms (pattern matching)

**Caching:**

- No caching implemented (each call queries Materials Project afresh)
- For repeated queries, consider external caching of MP structure data

**Comparison with Materials Handler:**

The Superconductors handler differs from the Materials handler in scope:

- **Materials handler**: General-purpose Materials Project queries (any property, any material)
- **Superconductors handler**: Specialized **family-aware** analysis of structural factors affecting superconductivity
- **Complementary use**: Use Materials handler to find cuprate materials, then Superconductors handler to analyze family-specific c-axis behavior

