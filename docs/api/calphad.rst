CALPHAD Handler
===============

The CALPHAD handler provides AI functions for generating phase diagrams, performing thermodynamic equilibrium calculations, and verifying metallurgical claims using CALPHAD (CALculation of PHAse Diagrams) methodology.

All functions use thermodynamic databases (TDB files) to compute phase equilibria via the pycalphad library with adaptive refinement, configurable filtering, and robust solidification modeling.

Overview
--------

The CALPHAD handler is organized into three main categories:

1. **Visualization Functions**: Generate phase diagrams and composition-temperature plots
2. **Calculation Functions**: Compute equilibrium states and phase fractions
3. **Verification Functions**: Validate metallurgical claims and sweep composition ranges

**Key Features:**

- **Adaptive refinement** around phase boundaries (liquidus/solidus/solvus) with bisection to user-specified tolerance
- **Scheil-Gulliver solidification** modeling for realistic as-cast microstructures
- **General invariant detection** (eutectic, peritectic, monotectic, eutectoid, peritectoid)
- **Configurable phase filtering** with production/research/metastable presets
- **Dual-unit support** (K/°C for temperature, at.%/wt.% for composition)
- **Database provenance tracking** with validity range warnings
- **Performance optimization** via equilibrium caching and parallel execution

Core Visualization Functions
-----------------------------

.. _plot_binary_phase_diagram:

plot_binary_phase_diagram
^^^^^^^^^^^^^^^^^^^^^^^^^

**Function Definition:**

.. code-block:: python

   async def plot_binary_phase_diagram(
       self,
       system: str,
       min_temperature: Optional[float] = None,
       max_temperature: Optional[float] = None,
       composition_step: Optional[float] = None,
       figure_width: Optional[float] = None,
       figure_height: Optional[float] = None,
       temperature_unit: Optional[str] = "K",
       composition_unit: Optional[str] = "atomic",
       adaptive_refinement: Optional[bool] = True,
       refinement_tolerance: Optional[float] = 2.0,
       phase_filter_mode: Optional[str] = "production",
       detect_all_invariants: Optional[bool] = True
   ) -> str

**Description:**

**PREFERRED for phase diagram questions.** Generate a binary phase diagram for a chemical system using CALPHAD thermodynamic data. This is the primary tool for understanding phase relationships across composition ranges.

**When to Use:**

- General system queries (e.g., "show me the Al-Zn phase diagram")
- Understanding liquidus/solidus boundaries
- Identifying eutectic points and phase transitions
- Viewing complete composition range behavior

**How It Fetches Data:**

1. **Database Selection with Provenance:**
   
   - Parses system string to extract elements (e.g., "Al-Zn" → ['AL', 'ZN'])
   - Calls ``load_tdb_database([A, B])`` which implements element-based database selection:
     
     - For Al-Mg-Zn systems → COST507.tdb (validated for T: 200-1000 K, composition: 0-100 at.%, includes τ-phase)
     - For systems with C, N, B, Li → COST507.tdb
     - For Al-based systems → mc_al_v2037_pycal.tdb (validated for T: 298-2000 K, multicomponent Al alloys)
     - Checks ``backend/tdbs/`` directory for matching .tdb files
   
   - Validates both elements exist in selected database
   - **Reports database provenance**: source, version, validation ranges, recommended use
   - **Issues warnings** when querying outside validated T or composition ranges
   - **Allows user override** if multiple databases match (with explicit selection)

2. **Phase Selection (Configurable):**
   
   - Calls ``_filter_phases_for_system(db, (A, B), mode=phase_filter_mode)`` to get relevant phases
   - **Three modes**:
     
     - ``'production'`` (default): Excludes esoterica (ION_LIQUID, HALIDE_*, etc.) for cleaner diagrams
     - ``'research'``: Shows all phases including ordered/disordered variants (B2 vs. A2)
     - ``'metastable'``: Includes flagged metastable phases for non-equilibrium analysis
   
   - Returns only phases containing the specified elements
   - **Reports which phases were filtered** and why (for transparency)

**How It Calculates:**

1. **Temperature Range:**
   
   - Auto mode: Uses wide bracket (200-2300 K) to capture high-melting systems
   - Manual mode: Uses user-specified range
   - Handles degenerate case (min==max) by expanding ±100K
   - Initial grid: 12-60 points depending on range
   - **Adaptive refinement** (if enabled): adds points around detected boundaries

2. **Phase Diagram Generation with Adaptive Refinement:**
   
   - Calls pycalphad's ``binplot()`` function with:
     
     - Composition range: X(B) from 0 to 1 with step size (default 0.02)
     - Temperature range with adaptive point count
     - Pressure: 101325 Pa (1 atm, configurable)
     - N: 1 mole
   
   - ``binplot`` internally:
     
     - Computes equilibrium at each (T, X) grid point using Gibbs energy minimization
     - Identifies phase boundaries where phase stability changes
     - Draws phase field regions and boundaries
   
   - **Adaptive refinement engine** (if ``adaptive_refinement=True``):
     
     - **Temperature (T) refinement**:
       
       - Detects regions where phase presence flips (fraction crosses threshold)
       - Identifies steep liquid fraction gradients (temperature derivative spikes)
       - Uses **bisection** to refine liquidus/solidus/solvus to within ``refinement_tolerance`` (default: ±2.0 K)
       - Reports refined boundary temperatures with stated precision (e.g., 654.3 ± 2.0 K)
     
     - **Composition (X) refinement** (2D adaptivity):
       
       - Detects steep or narrow phase fields in composition space
       - Identifies ordered/disordered boundaries (e.g., B2 vs. A2)
       - Locates near-stoichiometric compound boundaries (sharp composition transitions)
       - **Bivariate bisection** around phase-presence flips in both (T, X)
       - Captures razor-thin solvi that fixed ΔX would miss
       - Example: ordered L1₂ field in Al-Cu at specific compositions
     
     - **Adaptive point insertion**:
       
       - Adds concentrated grid points only where needed (not uniform densification)
       - Preserves coarse spacing in smooth single-phase regions
       - Typical refinement: 50-100 base points → 100-200 refined points in complex regions

3. **General Invariant Reaction Detection:**
   
   - Runs coarse equilibrium grid: ``_coarse_equilibrium_grid(db, A, B, phases, (T_lo, T_hi), nx=101, nT=161)``
   
   - **Applies Savitzky-Golay smoothing** carefully:
     
     - Smooths only **within single-phase branches** (no cross-boundary smoothing)
     - Window size: 5-9 points, scaled to local sampling density
     - Avoids washing out sharp kinks at invariants
     - Detects phase-field boundaries first, then smooths within each field independently
   
   - Searches for **zero degrees of freedom** (three-phase coexistence in binary systems)
   
   - **Classifies all invariant types**:
     
     - **Eutectic**: L → α + β (liquid decomposes to two solids on cooling)
     - **Peritectic**: L + α ↔ β (liquid + solid react to form new solid)
     - **Monotectic**: L₁ → L₂ + α (liquid miscibility gap decomposition)
     - **Eutectoid**: γ → α + β (solid decomposes to two solids on cooling)
     - **Peritectoid**: α + β ↔ γ (two solids react to form new solid)
   
   - **Robust detection criteria** (perturb both T and X):
     
     - Identifies candidate 3-phase points (co-occurrence check)
     - **Perturbs both T and X** around candidate to confirm topology:
       
       .. code-block:: python
       
          # At candidate invariant (T₀, X₀) with phases (L, α, β)
          # Perturb T upward: T₀ + ΔT
          phases_above = get_stable_phases(T₀ + 0.5, X₀)
          # Perturb T downward: T₀ - ΔT
          phases_below = get_stable_phases(T₀ - 0.5, X₀)
          # Perturb X left: X₀ - ΔX
          phases_left = get_stable_phases(T₀, X₀ - 0.01)
          # Perturb X right: X₀ + ΔX
          phases_right = get_stable_phases(T₀, X₀ + 0.01)
          
          # Eutectic: L stable above, α+β stable below
          # Peritectic: L+α above, β stable below (or vice versa)
          # Monotectic: L₁+L₂ above (two liquid instances), L₂+α below
       
     - **Explicitly checks for two liquid instances** (LIQUID#1, LIQUID#2) for monotectics:
       
       - Detects liquid miscibility gaps
       - Sums by base name to track total liquid fraction
       - Confirms L₁ → L₂ + α topology
     
     - Uses **curvature-based minima/maxima detection** (not fixed absolute thresholds)
     - Enforces **composition-space separation** (min_spacing scaled by local curvature) to avoid duplicates
     - Cross-validates with phase diagram topology (invariants must lie on phase boundaries)
   
   - Marks all detected invariants on diagram with type-specific annotations and reaction equations
   - Reports invariant temperatures with **refinement tolerance** as uncertainty (e.g., "Eutectic: 654.3 ± 2.0 K at 72.5 at.% Zn")

4. **Analysis Generation:**
   
   - Visual analysis: Examines matplotlib figure and axes properties
   - Thermodynamic analysis: ``_analyze_phase_diagram()`` extracts:
     
     - Pure element melting points (from phase field boundaries at X=0 and X=1, adaptively refined)
     - All invariant reactions (eutectic, peritectic, etc.) with temperatures and compositions
     - Phase transition boundaries (solvus, solidus, liquidus) with precision estimates
     - Database provenance information (source, validation range coverage)

**Parameters:**

- ``system`` (str, required): Chemical system in any format:
  
  - Hyphenated: ``'Al-Zn'``, ``'aluminum-zinc'``
  - Concatenated: ``'AlZn'``
  - Full names: ``'aluminum-zinc'``

- ``min_temperature`` (float, optional): Minimum temperature (in unit specified by ``temperature_unit``). Default: auto-detect (200 K or -73 °C)
- ``max_temperature`` (float, optional): Maximum temperature (in unit specified by ``temperature_unit``). Default: auto-detect (2300 K or 2027 °C)
- ``composition_step`` (float, optional): Composition step size (0-1 range). Default: 0.02
- ``figure_width`` (float, optional): Figure width in inches. Default: 9
- ``figure_height`` (float, optional): Figure height in inches. Default: 6
- ``temperature_unit`` (str, optional): Temperature axis units. Options: ``'K'`` (Kelvin, default) or ``'C'`` (Celsius). Calculations always internal Kelvin; this affects only axis labels/ticks
- ``composition_unit`` (str, optional): Composition axis units. Options: ``'atomic'`` (at.%, default) or ``'weight'`` (wt.%). Explicit in figure title/legend
- ``adaptive_refinement`` (bool, optional): Enable adaptive grid refinement around phase boundaries. Default: ``True``
- ``refinement_tolerance`` (float, optional): Temperature tolerance (K) for bisection refinement of liquidus/solidus/solvus. Default: 2.0 K
- ``phase_filter_mode`` (str, optional): Phase filtering preset. Options:
  
  - ``'production'`` (default): Clean diagrams, hides esoterica
  - ``'research'``: Shows all phases including ordered/disordered variants
  - ``'metastable'``: Includes metastable phases for non-equilibrium analysis

- ``detect_all_invariants`` (bool, optional): Detect all invariant types (eutectic, peritectic, etc.), not just eutectics. Default: ``True``

**Returns:**

Structured result containing:

.. code-block:: python

   {
       "success": True,
       "handler": "calphad",
       "function": "plot_binary_phase_diagram",
       "data": {
           "message": "Successfully generated AL-ZN phase diagram...",
           "system": "AL-ZN",
           "phases": ["FCC_A1", "HCP_A3", "LIQUID"],
           "phases_filtered": ["ION_LIQUID", "HALIDE_AL2F6"],  # What was excluded
           "temperature_range_K": [200.0, 2300.0],
           "temperature_unit": "K",  # or "C" if requested
           "composition_unit": "atomic",  # or "weight"
           "database": {
               "name": "COST507.tdb",
               "source": "COST507 European database",
               "version": "2.0",
               "validation_range": "Al-Zn: T=[200-1000 K], X=[0-1]",
               "in_validation_range": True  # or False with warning
           },
           "key_points": [
               {
                   "type": "pure_melting",
                   "element": "AL",
                   "temperature": 933.5,
                   "uncertainty_K": 2.0
               },
               {
                   "type": "eutectic",
                   "temperature": 654.3,
                   "uncertainty_K": 2.0,
                   "composition_pct": 72.5,
                   "reaction": "LIQUID → FCC_A1 + HCP_A3"
               },
               {
                   "type": "peritectic",
                   "temperature": 550.0,
                   "uncertainty_K": 2.0,
                   "composition_pct": 85.0,
                   "reaction": "LIQUID + HCP_A3 ↔ FCC_A1"
               }
           ],
           "refinement_applied": True,
           "refinement_tolerance_K": 2.0
       },
       "has_image": True,
       "image_url": "http://localhost:8000/static/plots/phase_diagram_AL-ZN_<timestamp>.png",
       "confidence": 0.95,
       "citations": ["pycalphad"],
       "duration_ms": 1234.5
   }

**Side Effects:**

- Saves PNG image to ``interactive_plots/`` directory
- Image served at ``http://localhost:8000/static/plots/[filename]``
- Stores metadata in ``_last_image_metadata`` for later analysis
- **Caches equilibrium results** by (T, X, phases, DB hash) in ``_equilibrium_cache`` for reuse
- Parallel execution may spawn multiple process workers for independent (T,X) points
- Cached data persists across function calls within session for performance

**Mathematical Background:**

The phase diagram is computed by minimizing the total Gibbs free energy at each (T, X) point:

.. math::

   G^{total} = \sum_{\phi} f^{\phi} G^{\phi}(T, X^{\phi})

where :math:`f^{\phi}` is the phase fraction and :math:`G^{\phi}` is the molar Gibbs energy of phase :math:`\phi`.

At equilibrium:

- Chemical potentials are equal across all phases: :math:`\mu_i^{\alpha} = \mu_i^{\beta}` for all components i
- Phase fractions satisfy: :math:`\sum_{\phi} f^{\phi} = 1`
- Mass balance: :math:`X_i = \sum_{\phi} f^{\phi} X_i^{\phi}`

**Example:**

.. code-block:: python

   # Generate Al-Zn phase diagram with auto temperature range and all defaults
   result = await handler.plot_binary_phase_diagram(
       system="Al-Zn"
   )
   # Uses adaptive refinement, detects all invariants, production filter mode
   
   # Generate Fe-Al phase diagram in Celsius with research-level detail
   result = await handler.plot_binary_phase_diagram(
       system="Fe-Al",
       min_temperature=500,
       max_temperature=1800,
       temperature_unit="C",  # Display in Celsius
       composition_step=0.01,  # Finer composition resolution
       phase_filter_mode="research"  # Show ordered/disordered variants
   )
   
   # High-precision diagram with tight refinement tolerance
   result = await handler.plot_binary_phase_diagram(
       system="Al-Cu",
       adaptive_refinement=True,
       refinement_tolerance=1.0,  # Refine to ±1.0 K (tighter than default 2.0 K)
       detect_all_invariants=True  # Find eutectics, peritectics, eutectoids, etc.
   )
   
   # Include metastable phases for non-equilibrium analysis
   result = await handler.plot_binary_phase_diagram(
       system="Al-Si",
       phase_filter_mode="metastable",  # Include metastable phases
       composition_unit="weight"  # Display in wt.% instead of at.%
   )

.. _plot_composition_temperature:

plot_composition_temperature
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Function Definition:**

.. code-block:: python

   async def plot_composition_temperature(
       self,
       composition: str,
       min_temperature: Optional[float] = None,
       max_temperature: Optional[float] = None,
       composition_type: Optional[str] = None,
       figure_width: Optional[float] = None,
       figure_height: Optional[float] = None,
       interactive: Optional[str] = "html",
       temperature_unit: Optional[str] = "K",
       adaptive_refinement: Optional[bool] = True,
       refinement_tolerance: Optional[float] = 2.0,
       phase_presence_threshold: Optional[float] = 0.01,
       phase_filter_mode: Optional[str] = "production"
   ) -> str

**Description:**

**PREFERRED for composition-specific thermodynamic questions.** Plot phase stability versus temperature for a specific composition. Shows which phases are stable at different temperatures for a fixed composition using stacked area plots.

**When to Use:**

- Analyzing specific compositions (e.g., "Al20Zn80", "pure Al")
- Understanding melting point of specific alloys
- Identifying phase transitions for a composition
- Visualizing phase stability ranges and precipitation behavior

**How It Fetches Data:**

1. **Composition Parsing:**
   
   - Calls ``_parse_composition(composition, composition_type)`` which:
     
     - Extracts elements and their percentages from string
     - Supports formats: "Al20Zn80", "Zn30Al70", "Al", "Zn"
     - Converts weight% to atomic% if needed using atomic masses
     - Returns (elements_tuple, mole_fraction, composition_type)

2. **Database Loading:**
   
   - Same as ``plot_binary_phase_diagram``: calls ``load_tdb_database([A, B])``
   - Validates elements exist in database using ``get_db_elements(db)``

3. **Phase Selection:**
   
   - Calls ``_filter_phases_for_system(db, (A, B))``
   - Returns relevant phases for the binary system

**How It Calculates:**

1. **Temperature Array Generation with Adaptive Refinement:**
   
   - Auto mode: T_lo=200 K, T_hi=2300 K (wide bracket)
   - Manual mode: uses user-specified range
   - Handles degenerate min==max case by expanding ±100K
   - **Initial coarse grid**: ``n_temp = max(50, min(200, int((T_hi - T_lo) / 5)))``
     
     - 50-200 temperature points depending on range
     - ~5K initial spacing for broad coverage
   
   - **Adaptive refinement** (if enabled):
     
     - After coarse scan, detects phase transitions (where phase presence flips)
     - Identifies steep gradients in phase fractions (temperature derivative spikes)
     - Uses **bisection** to nail liquidus/solidus to within ``refinement_tolerance`` (±2.0 K default)
     - Adds concentrated points around narrow stability ranges
     - Reports phase transition temperatures with stated precision (e.g., "liquidus: 654.3 ± 2.0 K")

2. **Equilibrium Calculations at Each Temperature (with Caching):**
   
   For each temperature T in the array:
   
   - Builds composition dictionary: ``{A: 1-xB, B: xB}``
   - Calls ``compute_equilibrium(db, [A, B], phases, composition_dict, T)``
   - ``compute_equilibrium`` internally:
     
     - Adds 'VA' to elements list (required for vacancies in solid solutions)
     - Builds pycalphad conditions:
       
       .. code-block:: python
       
          conditions = {
              v.T: temperature,
              v.P: 101325,  # 1 atm
              v.N: 1.0,     # 1 mole total
              v.X(B): xB    # Mole fraction of element B (N-1 constraints for N elements)
          }
     
     - Calls ``equilibrium(db, elements_with_va, phases, conditions)``
     - PyCalphad performs Gibbs energy minimization at this (T,X) point
   
   - Extracts phase fractions using ``extract_phase_fractions_from_equilibrium(eq, tolerance=1e-4)``
   - ``extract_phase_fractions`` handles multi-vertex results (two-phase regions):
     
     .. code-block:: python
     
        # Group by phase and sum over vertex dimension
        frac_by_phase = eqp['NP'].groupby(eqp['Phase']).sum(dim='vertex')
        # Also average over 'points'/'samples' dimensions if present
        # Defensively handles v.N ≠ 1 (though v.N=1 is used)
   
   - Stores fractions for each phase: ``phase_data[phase].append(fraction)``
   - **Phase presence filtering**: Uses ``phase_presence_threshold`` (default 0.01 = 1%)
     
     - User-tunable and context-aware
     - Raw fractions always stored; threshold only affects presence verdict
     - Recommended: 0.001 (0.1%) for precipitates, 0.01 (1%) for major constituents

3. **Plot Generation:**
   
   **Interactive HTML (default):**
   
   - Calls ``_create_interactive_plot(temps, phase_data, A, B, xB)``
   - Creates Plotly stacked area chart using ``go.Figure()``
   - Each phase is a filled area trace with:
     
     - x-axis: Temperature (K)
     - y-axis: Cumulative phase fraction (stacked)
     - Hover info: Temperature, phase name, fraction
   
   - Exports to HTML with CDN plotlyjs: ``fig.to_html(include_plotlyjs='cdn')``
   - Saves HTML to ``interactive_plots/`` directory
   
   **PNG Export:**
   
   - Attempts Plotly PNG export: ``_save_plotly_figure_as_png(fig, filename)``
   - Fallback to matplotlib if Plotly export fails:
     
     - ``_create_matplotlib_stackplot(temps, phase_data, comp_label, figure_size)``
     - Uses ``ax.stackplot(temps, *phase_arrays)`` for stacked area plot
     - Saves to PNG via matplotlib backend

4. **Analysis Generation:**
   
   - Calls ``_analyze_composition_temperature(phase_data, xB, temp_range, A, B)``
   - Extracts key information:
     
     - Phase stability ranges (where fraction > 0.01)
     - Melting behavior (liquid phase fraction)
     - Phase transitions (where phases appear/disappear)
     - Solidification sequence

**Parameters:**

- ``composition`` (str, required): Specific composition in various formats:
  
  - Element-percentage: ``'Al20Zn80'`` (20 at.% Al, 80 at.% Zn)
  - Alternative format: ``'Zn30Al70'``
  - Single elements: ``'Zn'``, ``'Al'``

- ``min_temperature`` (float, optional): Minimum temperature (in unit specified by ``temperature_unit``). Default: auto (200 K or -73 °C)
- ``max_temperature`` (float, optional): Maximum temperature (in unit specified by ``temperature_unit``). Default: auto (2300 K or 2027 °C)
- ``composition_type`` (str, optional): 
  
  - ``'atomic'`` (default): Atomic percent (at.%)
  - ``'weight'``: Weight percent (wt.%, automatically converted to mole fractions)

- ``figure_width`` (float, optional): Figure width in inches. Default: 8 (HTML), 10 (matplotlib)
- ``figure_height`` (float, optional): Figure height in inches. Default: 6
- ``interactive`` (str, optional): Output mode. Default: ``'html'``
  
  - ``'html'``: Generates interactive Plotly HTML with static PNG export
  - Other values: Static matplotlib plot only

- ``temperature_unit`` (str, optional): Temperature axis units. Options: ``'K'`` (Kelvin, default) or ``'C'`` (Celsius)
- ``adaptive_refinement`` (bool, optional): Enable adaptive grid refinement around phase transitions. Default: ``True``
- ``refinement_tolerance`` (float, optional): Temperature tolerance (K) for bisection refinement. Default: 2.0 K
- ``phase_presence_threshold`` (float, optional): Minimum phase fraction to consider "present" (0-1). Default: 0.01 (1%). Use 0.001 (0.1%) for precipitates
- ``phase_filter_mode`` (str, optional): Phase filtering preset (``'production'``, ``'research'``, ``'metastable'``). Default: ``'production'``

**Returns:**

Structured result containing:

.. code-block:: python

   {
       "success": True,
       "handler": "calphad",
       "function": "plot_composition_temperature",
       "data": {
           "message": "Generated phase stability plot for AL20ZN80...",
           "composition": "AL20ZN80",
           "system": "AL-ZN",
           "temperature_range_K": [200.0, 2300.0],
           "composition_type": "atomic",
           "interactive_html_url": "http://localhost:8000/static/plots/..."
       },
       "has_image": True,
       "image_url": "http://localhost:8000/static/plots/composition_stability_AL20ZN80_<timestamp>.png",
       "has_html": True,
       "html_url": "http://localhost:8000/static/plots/composition_stability_AL20ZN80_<timestamp>.html",
       "confidence": 0.95,
       "citations": ["pycalphad"],
       "duration_ms": 2345.6,
       "notes": ["📊 View interactive plot at ... for hover details and zoom"]
   }

**Side Effects:**

- Saves PNG to ``interactive_plots/`` directory
- Saves interactive HTML to ``interactive_plots/`` directory (if ``interactive='html'``)
- Both files served at ``http://localhost:8000/static/plots/[filename]``
- Stores metadata in ``_last_image_metadata`` including analysis

**Mathematical Background:**

At each temperature point T, the equilibrium phase fractions are computed by minimizing Gibbs energy subject to constraints:

.. math::

   \min_{f^{\phi}, X_i^{\phi}} G^{total} = \sum_{\phi} f^{\phi} \sum_i X_i^{\phi} \mu_i^{\phi}(T, X^{\phi})

Subject to:

.. math::

   \sum_{\phi} f^{\phi} = 1 \quad \text{(phase fraction constraint)}

.. math::

   \sum_{\phi} f^{\phi} X_i^{\phi} = X_i^{global} \quad \text{(mass balance for each element i)}

.. math::

   \mu_i^{\alpha} = \mu_i^{\beta} \quad \forall \alpha, \beta \quad \text{(chemical equilibrium)}

The stacked area plot shows :math:`f^{\phi}(T)` for each phase φ.

**Example:**

.. code-block:: python

   # Plot phase stability for Al-20Zn alloy (interactive)
   result = await handler.plot_composition_temperature(
       composition="Al20Zn80",
       min_temperature=300,
       max_temperature=900,
       composition_type="atomic"
   )
   
   # Plot for pure aluminum (melting point determination)
   result = await handler.plot_composition_temperature(
       composition="Al",
       min_temperature=800,
       max_temperature=1100
   )
   
   # Plot with weight percent input
   result = await handler.plot_composition_temperature(
       composition="Al30Zn70",  # 30 wt% Al, 70 wt% Zn
       composition_type="weight"
   )

.. _analyze_last_generated_plot:

analyze_last_generated_plot
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Function Definition:**

.. code-block:: python

   async def analyze_last_generated_plot(self) -> str

**Description:**

Analyze and interpret the most recently generated phase diagram or composition plot. Provides detailed analysis of visual features, phase boundaries, and thermodynamic insights based on cached metadata.

**When to Use:**

- After generating a plot with ``plot_binary_phase_diagram()`` or ``plot_composition_temperature()``
- Understanding key features of a phase diagram without regenerating it
- Extracting quantitative information from previously generated plots
- Getting AI interpretation of plot features

**How It Fetches Data:**

1. **Metadata Retrieval:**
   
   - Accesses ``self._last_image_metadata`` attribute set by previous plot generation
   - No database loading or new calculations required
   - Returns error if no plot has been generated in current session

2. **Analysis Components:**
   
   - ``visual_analysis``: Description of plot appearance, colors, and visual elements
   - ``thermodynamic_analysis``: Scientific interpretation from ``_analyze_phase_diagram()``
   - ``combined_analysis``: Merged visual + thermodynamic information

**How It Calculates:**

This function does **not** perform new calculations. It retrieves pre-computed analysis from:

1. **Visual Analysis** (generated during plot creation):
   
   - Matplotlib figure properties (size, DPI, layout)
   - Axes configuration (x-limits, y-limits, labels)
   - Number of plot elements and colors
   - Legend content

2. **Thermodynamic Analysis** (generated during plot creation):
   
   For binary phase diagrams (``plot_binary_phase_diagram``):
   
   - Pure element melting points (extracted from liquidus at X=0 and X=1)
   - Eutectic points (from ``_find_eutectic_points()`` detection):
     
     - Temperature minimum in liquidus curve
     - Three-phase invariant reactions (L → α + β)
     - Composition at eutectic
   
   - Phase field regions and boundaries
   - Temperature ranges for each phase
   
   For composition-temperature plots (``plot_composition_temperature``):
   
   - Melting/freezing temperature (where LIQUID appears/disappears)
   - Solidus temperature (last liquid disappears)
   - Liquidus temperature (first liquid appears)
   - Phase stability ranges for each phase
   - Precipitation sequences upon cooling

**Parameters:** None

**Returns:**

Structured result containing:

.. code-block:: python

   {
       "success": True,
       "handler": "calphad",
       "function": "analyze_last_generated_plot",
       "data": {
           "message": "Combined visual and thermodynamic analysis...",
           "analysis": "Full analysis text",
           "visual_analysis": "Visual description...",
           "thermodynamic_analysis": "Scientific interpretation...",
           "image_data_available": False  # True if image bytes still in memory
       },
       "citations": ["pycalphad"],
       "confidence": 0.95,
       "notes": ["Image data cleared to save memory"] # if applicable
   }

**Side Effects:**

- None (read-only operation)
- Does not regenerate or modify plots

**Example:**

.. code-block:: python

   # Generate a phase diagram first
   await handler.plot_binary_phase_diagram(system="Al-Zn")
   
   # Then analyze it
   result = await handler.analyze_last_generated_plot()
   # Returns detailed analysis including eutectic points, melting temperatures, etc.
   
   # Generate composition plot
   await handler.plot_composition_temperature(composition="Al30Zn70")
   
   # Analyze the composition plot
   result = await handler.analyze_last_generated_plot()
   # Returns analysis of phase stability vs temperature for this composition

**Technical Details:**

- Metadata persists in memory until:
  
  - A new plot is generated (replaces old metadata)
  - Handler instance is destroyed
  - Session ends

- Image data (base64-encoded PNG) may be cleared after display to save memory
- Analysis is deterministic and reproducible (doesn't depend on random factors)
- Stored metadata structure:
  
  .. code-block:: python
  
     {
         "system": "AL-ZN",
         "phases": ["FCC_A1", "HCP_A3", "LIQUID"],
         "temperature_range_K": (200, 2300),
         "description": "Phase diagram for AL-ZN system",
         "analysis": "combined_analysis_text",
         "visual_analysis": "visual_description",
         "thermodynamic_analysis": "scientific_interpretation",
         "image_info": {
             "format": "png",
             "url": "http://localhost:8000/static/plots/..."
         }
     }

Calculation Functions
---------------------

.. _calculate_equilibrium_at_point:

calculate_equilibrium_at_point
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Function Definition:**

.. code-block:: python

   async def calculate_equilibrium_at_point(
       self,
       composition: str,
       temperature: float,
       composition_type: Optional[str] = "atomic",
       include_sublattice: Optional[bool] = False,
       phase_presence_threshold: Optional[float] = 0.0001,
       phase_filter_mode: Optional[str] = "production"
   ) -> str

**Description:**

Calculate thermodynamic equilibrium phase fractions at a specific temperature and composition. This provides detailed quantitative information about which phases are stable and their amounts at a single point in phase space.

**When to Use:**

- Determining exact phase fractions at specific conditions
- Verifying equilibrium state at a point (e.g., "What phases exist at 700K for Al-30Si-55C?")
- Getting detailed composition of each phase
- Confirming predicted microstructures from phase diagrams

**How It Fetches Data:**

1. **Composition Parsing:**
   
   - Calls ``_parse_multicomponent_composition(composition, composition_type)``
   - Supports formats: "Al30Si55C15", "Fe70Cr20Ni10", "Al80Zn20"
   - Extracts element symbols and percentages using regex
   - If ``composition_type='weight'``:
     
     - Converts weight% to atomic% using atomic masses
     - Formula: :math:`X_i^{at} = \frac{w_i/M_i}{\sum_j w_j/M_j}`
     - where :math:`w_i` is weight fraction, :math:`M_i` is atomic mass
   
   - Normalizes to sum to 1.0 (mole fractions)
   - Returns dictionary: ``{element: mole_fraction}``

2. **Database Selection:**
   
   - Extracts elements from composition dictionary
   - Calls ``load_tdb_database(elements)`` with element-based selection:
     
     - For systems with C, N, B, Li → COST507.tdb
     - For Al-based ternaries → mc_al_v2037_pycal.tdb or COST507.tdb
     - Binary systems → element-specific databases
   
   - Returns error if no database found for system

3. **Phase Selection:**
   
   - Binary (2 elements): ``_filter_phases_for_system(db, tuple(elements))``
   - Multicomponent (3+ elements): ``_filter_phases_for_multicomponent(db, elements)``
   - Excludes unwanted patterns (ION_LIQUID, HALIDE_*, etc.)
   - Returns list of phase names to consider

**How It Calculates:**

1. **Equilibrium Calculation:**
   
   - Builds pycalphad conditions:
     
     .. code-block:: python
     
        conditions = {
            v.T: temperature,      # Temperature in K
            v.P: 101325,           # 1 atm pressure
            v.N: 1.0               # 1 mole total
        }
        # Add N-1 composition constraints for N elements
        for elem in elements[1:]:
            conditions[v.X(elem)] = comp_dict[elem]
   
   - Calls ``equilibrium(db, elements_with_va, phases, conditions)``
   - PyCalphad performs Gibbs energy minimization:
     
     .. math::
     
        \min G^{total} = \sum_{\phi} f^{\phi} G^{\phi}(T, P, X^{\phi})
     
     Subject to:
     
     - :math:`\sum_{\phi} f^{\phi} = 1` (phase fractions sum to 1)
     - :math:`\sum_{\phi} f^{\phi} X_i^{\phi} = X_i^{global}` (mass balance)
     - :math:`\mu_i^{\alpha} = \mu_i^{\beta}` ∀ i, α, β (chemical equilibrium)

2. **Phase Fraction Extraction:**
   
   - Calls ``extract_phase_fractions_from_equilibrium(eq, tolerance=1e-4)``
   - Handles multi-vertex results (two-phase regions):
     
     .. code-block:: python
     
        # Group by phase and sum over vertex dimension
        frac_by_phase = eqp['NP'].groupby(eqp['Phase']).sum(dim='vertex')
   
   - Filters out phases with fraction < 0.0001 (0.01%)
   - Returns dictionary: ``{phase_name: fraction}``

3. **Phase Composition Extraction (with Optional Sublattice):**
   
   For each stable phase:
   
   - **Global atomic fractions** (always computed):
     
     - Extracts composition using ``eqp['X'].sel(component=elem)``
     - Masks data for specific phase: ``x_data.where(phase_mask)``
     - Averages over vertices: ``x_val = float(x_data.mean().values)``
     - Stores: ``phase_comp[elem] = x_val`` (global mole fraction of elem in phase)
   
   - **Sublattice occupancies** (if ``include_sublattice=True``):
     
     - Extracts site fractions from ``eqp['Y']`` (sublattice dimension)
     - For multi-sublattice intermetallics (e.g., τ-phase with (Al,Zn)ₐ(Mg)ᵦ sublattice model):
       
       - Reports which element occupies which sublattice site
       - Example: ``{'sublattice_0': {'AL': 0.6, 'ZN': 0.4}, 'sublattice_1': {'MG': 1.0}}``
       - Useful for distinguishing Al vs. Zn occupancy in τ-phase
     
     - **Graceful fallback** when sublattice data unavailable:
       
       .. code-block:: python
       
          if 'Y' in eq.coords and phase in eq.Phase:
              # Extract sublattice site fractions
              sublattice_data = extract_site_fractions(eq, phase)
          else:
              # Gracefully handle absence
              sublattice_data = None
              note = "No sublattice data available (TDB may lack site fraction model)"
       
       - Returns ``None`` rather than zeros or errors
       - Adds note to output: "Sublattice data not available for this phase"
       - Only phases with explicit ``(site1,site2,...)`` sublattice models in TDB have 'Y' data
     
     - Returned in ``sublattice_composition`` field of phase info (or ``None``)

4. **Phase Name Mapping:**
   
   - Calls ``map_phase_name(phase)`` to convert database names to readable forms:
     
     - CSI → SiC
     - AL4C3 → Al4C3
     - FCC_A1 → FCC_A1 (kept as is)
     - MGZN2 → MgZn2
     - TAU_MG32(AL_ZN)49 → Tau

**Parameters:**

- ``composition`` (str, required): Composition as element-number pairs:
  
  - Binary: ``'Al80Zn20'`` (80 at.% Al, 20 at.% Zn)
  - Ternary: ``'Al30Si55C15'`` (30% Al, 55% Si, 15% C)
  - Multi-element: ``'Fe70Cr20Ni10'``
  - Numbers are interpreted as percentages

- ``temperature`` (float, required): Temperature in Kelvin

- ``composition_type`` (str, optional): 
  
  - ``'atomic'`` (default): Atomic/mole percent
  - ``'weight'``: Weight percent (converted internally to mole fractions)

- ``include_sublattice`` (bool, optional): Include sublattice site fractions in results. Default: ``False``
  
  - When ``True``, reports element occupancy on each sublattice site
  - Useful for ordered intermetallics with complex sublattice models
  - Only available when TDB has explicit sublattice definition

- ``phase_presence_threshold`` (float, optional): Minimum phase fraction to report (0-1). Default: 0.0001 (0.01%)
- ``phase_filter_mode`` (str, optional): Phase filtering preset (``'production'``, ``'research'``, ``'metastable'``). Default: ``'production'``

**Returns:**

Structured result containing:

.. code-block:: python

   {
       "success": True,
       "handler": "calphad",
       "function": "calculate_equilibrium_at_point",
       "data": {
           "message": "**Equilibrium at 1000.0 K for Al30.0Si55.0C15.0**\n...",
           "temperature_K": 1000.0,
           "composition": "AL30.0SI55.0C15.0",
           "phases": [
               {
                   "phase": "SiC",
                   "fraction": 0.45,
                   "composition": {"SI": 0.50, "C": 0.50}
               },
               {
                   "phase": "Al4C3",
                   "fraction": 0.35,
                   "composition": {"AL": 0.57, "C": 0.43}
               },
               {
                   "phase": "FCC_A1",
                   "fraction": 0.20,
                   "composition": {"AL": 0.95, "SI": 0.05}
               }
           ],
           "total_fraction": 1.00
       },
       "citations": ["pycalphad"],
       "confidence": 0.95
   }

**Side Effects:**

- None (pure calculation, no file I/O)

**Mathematical Background:**

The Gibbs free energy of phase φ at temperature T is:

.. math::

   G^{\phi}(T, P, X^{\phi}) = \sum_i X_i^{\phi} G_i^0(T) + RT \sum_i X_i^{\phi} \ln(X_i^{\phi}) + G^{ex}(T, X^{\phi})

where:

- :math:`G_i^0(T)` is the reference state Gibbs energy of component i
- :math:`RT \sum_i X_i^{\phi} \ln(X_i^{\phi})` is the ideal mixing term
- :math:`G^{ex}(T, X^{\phi})` is the excess Gibbs energy (from TDB parameters)
       
The equilibrium solver minimizes the total Gibbs energy subject to mass balance and chemical equilibrium constraints.

**Example:**

.. code-block:: python

   # Calculate equilibrium for Al-Si-C alloy at 1000K
   result = await handler.calculate_equilibrium_at_point(
       composition="Al30Si55C15",
       temperature=1000.0,
       composition_type="atomic"
   )
   # Returns: SiC (45%), Al4C3 (35%), FCC_A1 (20%)
   
   # Calculate for binary Al-Zn at room temperature
   result = await handler.calculate_equilibrium_at_point(
       composition="Al70Zn30",
       temperature=300.0
   )
   
   # Calculate with weight percent input
   result = await handler.calculate_equilibrium_at_point(
       composition="Fe70Cr20Ni10",  # wt%
       temperature=1200.0,
       composition_type="weight"
   )

.. _calculate_phase_fractions_vs_temperature:

calculate_phase_fractions_vs_temperature
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Function Definition:**

.. code-block:: python

   async def calculate_phase_fractions_vs_temperature(
       self,
       composition: str,
       min_temperature: float,
       max_temperature: float,
       temperature_step: Optional[float] = None,
       composition_type: Optional[str] = "atomic"
   ) -> str

**Description:**

Calculate how phase fractions change with temperature for a specific composition. This is essential for understanding precipitation behavior, dissolution, and phase transformation sequences.

**When to Use:**

- Understanding precipitation behavior (phase fraction increasing with cooling)
- Analyzing dissolution behavior (phase fraction decreasing with heating)
- Identifying phase transformation temperatures (where phases appear/disappear)
- Mapping solvus boundaries and phase stability ranges
- Predicting heat treatment behavior

**How It Fetches Data:**

Same as ``calculate_equilibrium_at_point``:

1. Parses composition using ``_parse_multicomponent_composition()``
2. Loads database via ``load_tdb_database(elements)``
3. Filters phases: binary → ``_filter_phases_for_system()``, multicomponent → ``_filter_phases_for_multicomponent()``

**How It Calculates:**

1. **Temperature Array Generation:**
   
   - Default step: 10 K
   - Creates array: ``temps = np.arange(min_temperature, max_temperature + step, step)``
   - Example: 300-1500 K with 10 K step → 121 temperature points

2. **Equilibrium Loop:**
   
   For each temperature T in array:
   
   .. code-block:: python
   
      for T in temps:
          # Build conditions (same as calculate_equilibrium_at_point)
          conditions = {v.T: T, v.P: 101325, v.N: 1.0}
          for elem in elements[1:]:
              conditions[v.X(elem)] = comp_dict[elem]
          
          # Calculate equilibrium
          eq = equilibrium(db, elements_with_va, phases, conditions)
          
          # Extract phase fractions with vertex handling
          temp_phases = extract_phase_fractions_from_equilibrium(eq, tolerance=1e-4)
          
          # Store fractions for all seen phases
          for phase in all_phases_seen:
              phase_fractions[phase].append(temp_phases.get(phase, 0.0))

3. **Trend Analysis:**
   
   For each phase that appears (max fraction > 1e-6):
   
   - ``frac_start = fractions[0]`` (at min_temperature)
   - ``frac_end = fractions[-1]`` (at max_temperature)
   - Determine trend:
     
     - If ``frac_end > frac_start + 0.01``: "increasing with temperature"
     - If ``frac_start > frac_end + 0.01``: "decreasing with temperature"
     - Otherwise: "stable"
   
   - Compute change magnitude: ``delta = frac_end - frac_start``

4. **Data Storage:**
   
   - Stores in ``self._last_phase_fraction_data`` for potential plotting or follow-up
   - Structure:
     
     .. code-block:: python
     
        {
            'temperatures': temps.tolist(),
            'phase_fractions': {phase: fractions_list},
            'composition': comp_dict,
            'composition_str': "AL30SI55C15"
        }

**Parameters:**

- ``composition`` (str, required): Composition as element-number pairs (e.g., ``'Al30Si55C15'``, ``'Al80Zn20'``)
- ``min_temperature`` (float, required): Minimum temperature in Kelvin
- ``max_temperature`` (float, required): Maximum temperature in Kelvin
- ``temperature_step`` (float, optional): Temperature step in Kelvin. Default: 10 K
- ``composition_type`` (str, optional): ``'atomic'`` (default, at.%) or ``'weight'`` (wt.%)

**Returns:**

Structured result containing:

.. code-block:: python

   {
       "success": True,
       "handler": "calphad",
       "function": "calculate_phase_fractions_vs_temperature",
       "data": {
           "message": "**Phase Fractions vs Temperature for AL30SI55C15**\n...",
           "composition": "AL30SI55C15",
           "temperature_range_K": [300, 1500],
           "temperature_points": 121,
           "phase_evolution": {
               "SIC": {"start": 0.45, "end": 0.40, "max": 0.50},
               "AL4C3": {"start": 0.35, "end": 0.30, "max": 0.40},
               "FCC_A1": {"start": 0.20, "end": 0.25, "max": 0.30},
               "LIQUID": {"start": 0.0, "end": 0.05, "max": 0.30}
           }
       },
       "citations": ["pycalphad"],
       "confidence": 0.95
   }

**Side Effects:**

- Stores phase fraction data in ``_last_phase_fraction_data`` for potential reuse
- This data can be accessed by other functions for plotting or analysis

**Use Cases and Interpretation:**

1. **Precipitation Analysis:**
   
   If a phase fraction increases with decreasing temperature (cooling):
   - The phase precipitates from solution upon cooling
   - Example: Theta phase in Al-Cu increasing from 0% at 500°C to 5% at 200°C
   - Useful for age-hardening heat treatment design

2. **Dissolution Analysis:**
   
   If a phase fraction decreases with increasing temperature (heating):
   - The phase dissolves into solution upon heating
   - Example: MgZn2 decreasing from 8% at 100°C to 0% at 400°C
   - Defines solution treatment temperature

3. **Solvus Temperature:**
   
   Temperature where phase completely dissolves (fraction → 0)
   - Critical for heat treatment design
   - Defines maximum solution treatment temperature

**Example:**

.. code-block:: python

   # Analyze phase evolution for Al-Si-C from 300-1500K
   result = await handler.calculate_phase_fractions_vs_temperature(
       composition="Al30Si55C15",
       min_temperature=300,
       max_temperature=1500,
       temperature_step=10
   )
   # Returns: SiC (decreasing), Al4C3 (decreasing), LIQUID (increasing with T)
   
   # Fine temperature resolution for critical range
   result = await handler.calculate_phase_fractions_vs_temperature(
       composition="Al96Cu4",
       min_temperature=400,
       max_temperature=600,
       temperature_step=5  # 5K steps for better resolution
   )
   
   # Weight percent input
   result = await handler.calculate_phase_fractions_vs_temperature(
       composition="Al92Mg8",  # wt%
       min_temperature=200,
       max_temperature=700,
       composition_type="weight"
   )

.. _analyze_phase_fraction_trend:

analyze_phase_fraction_trend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Function Definition:**

.. code-block:: python

   async def analyze_phase_fraction_trend(
       self,
       composition: str,
       phase_name: str,
       min_temperature: float,
       max_temperature: float,
       expected_trend: Optional[str] = None
   ) -> str

**Description:**

Analyze whether a specific phase increases or decreases with temperature. This function focuses on a single phase and provides detailed trend analysis, optionally verifying against expected behavior.

**When to Use:**

- Verifying claims like "Phase X increases with decreasing temperature"
- Testing statements about precipitation upon cooling (fraction increases as T decreases)
- Confirming dissolution behavior upon heating (fraction decreases as T increases)
- Understanding solvus behavior for a specific phase

**How It Fetches Data:**

Same as ``calculate_equilibrium_at_point``: parses composition, loads database, filters phases.

**How It Calculates:**

1. **Temperature Sampling:**
   
   - Creates 50 evenly-spaced temperature points: ``temps = np.linspace(min_temperature, max_temperature, 50)``
   - Provides good resolution while keeping computation reasonable

2. **Equilibrium at Each Temperature:**
   
   - Same loop as ``calculate_phase_fractions_vs_temperature``
   - But focuses only on tracking the specified phase

3. **Phase Instance Aggregation:**
   
   PyCalphad may label phase instances as ``SIC#1``, ``SIC#2``, etc. (different sublattice configurations or two-phase tie-line vertices). This function sums all instances:
   
   .. code-block:: python
   
      phase_frac = get_phase_fraction_by_base_name(temp_phases, phase_to_track)
      # Sums: SIC#1 + SIC#2 + SIC#3 = total SIC fraction
   
   Uses ``get_phase_fraction_by_base_name()`` which strips ``#`` suffix and sums matching base names.

4. **Trend Determination:**
   
   - ``frac_low_T = fractions[0]`` (at min_temperature)
   - ``frac_high_T = fractions[-1]`` (at max_temperature)
   - ``delta = frac_high_T - frac_low_T``
   
   Classification:
   
   - If ``abs(delta) < 0.001``: trend = "stable"
   - If ``delta > 0.001``: trend = "increases" (with increasing temperature, decreases upon cooling)
   - If ``delta < -0.001``: trend = "decreases" (with increasing temperature, increases upon cooling)

5. **Expected Trend Verification:**
   
   If ``expected_trend`` is provided, parses natural language patterns:
   
   .. code-block:: python
   
      # Pattern matching examples:
      "increasing with temperature" → check if frac_high_T > frac_low_T
      "decreasing temperature" or "upon cooling" → check if frac_low_T > frac_high_T
      "increases with cooling" → phase should be higher at LOW T (precipitation)
      "decreases upon heating" → phase should be lower at HIGH T (dissolution)
   
   Returns ✅ if trend matches expectation, ❌ otherwise.

**Parameters:**

- ``composition`` (str, required): Composition (e.g., ``'Al30Si55C15'``, ``'Al88Mg8Zn4'``)
- ``phase_name`` (str, required): Name of phase to analyze (e.g., ``'AL4C3'``, ``'SIC'``, ``'FCC_A1'``, ``'TAU'``)
- ``min_temperature`` (float, required): Minimum temperature in Kelvin
- ``max_temperature`` (float, required): Maximum temperature in Kelvin
- ``expected_trend`` (str, optional): Expected trend for verification:
  
  - ``'increase'`` / ``'decrease'`` / ``'stable'``
  - Context-aware: ``'increases with cooling'``, ``'decreases upon heating'``, ``'precipitates upon cooling'``

**Returns:**

Structured result containing:

.. code-block:: python

   {
       "success": True,
       "handler": "calphad",
       "function": "analyze_phase_fraction_trend",
       "data": {
           "message": "**Phase Fraction Analysis: SIC in AL30SI55C15**\n...",
           "phase": "SIC",
           "composition": "AL30SI55C15",
           "temperature_range_K": [300, 1500],
           "trend": "decreases",  # with increasing temperature
           "fraction_change": -0.05,
           "fraction_at_low_T": 0.50,
           "fraction_at_high_T": 0.45,
           "max_fraction": 0.50,
           "min_fraction": 0.42,
           "expected_trend": "increases with cooling",
           "matches_expectation": True  # Because decreasing with T = increasing with cooling
       },
       "citations": ["pycalphad"],
       "confidence": 0.95  # or 0.75 if doesn't match expectation
   }

**Side Effects:**

- None (pure calculation)

**Example:**

.. code-block:: python

   # Verify if SiC precipitates upon cooling (should increase as T decreases)
   result = await handler.analyze_phase_fraction_trend(
       composition="Al30Si55C15",
       phase_name="SIC",
       min_temperature=300,
       max_temperature=1500,
       expected_trend="increases with cooling"
   )
   # Returns: ✅ Verified - SIC fraction decreases with increasing T 
   #          (= increases with cooling, i.e., precipitates)
   
   # Analyze tau phase in Al-Mg-Zn without expectation
   result = await handler.analyze_phase_fraction_trend(
       composition="Al88Mg8Zn4",
       phase_name="TAU",
       min_temperature=200,
       max_temperature=600
   )
   # Returns trend description without verification
   
   # Check if FCC dissolves upon heating
   result = await handler.analyze_phase_fraction_trend(
       composition="Al96Cu4",
       phase_name="THETA",
       min_temperature=300,
       max_temperature=800,
       expected_trend="decreases upon heating"
   )

Advanced Verification Functions
--------------------------------

.. _verify_phase_formation_across_composition:

verify_phase_formation_across_composition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Function Definition:**

.. code-block:: python

   async def verify_phase_formation_across_composition(
       self,
       system: str,
       phase_name: str,
       composition_threshold: float,
       threshold_element: str,
       temperature: float = 300.0,
       composition_type: Optional[str] = "atomic",
       fixed_element: Optional[str] = None,
       fixed_composition: Optional[float] = None
   ) -> str

**Description:**

Verify phase formation statements across a composition range. This function tests claims like "beyond X% of element A, phase Y forms" by systematically sampling compositions and checking phase presence.

**When to Use:**

- Checking claims like "beyond 50% Al, phase X forms"
- Verifying "at compositions greater than X%, phase Y appears"
- Testing composition thresholds for phase stability
- Binary system analysis (varying one element)
- Ternary system analysis (varying one element while keeping another fixed)

**How It Fetches Data:**

1. **System Parsing:**
   
   - Extracts elements from system string: "Al-Mg-Zn" → ['AL', 'MG', 'ZN']
   - Validates threshold_element is in system
   - For ternary: validates fixed_element (must be different from threshold_element)

2. **Database Loading:**
   
   - ``load_tdb_database(elements)`` with same selection logic as other functions
   - For Al-Mg-Zn specifically, uses COST507.tdb (has tau phase data)

3. **Phase Selection and Name Mapping:**
   
   - Binary: ``_filter_phases_for_system(db, tuple(elements))``
   - Ternary: ``get_phases_for_elements(db, elements, self._phase_elements)``
   - **Category Mapping:** If ``phase_name`` is a category (e.g., "tau", "laves"):
     
     - Uses ``PHASE_CLASSIFICATION`` dictionary to find all matching database phases
     - Example: "tau" maps to ['TAU', 'TAU_MG32(AL_ZN)49'] in Al-Mg-Zn
     - All matching phases are aggregated in analysis

**How It Calculates:**

1. **Composition Sampling:**
   
   Binary system:
   
   .. code-block:: python
   
      # Sample 21 evenly-spaced points from 0 to 1
      threshold_compositions = np.linspace(0.0, 1.0, 21)
      
      # Add extra points near threshold for precision
      threshold_fraction = composition_threshold / 100.0
      threshold_nearby = [
          max(0.0, threshold_fraction - 0.05),
          max(0.0, threshold_fraction - 0.02),
          threshold_fraction,
          min(1.0, threshold_fraction + 0.02),
          min(1.0, threshold_fraction + 0.05)
      ]
      # Merge and sort unique compositions
   
   Ternary system:
   
   .. code-block:: python
   
      # Vary threshold_element, fix fixed_element, balance is third element
      comp_dict[threshold_elem] = x_threshold  # varies
      comp_dict[fixed_elem] = fixed_composition / 100.0  # fixed
      comp_dict[balance_elem] = 1.0 - x_threshold - fixed_composition/100.0  # balance

2. **Equilibrium at Each Composition:**
   
   For each sampled composition:
   
   - Builds composition dictionary with all elements
   - Calls ``compute_equilibrium(db, pycalphad_elements, phases, comp_dict, temperature)``
   - Extracts phase fractions using ``extract_phase_fractions_from_equilibrium(eq, tolerance=1e-4)``
   - Aggregates target phase fractions:
     
     .. code-block:: python
     
        # If checking category (e.g., "tau"), sum all matching phases
        target_names_upper = {'TAU', 'TAU_MG32(AL_ZN)49'}  # example
        phase_fraction = sum(
            frac for name, frac in phase_fractions.items()
            if base_name(name) in target_names_upper
        )
        phase_present = phase_fraction > 0.01  # 1% threshold
   
   - Stores result with at.% for all elements

3. **Threshold Analysis:**
   
   - Splits results into two groups:
     
     - ``below_threshold``: compositions where threshold_elem < composition_threshold
     - ``above_threshold``: compositions where threshold_elem ≥ composition_threshold
   
   - Counts phase presence in each group:
     
     - ``phase_count_below = sum(1 for r in below_threshold if r['phase_present'])``
     - ``phase_count_above = sum(1 for r in above_threshold if r['phase_present'])``
   
   - Computes frequencies:
     
     - ``fraction_below = phase_count_below / total_below``
     - ``fraction_above = phase_count_above / total_above``

4. **Verdict Determination:**
   
   Uses frequency-based comparison (not raw counts) with 5% tolerance (eps=0.05):
   
   - ✅ **VERIFIED**: ``fraction_above > 0 and fraction_below == 0``
     
     - Phase forms above threshold only, absent below
   
   - ⚠️ **PARTIALLY VERIFIED**: ``fraction_above >= eps and (fraction_above - fraction_below) > eps``
     
     - Phase forms more frequently above threshold but can appear below
   
   - ❌ **CONTRADICTED**: ``fraction_below >= eps and (fraction_below - fraction_above) > eps``
     
     - Phase is actually more frequent below threshold (opposite behavior)
   
   - ❌ **NOT VERIFIED**: Similar frequency above and below (no clear threshold)

**Parameters:**

- ``system`` (str, required): System specification:
  
  - Binary: ``'Fe-Al'``, ``'Al-Zn'``
  - Ternary: ``'Al-Mg-Zn'``

- ``phase_name`` (str, required): Phase to check:
  
  - Exact database name: ``'MGZN2'``, ``'TAU'``, ``'FCC_A1'``
  - Category name: ``'Laves'``, ``'tau'``, ``'fcc'``, ``'gamma'``

- ``composition_threshold`` (float, required): Threshold value in at.% (e.g., ``50.0`` for 50 at.%)
- ``threshold_element`` (str, required): Element being thresholded (e.g., ``'Al'`` in "beyond 50% Al")
- ``temperature`` (float, optional): Temperature in K for checking. Default: 300 K
- ``composition_type`` (str, optional): ``'atomic'`` (default) or ``'weight'``

**For Ternary Systems Only:**

- ``fixed_element`` (str, optional): Element to keep constant (e.g., ``'Zn'``)
- ``fixed_composition`` (float, optional): Fixed element composition in at.% (e.g., ``4.0`` for 4%)

**Returns:**

Structured result with markdown-formatted message including:

- Summary statistics (counts and frequencies)
- Example compositions from below/above threshold regions
- Verification verdict (✅/⚠️/❌)
- Detailed composition scan table with ~10 representative points

**Side Effects:**

- None (pure calculation)

**Example (Binary):**

.. code-block:: python

   # Check if tau phase forms beyond 50% Al in Fe-Al
   result = await handler.verify_phase_formation_across_composition(
       system="Fe-Al",
       phase_name="tau",
       composition_threshold=50.0,
       threshold_element="Al",
       temperature=300
   )
   # Samples Fe-Al compositions from 0-100% Al at 300K
   # Returns: ✅ VERIFIED if tau only appears when Al >= 50%

**Example (Ternary):**

.. code-block:: python

   # Check if tau forms above 8% Mg in Al-Mg-Zn with fixed 4% Zn
   result = await handler.verify_phase_formation_across_composition(
       system="Al-Mg-Zn",
       phase_name="tau",
       composition_threshold=8.0,
       threshold_element="Mg",
       temperature=300,
       fixed_element="Zn",
       fixed_composition=4.0
   )
   # Samples: Al-[0-18]Mg-4Zn compositions at 300K
   # Balance Al: 100 - Mg - 4 = [96-78]%
   # Returns verdict on whether tau appears above 8% Mg

.. _sweep_microstructure_claim_over_region:

sweep_microstructure_claim_over_region
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Function Definition:**

.. code-block:: python

   async def sweep_microstructure_claim_over_region(
       self,
       system: str,
       element_ranges: str,
       claim_type: str,
       expected_phases: Optional[str] = None,
       phase_to_check: Optional[str] = None,
       min_fraction: Optional[float] = None,
       max_fraction: Optional[float] = None,
       grid_points: int = 4,
       composition_type: str = "atomic",
       process_type: str = "as_cast",
       require_mechanical_desirability: bool = False
   ) -> Dict[str, Any]

**Description:**

Sweep composition space and evaluate whether a microstructure claim holds across an entire region. This function answers: "Does this claim hold for ALL compositions in the stated range?" not just "Does it hold for one specific composition?"

**When to Use:**

- Testing **universal** claims like "all Al-Mg-Zn alloys with Mg<8% and Zn<4% form fcc+tau"
- Validating design rules over composition regions
- Assessing generality/universality of metallurgical statements
- Finding exceptions to broad claims

**How It Fetches Data:**

1. **System and Range Parsing:**
   
   - Parses system string: "Al-Mg-Zn" → ['AL', 'MG', 'ZN']
   - Parses JSON element_ranges:
     
     .. code-block:: json
     
        {"MG": [0, 8], "ZN": [0, 4]}
     
     means Mg ∈ [0, 8) at.%, Zn ∈ [0, 4) at.%, Al (balance) = 100 - Mg - Zn
   
   - Determines balance element (first not in ranges)

2. **Grid Generation:**
   
   1D sweep (one element varying):
   
   .. code-block:: python
   
      el1_vals = np.linspace(el1_min, el1_max - 1e-6, grid_points)
      # Generates 4 points by default
   
   2D sweep (two elements varying):
   
   .. code-block:: python
   
      el1_vals = np.linspace(el1_min, el1_max - 1e-6, grid_points)
      el2_vals = np.linspace(el2_min, el2_max - 1e-6, grid_points)
      # Creates grid: 4x4 = 16 total compositions by default
      for v1 in el1_vals:
          for v2 in el2_vals:
              balance_val = 100 - v1 - v2
              if balance_val >= 0:  # Valid composition
                  compositions.append({balance: balance_val, el1: v1, el2: v2})

**How It Calculates:**

1. **Stratified Composition Sampling:**
   
   - **Uniform baseline grid**: Evenly-spaced points across composition range
   - **Threshold-concentrated sampling**: Additional points near ``composition_threshold`` boundaries
     
     - If threshold at 8% Mg: adds points at 6%, 7%, 7.5%, 8%, 8.5%, 9%, 10%
     - Captures sharp transitions in phase formation behavior
   
   - **Boundary oversampling**: Extra points along solvus/solidus from phase diagram analysis
   - Avoids **corner-weighting** (uniform grids over-represent corner compositions)
   - Total sampling: ``grid_points`` baseline + threshold refinements

2. **For Each Grid Point:**
   
   - Formats composition string: "Al92.0-Mg4.0-Zn4.0"
   - Calls ``fact_check_microstructure_claim()`` with same parameters:
     
     - Same claim_type, expected_phases, phase_to_check, min/max_fraction
     - Same process_type (as_cast = Scheil solidification or equilibrium_300K)
   
   - Receives verdict (True/False) and score (-2 to +2)

3. **Mechanical Desirability Check (if required):**
   
   If ``require_mechanical_desirability=True``:
   
   - Extracts ``mechanical_score`` from fact_check result
   - ``mechanical_ok = (mechanical_score > 0)``
   - ``overall_pass = microstructure_verdict AND mechanical_ok``
   
   This adds an additional filter: even if phases match, composition fails if mechanical properties are poor (brittle intermetallics dominant).

4. **Aggregation and Counter-Example Detection:**
   
   - Counts: ``pass_count``, ``fail_count``, ``mech_fail_count``
   - Computes: ``pass_fraction = pass_count / total_points``
   - **Identifies worst counter-examples**:
     
     - Sorts failed points by score (most negative first)
     - Extracts **top 3 worst-failing compositions**
     - For each counter-example, performs **local refinement** (bisection around failure point)
     - Reports exact composition where claim breaks down most severely
   
   - Stores grid results with:
     
     - composition dict and string
     - microstructure_verdict (phases match?)
     - mechanical_ok (good ductility?)
     - overall_pass (both conditions met)
     - score, phases list, error (if any)
     - **counter_example_flag** (True for worst failures)

5. **Overall Verdict:**
   
   Based on pass_fraction:
   
   - 1.00 (100%): "UNIVERSALLY SUPPORTED", score=+2, confidence=1.0
   - ≥0.90 (90%+): "MOSTLY SUPPORTED", score=+1, confidence=0.8
   - ≥0.50 (50-90%): "MIXED", score=0, confidence=0.5
   - >0 (<50%): "MOSTLY REJECTED", score=-1, confidence=0.7
   - 0 (0%): "UNIVERSALLY REJECTED", score=-2, confidence=1.0

**Parameters:**

- ``system`` (str, required): Chemical system (e.g., ``'Al-Mg-Zn'``)
- ``element_ranges`` (str, required): JSON dict of element ranges in at.% (default) or wt.%
- ``claim_type`` (str, required): ``'two_phase'``, ``'three_phase'``, or ``'phase_fraction'``
- ``expected_phases`` (str, optional): For two_phase/three_phase (e.g., ``'fcc+tau'``)
- ``phase_to_check`` (str, optional): For phase_fraction claims
- ``min_fraction`` (float, optional): Minimum phase fraction (0-1)
- ``max_fraction`` (float, optional): Maximum phase fraction (0-1)
- ``grid_points`` (int, optional): Points per element. Default: 4 (4×4=16 for 2D)
- ``composition_type`` (str, optional): ``'atomic'`` (default, at.%) [``'weight'`` not yet implemented]
- ``process_type`` (str, optional): ``'as_cast'`` (default, after solidification) or ``'equilibrium_300K'`` (infinite time at 300K)
- ``require_mechanical_desirability`` (bool, optional): Also check for good ductility. Default: False

**Returns:**

.. code-block:: python

   {
       "success": True,
       "message": "## Region Sweep Fact-Check Result\n...",
       "overall_verdict": "UNIVERSALLY SUPPORTED",  # or MOSTLY, MIXED, REJECTED
       "overall_score": 2,  # -2 to +2
       "confidence": 1.0,  # 0 to 1
       "pass_count": 16,
       "fail_count": 0,
       "total_points": 16,
       "pass_fraction": 1.0,
       "mechanical_fail_count": 0,  # if require_mechanical_desirability=True
       "microstructure_pass_count": 16,  # compositions matching phases
       "counter_examples": [  # Top 3 worst-failing compositions (if any failures)
           {
               "composition": {"AL": 80.0, "MG": 12.0, "ZN": 8.0},
               "composition_str": "Al80.0-Mg12.0-Zn8.0",
               "score": -2,
               "reason": "Forms FCC + Laves (30%) + Tau (10%), not FCC + Tau only",
               "phases": [("FCC_A1", 0.60, "fcc"), ("MGZN2", 0.30, "laves"), ("TAU", 0.10, "tau")],
               "local_refinement_applied": True
           }
       ],
       "grid_results": [  # first 20 points
           {
               "composition": {"AL": 88.0, "MG": 8.0, "ZN": 4.0},
               "composition_str": "Al88.0-Mg8.0-Zn4.0",
               "microstructure_verdict": True,
               "mechanical_ok": True,
               "overall_pass": True,
               "score": 2,
               "phases": [("FCC_A1", 0.85, "fcc"), ("TAU", 0.15, "tau")],
               "counter_example_flag": False,
               "error": None
           },
           ...
       ],
       "sampling_strategy": "stratified",  # stratified (threshold-aware) or uniform
       "refinement_points_added": 6,  # Extra points near thresholds
       "citations": ["pycalphad"]
   }

**Side Effects:**

- None (pure calculation, but calls many equilibrium calculations so can take time)

**Example:**

.. code-block:: python

   # Test if all Al-Mg<8%-Zn<4% alloys form fcc+tau after casting
   result = await handler.sweep_microstructure_claim_over_region(
       system="Al-Mg-Zn",
       element_ranges='{"MG": [0, 8], "ZN": [0, 4]}',
       claim_type="two_phase",
       expected_phases="fcc+tau",
       max_fraction=0.20,  # tau < 20%
       grid_points=4,  # 4x4 = 16 compositions
       process_type="as_cast"
   )
   # Returns: UNIVERSALLY SUPPORTED (score=+2) if all 16 points pass
   #          MOSTLY SUPPORTED (score=+1) if ≥90% pass
   #          MIXED (score=0) if 50-90% pass
   
   # Test with mechanical desirability filter
   result = await handler.sweep_microstructure_claim_over_region(
       system="Al-Mg-Zn",
       element_ranges='{"MG": [0, 12], "ZN": [0, 6]}',
       claim_type="two_phase",
       expected_phases="fcc+tau",
       grid_points=5,  # 5x5 = 25 compositions
       process_type="as_cast",
       require_mechanical_desirability=True  # Also check ductility
   )
   # Rejects compositions with too much brittle intermetallic phases

**Technical Details:**

- **Stratified sampling strategy**: concentrates points near thresholds and phase boundaries, not uniform
- Calls ``fact_check_microstructure_claim()`` (Scheil solidification) for each point
- Supports 1D (one element varies) and 2D (two elements vary) sweeps
- ``mechanical_desirability_score()`` evaluates (heuristic, see full rules above):
  
  - High FCC (>85%) with modest intermetallics (<15%) → +1 (ductile)
  - Very high intermetallics (>20%) or Laves (>15%) → -1 (brittle)
  - Otherwise → 0 (mixed)

- **Counter-example identification**: bisection refinement around worst-failing points
- **Execution time** scales as O((grid_points + refinements)^n_varying_elements × Scheil_time)
  
  - For 4×4 baseline grid + 6 refinements with ~3-5s per Scheil: ~2-3 minutes total
  - Parallel execution across CPU cores reduces wall time
  - Equilibrium caching reduces redundant calculations

- **Accuracy vs. speed trade-off**:
  
  - ``grid_points=3``: Fast screening (9-25 points)
  - ``grid_points=4`` (default): Balanced (16-36 points)
  - ``grid_points=5``: High resolution (25-49 points)
  - ``grid_points=6``: Very detailed (36-64 points)

.. _fact_check_microstructure_claim:

fact_check_microstructure_claim
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Function Definition:**

.. code-block:: python

   async def fact_check_microstructure_claim(
       self,
       system: str,
       composition: str,
       claim_type: str,
       expected_phases: Optional[str] = None,
       phase_to_check: Optional[str] = None,
       min_fraction: Optional[float] = None,
       max_fraction: Optional[float] = None,
       process_type: str = "as_cast",
       temperature: Optional[float] = None,
       composition_constraints: Optional[str] = None
   ) -> Dict[str, Any]

**Description:**

Evaluate microstructure claims for multicomponent alloys. Acts as an automated "materials expert witness" to verify metallurgical assertions using thermodynamic calculations. This is the core fact-checking function that ``sweep_microstructure_claim_over_region`` calls repeatedly.

**When to Use:**

- Fact-checking specific metallurgical statements
- Verifying a single alloy composition meets design criteria
- Evaluating claims from literature, specifications, or design rules
- Testing "what-if" scenarios for alloy development

**How It Fetches Data:**

1. **System and Composition Parsing:**
   
   - Parses system: "Al-Mg-Zn" → ['AL', 'MG', 'ZN']
   - Parses composition using ``parse_composition_string()``:
     
     - "Al88Mg8Zn4" → {AL: 88.0, MG: 8.0, ZN: 4.0} (at.%)
     - "88Al-8Mg-4Zn" → same
     - "Al-8Mg-4Zn" → same (Al is balance)
   
   - Converts to mole fractions: {AL: 0.88, MG: 0.08, ZN: 0.04}

2. **Composition Constraint Checking:**
   
   If ``composition_constraints`` provided:
   
   .. code-block:: json
   
      {"MG": {"lt": 8.0}, "ZN": {"lt": 4.0}}
   
   Checks:
   
   - MG < 8.0 at.% ? (if "lt")
   - MG ≤ 8.0 at.% ? (if "lte")
   - MG > 2.0 at.% ? (if "gt")
   - MG ≥ 2.0 at.% ? (if "gte")
   - 2.0 ≤ MG ≤ 8.0 at.% ? (if "between": [2.0, 8.0])
   
   If any constraint violated:
   
   - Sets verdict = False
   - Adjusts score to -1 (even if microstructure matched)
   - Appends violations to reasoning

3. **Database and Phase Selection:**
   
   - ``load_tdb_database(elements)``
   - ``get_phases_for_elements(db, elements, self._phase_elements)`` for ternary+
   - Returns phases relevant to this system

**How It Calculates:**

1. **Process Selection and Phase Fractions:**
   
   **Option A: as_cast (default) - Scheil-Gulliver Solidification**
   
   Simulates **non-equilibrium solidification** from melt using the Scheil-Gulliver model:
   
   .. code-block:: python
   
      # Find liquidus temperature (first liquid appears on cooling)
      T_liquidus = find_liquidus_temperature(...)
      
      # Scheil-Gulliver solidification: DRIVEN BY SOLID FRACTION INCREMENT
      # Assumption: Complete mixing in liquid, NO diffusion in solid
      T = T_liquidus
      f_solid = 0.0  # Initial solid fraction
      remaining_liquid_composition = initial_composition.copy()
      accumulated_solid_phases = {}  # Track cumulative phase fractions
      
      # Drive by Δf_s (solid fraction increment), NOT ΔT
      delta_fs = 0.01  # Initial solid fraction step (adaptive)
      
      while f_solid < 0.999:  # Until ~100% solid
          # (i) Impose small Δf_s increment
          f_solid_target = min(f_solid + delta_fs, 0.999)
          
          # (ii) Solve local equilibrium: find T where this f_s is achieved
          # Binary search for T that gives target solid fraction
          T_lo, T_hi = T - 10, T
          for _ in range(20):  # Bisection to find equilibrium T
              T_mid = (T_lo + T_hi) / 2
              eq = equilibrium(db, elements_with_va, phases,
                              {v.T: T_mid, v.P: 101325, 
                               v.X(...): remaining_liquid_composition})
              
              # Extract liquid and solid fractions
              # Handle LIQUID#1/LIQUID#2 for monotectics (sum by base name)
              liquid_frac = sum_phases_by_base_name(eq, "LIQUID")
              current_fs = 1.0 - liquid_frac
              
              if abs(current_fs - f_solid_target) < 1e-4:
                  break  # Converged
              elif current_fs < f_solid_target:
                  T_hi = T_mid  # Cool more
              else:
                  T_lo = T_mid  # Warm up
          
          T = T_mid  # Update temperature
          f_solid = f_solid_target
          
          # Extract solid phase fractions that formed in this step
          solid_phases_now = extract_solid_phases(eq, exclude="LIQUID")
          delta_solids = {ph: frac * delta_fs for ph, frac in solid_phases_now.items()}
          
          # Accumulate solid phases across all steps
          for phase, delta_frac in delta_solids.items():
              accumulated_solid_phases[phase] = accumulated_solid_phases.get(phase, 0) + delta_frac
          
          # (iii) Update liquid composition by STRICT MASS BALANCE
          # NOT the analytic C_L = C_0(1-f_s)^(k-1) (binary-only, assumes constant k)
          # Instead: C_L^new = (C_0 - sum(f_s^φ × C_s^φ)) / f_L^remaining
          for elem in elements:
              solid_contribution = sum(
                  accumulated_solid_phases[ph] * X_solid[ph][elem]
                  for ph in accumulated_solid_phases
              )
              remaining_liquid_composition[elem] = (
                  (initial_composition[elem] - solid_contribution) / (1.0 - f_solid)
              )
          
          # Normalize liquid composition (ensure sum = 1.0)
          total = sum(remaining_liquid_composition.values())
          for elem in elements:
              remaining_liquid_composition[elem] /= total
          
          # Adaptive stepping: shrink near terminal reactions
          if liquid_frac < 0.05:  # Last 5% liquid
              delta_fs = min(delta_fs, 0.005)  # Tighter steps (0.5%)
          elif phase_appearance_detected:
              delta_fs = min(delta_fs, 0.01)   # Moderate steps (1%)
          else:
              delta_fs = 0.01  # Normal steps
      
      # Optional: short isothermal back-diffusion knob
      # (Very small diffusion length to approximate late solid-state adjustments)
      if apply_back_diffusion:
          T_final = T - 20  # Cool slightly
          eq_final = equilibrium(db, elements_with_va, solid_phases,
                                {v.T: T_final, v.P: 101325,
                                 v.X(...): initial_composition})  # Global composition
          # Blend: 90% Scheil + 10% back-diffused
          precalc_fractions = blend_fractions(accumulated_solid_phases, eq_final, 0.9)
      else:
          precalc_fractions = accumulated_solid_phases
   
   This models **microsegregation** and **non-equilibrium intermetallic formation** during casting.
   
   **Why Scheil is more realistic than equilibrium:**
   
   - Accounts for coring (composition gradients in dendrites)
   - Predicts terminal eutectic reactions (low-melting constituent at dendrite boundaries)
   - Captures non-equilibrium phases that form during solidification but wouldn't exist at equilibrium
   - More accurate for as-cast mechanical properties (brittle intermetallics at grain boundaries)
   
   **Option B: equilibrium_300K**
   
   Full thermodynamic equilibrium at specified temperature (default 300K):
   
   .. code-block:: python
   
      T_ref = temperature or 300.0
      
      # Exclude LIQUID if T < 500K (metastable at low T)
      if T_ref < 500.0 and "LIQUID" in phases:
          phases = [p for p in phases if p != "LIQUID"]
      
      # Calculate equilibrium at T_ref
      eq = equilibrium(db, elements_with_va, phases,
                      {v.T: T_ref, v.P: 101325, ...})
      precalc_fractions = extract_phase_fractions_from_equilibrium(eq)
   
   This answers **"what is the equilibrium after infinite diffusion time?"**

2. **Phase Classification:**
   
   Maps database phase names to metallurgical categories using ``PHASE_CLASSIFICATION``:
   
   .. code-block:: python
   
      PHASE_CLASSIFICATION = {
          "FCC_A1": ("FCC", PhaseCategory.FCC, "face-centered cubic"),
          "HCP_A3": ("HCP", PhaseCategory.HCP, "hexagonal close-packed"),
          "BCC_A2": ("BCC", PhaseCategory.BCC, "body-centered cubic"),
          "TAU": ("Tau", PhaseCategory.TAU, "T phase"),
          "MGZN2": ("MgZn2", PhaseCategory.LAVES, "C14 Laves phase"),
          "AL3MG2": ("Al3Mg2", PhaseCategory.BETA, "β phase"),
          ...
      }
   
   Calls ``interpret_microstructure(precalc_fractions)`` which:
   
   - Groups phases by category
   - Returns list of ``PhaseInfo`` objects with base_name, fraction, category

3. **Claim Evaluation:**
   
   Creates ``AlloyFactChecker`` and adds appropriate checker based on ``claim_type``:
   
   **A. two_phase Claim**
   
   .. code-block:: python
   
      # Parse expected_phases: "fcc+tau" → [primary="fcc", secondary="tau"]
      checker = TwoPhaseChecker(
          db, elements, phases,
          primary_category=map_phase_to_category("fcc"),  # PhaseCategory.FCC
          secondary_category=map_phase_to_category("tau"),  # PhaseCategory.TAU
          secondary_max_fraction=max_fraction or 0.20,
          temperature=T_ref
      )
      checker.check(comp_molefrac, precalculated_fractions):
          # Finds phases in each category
          primary_phases = [p for p in phases if p.category == primary_category]
          secondary_phases = [p for p in phases if p.category == secondary_category]
          
          primary_frac = sum(p.fraction for p in primary_phases)
          secondary_frac = sum(p.fraction for p in secondary_phases)
          
          # Must have BOTH categories present
          has_both = (primary_frac > 0.01 and secondary_frac > 0.01)
          
          # Must have ONLY these two categories (no extras)
          other_frac = 1.0 - primary_frac - secondary_frac
          no_extras = (other_frac < 0.05)  # Tolerate <5% others
          
          # Secondary must be within bounds
          secondary_ok = (secondary_frac <= secondary_max_fraction)
          
          if has_both and no_extras and secondary_ok:
              return CheckResult(verdict=True, score=+2, confidence=0.95, ...)
          elif has_both and secondary_ok:  # Has extras
              return CheckResult(verdict=False, score=0, confidence=0.7, ...)
          else:
              return CheckResult(verdict=False, score=-2, confidence=0.9, ...)
   
   **B. three_phase Claim**
   
   Similar to two_phase but checks for three categories.
   
   **C. phase_fraction Claim**
   
   .. code-block:: python
   
      checker = PhaseFractionChecker(
          db, elements, phases,
          target_category=map_phase_to_category(phase_to_check),  # e.g., TAU
          min_fraction=min_fraction,  # e.g., 0.05
          max_fraction=max_fraction,  # e.g., 0.20
          temperature=T_ref
      )
      checker.check(comp_molefrac, precalculated_fractions):
          target_phases = [p for p in phases if p.category == target_category]
          target_frac = sum(p.fraction for p in target_phases)
          
          within_bounds = (
              (min_fraction is None or target_frac >= min_fraction) and
              (max_fraction is None or target_frac <= max_fraction)
          )
          
          if within_bounds:
              return CheckResult(verdict=True, score=+2, confidence=0.95, ...)
          else:
              return CheckResult(verdict=False, score=-2, confidence=0.9, ...)

4. **Mechanical Desirability Heuristic (for as_cast only):**
   
   .. code-block:: python
   
      # Extract phase categories
      phase_categories = {p.base_name: p.category.value for p in microstructure}
      
      mech_score, mech_interpretation = mechanical_desirability_score(
          precalc_fractions, phase_categories
      )
   
   **⚠️ HEURISTIC RULES (exposed and configurable):**
   
   These are simplified engineering guidelines, **NOT rigorous mechanical property predictions**:
   
   - **+1 (High ductility expected)**:
     
     - FCC > 85% AND (intermetallics < 15% OR tau < 10%)
     - Reasoning: Ductile matrix with modest strengthening precipitates
   
   - **-1 (Low ductility / brittleness risk)**:
     
     - Laves phases > 15% (C14, C15, C36 structures: MgZn₂, etc.)
     - Total intermetallics > 20%
     - Sigma phase present (>5%)
     - Reasoning: Large volume fraction of hard, brittle compounds
   
   - **-2 (High brittleness risk)**:
     
     - Laves > 25%
     - Sigma > 10%
   
   - **0 (Mixed / uncertain)**:
     
     - Intermediate fractions
     - Complex multi-phase mixtures
   
   **Important caveats:**
   
   - Actual ductility depends on **morphology** (coarse particles vs. fine precipitates)
   - **Grain size** effects not captured
   - **Specific intermetallic type** matters (Al₂Cu vs. MgZn₂ behave differently)
   - **Volume fraction** alone is insufficient; distribution and coherency matter
   - This is a **screening tool**, not a replacement for experimental testing
   
   **Phase-Specific Penalties (Default Configuration)**:
   
   Different intermetallic types have different impacts on ductility at the same volume fraction:
   
   .. code-block:: python
   
      PHASE_BRITTLENESS_WEIGHTS = {
          # Laves phases (C14, C15, C36): very brittle
          "MGZN2": 2.0,     # C14 Laves: high penalty
          "ALZN2": 2.0,
          
          # Sigma/Mu phases: extremely brittle, crack initiators
          "SIGMA": 3.0,
          "MU": 3.0,
          
          # Al-Cu phases: moderate brittleness (depending on morphology)
          "THETA": 1.0,     # θ-Al₂Cu: baseline penalty
          "THETA_PRIME": 0.5,  # θ' (coherent): lower penalty
          
          # Mg₂Si: moderate but can be managed
          "MG2SI": 1.2,
          
          # Al₃Mg₂ (β): very brittle in large amounts
          "AL3MG2": 1.8,
          
          # Default: unspecified intermetallics
          "DEFAULT": 1.0
      }
      
      # Effective brittleness score:
      brittleness = sum(
          fraction * PHASE_BRITTLENESS_WEIGHTS.get(phase, 1.0)
          for phase, fraction in phase_fractions.items()
          if is_intermetallic(phase)
      )
      
      # Apply weighted thresholds:
      if brittleness > 0.25:  # 25% weighted brittleness
          mechanical_score = -2
      elif brittleness > 0.15:
          mechanical_score = -1
      # ... etc.
   
   **User customization available:**
   
   - Adjust threshold values via configuration file
   - Override phase-specific weights (e.g., heavier penalty for Laves than θ-Al₂Cu)
   - Define custom brittleness categories for application-specific phases
   - Set morphology modifiers (e.g., reduce penalty for fine precipitates vs. coarse particles)
   - Enable/disable mechanical scoring entirely

5. **Final Verdict Assembly:**
   
   - If composition constraints violated → verdict=False, score adjusted
   - Otherwise uses checker result
   - Formats response with:
     
     - Verdict emoji (✓ / ✗)
     - Score text (+2/2, +1/2, 0/2, -1/2, -2/2)
     - Reasoning explanation
     - Mechanical desirability (if as_cast)
     - Calculated phase fractions (top 10)
     - Full report from fact checker

**Parameters:**

- ``system`` (str, required): Chemical system (e.g., ``'Al-Mg-Zn'``, ``'Fe-Cr-Ni'``)
- ``composition`` (str, required): Composition in at.% (various formats supported)
- ``claim_type`` (str, required): ``'two_phase'``, ``'three_phase'``, or ``'phase_fraction'``
- ``expected_phases`` (str, optional): For two_phase/three_phase (e.g., ``'fcc+tau'``)
- ``phase_to_check`` (str, optional): For phase_fraction claims (e.g., ``'tau'``)
- ``min_fraction`` (float, optional): Minimum phase fraction (0-1)
- ``max_fraction`` (float, optional): Maximum phase fraction (0-1)
- ``process_type`` (str, optional): ``'as_cast'`` (default, after solidification) or ``'equilibrium_300K'`` (infinite time)
- ``temperature`` (float, optional): Temperature in K (only for equilibrium_300K, default 300K)
- ``composition_constraints`` (str, optional): JSON constraints

**Returns:**

.. code-block:: python

   {
       "success": True,
       "message": "## Microstructure Fact-Check Result\n...",
       "verdict": True,  # or False
       "score": 2,  # -2 to +2
       "confidence": 0.95,  # 0 to 1
       "mechanical_score": 1.0,  # -1/0/+1 (only for as_cast)
       "mechanical_interpretation": "High ductility expected",
       "process_type": "as_cast",
       "supporting_data": {
           "phases": [
               ("FCC_A1", 0.85, "fcc"),
               ("TAU", 0.15, "tau")
           ],
           "composition_within_bounds": True,
           "composition_violations": []
       },
       "citations": ["pycalphad"]
   }

**Side Effects:**

- None (pure calculation)

**Example:**

.. code-block:: python

   # Fact-check: "Al-8Mg-4Zn forms fcc+tau with tau<20% after casting"
   result = await handler.fact_check_microstructure_claim(
       system="Al-Mg-Zn",
       composition="Al88Mg8Zn4",
       claim_type="two_phase",
       expected_phases="fcc+tau",
       max_fraction=0.20,
       process_type="as_cast"
   )
   # Returns: verdict=True, score=+2, confidence=0.95
   #          "SUPPORTED - Forms FCC (85%) + Tau (15%)"
   
   # Check with composition constraints
   result = await handler.fact_check_microstructure_claim(
       system="Al-Mg-Zn",
       composition="Al80Mg12Zn8",
       claim_type="two_phase",
       expected_phases="fcc+tau",
       max_fraction=0.20,
       composition_constraints='{"MG": {"lt": 8.0}, "ZN": {"lt": 4.0}}',
       process_type="as_cast"
   )
   # Returns: verdict=False (composition out of bounds)
   #          "MG=12.0 at.% is not < 8.0 at.%"
   
   # Check equilibrium at high temperature
   result = await handler.fact_check_microstructure_claim(
       system="Al-Cu",
       composition="Al96Cu4",
       claim_type="phase_fraction",
       phase_to_check="theta",
       max_fraction=0.10,
       process_type="equilibrium_300K",
       temperature=500.0
   )
   # Checks equilibrium at 500K (not as-cast)

Process Models
^^^^^^^^^^^^^^

**as_cast (default) - Scheil-Gulliver Solidification**:

- Simulates **non-equilibrium solidification** from the melt using Scheil-Gulliver model
- Answers: "What phases form during casting with no solid-state back-diffusion?"
- **Assumptions**:
  
  - Complete mixing in liquid (fast diffusion)
  - **Zero diffusion in solid** (frozen-in composition)
  - Local equilibrium at solid/liquid interface
  - Step-wise solidification from liquidus to solidus

- **Captures critical as-cast features**:
  
  - **Microsegregation** (coring in dendrites)
  - **Terminal eutectic reactions** (low-melting phases at grain boundaries)
  - **Non-equilibrium intermetallics** that form during solidification
  - **Realistic volume fractions** of brittle phases

- **More accurate than equilibrium** for:
  
  - As-cast mechanical properties
  - Brittle phase formation at dendrite boundaries
  - Solidification cracking susceptibility
  - Homogenization heat treatment planning

- Includes **mechanical desirability heuristic** scoring (see caveats above)

**equilibrium_300K - Full Thermodynamic Equilibrium**:

- Full thermodynamic equilibrium at specified temperature (default 300K)
- Answers: "What is the equilibrium state after infinite diffusion time?"
- **Assumes**:
  
  - Infinite time for solid-state diffusion
  - Complete homogenization
  - No kinetic barriers

- Excludes metastable phases (e.g., liquid at low T < 500 K)
- **More relevant for**:
  
  - Fully annealed/aged conditions
  - Long-term thermal exposure
  - Solvus temperature determination
  - Thermodynamic stability analysis

- Does **not** evaluate mechanical desirability (microstructure depends on heat treatment path)

**Choosing between models**:

- Use ``as_cast`` for: casting, welding, additive manufacturing, directional solidification
- Use ``equilibrium_300K`` for: annealed alloys, thermodynamic limits, phase transformation analysis

Database Support and Provenance
--------------------------------

Currently supported thermodynamic databases with **validation ranges** and **provenance tracking**:

**COST507.tdb** (COST507 European database)

- **Source**: European Cooperation in Science and Technology (COST) Action 507
- **Version**: 2.0
- **Validated systems**: Al-Zn, Al-Si, Al-Mg, Al-Cu, Al-Fe, Al-Mn, Al-Cr, ternaries with C, N, B, Li
- **Temperature range** (example): 200-1000 K (typical for Al-based systems below melting)
- **Composition range** (example): 0-100 at.% for validated binaries
- **⚠️ Note**: Ranges stated are **representative examples** based on literature assessments and calibration datasets
  
  - Exact validation ranges are **phase-specific** and **composition-dependent**
  - Some phases may have narrower validated regions (e.g., τ-phase in Al-Mg-Zn validated for specific composition corners)
  - The provenance system **reports database-claimed ranges**, not re-verified by this implementation
  - Always consult original TDB documentation for authoritative validation ranges

- **Special features**: 
  
  - Includes τ-phase (Mg₃₂(Al,Zn)₄₉) for Al-Mg-Zn system
  - Well-calibrated for Al-Mg-Zn with experimental validation (see original COST507 publications)
  - Contains Laves phases (MgZn₂, C14 structure)

- **Recommended for**: Al-Mg-Zn alloys, systems with interstitial elements (C, N, B)

**mc_al_v2037_pycal.tdb** (Multi-component Al database)

- **Source**: Custom multi-component aluminum alloy database
- **Version**: 2037 (pycalphad-compatible)
- **Validated systems**: Multi-component Al-based alloys (Al-Cu-Mg-Si-Zn-Mn-Fe-Cr)
- **Temperature range** (example): 298-2000 K (wide bracket for high-temperature applications)
- **Composition range** (example): Al-rich corner (Al > 80 at.%)
- **⚠️ Note**: Same caveat as COST507 – ranges are **representative examples**, phase-specific validation may be narrower

- **Special features**:
  
  - Optimized for commercial aluminum alloys (2xxx, 6xxx, 7xxx series)
  - Includes metastable phase descriptions
  - Better for complex multicomponent systems

- **Recommended for**: Commercial Al alloy development, multicomponent (4+ elements) systems

**Automatic Database Selection Logic**:

1. **Al-Mg-Zn systems** → COST507.tdb (best τ-phase data)
2. **Systems with C, N, B, Li** → COST507.tdb (interstitial element coverage)
3. **Multicomponent Al-rich** (4+ elements) → mc_al_v2037_pycal.tdb
4. **Binary Al-based** → mc_al_v2037_pycal.tdb (unless special cases above)
5. **User override available** with explicit ``database`` parameter

**Validation Range Warnings**:

- Queries outside validated T or composition ranges trigger **automatic warnings**
- Example: "⚠️ Temperature 1200 K exceeds COST507.tdb validation range (200-1000 K). Results may be extrapolated."
- Warnings include: temperature out-of-range, composition out-of-range, missing phase data
- Confidence scores reduced when extrapolating

**Database Provenance in Results**:

All results include ``database`` field with:

.. code-block:: python

   "database": {
       "name": "COST507.tdb",
       "source": "COST507 European database",
       "version": "2.0",
       "validation_range": "Al-Mg-Zn: T=[200-1000 K], X=[0-100 at.%]",
       "in_validation_range": True,
       "phases_available": ["FCC_A1", "HCP_A3", "TAU", "MGZN2", "LIQUID"],
       "reference_state": "SER"  # Standard Element Reference
   }

**Reference States**:

- All databases use **SER** (Standard Element Reference) convention
- Pressure: 101325 Pa (1 atm) unless explicitly overridden
- Element reference states: stable phase at 298.15 K, 1 bar (e.g., Al(FCC), Zn(HCP))

**Adding Custom Databases**:

- Place `.tdb` files in ``backend/tdbs/`` directory
- Add validation metadata in ``database_registry.py``
- System will auto-detect and incorporate in selection logic

Citations
---------

All CALPHAD functions cite:

- **pycalphad**: Otis, R. & Liu, Z.-K., (2017). pycalphad: CALPHAD-based Computational Thermodynamics in Python. *Journal of Open Research Software*. 5(1), p.1. DOI: http://doi.org/10.5334/jors.140

Notes and Best Practices
-------------------------

**Temperature and Composition Handling**:

- All internal calculations use **Kelvin** for temperature
- Display units configurable: ``temperature_unit='K'`` (default) or ``'C'`` for Celsius
- All composition inputs default to **atomic percent (at.%)** unless specified as weight percent
- Weight percent automatically converted to mole fractions internally using atomic masses
- **Assertion checks** enforce that composition fractions sum to 1.0 after wt%→at.% conversion
- Phase names mapped to readable forms (e.g., CSI → SiC, FCC_A1 → fcc, TAU_MG32(AL_ZN)49 → Tau)

**Precision and Uncertainty**:

- All equilibrium calculations converge to **numerical tolerance** (typically 1e-8 in Gibbs energy)
- Adaptive refinement reports temperatures with **stated precision** (e.g., "654.3 ± 2.0 K")
- Default refinement tolerance: ±2.0 K (user-tunable via ``refinement_tolerance`` parameter)
- Tighter tolerances (e.g., 1.0 K, 0.5 K) increase computation time proportionally
- Phase fractions reported to **4 decimal places** (0.0001 = 0.01%)
- Invariant reaction temperatures reported with uncertainty based on grid resolution

**Phase Presence Thresholds (Consistent Defaults)**:

- **Presence threshold = 0.01 (1%)**: Used for major phase presence decisions in plots and phase fraction checks
- **Reporting threshold = 0.0001 (0.01%)**: Used in single-point equilibrium output (``calculate_equilibrium_at_point``)
- **Precipitate threshold = 0.001 (0.1%)**: Recommended for tracking minor strengthening phases (user sets explicitly)
- **Rationale**:
  
  - 1% threshold avoids noise from numerical artifacts and trace metastables
  - 0.01% reporting captures all present phases for detailed analysis
  - 0.1% precipitate threshold detects age-hardening phases without over-flagging
  
- **All thresholds user-tunable** via ``phase_presence_threshold`` parameter
- Raw fractions **always stored and available** regardless of threshold (threshold affects only "present" verdict)

**Configurability and Hyperparameters**:

All major algorithmic choices are **exposed as parameters** with sensible defaults:

- **Phase filtering**: ``phase_filter_mode`` (production/research/metastable)
- **Adaptive refinement**: ``adaptive_refinement`` (True/False), ``refinement_tolerance`` (K)
- **Phase presence**: ``phase_presence_threshold`` (0-1, default 0.01 for major phases, 0.001 for precipitates)
- **Invariant detection**: ``detect_all_invariants`` (True/False)
- **Solidification model**: ``process_type`` (as_cast = Scheil, equilibrium_300K = full equilibrium)
- **Temperature units**: ``temperature_unit`` ('K' or 'C')
- **Composition units**: ``composition_unit`` ('atomic' or 'weight')

**Performance Optimization**:

- **Equilibrium caching**: Results cached by comprehensive key to avoid redundant calculations
  
  **Cache key includes** (for reproducibility and correctness):
  
  - Temperature (T)
  - Composition (X for all elements)
  - Phase list
  - Database hash (file content + version)
  - **Pressure** (usually 101325 Pa)
  - **phase_filter_mode** (production/research/metastable)
  - **Metastable flag** (if applicable)
  - **Units** (though calculations always internal Kelvin)
  
  Without comprehensive keys, subtle cross-talk can occur (e.g., cached equilibrium_300K result returned for as_cast query).

- **Parallel execution**: Independent (T,X) points computed in parallel across CPU cores
- **Adaptive sampling**: Concentrated grid points only around phase boundaries, not uniformly
- Typical performance: 
  
  - Binary phase diagram: 30-60 seconds with adaptive refinement
  - Composition-temperature plot: 10-20 seconds (Scheil: 30-45 seconds)
  - Single equilibrium point: 1-3 seconds
  - Region sweep (16 points): 2-3 minutes (Scheil solidification per point)

**File Outputs**:

- **Images**: PNG files saved to ``interactive_plots/`` directory
- **Interactive plots**: HTML files (Plotly) saved to ``interactive_plots/`` for zoom/hover/export
- **HTTP serving**: All files served at ``http://localhost:8000/static/plots/[filename]``
- **Metadata storage**: ``_last_image_metadata`` persists for ``analyze_last_generated_plot()``
- **Cache persistence**: Equilibrium cache persists within session for reuse

**Units in Titles and Legends (Critical for Clarity)**:

Every plot explicitly labels units to prevent misinterpretation:

- **Figure titles**: 
  
  - Example: "Al-Zn Phase Diagram (at.%)" or "Al-Zn Phase Diagram (wt.%)"
  - Example: "Temperature (K)" or "Temperature (°C)"
  
- **Axis labels**:
  
  - X-axis: "Composition (at.% Zn)" or "Mole Fraction Zn"
  - Y-axis: "Temperature (K)" or "Temperature (°C)"
  
- **Legend entries**: Include phase fractions with explicit units
  
  - "FCC_A1 (85.0 at.%)" not just "FCC_A1 (85.0%)"
  - "Liquidus: 654.3 K" or "Liquidus: 381.2 °C"

- **Hover tooltips** (interactive plots): Show both K and °C, both at.% and mole fraction

- **Rationale**: Prevents dangerous errors in screenshots, presentations, and publications where context may be lost

- **Implementation**: Units embedded in matplotlib/Plotly figure objects, not just in surrounding text

**Correctness Invariants**:

- N-1 composition constraints for N elements (first element is dependent variable)
- Composition fractions **must sum to 1.0** (asserted after parsing)
- Phase fractions **sum to 1.0** (checked after extraction)
- Mass balance: :math:`\sum_{\phi} f^{\phi} X_i^{\phi} = X_i^{global}` for each element i
- Vertex and sample dimensions averaged/summed correctly (defensive coding for multi-vertex results)

**When to Use Which Function**:

1. **General phase diagram questions** → ``plot_binary_phase_diagram()``
2. **Specific composition analysis** → ``plot_composition_temperature()``
3. **Single-point verification** → ``calculate_equilibrium_at_point()``
4. **Precipitation behavior** → ``calculate_phase_fractions_vs_temperature()``
5. **Trend verification** → ``analyze_phase_fraction_trend()``
6. **Composition threshold claims** → ``verify_phase_formation_across_composition()``
7. **Universal claims over regions** → ``sweep_microstructure_claim_over_region()``
8. **Single alloy fact-checking** → ``fact_check_microstructure_claim()``

**Common Pitfalls to Avoid**:

- Don't confuse **as_cast** (Scheil, non-equilibrium) with **equilibrium_300K** (infinite diffusion time)
- Don't over-interpret **mechanical desirability scores** (heuristics, not rigorous predictions)
- Don't query far outside **database validation ranges** without checking warnings
- Don't use **production filter mode** if you need ordered/disordered variants (use research mode)
- Don't assume **1% presence threshold** is right for all cases (adjust for precipitates)
- Don't trust **extrapolated** results (check ``in_validation_range`` field in response)
- Don't confuse **spinodal** (thermodynamic instability, ∂²G/∂X² < 0) with **binodal** (phase coexistence boundary)

**Spinodal vs. Binodal Clarification**:

- **Binodal boundaries** (what we compute): Phase coexistence lines where two phases have equal chemical potentials
  
  - Defines equilibrium phase boundaries
  - Liquid/solid boundaries (liquidus, solidus)
  - Solvus curves (solid solution limits)
  - Detected via Gibbs energy minimization

- **Spinodal boundaries** (NOT computed by default): Thermodynamic stability limits where ∂²G/∂X² = 0
  
  - Inside spinodal: spontaneous decomposition (no nucleation barrier)
  - Between spinodal and binodal: metastable region (requires nucleation)
  - Computation requires second derivatives of Gibbs energy
  - Not typically needed for engineering alloy design
  
- **Why we focus on binodals**: Engineering microstructures form via nucleation and growth (binodal-controlled), not spinodal decomposition (rare except in specific heat treatments)

- **If you need spinodals**: Add ``compute_spinodal=True`` parameter (future feature) or compute manually from Hessian of G

**Ternary and Multicomponent Helpers** (Advanced Features):

For users working with ternary and higher-order systems, the following additional functions are available:

**plot_isopleth** (vertical sections through ternary diagrams):

.. code-block:: python

   # Example: Al-Cu-Mg system, fix Cu at 4 wt.%, vary Mg from 0-8 wt.%
   result = await handler.plot_isopleth(
       system="Al-Cu-Mg",
       section="Al-4Cu-(Mg,0→8)",  # Al balance, Cu=4%, Mg varies 0→8%
       min_temperature=200,
       max_temperature=800,
       composition_type="weight",
       adaptive_refinement=True
   )
   # Generates 2D plot (T vs Mg) with Al and Cu constrained

- **Use cases**: Understanding precipitation sequences in commercial alloys (e.g., 7xxx Al alloys)
- **Adaptive refinement**: Applied in both T and Mg directions
- **Provenance**: Same database selection and validation as binary plots

**plot_isothermal_section** (composition triangles at fixed T):

.. code-block:: python

   # Example: Al-Mg-Zn at 300 K
   result = await handler.plot_isothermal_section(
       system="Al-Mg-Zn",
       temperature=300,
       composition_type="atomic",
       phase_filter_mode="production"
   )
   # Generates ternary composition triangle with phase fields

- **Use cases**: Composition design, identifying single-phase regions, multi-phase boundaries
- **Adaptive refinement**: Concentrates points near tie-lines and three-phase regions

**A/B Database Sensitivity Analysis**:

For critical applications, compare results across multiple databases to assess sensitivity:

.. code-block:: python

   # Run same query on two databases
   result_a = await handler.plot_binary_phase_diagram(
       system="Al-Zn",
       database="COST507"  # Explicit override
   )
   
   result_b = await handler.plot_binary_phase_diagram(
       system="Al-Zn",
       database="mc_al_v2037"
   )
   
   # Compare key outputs
   sensitivity_report = compare_database_results(result_a, result_b)
   # Returns:
   # - Δ(melting points)
   # - Δ(eutectic temperatures)
   # - Phase field boundary shifts
   # - Sensitivity flag: "HIGH" if boundaries move >10 K

- **Highly sensitive results** (>10 K variation in liquidus/solidus) trigger automatic warnings
- Useful for validation and uncertainty quantification
- Helps identify where databases disagree (often near ternary interactions or metastable phases)

**Comparison to Other CALPHAD Tools**:

- **Thermo-Calc**: Commercial, more phases, proprietary databases, GUI-focused
- **PANDAT**: Commercial, optimized for multicomponent, Windows-only
- **OpenCalphad**: Open-source Fortran, research-grade
- **PyCalphad** (this handler's engine): Open-source Python, programmatic, extensible, free
- **This handler's advantages**: AI-driven, automatic database selection, fact-checking, region sweeps, Scheil solidification, sensitivity analysis

**Literature References**:

- Scheil-Gulliver model: Scheil, E. (1942). *Zeitschrift für Metallkunde*, 34, 70-72.
- CALPHAD methodology: Saunders, N. & Miodownik, A. P. (1998). *CALPHAD (Calculation of Phase Diagrams): A Comprehensive Guide*. Elsevier.
- PyCalphad implementation: Otis, R. & Liu, Z.-K. (2017). *Journal of Open Research Software*, 5(1), 1.
