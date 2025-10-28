CALPHAD Handler
===============

The CALPHAD handler provides AI functions for generating phase diagrams, performing thermodynamic equilibrium calculations, and verifying metallurgical claims using CALPHAD (CALculation of PHAse Diagrams) methodology.

All functions use thermodynamic databases (TDB files) to compute phase equilibria via the pycalphad library.

Overview
--------

The CALPHAD handler is organized into three main categories:

1. **Visualization Functions**: Generate phase diagrams and composition-temperature plots
2. **Calculation Functions**: Compute equilibrium states and phase fractions
3. **Verification Functions**: Validate metallurgical claims and sweep composition ranges

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
       figure_height: Optional[float] = None
   ) -> str

**Description:**

**PREFERRED for phase diagram questions.** Generate a binary phase diagram for a chemical system using CALPHAD thermodynamic data. This is the primary tool for understanding phase relationships across composition ranges.

**When to Use:**

- General system queries (e.g., "show me the Al-Zn phase diagram")
- Understanding liquidus/solidus boundaries
- Identifying eutectic points and phase transitions
- Viewing complete composition range behavior

**How It Fetches Data:**

1. **Database Selection:**
   
   - Parses system string to extract elements (e.g., "Al-Zn" → ['AL', 'ZN'])
   - Calls ``load_tdb_database([A, B])`` which implements element-based database selection:
     
     - For Al-Mg-Zn systems → COST507.tdb
     - For systems with C, N, B, Li → COST507.tdb
     - For Al-based systems → mc_al_v2037_pycal.tdb
     - Checks ``backend/tdbs/`` directory for matching .tdb files
   
   - Validates both elements exist in selected database

2. **Phase Selection:**
   
   - Calls ``_filter_phases_for_system(db, (A, B))`` to get relevant phases
   - Excludes phases with unwanted patterns (ION_LIQUID, HALIDE_*, etc.)
   - Returns only phases containing the specified elements

**How It Calculates:**

1. **Temperature Range:**
   
   - Auto mode: Uses wide bracket (200-2300 K) to capture high-melting systems
   - Manual mode: Uses user-specified range
   - Handles degenerate case (min==max) by expanding ±100K
   - Clamps temperature points: 12-60 points depending on range

2. **Phase Diagram Generation:**
   
   - Calls pycalphad's ``binplot()`` function with:
     
     - Composition range: X(B) from 0 to 1 with step size (default 0.02)
     - Temperature range with adaptive point count
     - Pressure: 101325 Pa (1 atm)
     - N: 1 mole
   
   - ``binplot`` internally:
     
     - Computes equilibrium at each (T, X) grid point using Gibbs energy minimization
     - Identifies phase boundaries where phase stability changes
     - Draws phase field regions and boundaries

3. **Eutectic Detection:**
   
   - Runs coarse equilibrium grid: ``_coarse_equilibrium_grid(db, A, B, phases, (T_lo, T_hi), nx=101, nT=161)``
   - Extracts liquidus/solidus data from equilibrium results
   - Identifies eutectic points using ``_find_eutectic_points()`` with:
     
     - delta_T=10.0 K tolerance
     - min_spacing=0.03 composition separation
     - eps_drop=0.1 K sensitivity for temperature minima
   
   - Marks eutectics on diagram with annotations

4. **Analysis Generation:**
   
   - Visual analysis: Examines matplotlib figure and axes properties
   - Thermodynamic analysis: ``_analyze_phase_diagram()`` extracts:
     
     - Pure element melting points (from phase field boundaries at X=0 and X=1)
     - Eutectic temperatures and compositions
     - Phase transition boundaries

**Parameters:**

- ``system`` (str, required): Chemical system in any format:
  
  - Hyphenated: ``'Al-Zn'``, ``'aluminum-zinc'``
  - Concatenated: ``'AlZn'``
  - Full names: ``'aluminum-zinc'``

- ``min_temperature`` (float, optional): Minimum temperature in Kelvin. Default: auto-detect (200 K)
- ``max_temperature`` (float, optional): Maximum temperature in Kelvin. Default: auto-detect (2300 K)
- ``composition_step`` (float, optional): Composition step size (0-1 range). Default: 0.02
- ``figure_width`` (float, optional): Figure width in inches. Default: 9
- ``figure_height`` (float, optional): Figure height in inches. Default: 6

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
           "temperature_range_K": [200.0, 2300.0],
           "key_points": [
               {"type": "pure_melting", "element": "AL", "temperature": 933.5},
               {"type": "eutectic", "temperature": 654.3, "composition_pct": 72.5, "reaction": "LIQUID → FCC_A1 + HCP_A3"}
           ]
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
- Caches equilibrium data in ``_cached_eq_coarse`` (cleaned up after analysis)

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

   # Generate Al-Zn phase diagram with auto temperature range
   result = await handler.plot_binary_phase_diagram(
       system="Al-Zn"
   )
   
   # Generate Fe-Al phase diagram with specific temperature range
   result = await handler.plot_binary_phase_diagram(
       system="Fe-Al",
       min_temperature=500,
       max_temperature=1800,
       composition_step=0.01  # Finer composition resolution
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
       interactive: Optional[str] = "html"
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

1. **Temperature Array Generation:**
   
   - Auto mode: T_lo=200 K, T_hi=2300 K (wide bracket)
   - Manual mode: uses user-specified range
   - Handles degenerate min==max case by expanding ±100K
   - Adaptive point count: ``n_temp = max(50, min(200, int((T_hi - T_lo) / 5)))``
     
     - 50-200 temperature points depending on range
     - ~5K spacing for reasonable resolution

2. **Equilibrium Calculations at Each Temperature:**
   
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
   
   - Stores fractions for each phase: ``phase_data[phase].append(fraction)``

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

- ``min_temperature`` (float, optional): Minimum temperature in Kelvin. Default: auto (200 K)
- ``max_temperature`` (float, optional): Maximum temperature in Kelvin. Default: auto (2300 K)
- ``composition_type`` (str, optional): 
  
  - ``'atomic'`` (default): Atomic percent (at.%)
  - ``'weight'``: Weight percent (wt.%, automatically converted to mole fractions)

- ``figure_width`` (float, optional): Figure width in inches. Default: 8 (HTML), 10 (matplotlib)
- ``figure_height`` (float, optional): Figure height in inches. Default: 6
- ``interactive`` (str, optional): Output mode. Default: ``'html'``
  
  - ``'html'``: Generates interactive Plotly HTML with static PNG export
  - Other values: Static matplotlib plot only

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
       composition_type: Optional[str] = "atomic"
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

3. **Phase Composition Extraction:**
   
   For each stable phase:
   
   - Extracts composition using ``eqp['X'].sel(component=elem)``
   - Masks data for specific phase: ``x_data.where(phase_mask)``
   - Averages over vertices: ``x_val = float(x_data.mean().values)``
   - Stores: ``phase_comp[elem] = x_val`` (mole fraction of elem in phase)

4. **Phase Name Mapping:**
   
   - Calls ``map_phase_name(phase)`` to convert database names to readable forms:
     
     - CSI → SiC
     - AL4C3 → Al4C3
     - FCC_A1 → FCC_A1 (kept as is)
     - MGZN2 → MgZn2

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

1. **For Each Grid Point:**
   
   - Formats composition string: "Al92.0-Mg4.0-Zn4.0"
   - Calls ``fact_check_microstructure_claim()`` with same parameters:
     
     - Same claim_type, expected_phases, phase_to_check, min/max_fraction
     - Same process_type (as_cast or equilibrium_300K)
   
   - Receives verdict (True/False) and score (-2 to +2)

2. **Mechanical Desirability Check (if required):**
   
   If ``require_mechanical_desirability=True``:
   
   - Extracts ``mechanical_score`` from fact_check result
   - ``mechanical_ok = (mechanical_score > 0)``
   - ``overall_pass = microstructure_verdict AND mechanical_ok``
   
   This adds an additional filter: even if phases match, composition fails if mechanical properties are poor (brittle intermetallics dominant).

3. **Aggregation:**
   
   - Counts: ``pass_count``, ``fail_count``, ``mech_fail_count``
   - Computes: ``pass_fraction = pass_count / total_points``
   - Stores grid results with:
     
     - composition dict and string
     - microstructure_verdict (phases match?)
     - mechanical_ok (good ductility?)
     - overall_pass (both conditions met)
     - score, phases list, error (if any)

4. **Overall Verdict:**
   
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
       "grid_results": [  # first 20 points
           {
               "composition": {"AL": 88.0, "MG": 8.0, "ZN": 4.0},
               "composition_str": "Al88.0-Mg8.0-Zn4.0",
               "microstructure_verdict": True,
               "mechanical_ok": True,
               "overall_pass": True,
               "score": 2,
               "phases": [("FCC_A1", 0.85, "fcc"), ("TAU", 0.15, "tau")],
               "error": None
           },
           ...
       ],
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

- Uses nested loops to generate composition grid
- Calls ``fact_check_microstructure_claim()`` for each point (see next section)
- Supports 1D (one element varies) and 2D (two elements vary) sweeps
- ``mechanical_desirability_score()`` evaluates:
  
  - High FCC (>85%) with modest intermetallics (<15%) → +1 (ductile)
  - Very high intermetallics (>20%) or Laves (>15%) → -1 (brittle)
  - Otherwise → 0 (mixed)

- Execution time scales as O(grid_points^n_varying_elements × equilibrium_time)
- For 4×4 grid with ~2s per equilibrium: ~60 seconds total

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
   
   **Option A: as_cast (default)**
   
   Simulates slow solidification from melt:
   
   .. code-block:: python
   
      # Find liquidus and solidus temperatures
      T_liquidus, T_solidus = find_liquidus_solidus_temperatures(...)
      # Calculates by sweeping T and checking liquid fraction
      
      # Set as-cast temperature: ~20K below solidus
      T_ascast = T_solidus - 20.0
      
      # Calculate equilibrium just after solidification
      # Excludes LIQUID phase (already frozen)
      solid_phases = [p for p in phases if p != "LIQUID"]
      eq = equilibrium(db, elements_with_va, solid_phases, 
                      {v.T: T_ascast, v.P: 101325, ...})
      
      # Extract phase fractions
      precalc_fractions = extract_phase_fractions_from_equilibrium(eq)
   
   This approximates **"what you get after the alloy freezes"** without infinite solid-state diffusion.
   
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

4. **Mechanical Desirability (for as_cast only):**
   
   .. code-block:: python
   
      # Extract phase categories
      phase_categories = {p.base_name: p.category.value for p in microstructure}
      
      mech_score, mech_interpretation = mechanical_desirability_score(
          precalc_fractions, phase_categories
      )
      
      # Rules of thumb:
      # High FCC (>85%) + modest intermetallics (<15%) → +1 (ductile)
      # Very high intermetallics (>20%) or Laves (>15%) → -1 (brittle)
      # Otherwise → 0 (mixed)

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

**as_cast (default)**:

- Simulates slow solidification from the melt
- Answers: "What phases form after the alloy freezes?"
- Uses solidification path: finds liquidus→solidus, calculates equilibrium ~20K below solidus
- More realistic for cast alloys (avoids infinite solid-state diffusion assumption)
- Includes mechanical desirability scoring

**equilibrium_300K**:

- Full thermodynamic equilibrium at specified temperature (default 300K)
- Answers: "What is the equilibrium state after infinite diffusion time?"
- Excludes metastable phases (e.g., liquid at low T)
- More relevant for fully annealed/aged conditions
- Does not evaluate mechanical desirability

Database Support
----------------

Currently supported thermodynamic databases:

- ``COST507.tdb``: Al-based systems (Al-Zn, Al-Si, Al-Mg, etc.)
- ``mc_al_v2037_pycal.tdb``: Multi-component aluminum alloys

Available systems include: Al-Zn, Al-Si, Al-Mg, Al-Cu, Fe-Al, and more.

Citations
---------

All CALPHAD functions cite:

- **pycalphad**: Otis, R. & Liu, Z.-K., (2017). pycalphad: CALPHAD-based Computational Thermodynamics in Python. *Journal of Open Research Software*. 5(1), p.1. DOI: http://doi.org/10.5334/jors.140

Notes
-----

- All temperature inputs are in Kelvin
- All composition inputs default to atomic percent (at.%) unless specified as weight percent
- Weight percent is automatically converted to mole fractions internally
- Phase names are mapped to readable forms (e.g., CSI → SiC, FCC_A1 → fcc)
- Images and HTML files are saved to ``interactive_plots/`` and served via HTTP
