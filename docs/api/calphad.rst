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

**PREFERRED for phase diagram questions.**

Generate a binary phase diagram for a chemical system using CALPHAD thermodynamic data. This is the primary tool for understanding phase relationships across composition ranges.

**When to Use:**

- General system queries (e.g., "show me the Al-Zn phase diagram")
- Understanding liquidus/solidus boundaries
- Identifying eutectic points and phase transitions
- Viewing complete composition range behavior

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

Success message containing:

- System identification (e.g., "AL-ZN")
- List of phases present
- Temperature range used
- Pure element melting points
- Eutectic points (temperature, composition, and reaction)
- Key phase boundaries

**Side Effects:**

- Saves PNG image to ``interactive_plots/`` directory
- Image served at ``http://localhost:8000/static/plots/[filename]``
- Stores metadata in ``_last_image_metadata`` for later analysis

**Example:**

.. code-block:: python

   # Generate Al-Zn phase diagram
   result = await handler.plot_binary_phase_diagram(
       system="Al-Zn",
       min_temperature=300,
       max_temperature=1000
   )
   # Returns: "Successfully generated AL-ZN phase diagram showing phases: FCC_A1, HCP_A3, LIQUID..."

**Technical Details:**

- Uses pycalphad's ``binplot`` for visualization
- Automatically detects and marks eutectic points
- Wide auto temperature range (200-2300 K) ensures high-melting systems are captured
- Currently supports systems available in loaded TDB databases (Al-Zn, Al-Si, Fe-Al, etc.)

.. _plot_composition_temperature:

plot_composition_temperature
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**PREFERRED for composition-specific thermodynamic questions.**

Plot phase stability versus temperature for a specific composition. Shows which phases are stable at different temperatures for a fixed composition.

**When to Use:**

- Analyzing specific compositions (e.g., "Al20Zn80", "pure Al")
- Understanding melting point of specific alloys
- Identifying phase transitions for a composition
- Visualizing phase stability ranges

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

- ``figure_width`` (float, optional): Figure width in inches. Default: 8
- ``figure_height`` (float, optional): Figure height in inches. Default: 6
- ``interactive`` (str, optional): Output mode. Default: ``'html'``
  
  - ``'html'``: Generates interactive Plotly HTML with static PNG export

**Returns:**

Success message containing:

- Composition string (e.g., "AL20ZN80")
- System identification
- Temperature range
- Phase stability information

**Side Effects:**

- Saves PNG to ``interactive_plots/`` directory
- Saves interactive HTML to ``interactive_plots/`` directory (if ``interactive='html'``)
- Both files served at ``http://localhost:8000/static/plots/[filename]``

**Example:**

.. code-block:: python

   # Plot phase stability for Al-20Zn alloy
   result = await handler.plot_composition_temperature(
       composition="Al20Zn80",
       min_temperature=300,
       max_temperature=900,
       composition_type="atomic"
   )

**Technical Details:**

- Computes equilibrium at 50-200 temperature points (adaptive)
- Uses stacked area plot to show phase fraction evolution
- Handles both atomic and weight percent inputs
- Interactive HTML includes hover tooltips and zoom capabilities

.. _analyze_last_generated_plot:

analyze_last_generated_plot
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Analyze and interpret the most recently generated phase diagram or composition plot. Provides detailed analysis of visual features, phase boundaries, and thermodynamic insights.

**When to Use:**

- After generating a plot, to get detailed interpretation
- Understanding key features of a phase diagram
- Extracting quantitative information from plots

**Parameters:** None

**Returns:**

Success message containing:

- Visual analysis of the plot content
- Thermodynamic analysis (phases, boundaries, invariant points)
- Phase stability information
- Key temperatures and compositions

**Technical Details:**

- Accesses cached metadata from last plot generation
- No re-computation required (uses stored analysis)
- Image data may be cleared to save memory after display

Calculation Functions
---------------------

.. _calculate_equilibrium_at_point:

calculate_equilibrium_at_point
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Calculate thermodynamic equilibrium phase fractions at a specific temperature and composition. Use to verify phase amounts at a single condition.

**When to Use:**

- Determining exact phase fractions at specific conditions
- Verifying equilibrium state at a point
- Getting detailed composition of each phase

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

Formatted text with:

- Temperature in K and °C
- Input composition
- List of stable phases with:
  
  - Phase name (mapped to readable form, e.g., CSI → SiC)
  - Phase fraction (%)
  - Composition of each phase

- Total phase fraction verification

**Example:**

.. code-block:: python

   # Calculate equilibrium for Al-Si-C alloy at 1000K
   result = await handler.calculate_equilibrium_at_point(
       composition="Al30Si55C15",
       temperature=1000.0,
       composition_type="atomic"
   )
   # Returns formatted text: "Equilibrium at 1000.0 K for Al30.0Si55.0C15.0..."

**Technical Details:**

- Uses pycalphad's ``equilibrium`` function
- Automatically selects appropriate TDB database
- Handles multi-component systems (2+ elements)
- Phase names mapped to readable forms (e.g., FCC_A1, AL4C3, SIC)

.. _calculate_phase_fractions_vs_temperature:

calculate_phase_fractions_vs_temperature
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Calculate how phase fractions change with temperature for a specific composition. Essential for understanding precipitation, dissolution, and phase transformations.

**When to Use:**

- Understanding precipitation behavior (phase fraction increasing with cooling)
- Analyzing dissolution behavior (phase fraction decreasing with heating)
- Identifying phase transformation temperatures
- Mapping solvus boundaries

**Parameters:**

- ``composition`` (str, required): Composition as element-number pairs (e.g., ``'Al30Si55C15'``)
- ``min_temperature`` (float, required): Minimum temperature in Kelvin
- ``max_temperature`` (float, required): Maximum temperature in Kelvin
- ``temperature_step`` (float, optional): Temperature step in Kelvin. Default: 10
- ``composition_type`` (str, optional): ``'atomic'`` (default) or ``'weight'``

**Returns:**

Formatted text containing:

- Temperature range (K and °C)
- Composition
- Number of temperature points computed
- Phase evolution for each phase:
  
  - Fraction at start temperature
  - Fraction at end temperature
  - Trend (increasing/decreasing/stable)
  - Magnitude of change

**Example:**

.. code-block:: python

   # Analyze phase evolution for Al-Si-C from 300-1500K
   result = await handler.calculate_phase_fractions_vs_temperature(
       composition="Al30Si55C15",
       min_temperature=300,
       max_temperature=1500,
       temperature_step=10
   )

**Technical Details:**

- Computes equilibrium at each temperature step
- Handles multi-component systems
- Stores data internally for potential follow-up analysis
- Identifies and reports trends (increasing/decreasing/stable)

.. _analyze_phase_fraction_trend:

analyze_phase_fraction_trend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Analyze whether a specific phase increases or decreases with temperature. Use to verify statements about precipitation or dissolution behavior.

**When to Use:**

- Verifying claims like "Phase X increases with decreasing temperature"
- Testing statements about precipitation upon cooling
- Confirming dissolution behavior upon heating

**Parameters:**

- ``composition`` (str, required): Composition (e.g., ``'Al30Si55C15'``)
- ``phase_name`` (str, required): Name of phase to analyze (e.g., ``'AL4C3'``, ``'SIC'``, ``'FCC_A1'``)
- ``min_temperature`` (float, required): Minimum temperature in Kelvin
- ``max_temperature`` (float, required): Maximum temperature in Kelvin
- ``expected_trend`` (str, optional): Expected trend for verification:
  
  - ``'increase'`` / ``'decrease'`` / ``'stable'``
  - Can include context: ``'increases with cooling'``, ``'decreases upon heating'``

**Returns:**

Formatted analysis containing:

- Phase name and composition
- Temperature range
- Fraction at low and high temperatures
- Change magnitude
- Maximum and minimum fractions observed
- Trend description
- Verification result (if ``expected_trend`` provided): ✅ matches or ❌ does not match

**Example:**

.. code-block:: python

   # Verify if SiC precipitates upon cooling
   result = await handler.analyze_phase_fraction_trend(
       composition="Al30Si55C15",
       phase_name="SIC",
       min_temperature=300,
       max_temperature=1500,
       expected_trend="increases with cooling"
   )

**Technical Details:**

- Samples 50 temperature points across range
- Sums all instances of phase (e.g., SIC#1 + SIC#2)
- Compares trend against expected behavior
- Handles natural language trend descriptions

Advanced Verification Functions
--------------------------------

.. _verify_phase_formation_across_composition:

verify_phase_formation_across_composition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Verify phase formation statements across a composition range. Use to check claims about composition-dependent phase appearance.

**When to Use:**

- Checking claims like "beyond 50% Al, phase X forms"
- Verifying "at compositions greater than X%, phase Y appears"
- Testing composition thresholds for phase stability
- Binary or ternary system analysis

**Parameters:**

- ``system`` (str, required): System specification:
  
  - Binary: ``'Fe-Al'``, ``'Al-Zn'``
  - Ternary: ``'Al-Mg-Zn'``

- ``phase_name`` (str, required): Phase to check:
  
  - Exact database name: ``'MGZN2'``, ``'TAU'``, ``'FCC_A1'``
  - Category name: ``'Laves'``, ``'tau'``, ``'fcc'``, ``'gamma'``

- ``composition_threshold`` (float, required): Threshold value in at.% (e.g., ``50.0`` for 50 at.%)
- ``threshold_element`` (str, required): Element being thresholded (e.g., ``'Al'`` in "beyond 50% Al")
- ``temperature`` (float, optional): Temperature in K for checking. Default: 300
- ``composition_type`` (str, optional): ``'atomic'`` (default) or ``'weight'``

**For Ternary Systems Only:**

- ``fixed_element`` (str, optional): Element to keep constant (e.g., ``'Zn'``)
- ``fixed_composition`` (float, optional): Fixed element composition in at.% (e.g., ``4.0`` for 4%)

**Returns:**

Detailed analysis including:

- Summary statistics (phase presence below/above threshold)
- Example compositions and their phases
- Verification verdict:
  
  - ✅ **VERIFIED**: Phase forms above threshold only
  - ⚠️ **PARTIALLY VERIFIED**: Forms more frequently above threshold
  - ❌ **CONTRADICTED**: Opposite behavior or no clear threshold

- Detailed composition scan table

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

**Technical Details:**

- Samples 20+ compositions with extra points near threshold
- Computes equilibrium at each composition
- Aggregates phases if category name given (e.g., "tau" matches all tau variants)
- Uses frequency-based comparison (not raw counts)

.. _sweep_microstructure_claim_over_region:

sweep_microstructure_claim_over_region
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sweep composition space and evaluate whether a microstructure claim holds across an entire region. Tests universal claims over composition ranges.

**When to Use:**

- Testing claims like "all Al-Mg-Zn alloys with Mg<8% and Zn<4% form fcc+tau"
- Validating design rules over composition regions
- Assessing universality of metallurgical statements

**Parameters:**

- ``system`` (str, required): Chemical system (e.g., ``'Al-Mg-Zn'``)

- ``element_ranges`` (str, required): JSON dict of element ranges. Example:
  
  .. code-block:: json

     {"MG": [0, 8], "ZN": [0, 4]}

  Units determined by ``composition_type``. Remaining composition is balance element.

- ``claim_type`` (str, required): Type of claim to verify:
  
  - ``'two_phase'``: Expects exactly two phases (matrix + secondary)
  - ``'three_phase'``: Expects exactly three phases
  - ``'phase_fraction'``: Checks specific phase fraction bounds

- ``expected_phases`` (str, optional): Expected phases for two_phase/three_phase claims (e.g., ``'fcc+tau'``, ``'fcc+tau+gamma'``)
- ``phase_to_check`` (str, optional): Phase name for phase_fraction claims
- ``min_fraction`` (float, optional): Minimum phase fraction (0-1)
- ``max_fraction`` (float, optional): Maximum phase fraction (0-1)
- ``grid_points`` (int, optional): Number of grid points per element. Default: 4 (16 total for 2D)
- ``composition_type`` (str, optional): ``'atomic'`` (default, at.%) or ``'weight'`` (wt.%, not yet implemented)
- ``process_type`` (str, optional): 
  
  - ``'as_cast'`` (default): After solidification
  - ``'equilibrium_300K'``: Infinite-time room temperature equilibrium

- ``require_mechanical_desirability`` (bool, optional): Also require positive mechanical desirability. Default: False

**Returns:**

Dictionary containing:

- Overall verdict: "UNIVERSALLY SUPPORTED", "MOSTLY SUPPORTED", "MIXED", "MOSTLY REJECTED", "UNIVERSALLY REJECTED"
- Score: -2 to +2
- Confidence: 0-1
- Pass/fail counts and fraction
- Sample of failed compositions with reasons
- Grid results (first 20 points)

**Example:**

.. code-block:: python

   # Test if all Al-Mg<8%-Zn<4% alloys form fcc+tau after casting
   result = await handler.sweep_microstructure_claim_over_region(
       system="Al-Mg-Zn",
       element_ranges='{"MG": [0, 8], "ZN": [0, 4]}',
       claim_type="two_phase",
       expected_phases="fcc+tau",
       max_fraction=0.20,
       grid_points=4,
       process_type="as_cast"
   )

**Technical Details:**

- Generates grid over specified composition ranges
- Evaluates claim at each grid point using ``fact_check_microstructure_claim``
- Supports 1D and 2D sweeps
- Can filter by mechanical desirability (ductile vs brittle phases)

.. _fact_check_microstructure_claim:

fact_check_microstructure_claim
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Evaluate microstructure claims for multicomponent alloys. Acts as an automated "materials expert witness" to verify metallurgical assertions.

**When to Use:**

- Fact-checking metallurgical statements
- Verifying alloy compositions meet design criteria
- Evaluating claims from literature or specifications

**Parameters:**

- ``system`` (str, required): Chemical system (e.g., ``'Al-Mg-Zn'``, ``'Fe-Cr-Ni'``)

- ``composition`` (str, required): Composition in at.%:
  
  - Concatenated: ``'Al88Mg8Zn4'``
  - Hyphenated: ``'88Al-8Mg-4Zn'``
  - Element-first: ``'Al-8Mg-4Zn'``

- ``claim_type`` (str, required): Type of claim (see ``sweep_microstructure_claim_over_region``)

- ``expected_phases`` (str, optional): For two_phase/three_phase claims

- ``phase_to_check`` (str, optional): For phase_fraction claims

- ``min_fraction`` (float, optional): Minimum phase fraction (0-1)

- ``max_fraction`` (float, optional): Maximum phase fraction (0-1)

- ``process_type`` (str, optional): 
  
  - ``'as_cast'`` (default): Simulates slow solidification from melt
  - ``'equilibrium_300K'``: Infinite diffusion time at room temperature

- ``temperature`` (float, optional): Temperature in K (only used for ``'equilibrium_300K'``)

- ``composition_constraints`` (str, optional): JSON string of composition constraints. Example:
  
  .. code-block:: json

     {"MG": {"lt": 8.0}, "ZN": {"lt": 4.0}}

  Supported operators: ``lt``, ``lte``, ``gt``, ``gte``, ``between``

**Returns:**

Dictionary containing:

- **verdict**: True/False (claim supported or rejected)
- **score**: -2 to +2
  
  - +2: Fully correct
  - +1: Mostly correct
  - 0: Partially correct
  - -1: Mostly wrong
  - -2: Completely wrong

- **confidence**: 0-1
- **reasoning**: Explanation of verdict
- **mechanical_score**: -1/0/+1 (brittleness/desirability for as_cast)
- **supporting_data**: Phase fractions and details
- **citations**: ["pycalphad"]

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
   # Returns: verdict=True/False, score=-2 to +2, reasoning, etc.

**Technical Details:**

- Uses sophisticated solidification simulation for ``'as_cast'``
- Maps CALPHAD phase names to metallurgical categories (fcc, bcc, tau, Laves, etc.)
- Evaluates mechanical desirability based on phase distribution
- Checks composition constraints if provided

Process Models
^^^^^^^^^^^^^^

**as_cast (default)**:

- Simulates slow solidification from the melt
- Answers: "What phases form after the alloy freezes?"
- Uses solidification path and freezing range temperature
- More realistic for cast alloys

**equilibrium_300K**:

- Full thermodynamic equilibrium at specified temperature
- Answers: "What is the equilibrium state after infinite diffusion time?"
- Excludes metastable phases (e.g., liquid at low T)
- More relevant for fully annealed/aged conditions

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
