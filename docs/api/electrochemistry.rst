Electrochemistry Handler
========================

The Electrochemistry handler provides AI functions for battery electrode analysis, voltage calculations, and electrochemical property predictions using Materials Project's DFT-calculated database and thermodynamic modeling.

All calculations are based on rigorous convex hull analysis and first-principles density functional theory (DFT) data.

Overview
--------

The Electrochemistry handler is organized into three main categories:

1. **Search and Discovery**: Finding electrode materials with specific properties
2. **Calculation and Analysis**: Computing voltages, capacities, and energy densities
3. **Stability and Mechanism**: Analyzing thermodynamic stability and lithiation mechanisms

Core Functions
--------------

.. _search_battery_electrodes:

search_battery_electrodes
^^^^^^^^^^^^^^^^^^^^^^^^^

Search for battery electrode materials and their voltage profiles from the Materials Project insertion electrodes database.

**When to Use:**

- Finding electrode materials by formula or elements
- Searching for electrodes with specific voltage ranges
- Filtering by gravimetric capacity
- General battery electrode queries

**Parameters:**

- ``formula`` (str, optional): Chemical formula of electrode material (e.g., ``'AlMg'``, ``'Al2Mg3'``, ``'LiCoO2'``)
- ``elements`` (str, optional): Comma-separated elements in electrode (e.g., ``'Al,Mg'`` or ``'Li,Co,O'``)
- ``working_ion`` (str, optional): Working ion for the battery (e.g., ``'Li'``, ``'Na'``, ``'Mg'``). Default: ``'Li'``
- ``min_capacity`` (float, optional): Minimum gravimetric capacity in mAh/g
- ``max_capacity`` (float, optional): Maximum gravimetric capacity in mAh/g  
- ``min_voltage`` (float, optional): Minimum average voltage vs working ion in V
- ``max_voltage`` (float, optional): Maximum average voltage vs working ion in V
- ``max_entries`` (int, optional): Maximum number of results to return. Default: 10

**Returns:**

Dictionary containing:

- ``count``: Number of electrode materials found
- ``electrodes``: List of electrode data with:
  
  - ``material_id``: Materials Project ID
  - ``formula``: Chemical formula
  - ``framework``: Host framework formula
  - ``average_voltage``: Average voltage vs working ion (V)
  - ``min_voltage``: Minimum voltage (V)
  - ``max_voltage``: Maximum voltage (V)
  - ``capacity_grav``: Gravimetric capacity (mAh/g)
  - ``capacity_vol``: Volumetric capacity (Ah/L)
  - ``energy_grav``: Gravimetric energy density (Wh/kg)
  - ``energy_vol``: Volumetric energy density (Wh/L)
  - ``stability``: Energy above hull (eV/atom)

- ``query``: Query parameters used

**Example:**

.. code-block:: python

   # Search for Al-Mg electrodes for Li-ion batteries
   result = await handler.search_battery_electrodes(
       elements="Al,Mg",
       working_ion="Li",
       min_voltage=0.0,
       max_voltage=1.0,
       max_entries=5
   )

**Technical Details:**

- First searches Materials Project insertion_electrodes database (pre-computed voltage profiles)
- Falls back to convex hull computation if no curated data available
- Post-filters results to ensure framework matches requested elements
- All voltages are vs. working ion reference (e.g., Li/Li⁺)

.. _calculate_voltage_from_formation_energy:

calculate_voltage_from_formation_energy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Calculate electrode voltage from Materials Project DFT entries using convex hull analysis and PyMatGen's InsertionElectrode class.

**When to Use:**

- Computing voltage for novel electrode materials not in curated database
- Getting thermodynamically rigorous voltage predictions
- Understanding voltage from first principles

**Parameters:**

- ``electrode_formula`` (str, required): Chemical formula of electrode material (e.g., ``'Al3Mg2'``, ``'AlMg'``)
- ``working_ion`` (str, optional): Working ion element symbol (e.g., ``'Li'``, ``'Na'``, ``'Mg'``). Default: ``'Li'``
- ``temperature`` (float, optional): Temperature in Kelvin (currently unused - 0K hull used). Default: 298.15

**Returns:**

Dictionary containing:

- ``calculation_method``: Method used (``'insertion_electrode'`` or ``'phase_diagram_line_scan'``)
- ``calculated_voltage``: Average voltage vs working ion (V)
- ``chemical_system``: Chemical system analyzed
- ``framework_formula``: Host framework formula
- ``voltage_range``: Dictionary with min, max, and average voltages (V)
- ``capacity_grav``: Gravimetric capacity (mAh/g)
- ``energy_grav``: Gravimetric energy density (Wh/kg)
- ``electrode_material``: Material properties including formation energy

**Example:**

.. code-block:: python

   # Calculate voltage for Al-Mg alloy vs Li
   result = await handler.calculate_voltage_from_formation_energy(
       electrode_formula="AlMg",
       working_ion="Li"
   )

**Technical Details:**

- Uses two-phase equilibria on convex hull for physically valid voltages
- All data from consistent set of ComputedEntry objects from Materials Project
- Falls back to phase diagram line scan if InsertionElectrode fails
- Returns error if no suitable framework found or voltage is unphysical

.. _get_voltage_profile:

get_voltage_profile
^^^^^^^^^^^^^^^^^^^

Get detailed voltage profile and phase evolution data for a specific electrode material. Shows how voltage changes during charge/discharge cycles.

**When to Use:**

- Visualizing full charge/discharge curve
- Understanding voltage vs capacity relationship
- Analyzing voltage plateaus and phase transitions

**Parameters:**

- ``material_id`` (str, required): Materials Project ID of the electrode (e.g., ``'mp-12345'``) or battery ID
- ``working_ion`` (str, optional): Working ion element symbol (e.g., ``'Li'``, ``'Na'``). Default: ``'Li'``

**Returns:**

Dictionary containing:

- ``material_id``: Materials Project ID
- ``formula``: Electrode formula
- ``framework``: Host framework formula
- ``working_ion``: Working ion symbol
- ``voltage_profile``: List of voltage-capacity points
- ``average_voltage``: Average voltage (V)
- ``capacity_grav``: Gravimetric capacity (mAh/g)
- ``energy_grav``: Gravimetric energy density (Wh/kg)
- ``voltage_steps``: Number of voltage plateaus

**Example:**

.. code-block:: python

   # Get voltage profile for a specific electrode
   result = await handler.get_voltage_profile(
       material_id="mp-12345",
       working_ion="Li"
   )

**Technical Details:**

- Retrieves pre-computed voltage profiles from Materials Project
- Returns full charge/discharge curve data
- Identifies voltage plateaus and phase transitions

.. _compare_electrode_materials:

compare_electrode_materials
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Compare multiple electrode materials side-by-side for battery applications. **USE THIS** for questions like "does X increase voltage vs Y", "compare Al vs AlMg", "which is better: A or B".

**When to Use:**

- Comparing voltages of different electrode materials
- Evaluating effect of alloying on electrochemical performance
- Determining which material has higher/lower voltage or capacity

**Parameters:**

- ``formulas`` (str, required): Comma-separated list of chemical formulas to compare (e.g., ``'Al,AlMg'`` or ``'LiCoO2,LiFePO4'``)
- ``working_ion`` (str, optional): Working ion for comparison (e.g., ``'Li'``, ``'Na'``). Default: ``'Li'``

**Returns:**

Dictionary containing:

- ``working_ion``: Working ion used
- ``comparison``: List of comparison results for each material with:
  
  - ``formula``: Material formula
  - ``data``: Voltage, capacity, energy density data
  - ``source``: Data source (``'electrodes_database'`` or ``'calculated_from_formation_energy'``)

- ``count``: Number of materials compared
- ``summary``: Textual comparison summary including:
  
  - Which material has higher/lower voltage
  - Voltage differences
  - Capacity and energy density comparisons

**Example:**

.. code-block:: python

   # Compare Al and AlMg as Li-ion battery anodes
   result = await handler.compare_electrode_materials(
       formulas="Al,AlMg",
       working_ion="Li"
   )

**Technical Details:**

- Tries electrodes database first, falls back to formation energy calculation
- Automatically generates comparison summary
- All data from Materials Project DFT calculations - no heuristics

Stability and Mechanism Analysis
---------------------------------

.. _check_composition_stability:

check_composition_stability
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Check if a composition is thermodynamically stable (on convex hull). **USE THIS** to determine if a material can exist as a stable phase or if it decomposes.

**When to Use:**

- Determining if a material is thermodynamically stable
- Finding decomposition products
- Questions about "can form", "stable phase", "energy above hull"

**Parameters:**

- ``composition`` (str, required): Chemical composition to check (e.g., ``'Cu8LiAl'``, ``'Li3Al2'``, ``'Cu80Li10Al10'``)

**Returns:**

Dictionary containing:

- ``composition``: Input composition
- ``is_stable``: Boolean indicating if composition is on convex hull
- ``energy_above_hull``: Energy above hull in eV/atom (0 = stable, None if no entry exists)
- ``material_id``: Materials Project ID if entry exists
- ``decomposition``: List of decomposition products if unstable, with:
  
  - ``formula``: Decomposition product formula
  - ``fraction``: Mole fraction
  - ``material_id``: Materials Project ID

- ``hull_energy``: Energy on convex hull (eV/atom)
- ``formation_energy``: Formation energy per atom (eV/atom)

**Example:**

.. code-block:: python

   # Check if Cu8LiAl is thermodynamically stable
   result = await handler.check_composition_stability(
       composition="Cu8LiAl"
   )

**Technical Details:**

- Uses 0 K convex hull from DFT calculations
- Returns energy above hull (E_above_hull = 0 means stable)
- Provides decomposition products if unstable
- Returns None for E_above_hull if composition not in database

.. _analyze_anode_viability:

analyze_anode_viability
^^^^^^^^^^^^^^^^^^^^^^^

Comprehensive analysis of a composition as a potential battery anode, including stability check and voltage calculation. **USE THIS** for questions about whether a material "can form an anode" or "is suitable as anode".

**When to Use:**

- Evaluating materials as potential battery anodes
- Combined stability + voltage analysis
- Determining anode viability

**Parameters:**

- ``composition`` (str, required): Chemical composition to analyze (e.g., ``'Cu8LiAl'``, ``'AlMg'``, ``'Li3Al2'``)
- ``working_ion`` (str, optional): Working ion for battery (e.g., ``'Li'``, ``'Na'``). Default: ``'Li'``

**Returns:**

Dictionary containing:

- ``composition``: Input composition
- ``working_ion``: Working ion used
- ``is_stable``: Thermodynamic stability boolean
- ``energy_above_hull``: Energy above hull (eV/atom)
- ``voltage_data``: Voltage calculation results (if stable enough) with:
  
  - ``average_voltage``: Average voltage vs working ion (V)
  - ``capacity_grav``: Gravimetric capacity (mAh/g)
  - ``energy_grav``: Gravimetric energy density (Wh/kg)

- ``decomposition``: Decomposition products (if unstable)
- ``viability_assessment``: Textual assessment of anode viability

**Example:**

.. code-block:: python

   # Analyze AlMg as Li-ion anode
   result = await handler.analyze_anode_viability(
       composition="AlMg",
       working_ion="Li"
   )

**Technical Details:**

- First checks thermodynamic stability
- If viable (E_above_hull < 0.1 eV/atom), calculates voltage
- Provides comprehensive viability assessment
- Returns decomposition products if unstable

.. _analyze_lithiation_mechanism:

analyze_lithiation_mechanism
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Analyze the lithiation mechanism of a host material. Reports phase evolution, two-phase vs single-phase reactions, and equilibrium phases at each voltage step. **USE THIS** for questions about "two-phase reaction", "lithiation mechanism", "phase evolution", "what phases form", "initial reaction".

**When to Use:**

- Understanding lithiation mechanism
- Identifying two-phase vs single-phase reactions
- Analyzing phase evolution during charge/discharge
- Determining equilibrium phases at each voltage plateau

**Parameters:**

- ``host_composition`` (str, required): Host material composition (e.g., ``'AlCu'``, ``'CuAl'``, ``'Al'``, ``'Mg'``). Do NOT include Li.
- ``working_ion`` (str, optional): Working ion for battery (e.g., ``'Li'``, ``'Na'``). Default: ``'Li'``
- ``max_x`` (float, optional): Maximum Li per host atom to analyze. Default: 3.0
- ``room_temp`` (bool, optional): Filter out phases hard to form at room temperature (E_hull > 0.03 eV/atom). Default: True

**Returns:**

Dictionary containing:

- ``host_composition``: Host material analyzed
- ``working_ion``: Working ion symbol
- ``max_x``: Maximum x in Li_x(Host) analyzed
- ``voltage_plateaus``: List of voltage plateaus with:
  
  - ``voltage``: Plateau voltage (V)
  - ``x_range``: Composition range [x_start, x_end]
  - ``reaction_type``: ``'two-phase'`` or ``'single-phase'``
  - ``phases_present``: List of equilibrium phases
  - ``phase_reaction``: Phase transformation description

- ``initial_reaction``: Description of first lithiation step
- ``full_sequence``: Complete lithiation sequence description
- ``room_temp_filtered``: Whether room temperature filtering was applied

**Example:**

.. code-block:: python

   # Analyze lithiation mechanism of Al
   result = await handler.analyze_lithiation_mechanism(
       host_composition="Al",
       working_ion="Li",
       max_x=3.0,
       room_temp=True
   )

**Technical Details:**

- Computes convex hull of G(x) vs x for Li_x(Host)
- Identifies voltage plateaus from hull segments
- Determines equilibrium phases from endpoint decompositions
- Reports two-state plateaus vs single-phase regions
- Filters metastable phases if room_temp=True (E_hull > 0.03 eV/atom)

.. _estimate_ion_hopping_barrier:

estimate_ion_hopping_barrier
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Estimate the ion hopping/diffusion barrier (activation energy, eV) for an intercalating ion moving between sites in an electrode material. **USE THIS** for questions about ion mobility, diffusion barriers, or lithium hopping in electrodes.

**When to Use:**

- Estimating ion diffusion barriers
- Understanding ion mobility in electrodes
- Questions about activation energy for ion hopping
- Comparing diffusion rates between materials

**Parameters:**

- ``host_material`` (str, required): Host electrode material formula (e.g., ``'C6'`` (graphite), ``'LiFePO4'``, ``'TiS2'``)
- ``ion`` (str, optional): Intercalating ion (e.g., ``'Li'``, ``'Na'``, ``'Mg'``). Default: ``'Li'``
- ``structure_type`` (str, optional): Structure type/dimensionality:
  
  - ``'layered'``: 2D layered materials (graphite, TMDs)
  - ``'1D-channel'``: 1D channel structures
  - ``'3D'``: 3D frameworks (spinels, garnets)
  - ``'olivine'``: Olivine structures (LiFePO4)
  - Optional - will auto-classify if not provided

**Returns:**

Dictionary containing:

- ``host_material``: Host material formula
- ``ion``: Intercalating ion symbol
- ``structure_type``: Classified structure type
- ``activation_energy_eV``: Estimated activation energy (eV)
- ``energy_range_eV``: Typical range [min, max] (eV)
- ``method``: Estimation method (``'structure_heuristic_v1'``)
- ``descriptors``: Additional information:
  
  - ``literature_value``: True if from known literature benchmark
  - ``note``: Description of material class
  - ``structure_type``: Structure classification

**Example:**

.. code-block:: python

   # Estimate Li hopping barrier in graphite
   result = await handler.estimate_ion_hopping_barrier(
       host_material="C6",
       ion="Li",
       structure_type="layered"
   )

**Technical Details:**

- Uses structure-based heuristics and literature benchmark values
- Exact formula matches return high-confidence literature values
- Structure-based estimates for unknown materials
- Auto-classifies structure type if not provided
- Caveats: Actual barriers depend on crystallographic pathway, site occupancy, lattice strain

**Known Literature Benchmarks:**

- Li in C6 (graphite): 0.30-0.40 eV
- Li in LiFePO4: 0.20-0.55 eV
- Na in hard carbon: 0.10-0.40 eV
- Li in TiS2: 0.25-0.38 eV

**Structure-Based Estimates:**

- Layered (2D): 0.20-0.40 eV
- 1D channels: 0.30-0.60 eV
- 3D frameworks: 0.40-0.80 eV
- Olivines: 0.50-0.70 eV

Database and Citations
----------------------

**Data Sources:**

- **Materials Project**: DFT-calculated formation energies, crystal structures, electrochemical properties
- **PyMatGen**: InsertionElectrode class for voltage calculations, convex hull analysis

**Calculation Methods:**

- **Convex Hull Analysis**: Thermodynamically rigorous voltage calculations based on phase equilibria
- **0 K DFT**: All energies from density functional theory at 0 K
- **InsertionElectrode**: PyMatGen's two-phase equilibrium voltage calculator

**Citations:**

All Electrochemistry functions cite:

- **Materials Project**: Jain, A. et al. (2013). The Materials Project: A materials genome approach to accelerating materials innovation. *APL Materials*, 1(1), 011002. DOI: 10.1063/1.4812323
- **PyMatGen**: Ong, S. P. et al. (2013). Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis. *Computational Materials Science*, 68, 314-319. DOI: 10.1016/j.commatsci.2012.10.028

Notes
-----

- All voltages are reported vs. working ion reference (e.g., Li/Li⁺, Na/Na⁺)
- Capacities in mAh/g (gravimetric) or Ah/L (volumetric)
- Energy densities in Wh/kg (gravimetric) or Wh/L (volumetric)
- Stability threshold: E_above_hull < 0.03 eV/atom considered synthesizable at room temperature
- All thermodynamic data from 0 K DFT calculations
- Framework compositions verified to match requested elements in search results
