Electrochemistry Handler
========================

Functions for battery electrode analysis, voltage calculations, and electrochemical properties.

.. _search_battery_electrodes:

search_battery_electrodes
--------------------------

Search for battery electrode materials with voltage profiles.

**Parameters:**

- ``working_ion`` (str): Working ion symbol (e.g., 'Li', 'Na', 'Mg')
- ``elements`` (List[str], optional): List of elements to include
- ``exclude_elements`` (List[str], optional): List of elements to exclude
- ``max_energy_above_hull`` (float, optional): Maximum energy above hull in eV/atom (default: 0.05)
- ``min_voltage`` (float, optional): Minimum average voltage in V
- ``max_voltage`` (float, optional): Maximum average voltage in V
- ``min_capacity`` (float, optional): Minimum gravimetric capacity in mAh/g
- ``max_capacity`` (float, optional): Maximum gravimetric capacity in mAh/g

**Returns:** Dictionary containing matching electrode materials with voltage and capacity data

.. _calculate_voltage_from_formation_energy:

calculate_voltage_from_formation_energy
----------------------------------------

Calculate electrode voltage from DFT formation energies.

**Parameters:**

- ``electrode_formula`` (str): Electrode material formula (e.g., 'AlMg', 'LiFePO4')
- ``working_ion`` (str): Working ion symbol (e.g., 'Li', 'Na')
- ``max_ion_insertion`` (int, optional): Maximum number of ions that can be inserted

**Returns:** Dictionary containing calculated voltage, capacity, and energy density

.. _compare_electrode_materials:

compare_electrode_materials
----------------------------

Compare two electrode materials side-by-side.

**Parameters:**

- ``material1_formula`` (str): First material formula
- ``material2_formula`` (str): Second material formula
- ``working_ion`` (str): Working ion symbol (e.g., 'Li')

**Returns:** Dictionary containing comparison of voltage, capacity, energy density, and stability

.. _check_composition_stability:

check_composition_stability
----------------------------

Check thermodynamic stability of a composition using convex hull analysis.

**Parameters:**

- ``composition`` (str): Composition to check (e.g., 'AlMg', 'Cu80Li10Al10')
- ``include_plot`` (bool, optional): Whether to generate stability plot (default: False)

**Returns:** Dictionary containing stability analysis, energy above hull, and decomposition products

.. _analyze_anode_viability:

analyze_anode_viability
------------------------

Comprehensive analysis of a material's viability as a battery anode.

**Parameters:**

- ``composition`` (str): Anode material composition
- ``working_ion`` (str): Working ion symbol (e.g., 'Li')
- ``max_voltage_threshold`` (float, optional): Maximum acceptable voltage in V (default: 1.0)

**Returns:** Dictionary containing stability, voltage, capacity, and overall viability assessment

.. _get_voltage_profile:

get_voltage_profile
-------------------

Get detailed voltage profile for an electrode material.

**Parameters:**

- ``material_id`` (str): Materials Project ID (e.g., 'mp-149')
- ``working_ion`` (str): Working ion symbol (e.g., 'Li')

**Returns:** Dictionary containing voltage curve data points and statistics

.. _analyze_lithiation_mechanism:

analyze_lithiation_mechanism
-----------------------------

Analyze phase evolution during lithiation of an electrode material.

**Parameters:**

- ``electrode_formula`` (str): Electrode material formula
- ``working_ion`` (str): Working ion symbol (e.g., 'Li')
- ``max_ion_insertion`` (int, optional): Maximum ions to insert

**Returns:** Dictionary containing lithiation pathway, phase transformations, and voltage steps
