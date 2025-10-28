Solutes Handler
===============

The Solutes handler provides AI functions for analyzing solute effects on lattice parameters, atomic interactions, and solid solution behavior in alloy systems.

Overview
--------

This handler enables:

1. **Lattice Parameter Effects**: Analyze how solute atoms affect host lattice parameters
2. **Solute Comparison**: Compare lattice effects between different solute elements
3. **Quantitative Calculations**: Calculate specific lattice expansion/contraction values
4. **Reference Data**: Access reference lattice parameter data for pure elements

These functions are essential for understanding solid solution strengthening, lattice distortions, and compositional effects in alloys.

Functions
---------

.. _analyze_solute_lattice_effect:

analyze_solute_lattice_effect
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Analyze how a solute element affects the lattice parameters of a host material. Returns qualitative assessment and magnitude of lattice distortion.

**When to Use:**

- Understanding solute-induced lattice distortions
- Predicting solid solution strengthening effects
- Analyzing size mismatch between host and solute atoms

**Parameters:**

- ``host_element`` (str, required): Host element symbol (e.g., ``'Al'``, ``'Fe'``, ``'Cu'``)
- ``solute_element`` (str, required): Solute element symbol (e.g., ``'Mg'``, ``'Zn'``, ``'Ni'``)

**Returns:**

Dictionary containing:

- ``host_element``: Host element analyzed
- ``solute_element``: Solute element analyzed
- ``host_lattice_parameter``: Host lattice parameter (Å)
- ``solute_atomic_radius``: Solute atomic radius (Å)
- ``lattice_effect``: Qualitative effect (``'expansion'``, ``'contraction'``, ``'minimal'``)
- ``size_mismatch``: Percent size mismatch
- ``description``: Textual description of effect

**Example:**

.. code-block:: python

   # Analyze effect of Mg solute in Al host
   result = await handler.analyze_solute_lattice_effect(
       host_element="Al",
       solute_element="Mg"
   )

.. _compare_solute_lattice_effects:

compare_solute_lattice_effects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Compare lattice effects of multiple solute elements in the same host material. Ranks solutes by magnitude of lattice distortion.

**When to Use:**

- Comparing multiple potential solute additions
- Ranking solutes by lattice distortion magnitude
- Optimizing alloy composition selection

**Parameters:**

- ``host_element`` (str, required): Host element symbol
- ``solute_elements`` (List[str], required): List of solute element symbols to compare

**Returns:**

Dictionary containing:

- ``host_element``: Host element
- ``solute_comparisons``: List of comparisons for each solute
- ``ranking``: Solutes ranked by distortion magnitude
- ``summary``: Comparison summary

**Example:**

.. code-block:: python

   # Compare Mg, Zn, and Cu solutes in Al
   result = await handler.compare_solute_lattice_effects(
       host_element="Al",
       solute_elements=["Mg", "Zn", "Cu"]
   )

.. _calculate_solute_lattice_effect:

calculate_solute_lattice_effect
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Calculate quantitative lattice parameter change due to solute addition. Provides numerical estimates of lattice expansion or contraction.

**When to Use:**

- Obtaining quantitative lattice parameter changes
- Calculating specific dimensional changes in alloys
- Predicting unit cell volume changes

**Parameters:**

- ``host_element`` (str, required): Host element symbol
- ``solute_element`` (str, required): Solute element symbol
- ``solute_concentration`` (float, required): Solute atomic fraction (e.g., ``0.05`` for 5 at.%)

**Returns:**

Dictionary containing:

- ``host_element``: Host element
- ``solute_element``: Solute element
- ``solute_concentration``: Atomic fraction
- ``lattice_parameter_change``: Change in lattice parameter (Å)
- ``lattice_parameter_change_percent``: Percent change
- ``volume_change_percent``: Percent volume change
- ``calculation_method``: Method used for calculation

**Example:**

.. code-block:: python

   # Calculate lattice change for 5% Mg in Al
   result = await handler.calculate_solute_lattice_effect(
       host_element="Al",
       solute_element="Mg",
       solute_concentration=0.05
   )

.. _get_solute_reference_data:

get_solute_reference_data
^^^^^^^^^^^^^^^^^^^^^^^^^

Get reference lattice parameter and atomic radius data for pure elements. Provides baseline data for solute analysis.

**When to Use:**

- Accessing reference lattice parameters
- Getting atomic radii for calculations
- Validating solute analysis data

**Parameters:**

- ``element`` (str, required): Element symbol

**Returns:**

Dictionary containing:

- ``element``: Element symbol
- ``lattice_parameter``: Lattice parameter (Å)
- ``atomic_radius``: Atomic radius (Å)
- ``crystal_structure``: Crystal structure type
- ``data_source``: Source of reference data

**Example:**

.. code-block:: python

   # Get reference data for Al
   result = await handler.get_solute_reference_data(
       element="Al"
   )

Technical Details
-----------------

**Calculation Methods:**

- Atomic radius comparisons for qualitative assessments
- Vegard's law for linear lattice parameter interpolation
- Size mismatch calculations based on atomic radii
- Volume changes from lattice parameter changes

**Data Sources:**

- Crystallographic reference data for pure elements
- Atomic radii from standard tables
- Materials Project for validation data

**Assumptions:**

- Ideal solid solution behavior (Vegard's law)
- Small solute concentrations (<10-15 at.%)
- No ordering or precipitation effects
- Room temperature properties

Citations
---------

All Solutes functions cite crystallographic databases and Materials Project data.

Notes
-----

- Lattice parameters in Ångströms (Å)
- Atomic fractions (not weight fractions)
- Assumes random solid solution (no ordering)
- Results most accurate for dilute solutions (<10 at.%)
- Size mismatch > 15% may indicate limited solubility

