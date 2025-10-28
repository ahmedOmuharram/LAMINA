Materials Handler
=================

The Materials handler provides AI functions for searching and retrieving material information from the Materials Project database, which contains DFT-calculated properties for over 200,000 materials.

Overview
--------

The Materials handler provides access to:

1. **Material Search**: Find materials by composition, formula, or elements
2. **Property-Based Search**: Search materials by characteristics (band gap, crystal system, mechanical properties, etc.)
3. **Detailed Material Information**: Retrieve comprehensive data for specific materials
4. **Elastic and Mechanical Properties**: Access elastic moduli, Poisson's ratio, and related properties
5. **Alloy Analysis**: Find and analyze alloy compositions
6. **Property Comparison**: Compare properties between materials
7. **Doping Analysis**: Analyze effects of doping on material properties

Core Search Functions
---------------------

.. _get_material:

get_material
^^^^^^^^^^^^

Query materials by their chemical system, formula, or elements. Returns material IDs and formulas matching the search criteria.

**When to Use:**

- Searching for materials by chemical system (e.g., Li-Fe-O)
- Finding materials with specific formulas
- Querying materials containing specific elements

**Parameters:**

- ``chemsys`` (str, optional): Chemical system(s) or comma-separated list (e.g., ``'Li-Fe-O'``, ``'Si-*'``). Use chemical symbols directly.
- ``formula`` (str, optional): Formula(s), anonymized formula, or wildcard(s) (e.g., ``'Li2FeO3'``, ``'Fe2O3'``, ``'Fe*O*'``).
- ``element`` (str, optional): Element(s) or comma-separated list (e.g., ``'Li,Fe,O'``). Use chemical symbols directly.
- ``page`` (int, optional): Page number. Default: 1
- ``per_page`` (int, optional): Items per page. Default: 10

**Returns:**

Dictionary containing:

- ``materials``: List of materials with:
  
  - ``material_id``: Materials Project ID (e.g., ``'mp-12345'``)
  - ``formula_pretty``: Prettified chemical formula
  - ``formula_anonymous``: Anonymous formula
  - ``chemsys``: Chemical system
  - ``elements``: List of elements
  - ``nelements``: Number of elements

- ``count``: Total number of materials found
- ``page``: Current page number
- ``per_page``: Items per page

**Example:**

.. code-block:: python

   # Search for Li-Fe-O system materials
   result = await handler.get_material(
       chemsys="Li-Fe-O",
       page=1,
       per_page=10
   )

.. _get_material_by_char:

get_material_by_char
^^^^^^^^^^^^^^^^^^^^

Fetch materials by their characteristics (band gap, mechanical properties, magnetic properties, etc.). Highly flexible search with extensive filtering options.

**When to Use:**

- Finding materials with specific properties
- Filtering by band gap range
- Searching for materials with specific crystal systems
- Filtering by mechanical, electronic, or magnetic properties

**Parameters (Selected Key Parameters):**

**Electronic Properties:**

- ``band_gap`` (List[float], optional): Min,max range of band gap in eV (e.g., ``[1.2, 3.0]``)
- ``efermi`` (List[float], optional): Min,max Fermi energy in eV
- ``is_gap_direct`` (bool, optional): Whether material has direct band gap
- ``is_metal`` (bool, optional): Whether material is a metal

**Mechanical Properties:**

- ``k_vrh`` (List[float], optional): Min,max Voigt-Reuss-Hill bulk modulus in GPa
- ``g_vrh`` (List[float], optional): Min,max Voigt-Reuss-Hill shear modulus in GPa
- ``poisson_ratio`` (List[float], optional): Min,max Poisson's ratio
- ``elastic_anisotropy`` (List[float], optional): Min,max elastic anisotropy

**Magnetic Properties:**

- ``total_magnetization`` (List[float], optional): Min,max total magnetization in Bohr magnetons/atom
- ``magnetic_ordering`` (str, optional): Magnetic ordering (``'paramagnetic'``, ``'ferromagnetic'``, ``'antiferromagnetic'``, ``'ferrimagnetic'``)
- ``num_magnetic_sites`` (List[int], optional): Min,max number of magnetic sites

**Dielectric Properties:**

- ``e_total`` (List[float], optional): Min,max total dielectric constant
- ``e_electronic`` (List[float], optional): Min,max electronic dielectric constant
- ``e_ionic`` (List[float], optional): Min,max ionic dielectric constant
- ``piezoelectric_modulus`` (List[float], optional): Min,max piezoelectric modulus in C/m²

**Thermodynamic Properties:**

- ``formation_energy`` (List[float], optional): Min,max formation energy in eV/atom
- ``energy_above_hull`` (List[float], optional): Min,max energy above hull in eV/atom
- ``is_stable`` (bool, optional): Whether material lies on convex energy hull

**Structural Properties:**

- ``crystal_system`` (str, optional): Crystal system (``'Triclinic'``, ``'Monoclinic'``, ``'Orthorhombic'``, ``'Tetragonal'``, ``'Trigonal'``, ``'Hexagonal'``, ``'Cubic'``)
- ``spacegroup_number`` (int, optional): Spacegroup number
- ``spacegroup_symbol`` (str, optional): Spacegroup symbol
- ``density`` (List[float], optional): Min,max density
- ``volume`` (List[float], optional): Min,max volume in bohr³
- ``num_sites`` (List[int], optional): Min,max number of sites
- ``nelements`` (List[int], optional): Min,max number of elements

**Composition Filters:**

- ``elements`` (List[str], optional): List of elements (e.g., ``['Li', 'Fe', 'O']``)
- ``exclude_elements`` (str, optional): Elements to exclude (e.g., ``'Li,Fe,O'``)
- ``possible_species`` (str, optional): Possible species of material

**Other:**

- ``theoretical`` (bool, optional): Whether entry is theoretical or experimental
- ``page`` (int, optional): Page number. Default: 1
- ``per_page`` (int, optional): Items per page. Default: 10

**Returns:**

Dictionary containing:

- ``materials``: List of materials matching criteria
- ``count``: Number of materials found
- ``filters_applied``: Summary of applied filters

**Example:**

.. code-block:: python

   # Find semiconductors with band gap 1-3 eV
   result = await handler.get_material_by_char(
       band_gap=[1.0, 3.0],
       is_metal=False,
       is_stable=True,
       per_page=10
   )

.. _get_material_details_by_ids:

get_material_details_by_ids
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fetch detailed information for one or more materials using their Materials Project IDs. Returns comprehensive property data.

**When to Use:**

- Retrieving complete material data for known IDs
- Getting detailed properties after initial search
- Accessing all available fields for specific materials

**Parameters:**

- ``material_ids`` (List[str], required): List of material IDs (e.g., ``['mp-149', 'mp-150', 'mp-151']``)
- ``fields`` (List[str], optional): List of specific fields to include. Available fields include:
  
  - Basic: ``'material_id'``, ``'formula_pretty'``, ``'formula_anonymous'``, ``'chemsys'``, ``'elements'``, ``'nelements'``
  - Structural: ``'structure'``, ``'nsites'``, ``'volume'``, ``'density'``, ``'symmetry'``
  - Energetic: ``'energy_per_atom'``, ``'formation_energy_per_atom'``, ``'energy_above_hull'``
  - Electronic: ``'band_gap'``, ``'cbm'``, ``'vbm'``, ``'efermi'``, ``'is_gap_direct'``, ``'is_metal'``, ``'bandstructure'``, ``'dos'``
  - Magnetic: ``'is_magnetic'``, ``'ordering'``, ``'total_magnetization'``, ``'num_magnetic_sites'``
  - Mechanical: ``'bulk_modulus'``, ``'shear_modulus'``, ``'universal_anisotropy'``, ``'homogeneous_poisson'``
  - Dielectric: ``'e_total'``, ``'e_ionic'``, ``'e_electronic'``
  - Surface: ``'weighted_surface_energy'``, ``'weighted_work_function'``, ``'surface_anisotropy'``

- ``all_fields`` (bool, optional): Whether to return all document fields. Default: True
- ``page`` (int, optional): Page number. Default: 1
- ``per_page`` (int, optional): Items per page. Default: 10

**Returns:**

Dictionary containing:

- ``materials``: List of material documents with requested fields
- ``count``: Number of materials returned

**Example:**

.. code-block:: python

   # Get detailed info for specific materials
   result = await handler.get_material_details_by_ids(
       material_ids=['mp-149', 'mp-30'],
       all_fields=True
   )

Property Analysis Functions
---------------------------

.. _get_elastic_properties:

get_elastic_properties
^^^^^^^^^^^^^^^^^^^^^^

Get elastic and mechanical properties (bulk modulus, shear modulus, Poisson's ratio, etc.) for a specific material.

**When to Use:**

- Retrieving mechanical properties
- Understanding stiffness and elastic behavior
- Comparing mechanical properties between materials

**Parameters:**

- ``material_id`` (str, required): Material ID (e.g., ``'mp-81'`` for Ag, ``'mp-30'`` for Cu)

**Returns:**

Dictionary containing:

- ``material_id``: Materials Project ID
- ``formula``: Chemical formula
- ``bulk_modulus``: Bulk modulus information:
  
  - ``vrh``: Voigt-Reuss-Hill average (GPa)
  - ``voigt``: Voigt bound (GPa)
  - ``reuss``: Reuss bound (GPa)

- ``shear_modulus``: Shear modulus information:
  
  - ``vrh``: Voigt-Reuss-Hill average (GPa)
  - ``voigt``: Voigt bound (GPa)
  - ``reuss``: Reuss bound (GPa)

- ``universal_anisotropy``: Universal anisotropy index
- ``homogeneous_poisson``: Poisson's ratio

**Example:**

.. code-block:: python

   # Get elastic properties for Ag
   result = await handler.get_elastic_properties(
       material_id="mp-81"
   )

.. _find_alloy_compositions:

find_alloy_compositions
^^^^^^^^^^^^^^^^^^^^^^^

Find materials with specific alloy compositions (e.g., Ag-Cu alloys with ~12.5% Cu). Useful for identifying alloys with target compositions.

**When to Use:**

- Finding alloys with specific compositions
- Searching for binary alloys
- Filtering by stability criteria

**Parameters:**

- ``elements`` (List[str], required): List of elements in the alloy (e.g., ``['Ag', 'Cu']``)
- ``target_composition`` (Dict[str, float], optional): Target atomic fractions as dictionary (e.g., ``{'Ag': 0.875, 'Cu': 0.125}`` for 12.5% Cu). If None, returns all compositions.
- ``tolerance`` (float, optional): Tolerance for composition matching. Default: 0.05
- ``is_stable`` (bool, optional): Whether to filter for stable materials only. Default: True
- ``ehull_max`` (float, optional): Maximum energy above hull for metastable entries in eV/atom. Default: 0.20
- ``require_binaries`` (bool, optional): Whether to require exactly 2 elements. Default: True

**Returns:**

Dictionary containing:

- ``alloys``: List of matching alloys with:
  
  - ``material_id``: Materials Project ID
  - ``formula``: Chemical formula
  - ``composition``: Atomic fractions
  - ``energy_above_hull``: Stability (eV/atom)
  - ``is_stable``: Boolean stability flag
  - ``composition_match``: Distance from target composition

- ``count``: Number of alloys found
- ``target_composition``: Target composition queried

**Example:**

.. code-block:: python

   # Find Ag-Cu alloys with ~12.5% Cu
   result = await handler.find_alloy_compositions(
       elements=['Ag', 'Cu'],
       target_composition={'Ag': 0.875, 'Cu': 0.125},
       tolerance=0.05,
       is_stable=True
   )

.. _compare_material_properties:

compare_material_properties
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Compare a specific property (e.g., bulk modulus) between two materials and calculate percent change.

**When to Use:**

- Comparing properties between two materials
- Calculating percent change in properties
- Understanding property differences

**Parameters:**

- ``material_id1`` (str, required): First material ID
- ``material_id2`` (str, required): Second material ID
- ``property_name`` (str, optional): Property to compare. Options: ``'bulk_modulus'``, ``'shear_modulus'``, ``'poisson_ratio'``, ``'universal_anisotropy'``. Default: ``'bulk_modulus'``

**Returns:**

Dictionary containing:

- ``material_id1``: First material ID
- ``material_id2``: Second material ID
- ``property_name``: Property compared
- ``value1``: Property value for material 1
- ``value2``: Property value for material 2
- ``difference``: Absolute difference
- ``percent_change``: Percent change from material 1 to material 2
- ``comparison``: Textual comparison summary

**Example:**

.. code-block:: python

   # Compare bulk modulus of Ag and Cu
   result = await handler.compare_material_properties(
       material_id1="mp-81",  # Ag
       material_id2="mp-30",  # Cu
       property_name="bulk_modulus"
   )

.. _analyze_doping_effect:

analyze_doping_effect
^^^^^^^^^^^^^^^^^^^^^

Analyze the effect of doping a host material with a dopant element on a specific property. Compares pure host material with doped alloy.

**When to Use:**

- Understanding how doping affects material properties
- Comparing pure vs doped materials
- Analyzing property changes with doping concentration

**Parameters:**

- ``host_element`` (str, required): Host element symbol (e.g., ``'Ag'``)
- ``dopant_element`` (str, required): Dopant element symbol (e.g., ``'Cu'``)
- ``dopant_concentration`` (float, required): Dopant atomic fraction (e.g., ``0.125`` for 12.5% doping)
- ``property_name`` (str, optional): Property to analyze. Options: ``'bulk_modulus'``, ``'shear_modulus'``, ``'poisson_ratio'``, etc. Default: ``'bulk_modulus'``

**Returns:**

Dictionary containing:

- ``host_element``: Host element symbol
- ``dopant_element``: Dopant element symbol
- ``dopant_concentration``: Dopant atomic fraction
- ``property_name``: Property analyzed
- ``host_material``: Pure host material information:
  
  - ``material_id``: Materials Project ID
  - ``formula``: Chemical formula
  - ``property_value``: Property value

- ``doped_material``: Doped material information:
  
  - ``material_id``: Materials Project ID
  - ``formula``: Chemical formula
  - ``actual_composition``: Actual atomic fractions
  - ``property_value``: Property value

- ``doping_effect``: Analysis of doping effect:
  
  - ``absolute_change``: Absolute change in property
  - ``percent_change``: Percent change
  - ``effect_description``: Textual description

**Example:**

.. code-block:: python

   # Analyze effect of 12.5% Cu doping on Ag bulk modulus
   result = await handler.analyze_doping_effect(
       host_element="Ag",
       dopant_element="Cu",
       dopant_concentration=0.125,
       property_name="bulk_modulus"
   )

Database and Citations
----------------------

**Data Source:**

- **Materials Project**: Over 200,000 DFT-calculated materials with crystal structures, formation energies, electronic properties, elastic properties, magnetic properties, and thermodynamic data

**Calculation Methods:**

- All properties from density functional theory (DFT) calculations
- Exchange-correlation functional: Perdew-Burke-Ernzerhof (PBE)
- Elastic properties from DFT perturbation calculations

**Citations:**

All Materials functions cite:

- **Materials Project**: Jain, A. et al. (2013). The Materials Project: A materials genome approach to accelerating materials innovation. *APL Materials*, 1(1), 011002. DOI: 10.1063/1.4812323
- **PyMatGen**: Ong, S. P. et al. (2013). Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis. *Computational Materials Science*, 68, 314-319. DOI: 10.1016/j.commatsci.2012.10.028

Available Properties
--------------------

The Materials Project database provides extensive material properties:

**Basic Information:**

- material_id, formula_pretty, formula_anonymous, chemsys, elements, nelements, composition, nsites

**Structural Properties:**

- structure, volume, density, density_atomic, symmetry, crystal_system, spacegroup_number, spacegroup_symbol

**Energetic Properties:**

- energy_per_atom, formation_energy_per_atom, energy_above_hull, equilibrium_reaction_energy_per_atom, is_stable, decomposes_to

**Electronic Properties:**

- band_gap, cbm, vbm, efermi, is_gap_direct, is_metal, bandstructure, dos

**Magnetic Properties:**

- is_magnetic, ordering, total_magnetization, total_magnetization_normalized_vol, total_magnetization_normalized_formula_units, num_magnetic_sites, num_unique_magnetic_sites, types_of_magnetic_species

**Mechanical Properties:**

- bulk_modulus (k_vrh, k_voigt, k_reuss), shear_modulus (g_vrh, g_voigt, g_reuss), universal_anisotropy, homogeneous_poisson, elastic_anisotropy

**Dielectric Properties:**

- e_total, e_ionic, e_electronic, piezoelectric_modulus

**Surface Properties:**

- weighted_surface_energy, weighted_work_function, surface_anisotropy, shape_factor, has_reconstructed

Notes
-----

- All energies in eV/atom
- All pressures in GPa for mechanical properties
- Formation energies relative to elemental references
- Energy above hull indicates thermodynamic stability (0 = stable)
- Elastic moduli: VRH = Voigt-Reuss-Hill average (recommended value)
- Band gaps from DFT may underestimate experimental values
- Use ``all_fields=True`` in ``get_material_details_by_ids`` for comprehensive data
