Materials Handler
=================

Functions for searching and retrieving material information from the Materials Project database.

.. _materials-data-sources:

Data Sources
============

The Materials Handler integrates multiple data sources to provide comprehensive material property information:

**Materials Project Database**
   The primary data source containing DFT-calculated properties for over 200,000 materials. Provides crystal structures, formation energies, electronic properties, elastic properties, magnetic properties, and thermodynamic data. All calculations are performed using density functional theory (DFT) with the Perdew-Burke-Ernzerhof (PBE) exchange-correlation functional.

**PyCalphad with TDB Databases**
   Thermodynamic database files (TDB) used for CALPHAD (CALculation of PHAse Diagrams) calculations. The system includes multiple TDB files for different material systems, enabling phase diagram calculations and thermodynamic property predictions.

**PyMatgen**
   Python Materials Genomics library used for materials analysis, structure manipulation, and property calculations. Provides utilities for crystal structure analysis, composition handling, and materials property computations.

.. _materials-available-fields:

Available Fields
================

The Materials Project database provides extensive material properties organized by category:

**Basic Information:**

- ``material_id``: Materials Project ID

- ``formula_pretty``: Prettified chemical formula

- ``formula_anonymous``: Anonymous chemical formula

- ``chemsys``: Chemical system

- ``elements``: List of elements

- ``nelements``: Number of elements

- ``composition``: Composition dictionary

- ``composition_reduced``: Reduced composition

- ``nsites``: Number of sites

**Structural Properties:**

- ``structure``: Crystal structure object

- ``volume``: Unit cell volume

- ``density``: Density

- ``density_atomic``: Atomic density

- ``symmetry``: Symmetry information

- ``spacegroup``: Space group information

**Electronic Properties:**

- ``band_gap``: Band gap in eV

- ``cbm``: Conduction band minimum

- ``vbm``: Valence band maximum

- ``efermi``: Fermi energy

- ``is_gap_direct``: Whether band gap is direct

- ``is_metal``: Whether material is metallic

- ``bandstructure``: Band structure data

- ``dos``: Density of states

- ``dos_energy_up``: DOS energy up

- ``dos_energy_down``: DOS energy down

**Thermodynamic Properties:**

- ``formation_energy_per_atom``: Formation energy per atom

- ``energy_per_atom``: Energy per atom

- ``uncorrected_energy_per_atom``: Uncorrected energy per atom

- ``energy_above_hull``: Energy above convex hull

- ``equilibrium_reaction_energy_per_atom``: Equilibrium reaction energy

- ``is_stable``: Whether material is stable

- ``decomposes_to``: Decomposition products

**Elastic Properties:**

- ``bulk_modulus``: Bulk modulus

- ``shear_modulus``: Shear modulus

- ``universal_anisotropy``: Universal anisotropy

- ``homogeneous_poisson``: Homogeneous Poisson ratio

**Magnetic Properties:**

- ``is_magnetic``: Whether material is magnetic

- ``ordering``: Magnetic ordering

- ``total_magnetization``: Total magnetization

- ``total_magnetization_normalized_vol``: Volume-normalized magnetization

- ``total_magnetization_normalized_formula_units``: Formula-normalized magnetization

- ``num_magnetic_sites``: Number of magnetic sites

- ``num_unique_magnetic_sites``: Number of unique magnetic sites

- ``types_of_magnetic_species``: Types of magnetic species


**Dielectric Properties:**

- ``e_total``: Total dielectric constant

- ``e_ionic``: Ionic dielectric constant

- ``e_electronic``: Electronic dielectric constant

**Surface Properties:**

- ``weighted_surface_energy``: Weighted surface energy

- ``weighted_surface_energy_EV_PER_ANG2``: Surface energy in eV/ang^2

- ``weighted_work_function``: Weighted work function

- ``surface_anisotropy``: Surface anisotropy

- ``shape_factor``: Shape factor

- ``has_reconstructed``: Whether has reconstructed surfaces

**Other Properties:**

- ``n``: Number of atoms

- ``e_ij_max``: Maximum elastic constant

- ``possible_species``: Possible species

- ``has_props``: Available properties

- ``theoretical``: Whether theoretical

- ``xas``: X-ray absorption spectroscopy data

- ``grain_boundaries``: Grain boundary data

**Metadata (not typically used by the user):**

- ``es_source_calc_id``: Source calculation ID

- ``builder_meta``

- ``property_name``

- ``deprecated``

- ``deprecation_reasons``

- ``last_updated``

- ``origins``

- ``warnings``

- ``task_ids``

- ``database_Ids``

.. _get_material:

get_material
------------

Query materials by their chemical system and return their material IDs and formula.

**Parameters:**

- ``chemsys`` (str, optional): Chemical system(s) or comma-separated list (e.g., "Li-Fe-O", "Si-*")

- ``formula`` (str, optional): Formula(s), anonymized formula, or wildcard(s) (e.g., "Li2FeO3", "Fe2O3", "Fe*O*")

- ``element`` (str, optional): Element(s) or comma-separated list (e.g., "Li,Fe,O")

- ``page`` (int, optional): Page number (default 1)

- ``per_page`` (int, optional): Items per page (default 10)

**Returns:** Dictionary containing total_count, page info, and list of materials with material_id and formula_pretty

**Implementation:**

The function queries the Materials Project database through the following process:

1. **API Integration**: Uses the Materials Project API (`mpr.materials.summary.search`) to search the database of >200,000 materials

2. **Parameter Processing**: Converts input parameters (chemsys, formula, elements) into the Materials Project API format, handling various input formats like comma-separated lists and wildcards

3. **Database Query**: Executes the search against the Materials Project database with pagination support (default 10 results per page; can be changed using the ``per_page`` parameter)

4. **Data Retrieval**: Returns material IDs and formulas for materials matching the search criteria, as well as pagination information

**Data Source**: See :ref:`Data Sources <materials-data-sources>` for detailed information about the integrated data sources.

.. _get_material_by_char:

get_material_by_char
--------------------

Fetch materials by their characteristics (properties).

**Parameters:**

**Electronic Properties:**

- ``band_gap`` (List[float], optional): Min,max range of band gap in eV

- ``efermi`` (List[float], optional): Min,max fermi energy in eV

- ``is_gap_direct`` (bool, optional): Whether the material has a direct band gap

- ``is_metal`` (bool, optional): Whether the material is considered a metal

**Dielectric Properties:**

- ``e_electronic`` (List[float], optional): Min,max electronic dielectric constant

- ``e_ionic`` (List[float], optional): Min,max ionic dielectric constant

- ``e_total`` (List[float], optional): Min,max total dielectric constant

**Elastic Properties:**

- ``k_reuss`` (List[float], optional): Min,max Reuss bulk modulus in GPa

- ``k_voigt`` (List[float], optional): Min,max Voigt bulk modulus in GPa

- ``k_vrh`` (List[float], optional): Min,max Voigt-Reuss-Hill bulk modulus in GPa

- ``g_reuss`` (List[float], optional): Min,max Reuss grain boundary energy in eV/atom

- ``g_voigt`` (List[float], optional): Min,max Voigt grain boundary energy in eV/atom

- ``g_vrh`` (List[float], optional): Min,max Voigt-Reuss-Hill grain boundary energy in eV/atom

- ``poisson_ratio`` (List[float], optional): Min,max Poisson's ratio

- ``elastic_anisotropy`` (List[float], optional): Min,max elastic anisotropy

**Thermodynamic Properties:**

- ``formation_energy`` (List[float], optional): Min,max formation energy in eV/atom

- ``energy_above_hull`` (List[float], optional): Min,max energy above hull in eV/atom

- ``equilibrium_reaction_energy`` (List[float], optional): Min,max equilibrium reaction energy in eV/atom

- ``total_energy`` (List[float], optional): Min,max total energy in eV/atom

- ``uncorrected_energy`` (List[float], optional): Min,max uncorrected energy in eV/atom

**Magnetic Properties:**

- ``total_magnetization`` (List[float], optional): Min,max total magnetization in Bohr magnetons/atom

- ``total_magnetization_normalized_formula_units`` (List[float], optional): Min,max total magnetization normalized to formula units

- ``total_magnetization_normalized_vol`` (List[float], optional): Min,max total magnetization normalized to volume

- ``magnetic_ordering`` (str, optional): Magnetic ordering ('paramagnetic', 'ferromagnetic', 'antiferromagnetic', 'ferrimagnetic')

- ``num_magnetic_sites`` (List[int], optional): Min,max number of magnetic sites

- ``num_unique_magnetic_sites`` (List[int], optional): Min,max number of unique magnetic sites

**Structural Properties:**

- ``crystal_system`` (str, optional): Crystal system ('Triclinic', 'Monoclinic', 'Orthorhombic', 'Tetragonal', 'Trigonal', 'Hexagonal', 'Cubic')

- ``spacegroup_number`` (int, optional): Spacegroup number of material

- ``spacegroup_symbol`` (str, optional): Spacegroup symbol of material

- ``density`` (List[float], optional): Min,max density range

- ``volume`` (List[float], optional): Min,max volume in bohr^3

- ``n`` (List[int], optional): Min,max number of atoms

- ``nelements`` (List[int], optional): Min,max number of elements

- ``num_sites`` (List[int], optional): Min,max number of sites

**Surface Properties:**

- ``weighted_surface_energy`` (List[float], optional): Min,max weighted surface energy in eV/ang^2

- ``weighted_work_function`` (List[float], optional): Min,max weighted work function in eV

- ``surface_energy_anisotropy`` (List[float], optional): Min,max surface energy anisotropy

- ``surface_anisotropy`` (List[float], optional): Min,max surface anisotropy

- ``has_reconstructed`` (bool, optional): Whether the entry has reconstructed surfaces

- ``shape_factor`` (List[float], optional): Min,max shape factor

**Piezoelectric Properties:**

- ``piezoelectric_modulus`` (List[float], optional): Min,max piezoelectric modulus in C/m^2

**Composition and Elements:**

- ``elements`` (List[str], optional): List of elements (e.g., ['Li', 'Fe', 'O'])

- ``exclude_elements`` (str, optional): Elements to exclude (e.g., 'Li,Fe,O')

- ``possible_species`` (str, optional): Possible species of material (e.g., 'Li,Fe,O')

**Stability and Classification:**

- ``is_stable`` (bool, optional): Whether the material lies on the convex energy hull

- ``theoretical`` (bool, optional): Whether the entry is theoretical (true) or experimental (false)

- ``has_props`` (str, optional): Calculated properties available

**Conditions:**

- ``temperature`` (float, optional): Temperature in Kelvin

- ``pressure`` (float, optional): Pressure in GPa

**Pagination:**

- ``page`` (int, optional): Page number (default 1)

- ``per_page`` (int, optional): Items per page (default 10; can be changed using the per_page parameter)

**Returns:** Dictionary containing matching materials with their properties, as well as pagination information

**Implementation:**

The function searches the Materials Project database using property-based filters through the following process:

1. **API Integration**: Uses the Materials Project API (`mpr.materials.summary.search`) to search the database of >200,000 materials

2. **Parameter Validation**: Validates range parameters (min,max pairs) and converts various input formats (lists, CSV strings) into the Materials Project API format

3. **Selector Validation**: Ensures at least one selector is provided (either identity selectors like elements/formula OR numeric/range filters like band_gap/formation_energy)

4. **Field Selection**: Automatically includes material_id, formula_pretty, elements, and chemsys in the response fields

5. **Database Query**: Executes the search against the Materials Project database with pagination support

6. **Data Retrieval**: Returns materials matching the specified property criteria along with pagination metadata

**Supported Property Ranges**: The function supports 30+ property ranges including band_gap, density, formation_energy, elastic properties, magnetic properties, and structural parameters

**Data Source**: See :ref:`Data Sources <materials-data-sources>` for detailed information about the integrated data sources.

.. _get_material_details_by_ids:

get_material_details_by_ids
---------------------------

Fetch one or more materials by their material IDs and return detailed information.

**Parameters:**

- ``material_ids`` (List[str]): List of material IDs (e.g., ['mp-149', 'mp-150', 'mp-151'])

- ``fields`` (List[str], optional): List of fields to include (see :ref:`Available Fields <materials-available-fields>`)

- ``all_fields`` (bool, optional): Whether to return all document fields (default True)

- ``page`` (int, optional): Page number (default 1)

- ``per_page`` (int, optional): Items per page (default 10; can be changed using the per_page parameter)

**Returns:** Dictionary containing detailed material information, as well as pagination information

**Implementation:**

The function retrieves detailed material information through the following process:

1. **API Integration**: Uses the Materials Project API (`mpr.materials.summary.search`) to fetch detailed data for specific material IDs

2. **Field Selection**: Allows users to specify which fields to return, or returns all fields by default (all_fields=True)

3. **Material ID Processing**: Handles both list and JSON string formats for material IDs, with automatic CSV parsing

4. **Data Retrieval**: Fetches comprehensive material data including structural, electronic, thermodynamic, and magnetic properties

5. **Pagination**: Supports pagination for large result sets with configurable page size

6. **Data Processing**: Converts MPRester documents to plain dictionaries and includes pagination metadata

**Data Source**: See :ref:`Data Sources <materials-data-sources>` for detailed information about the integrated data sources.

.. _get_elastic_properties:

get_elastic_properties
----------------------

Get elastic and mechanical properties for a material.

**Parameters:**

- ``material_id`` (str): Material ID (e.g., 'mp-81' for Ag, 'mp-30' for Cu)

**Returns:** Dictionary containing elastic properties including bulk modulus, shear modulus, Poisson's ratio, universal anisotropy, and stability information

**Implementation:**

The function retrieves elastic and mechanical properties through the following process:

1. **API Integration**: Uses the Materials Project API (`mpr.materials.summary.search`) to fetch material data with specific elastic property fields

2. **Field Selection**: Queries for essential elastic properties including bulk_modulus, shear_modulus, universal_anisotropy, homogeneous_poisson, energy_above_hull, and is_stable

3. **Data Processing**: Extracts and processes bulk modulus and shear modulus data, handling both dictionary and object formats from the Materials Project API

4. **Property Extraction**: Returns Voigt-Reuss-Hill (VRH) averages for bulk and shear moduli, along with Voigt and Reuss bounds for comprehensive mechanical characterization

5. **Stability Information**: Includes material stability status and energy above hull for context on thermodynamic stability

**Data Source**: See :ref:`Data Sources <materials-data-sources>` for detailed information about the integrated data sources.

.. _find_alloy_compositions:

find_alloy_compositions
-----------------------

Find materials with specific alloy compositions.

**Parameters:**

- ``elements`` (List[str]): List of elements in the alloy (e.g., ['Ag', 'Cu'])

- ``target_composition`` (Dict[str, float], optional): Target atomic fractions (e.g., {'Ag': 0.875, 'Cu': 0.125})

- ``tolerance`` (float, optional): Tolerance for composition matching (default 0.05)

- ``is_stable`` (bool, optional): Filter for stable materials only (default True)

- ``ehull_max`` (float, optional): Maximum energy above hull for metastable entries in eV/atom (default 0.20)

- ``require_binaries`` (bool, optional): Require exactly 2 elements (default True)

**Returns:** Dictionary containing matching alloy materials with composition analysis, elastic properties, and stability information

**Implementation:**

The function searches for alloy compositions through the following process:

1. **API Integration**: Uses the Materials Project API (`mpr.materials.summary.search`) to search for materials in the specified chemical system

2. **Composition Filtering**: Searches by chemical system (e.g., "Ag-Cu") and optionally filters by number of elements and stability criteria

3. **Composition Matching**: Calculates atomic fractions for each material and compares against target composition with specified tolerance

4. **Fallback Strategy**: If no exact matches are found within tolerance, returns the closest match with composition distance metrics

5. **Elastic Properties**: Includes bulk modulus data when available for mechanical property analysis

6. **Stability Analysis**: Filters by energy above hull (stable: ≤1meV/atom, metastable: ≤ehull_max) and provides stability information

**Data Source**: See :ref:`Data Sources <materials-data-sources>` for detailed information about the integrated data sources.

.. _compare_material_properties:

compare_material_properties
----------------------------

Compare a specific property between two materials.

**Parameters:**

- ``material_id1`` (str): First material ID

- ``material_id2`` (str): Second material ID

- ``property_name`` (str, optional): Property to compare (default 'bulk_modulus')

**Returns:** Dictionary containing comparison results including absolute difference, percent change, ratio, and interpretation

**Implementation:**

The function compares material properties through the following process:

1. **Property Retrieval**: Uses `get_elastic_properties` to fetch detailed property data for both materials

2. **Property Extraction**: Extracts the specified property value from each material's property data, handling nested structures for bulk_modulus and shear_modulus

3. **Comparison Calculation**: Computes absolute difference, percent change, and ratio between the two property values

4. **Unit Handling**: Automatically determines appropriate units (GPa for elastic properties) and includes unit information in results

5. **Interpretation**: Provides qualitative interpretation of the change magnitude (negligible, higher, or lower)

6. **Error Handling**: Validates that both materials have the requested property data before performing comparisons

Currently only used for comparing the effect of doping a material with a dopant element on a specific property.

**Data Source**: See :ref:`Data Sources <materials-data-sources>` for detailed information about the integrated data sources.

.. _analyze_doping_effect:

analyze_doping_effect
---------------------

Analyze the effect of doping a host material with a dopant element on a specific property.

**Parameters:**

- ``host_element`` (str): Host element symbol (e.g., 'Ag')

- ``dopant_element`` (str): Dopant element symbol (e.g., 'Cu')

- ``dopant_concentration`` (float): Dopant atomic fraction (e.g., 0.125 for 12.5% doping)

- ``property_name`` (str, optional): Property to analyze (default 'bulk_modulus')

**Returns:** Dictionary containing comprehensive doping effect analysis including pure element properties, alloy comparisons, and theoretical estimates

**Implementation:**

The function analyzes doping effects through the following process:

1. **Pure Host Material**: Searches for and retrieves properties of the pure host element using Materials Project API

2. **Alloy Search**: Uses `find_alloy_compositions` to locate materials with the target doping concentration, searching both stable and metastable entries

3. **Property Comparison**: Compares doped alloy properties against pure host using `compare_material_properties` for each found alloy

4. **Theoretical Estimation**: When no alloys are found, computes Voigt-Reuss-Hill (VRH) bounds using pure element properties as a theoretical estimate:

   .. math::
      K_V = (1-x) \cdot K_{host} + x \cdot K_{dopant}
      
      K_R = \frac{1}{\frac{1-x}{K_{host}} + \frac{x}{K_{dopant}}}
      
      K_{VRH} = \frac{1}{2}(K_V + K_R)

   where :math:`x` is the dopant concentration, :math:`K_{host}` and :math:`K_{dopant}` are the bulk moduli of pure host and dopant elements, :math:`K_V` is the Voigt upper bound, :math:`K_R` is the Reuss lower bound, and :math:`K_{VRH}` is the Voigt-Reuss-Hill average.

5. **Comprehensive Analysis**: Provides detailed comparison including composition deviations, energy above hull, and statistical summaries of property changes

6. **Fallback Strategies**: Implements multiple fallback approaches including closest match selection and mixture model estimates when exact compositions aren't available

**Data Source**: See :ref:`Data Sources <materials-data-sources>` for detailed information about the integrated data sources.
