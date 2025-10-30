Materials Handler
=================

The Materials handler provides AI functions for searching and retrieving material information from the Materials Project database, which contains DFT-calculated properties for over 200,000 materials.

All functions query the Materials Project API via the `mp-api` client library, with automatic pagination, field selection, and data transformation.

Overview
--------

The Materials handler provides comprehensive access to:

1. **Material Search**: Find materials by composition, formula, or elements (``mp_search_by_composition``)
2. **Material Lookup by ID**: Get materials by Materials Project ID (``mp_get_by_id``)
3. **Property-Based Search**: Search materials by characteristics (band gap, crystal system, mechanical properties, etc.)
4. **Detailed Material Information**: Retrieve comprehensive data for specific materials
5. **Elastic and Mechanical Properties**: Access elastic moduli, Poisson's ratio, and anisotropy
6. **Alloy Analysis**: Find and analyze alloy compositions with composition matching
7. **Property Comparison**: Compare properties between materials with percent change calculations
8. **Doping Analysis**: Analyze effects of doping on material properties using database entries or mixture models

Core Search Functions
---------------------

.. _mp_search_by_composition:

mp_search_by_composition
^^^^^^^^^^^^^^^^^^^^^^^^^

**Function Definition:**

.. code-block:: python

   async def mp_search_by_composition(
       self,
       chemsys: Optional[str] = None,
       formula: Optional[str] = None,
       element: Optional[str] = None,
       page: int = 1,
       per_page: int = 10
   ) -> Dict[str, Any]

**Description:**

Search materials by their chemical system, formula, or elements. This is the primary entry point for composition-based material search. Returns material IDs and basic formula information.

**When to Use:**

- Searching for materials by chemical system (e.g., "Li-Fe-O")
- Finding materials with specific formulas (e.g., "Fe2O3")
- Querying materials containing specific elements (e.g., "Li,Fe,O")
- Initial broad searches before narrowing with property filters

**How It Fetches Data:**

1. **Parameter Validation:**
   
   - At least one of ``chemsys``, ``formula``, or ``element`` must be provided
   - All three can be provided simultaneously (intersection of results)
   - Parameters are passed directly to Materials Project API without transformation

2. **API Query Construction:**
   
   - Calls ``mp_search_by_composition(params)`` which builds kwargs for ``mpr.materials.summary.search()``
   - Internally uses ``_build_summary_search_kwargs()`` to parse parameters
   - Constructs search criteria for Materials Project Summary endpoint

3. **Search Execution:**
   
   .. code-block:: python
   
      # In mp_search_by_composition():
      kwargs = {
          "chemsys": chemsys,  # e.g., "Li-Fe-O"
          "formula": formula,  # e.g., "Fe2O3"
          "elements": element.split(',') if element else None  # parsed to list
      }
      docs = mpr.materials.summary.search(**kwargs)

4. **Result Processing:**
   
   - Converts Materials Project Document objects to dictionaries via ``_convert_docs_to_dicts()``
   - Applies pagination using ``_slice_for_page(data_all, page, per_page)``
   - Computes total count via ``_total_count_for_summary()`` for all matching materials (ignoring pagination)
   - Returns standardized result envelope with metadata

**Parameters:**

- ``chemsys`` (str, optional): Chemical system(s) or comma-separated list
  
  - Format: ``'Li-Fe-O'`` (hyphen-separated elements)
  - Wildcards: ``'Si-*'`` (all systems containing Si)
  - Multiple: ``'Li-Fe-O,Si-O'`` (union of systems)

- ``formula`` (str, optional): Chemical formula(s)
  
  - Exact formula: ``'Li2FeO3'``, ``'Fe2O3'``
  - Wildcards: ``'Fe*O*'`` (any formula matching pattern)
  - Anonymous formula: ``'AB2'`` (stoichiometry pattern)

- ``element`` (str, optional): Element(s) or comma-separated list
  
  - Single: ``'Li'`` (all materials containing Li)
  - Multiple: ``'Li,Fe,O'`` (materials containing all three)
  - Use chemical symbols directly (case-sensitive)

- ``page`` (int, optional): Page number for pagination. Default: 1
- ``per_page`` (int, optional): Items per page (max 100). Default: 10

**Returns:**

Dictionary containing:

.. code-block:: python

   {
       "success": bool,
       "handler": str,
       "function": str,
       "data": {
           "total_count": int,
           "page": int,
           "per_page": int,
           "total_pages": int,
           "data": [
               {
                   "material_id": str,
                   "formula_pretty": str,
                   "formula_anonymous": str,
                   "chemsys": str,
                   "elements": List[str],
                   "num_elements": int,
                   "nsites": int,
                   "volume": float,
                   "density": float
               },
               ...
           ]
       },
       "confidence": float,
       "citations": List[str],
       "duration_ms": float
   }

**Side Effects:**

- None (read-only API query)
- Results are not cached (fresh query each time)

**Example:**

.. code-block:: python

   # Search for Li-Fe-O system materials
   result = await handler.mp_search_by_composition(
       chemsys="Li-Fe-O",
       per_page=10
   )
   
   # Search for specific formula
   result = await handler.mp_search_by_composition(
       formula="Fe2O3"
   )
   
   # Search for materials containing specific elements
   result = await handler.mp_search_by_composition(
       element="Li,Fe,O",
       page=2,
       per_page=20
   )

.. _mp_get_by_characteristic:

mp_get_by_characteristic
^^^^^^^^^^^^^^^^^^^^^^^^^

**Function Definition:**

.. code-block:: python

   async def mp_get_by_characteristic(
       self,
       band_gap: Optional[List[float]] = None,
       crystal_system: Optional[str] = None,
       elements: Optional[List[str]] = None,
       is_stable: Optional[bool] = None,
       k_vrh: Optional[List[float]] = None,
        # ... many more parameters (see :ref:`Parameters <parameters_section>` section for complete list)
       page: int = 1,
       per_page: int = 10
   ) -> Dict[str, Any]

**Description:**

Fetch materials by their characteristics (band gap, mechanical properties, magnetic properties, etc.). This is the most flexible search function with extensive filtering capabilities across all material properties.

**When to Use:**

- Finding materials with specific property ranges (e.g., band gap 1-3 eV)
- Filtering by crystal structure (e.g., cubic systems only)
- Searching for materials with target mechanical properties (e.g., bulk modulus > 100 GPa)
- Combining multiple property constraints (e.g., stable semiconductors with direct band gap)

**How It Fetches Data:**

1. **Parameter Processing:**
   
   - Accepts 50+ optional **search parameters** (function inputs) covering all material properties
   - **Note**: Parameters are used for filtering/searching; **fields** (see Fields section) are the returned data properties
   - Range parameters use ``[min, max]`` format (both values required)
   - Boolean flags for categorical properties (``is_metal``, ``is_stable``, etc.)
   - All parameters are optional; at least one must be provided

2. **Range Parameter Handling:**
   
   In ``_build_summary_search_kwargs()``, range parameters are validated:
   
   .. code-block:: python
   
      RANGE_KEYS = {
          "band_gap", "density", "e_electronic", "formation_energy",
          "k_vrh", "g_vrh", "poisson_ratio", "energy_above_hull", ...
      }
      
      for key in RANGE_KEYS:
          if key in params and params[key] is not None:
              val = params[key]
              if isinstance(val, (list, tuple)) and len(val) == 2:
                  kwargs[key] = tuple(val)  # Convert to tuple for API
              else:
                  # Validation error: both min and max required
                  errors.append(f"{key} must be [min, max] with both values")

3. **Field Selection (Returned Data):**
   
   - Default returned fields: ``["material_id", "formula_pretty", "elements", "chemsys"]``
   - Automatically includes ``material_id`` if not present
   - Custom fields can be specified via API parameters (not exposed in AI function)

4. **API Query Construction:**
   
   .. code-block:: python
   
      # In mp_get_by_characteristic():
      kwargs = {
          "band_gap": (1.0, 3.0),  # Range tuple
          "is_metal": False,  # Boolean filter
          "is_stable": True,
          "crystal_system": "Cubic",  # String filter
          "elements": ["Li", "Fe", "O"],  # List filter
          "fields": ["material_id", "formula_pretty", "band_gap", "energy_above_hull"]
      }
      docs = mpr.materials.summary.search(**kwargs)

5. **Result Processing:**
   
   Same as ``mp_search_by_composition``: convert docs → paginate → compute total count → return envelope

**Parameters:**

See the :ref:`Parameters <parameters_section>` section for a complete list of all available search parameters organized by category.

**Returns:**

Dictionary with same structure as ``mp_search_by_composition``, including filtered results and pagination metadata.

**Example:**

.. code-block:: python

   # Find stable semiconductors with band gap 1-3 eV
   result = await handler.mp_get_by_characteristic(
       band_gap=[1.0, 3.0],
       is_metal=False,
       is_stable=True,
       per_page=10
   )
   
   # Find high bulk modulus cubic materials
   result = await handler.mp_get_by_characteristic(
       k_vrh=[100, 500],
       crystal_system="Cubic",
       is_stable=True
   )
   
   # Find ferromagnetic materials
   result = await handler.mp_get_by_characteristic(
       magnetic_ordering="FM",  # Use enum value: FM, AFM, FiM, NM, PM
       is_stable=True
   )

.. _mp_get_by_id:

mp_get_by_id
^^^^^^^^^^^^

**Function Definition:**

.. code-block:: python

   async def mp_get_by_id(
       self,
       material_ids: List[str],
       fields: Optional[List[str]] = None,
       page: int = 1,
       per_page: int = 10
   ) -> Dict[str, Any]

**Description:**

Get materials by their Materials Project IDs. By default, returns basic material information (IDs, formulas, elements, chemical systems). Custom fields can be specified via the ``fields`` parameter to retrieve specific properties. Use ``mp_get_material_details()`` for comprehensive property data with all available fields.

**When to Use:**

- Retrieving basic material information when you already have the Materials Project ID
- Quick lookups of material IDs to verify existence
- Getting IDs, formulas, and elements for known materials
- Retrieving specific fields by providing the ``fields`` parameter
- Use ``mp_get_material_details()`` for comprehensive property data with all fields

**How It Fetches Data:**

1. **Material ID Parsing:**
   
   - Accepts list of material IDs: ``['mp-149', 'mp-150', 'mp-151']``
   - Also accepts JSON string: ``'["mp-149", "mp-150"]'`` (parsed automatically)
   - Or CSV string: ``'mp-149,mp-150,mp-151'`` (split on comma)
   - Converts to comma-separated string for API: ``"mp-149,mp-150,mp-151"``

2. **Field Selection:**
   
   - Default fields: ``["material_id", "formula_pretty", "elements", "chemsys"]``
   - Custom fields can be specified via ``fields`` parameter
   - Uses ``mp_get_material_details()`` internally
   - If ``fields`` parameter is provided, uses those fields instead of defaults

3. **API Query:**
   
   .. code-block:: python
   
      # In mp_get_by_id():
      search_kwargs = {
          "material_ids": "mp-149,mp-150,mp-151",  # CSV string
          "fields": fields if fields else ["material_id", "formula_pretty", "elements", "chemsys"]  # Default or custom
      }
      docs = mpr.materials.summary.search(**search_kwargs)

4. **Result Processing:**
   
   Same as ``mp_search_by_composition``: convert docs → paginate → compute total count → return envelope

**Parameters:**

- ``material_ids`` (List[str], required): List of material IDs (e.g., ``['mp-149', 'mp-30', 'mp-81']``)
- ``fields`` (List[str], optional): Specific data fields to return in response (see Fields section for full list of available returned properties). If not provided, returns basic fields: ``['material_id', 'formula_pretty', 'elements', 'chemsys']``
- ``page`` (int, optional): Page number. Default: 1
- ``per_page`` (int, optional): Items per page. Default: 10

**Returns:**

Dictionary with same structure as ``mp_search_by_composition``, containing basic material information.

**Example:**

.. code-block:: python

   # Get basic info for materials by ID
   result = await handler.mp_get_by_id(
       material_ids=['mp-149', 'mp-30']
   )
   
   # Get specific fields for materials by ID
   result = await handler.mp_get_by_id(
       material_ids=['mp-149', 'mp-30'],
       fields=['material_id', 'formula_pretty', 'band_gap', 'energy_above_hull']
   )

.. _mp_get_material_details:

mp_get_material_details
^^^^^^^^^^^^^^^^^^^^^^^

**Function Definition:**

.. code-block:: python

   async def mp_get_material_details(
       self,
       material_ids: List[str],
       fields: Optional[List[str]] = None,
       all_fields: bool = True,
       page: int = 1,
       per_page: int = 10
   ) -> Dict[str, Any]

**Description:**

Fetch detailed information for one or more materials using their Materials Project IDs. This retrieves comprehensive property data for known materials.

**When to Use:**

- Retrieving complete material data after initial search
- Getting detailed properties for specific materials
- Accessing all available fields for analysis
- Following up on material IDs from other functions

**How It Fetches Data:**

1. **Material ID Parsing:**
   
   - Accepts list of material IDs: ``['mp-149', 'mp-150', 'mp-151']``
   - Also accepts JSON string: ``'["mp-149", "mp-150"]'`` (parsed automatically)
   - Or CSV string: ``'mp-149,mp-150,mp-151'`` (split on comma)
   - Converts to comma-separated string for API: ``"mp-149,mp-150,mp-151"``

2. **Field Selection Logic:**
   
   .. code-block:: python
   
      # In mp_get_material_details():
      if fields is not None:
          if isinstance(fields, str):
              fields = [f.strip() for f in fields.split(",")]
          if "material_id" not in fields:
              fields.append("material_id")  # Always include ID
          
          # Map old field names to new API field names
          field_mapping = {
              "formula": "formula_pretty",
              "magnetic_ordering": "ordering"
          }
          fields = [field_mapping.get(f, f) for f in fields]
          kwargs["fields"] = fields
      elif not all_fields:
          kwargs["fields"] = ["material_id"]  # Minimal response
      # else: all_fields=True means return everything (no fields specified)

3. **API Query:**
   
   .. code-block:: python
   
      search_kwargs = {
          "material_ids": "mp-149,mp-150,mp-151",  # CSV string
          "all_fields": True,  # or False with specific fields
          "fields": [...] if specified else None
      }
      docs = mpr.materials.summary.search(**search_kwargs)

4. **Data Conversion:**
   
   - Documents contain pymatgen Structure objects, pydantic models, etc.
   - ``_convert_docs_to_dicts()`` serializes all fields to JSON-compatible dicts
   - Handles special types: Structure → dict, Composition → dict, enums → strings

**Parameters:**

- ``material_ids`` (List[str], required): List of material IDs (e.g., ``['mp-149', 'mp-30']``)
- ``fields`` (List[str], optional): Specific data fields to return in response (see Fields section for full list of available returned properties)
- ``all_fields`` (bool, optional): Return all available fields. Default: True
- ``page`` (int, optional): Page number. Default: 1
- ``per_page`` (int, optional): Items per page. Default: 10

**Returns:**

Dictionary containing full material documents with all requested fields.

**Example:**

.. code-block:: python

   # Get all fields for specific materials
   result = await handler.mp_get_material_details(
       material_ids=['mp-149', 'mp-30'],
       all_fields=True
   )
   
   # Get only specific fields
   result = await handler.mp_get_material_details(
       material_ids=['mp-81'],
       fields=['material_id', 'formula_pretty', 'band_gap', 'energy_above_hull'],
       all_fields=False
   )

Property Analysis Functions
---------------------------

.. _get_elastic_properties:

get_elastic_properties
^^^^^^^^^^^^^^^^^^^^^^

**Function Definition:**

.. code-block:: python

   async def get_elastic_properties(
       self,
       material_id: str
   ) -> Dict[str, Any]

**Description:**

Get elastic and mechanical properties (bulk modulus, shear modulus, Poisson's ratio, etc.) for a specific material. Extracts moduli in both Voigt, Reuss, and VRH (Voigt-Reuss-Hill average) forms.

**When to Use:**

- Retrieving mechanical properties for stiffness analysis
- Comparing elastic behavior between materials
- Understanding anisotropy and compliance
- Designing materials for structural applications

**How It Fetches Data:**

1. **API Query:**
   
   Calls ``mpr.materials.summary.search()`` with specific elastic-related fields:
   
   .. code-block:: python
   
      docs = mpr.materials.summary.search(
          material_ids=material_id,
          fields=[
              "material_id", "formula_pretty", "composition",
              "bulk_modulus", "shear_modulus", "universal_anisotropy",
              "homogeneous_poisson", "energy_above_hull", "is_stable"
          ]
      )

2. **Data Extraction:**
   
   - ``bulk_modulus`` and ``shear_modulus`` are objects/dicts with ``vrh``, ``voigt``, ``reuss`` attributes
   - Handles both dictionary format (from API) and object format (from pymatgen)
   - Extracts VRH average (recommended value), Voigt bound (upper), Reuss bound (lower)

3. **Moduli Interpretation:**
   
   - **Voigt bound**: Upper bound assuming uniform strain (iso-strain)
   - **Reuss bound**: Lower bound assuming uniform stress (iso-stress)
   - **VRH average**: ``(Voigt + Reuss) / 2`` - recommended for polycrystalline aggregates
   - All values in GPa

**Parameters:**

- ``material_id`` (str, required): Material ID (e.g., ``'mp-81'`` for Ag, ``'mp-30'`` for Cu)

**Returns:**

Dictionary containing:

.. code-block:: python

   {
       "success": bool,
       "handler": str,
       "function": str,
       "data": {
           "material_id": str,
           "formula": str,
           "composition": Dict[str, float],
           "is_stable": bool,
           "energy_above_hull": float,
           "bulk_modulus": {
               "k_vrh": float,
               "k_voigt": float,
               "k_reuss": float,
               "unit": str
           },
           "shear_modulus": {
               "g_vrh": float,
               "g_voigt": float,
               "g_reuss": float,
               "unit": str
           },
           "universal_anisotropy": float,
           "poisson_ratio": float
       },
       "confidence": float,
       "citations": List[str]
   }

**Example:**

.. code-block:: python

   # Get elastic properties for silver
   result = await handler.get_elastic_properties(
       material_id="mp-81"
   )

.. _find_alloy_compositions:

find_alloy_compositions
^^^^^^^^^^^^^^^^^^^^^^^

**Function Definition:**

.. code-block:: python

   async def find_alloy_compositions(
       self,
       elements: List[str],
       target_composition: Optional[Dict[str, float]] = None,
       tolerance: float = 0.05,
       is_stable: bool = True,
       ehull_max: float = 0.20,
       require_binaries: bool = True
   ) -> Dict[str, Any]

**Description:**

Find materials with specific alloy compositions (e.g., Ag-Cu alloys with ~12.5% Cu). Supports composition matching with tolerance and stability filtering.

**When to Use:**

- Finding database entries for specific alloy compositions
- Searching for binary alloys near target compositions
- Discovering stable vs. metastable alloy phases
- Identifying closest matches when exact composition unavailable

**How It Fetches Data:**

1. **Chemical System Construction:**
   
   - Sorts elements alphabetically: ``['Ag', 'Cu']`` → ``"Ag-Cu"``
   - Queries entire chemical system first, then filters by composition

2. **Stability Filtering:**
   
   .. code-block:: python
   
      search_kwargs = {
          "chemsys": "Ag-Cu",
          "num_elements": 2 if require_binaries else None,
          "fields": ["material_id", "formula_pretty", "composition",
                    "energy_above_hull", "is_stable",
                    "bulk_modulus", "shear_modulus"]
      }
      
      if is_stable:
          search_kwargs["energy_above_hull"] = (0, 1e-3)  # Essentially 0
      else:
          search_kwargs["energy_above_hull"] = (0, ehull_max)  # e.g., 0-0.20 eV/atom

3. **Composition Matching:**
   
   For each material in system:
   
   .. code-block:: python
   
      comp_dict = doc.composition.as_dict()
      total_atoms = sum(comp_dict.values())
      fractions = {el: count/total_atoms for el, count in comp_dict.items()}
      
      # Check if within tolerance
      matches_target = True
      max_deviation = 0.0
      if target_composition:
          for el, target_frac in target_composition.items():
              actual_frac = fractions.get(el, 0.0)
              deviation = abs(actual_frac - target_frac)
              max_deviation = max(max_deviation, deviation)
              if deviation > tolerance:
                  matches_target = False

4. **Closest Match Fallback:**
   
   If no materials within tolerance:
   
   .. code-block:: python
   
      # Calculate L1 distance to target
      def composition_distance(mat):
          fracs = mat["atomic_fractions"]
          return sum(abs(fracs.get(el, 0.0) - target_composition.get(el, 0.0))
                    for el in target_composition)
      
      # Find closest match
      closest = min(all_candidates, key=composition_distance)
      closest["closest_match"] = True
      materials = [closest]

**Parameters:**

- ``elements`` (List[str], required): List of elements (e.g., ``['Ag', 'Cu']``)
- ``target_composition`` (Dict[str, float], optional): Target atomic fractions (e.g., ``{'Ag': 0.875, 'Cu': 0.125}'``)
  
  - If None, returns all compositions in system
  - Fractions must sum to 1.0
  - Atomic fractions (not weight percent)

- ``tolerance`` (float, optional): Composition matching tolerance. Default: 0.05 (±5 at.%)
- ``is_stable`` (bool, optional): Filter for stable materials only (Ehull ≈ 0). Default: True
- ``ehull_max`` (float, optional): Max energy above hull for metastable entries in eV/atom. Default: 0.20
- ``require_binaries`` (bool, optional): Require exactly 2 elements. Default: True

**Returns:**

Dictionary containing list of matching alloys with composition info, energy above hull, and closest match indicator.

**Example:**

.. code-block:: python

   # Find Ag-Cu alloys with ~12.5% Cu (87.5% Ag)
   result = await handler.find_alloy_compositions(
       elements=['Ag', 'Cu'],
       target_composition={'Ag': 0.875, 'Cu': 0.125},
       tolerance=0.05,
       is_stable=True
   )

.. _compare_material_properties:

compare_material_properties
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Function Definition:**

.. code-block:: python

   async def compare_material_properties(
       self,
       material_id1: str,
       material_id2: str,
       property_name: str = "bulk_modulus"
   ) -> Dict[str, Any]

**Description:**

Compare a specific property between two materials and calculate percent change, absolute difference, and ratio.

**When to Use:**

- Quantifying property differences between materials
- Calculating percent change for doping or composition effects
- Comparing candidate materials for design selection

**How It Calculates:**

1. **Property Extraction:**
   
   - Calls ``get_elastic_properties()`` for both materials
   - Extracts specified property from standardized result
   - For moduli, uses VRH average value

2. **Comparison Calculation:**
   
   .. code-block:: python
   
      # Calculate differences
      absolute_diff = val2 - val1
      percent_change = (absolute_diff / val1) * 100.0 if val1 != 0 else None
      ratio = val2 / val1 if val1 != 0 else None
      
      # Interpretation
      if abs(percent_change) < 1:
          interpretation = "Negligible change"
      elif percent_change > 0:
          interpretation = f"Material 2 has {abs(percent_change):.1f}% higher {property_name}"
      else:
          interpretation = f"Material 2 has {abs(percent_change):.1f}% lower {property_name}"

**Parameters:**

- ``material_id1`` (str, required): First material ID
- ``material_id2`` (str, required): Second material ID
- ``property_name`` (str, optional): Property to compare: ``'bulk_modulus'``, ``'shear_modulus'``, ``'poisson_ratio'``, ``'universal_anisotropy'``. Default: ``'bulk_modulus'``

**Returns:**

Dictionary with comparison including absolute difference, percent change, ratio, and interpretation.

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

**Function Definition:**

.. code-block:: python

   async def analyze_doping_effect(
       self,
       host_element: str,
       dopant_element: str,
       dopant_concentration: float,
       property_name: str = "bulk_modulus"
   ) -> Dict[str, Any]

**Description:**

Analyze the effect of doping a host material with a dopant element. Compares pure host material with doped alloy. Uses database entries when available or Voigt-Reuss-Hill mixture bounds as fallback.

**When to Use:**

- Understanding how doping affects material properties
- Predicting property changes with composition
- Comparing pure vs doped materials
- Designing alloys with target properties

**How It Calculates:**

1. **Pure Host Material:**
   
   - Searches for pure element: ``elements=[host_element], num_elements=1, is_stable=True``
   - Retrieves elastic properties via ``get_elastic_properties()``

2. **Doped Alloy Search:**
   
   Constructs target composition and searches with tolerance:
   
   .. code-block:: python
   
      target_comp = {
          host_element: 1.0 - dopant_concentration,
          dopant_element: dopant_concentration
      }
      
      # Try stable entries first (Ehull ≈ 0)
      alloys = find_alloy_compositions(
          mpr, [host_element, dopant_element],
          target_composition=target_comp,
          tolerance=0.05,  # ±5 at.%
          is_stable=True
      )
      
      # Fallback: allow metastable entries (Ehull ≤ 0.20 eV/atom)
      if no results:
          alloys = find_alloy_compositions(..., is_stable=False, ehull_max=0.20)

3. **VRH Mixture Model Fallback:**
   
   If no database entries found, computes rigorous Voigt-Reuss-Hill bounds from pure elements:
   
   .. code-block:: python
   
      # Get pure dopant properties
      dopant_props = get_elastic_properties(mpr, dopant_element)
      
      # Extract bulk moduli
      K_host = host_data["bulk_modulus"]["k_vrh"]
      K_dopant = dopant_data["bulk_modulus"]["k_vrh"]
      x = dopant_concentration
      
      # Voigt bound (iso-strain, upper bound)
      K_V = (1 - x) * K_host + x * K_dopant
      
      # Reuss bound (iso-stress, lower bound)
      K_R = 1.0 / ((1 - x) / K_host + x / K_dopant)
      
      # VRH average (recommended value)
      K_VRH = 0.5 * (K_V + K_R)
      
      # Percent changes
      pct_vrh = 100 * (K_VRH - K_host) / K_host
      pct_voigt = 100 * (K_V - K_host) / K_host
      pct_reuss = 100 * (K_R - K_host) / K_host

4. **Property Comparison:**
   
   For each matching alloy in database:
   
   - Retrieves elastic properties
   - Calls ``compare_material_properties()`` to compute percent change
   - Reports actual vs. requested composition
   - Flags closest match if tolerance exceeded

**Parameters:**

- ``host_element`` (str, required): Host element symbol (e.g., ``'Ag'``)
- ``dopant_element`` (str, required): Dopant element symbol (e.g., ``'Cu'``)
- ``dopant_concentration`` (float, required): Dopant atomic fraction (e.g., ``0.125`` for 12.5%)
- ``property_name`` (str, optional): Property to analyze. Default: ``'bulk_modulus'``

**Returns:**

Dictionary containing:

- Pure host material properties
- Pure dopant material properties (if VRH used)
- List of doped alloy comparisons from database
- VRH mixture model estimate with bounds
- Summary statistics (avg, min, max percent change)
- Notes about metastable entries, closest matches, and VRH bounds

**Example:**

.. code-block:: python

   # Analyze 12.5% Cu doping effect on Ag bulk modulus
   result = await handler.analyze_doping_effect(
       host_element="Ag",
       dopant_element="Cu",
       dopant_concentration=0.125,
       property_name="bulk_modulus"
   )

.. _parameters_section:

Parameters
----------

Below are the available search parameters organized by category. Range parameters use ``[min, max]`` format with both values required:

**Electronic Properties:**

- ``band_gap`` (List[float]): Min,max band gap in eV (e.g., ``[1.2, 3.0]``)
- ``efermi`` (List[float]): Min,max Fermi energy in eV
- ``is_gap_direct`` (bool): Whether material has direct band gap
- ``is_metal`` (bool): Whether material is a metal

**Mechanical Properties:**

- ``k_vrh`` (List[float]): Min,max Voigt-Reuss-Hill bulk modulus in GPa
- ``k_voigt`` (List[float]): Min,max Voigt bulk modulus in GPa
- ``k_reuss`` (List[float]): Min,max Reuss bulk modulus in GPa
- ``g_vrh`` (List[float]): Min,max Voigt-Reuss-Hill shear modulus in GPa
- ``g_voigt`` (List[float]): Min,max Voigt shear modulus in GPa
- ``g_reuss`` (List[float]): Min,max Reuss shear modulus in GPa
- ``poisson_ratio`` (List[float]): Min,max Poisson's ratio
- ``elastic_anisotropy`` (List[float]): Min,max elastic anisotropy

**Magnetic Properties:**

- ``total_magnetization`` (List[float]): Min,max magnetization in μ_B/atom
- ``total_magnetization_normalized_vol`` (List[float]): Min,max magnetization normalized to volume in μ_B/Å³
- ``total_magnetization_normalized_formula_units`` (List[float]): Min,max magnetization normalized to formula units in μ_B/formula unit
- ``magnetic_ordering`` (str): Magnetic ordering type. Accepts: ``'FM'`` (ferromagnetic), ``'AFM'`` (antiferromagnetic), ``'FiM'`` (ferrimagnetic), ``'NM'`` (non-magnetic), ``'PM'`` (paramagnetic), or human-readable strings
- ``num_magnetic_sites`` (List[int]): Min,max number of magnetic sites
- ``num_unique_magnetic_sites`` (List[int]): Min,max number of unique magnetic sites

**Thermodynamic Properties:**

- ``formation_energy`` (List[float]): Min,max formation energy in eV/atom
- ``energy_above_hull`` (List[float]): Min,max energy above hull in eV/atom
- ``equilibrium_reaction_energy`` (List[float]): Min,max equilibrium reaction energy in eV/atom
- ``total_energy`` (List[float]): Min,max total energy in eV/atom
- ``uncorrected_energy`` (List[float]): Min,max uncorrected energy in eV/atom
- ``is_stable`` (bool): Whether material lies on convex energy hull (Ehull = 0)

**Structural Properties:**

- ``crystal_system`` (str): Crystal system. Options: ``'Triclinic'``, ``'Monoclinic'``, ``'Orthorhombic'``, ``'Tetragonal'``, ``'Trigonal'``, ``'Hexagonal'``, ``'Cubic'``
- ``spacegroup_number`` (int): International spacegroup number (1-230)
- ``spacegroup_symbol`` (str): Hermann-Mauguin spacegroup symbol
- ``density`` (List[float]): Min,max density in g/cm³
- ``volume`` (List[float]): Min,max unit cell volume in Å³
- ``num_elements`` (List[int]): Min,max number of elements
- ``num_sites`` (List[int]): Min,max number of sites

**Dielectric Properties:**

- ``e_total`` (List[float]): Min,max total dielectric constant
- ``e_ionic`` (List[float]): Min,max ionic dielectric constant
- ``e_electronic`` (List[float]): Min,max electronic dielectric constant
- ``n`` (List[float]): Min,max refractive index
- ``piezoelectric_modulus`` (List[float]): Min,max piezoelectric modulus in C/m²

**Surface Properties:**

- ``weighted_surface_energy`` (List[float]): Min,max weighted surface energy in J/m²
- ``weighted_work_function`` (List[float]): Min,max weighted work function in eV
- ``surface_energy_anisotropy`` (List[float]): Min,max surface energy anisotropy
- ``surface_anisotropy`` (List[float]): Min,max surface anisotropy
- ``shape_factor`` (List[float]): Min,max shape factor
- ``has_reconstructed`` (bool): Whether entry has reconstructed surfaces

**Composition Filters:**

- ``elements`` (List[str]): List of elements material must contain (e.g., ``['Li', 'Fe', 'O']``)
- ``exclude_elements`` (str): Comma-separated elements to exclude (e.g., ``'Li,Fe,O'``)
- ``possible_species`` (str): Possible species in material (e.g., ``'Li,Fe,O'``)

**Metadata Filters:**

- ``theoretical`` (bool): Whether entry is theoretical (true) or experimental/experimentally observed (false)
- ``has_props`` (str): Calculated properties available (list of HasProps or strings)

Fields
------

The Materials Project database provides extensive material properties. Below are the available fields organized by category:

**Basic Information:**

- ``material_id``: Materials Project ID (e.g., 'mp-149')
- ``formula_pretty``: Prettified chemical formula (e.g., 'Fe2O3')
- ``formula_anonymous``: Anonymous formula showing stoichiometry (e.g., 'A2B3')
- ``chemsys``: Chemical system (e.g., 'Fe-O')
- ``elements``: List of element symbols
- ``num_elements``: Number of elements in composition
- ``composition``: Full composition dictionary
- ``composition_reduced``: Reduced composition
- ``nsites``: Number of sites in unit cell

**Structural Properties:**

- ``structure``: Full crystal structure (pymatgen Structure object)
- ``volume``: Unit cell volume in Å³
- ``density``: Density in g/cm³
- ``density_atomic``: Atomic density
- ``symmetry``: Symmetry information
- ``crystal_system``: Crystal system (Triclinic, Monoclinic, Orthorhombic, Tetragonal, Trigonal, Hexagonal, Cubic)
- ``spacegroup_number``: International spacegroup number (1-230)
- ``spacegroup_symbol``: Hermann-Mauguin spacegroup symbol

**Energetic Properties:**

- ``energy_per_atom``: Total energy per atom in eV/atom
- ``uncorrected_energy_per_atom``: Uncorrected energy per atom
- ``formation_energy_per_atom``: Formation energy in eV/atom (relative to elemental references)
- ``energy_above_hull``: Energy above convex hull in eV/atom (stability indicator)
- ``is_stable``: Boolean indicating if on convex hull (Ehull = 0)
- ``equilibrium_reaction_energy_per_atom``: Equilibrium reaction energy
- ``decomposes_to``: Products of decomposition reaction

**Electronic Properties:**

- ``band_gap``: Band gap in eV (DFT-PBE, may underestimate experimental values)
- ``cbm``: Conduction band minimum in eV
- ``vbm``: Valence band maximum in eV
- ``efermi``: Fermi energy in eV
- ``is_gap_direct``: Boolean indicating direct vs indirect band gap
- ``is_metal``: Boolean indicating metallic behavior
- ``bandstructure``: Full band structure data
- ``dos``: Density of states data
- ``dos_energy_up``: Spin-up DOS
- ``dos_energy_down``: Spin-down DOS

**Magnetic Properties:**

- ``is_magnetic``: Boolean indicating magnetic ordering
- ``ordering``: Magnetic ordering type (paramagnetic, ferromagnetic, antiferromagnetic, ferrimagnetic)
- ``total_magnetization``: Total magnetization in μ_B/atom
- ``total_magnetization_normalized_vol``: Magnetization per volume in μ_B/Å³
- ``total_magnetization_normalized_formula_units``: Magnetization per formula unit
- ``num_magnetic_sites``: Number of magnetic sites
- ``num_unique_magnetic_sites``: Number of unique magnetic sites
- ``types_of_magnetic_species``: List of magnetic element types

**Mechanical Properties:**

- ``bulk_modulus``: Bulk modulus (VRH, Voigt, Reuss) in GPa
  
  - ``k_vrh``: Voigt-Reuss-Hill average (recommended)
  - ``k_voigt``: Voigt bound (upper, iso-strain)
  - ``k_reuss``: Reuss bound (lower, iso-stress)

- ``shear_modulus``: Shear modulus (VRH, Voigt, Reuss) in GPa
  
  - ``g_vrh``: Voigt-Reuss-Hill average
  - ``g_voigt``: Voigt bound
  - ``g_reuss``: Reuss bound

- ``universal_anisotropy``: Universal elastic anisotropy index
- ``homogeneous_poisson``: Poisson's ratio (dimensionless)
- ``elastic_anisotropy``: Elastic anisotropy

**Dielectric Properties:**

- ``e_total``: Total dielectric constant
- ``e_ionic``: Ionic contribution to dielectric constant
- ``e_electronic``: Electronic contribution to dielectric constant
- ``n``: Refractive index
- ``e_ij_max``: Maximum dielectric tensor component
- ``piezoelectric_modulus``: Piezoelectric modulus in C/m²

**Surface Properties:**

- ``weighted_surface_energy``: Weighted surface energy in J/m²
- ``weighted_work_function``: Weighted work function in eV
- ``surface_anisotropy``: Surface energy anisotropy
- ``shape_factor``: Wulff shape factor
- ``has_reconstructed``: Boolean indicating surface reconstruction

**Metadata:**

- ``builder_meta``: Builder metadata
- ``deprecated``: Boolean indicating deprecated status
- ``deprecation_reasons``: Reasons for deprecation
- ``last_updated``: Last update timestamp
- ``origins``: Data origin information
- ``warnings``: Calculation warnings
- ``task_ids``: Associated task IDs
- ``theoretical``: Boolean indicating theoretical vs experimental
- ``possible_species``: Possible species in material
- ``has_props``: List of available calculated properties
- ``database_ids``: External database IDs

Database and Methodology
------------------------

**Data Source:**

The Materials Project database contains DFT-calculated properties for over 200,000 materials, computed using high-throughput ab initio calculations.

**Calculation Methodology:**

**Density Functional Theory (DFT):**

- **Exchange-correlation functional**: Perdew-Burke-Ernzerhof (PBE) generalized gradient approximation
- **Basis sets**: Projector augmented wave (PAW) pseudopotentials
- **Software**: VASP (Vienna Ab initio Simulation Package)
- **k-point density**: Converged meshes for each structure type
- **Energy cutoff**: Material-specific cutoffs (typically 520 eV)

**Elastic Properties:**

- Computed via **finite strain-stress method**
- Apply small strains to structure, compute DFT stress tensor response
- Fit elastic constants from stress-strain relationship
- Voigt, Reuss, and VRH averages computed from elastic tensor

**Thermodynamic Stability:**

- **Convex hull construction**: Computed from formation energies of all phases in chemical system
- **Energy above hull**: Perpendicular distance from convex hull surface
- **Ehull = 0**: Stable (on hull)
- **Ehull > 0**: Metastable (decomposes to products with lower energy)
- **Ehull > 0.20 eV/atom**: Likely synthesizable metastable phase

**Band Gap Limitations:**

- PBE-GGA **systematically underestimates** band gaps (typically 30-50% error)
- Metals and semimetals generally well-described
- Band structures and DOS are GGA(PBE) or GGA+U; hybrid functionals (HSE06) or GW corrections are not standard in MP core database
- For experimental validation, always compare with measured band gaps

**API Access:**

- All queries via **mp-api** Python client (Materials Project REST API v2)
- Streaming/chunking with ``chunk_size`` and ``num_chunks`` parameters (not page/per_page)
- Rate limiting: Burst throttle ≈ 25 requests/second (requires API key authentication)
- Handler implements client-side pagination for consistency with other handlers

Citations
---------

All Materials functions cite:

- **Materials Project**: Jain, A. et al. (2013). The Materials Project: A materials genome approach to accelerating materials innovation. *APL Materials*, 1(1), 011002. DOI: 10.1063/1.4812323

- **PyMatGen**: Ong, S. P. et al. (2013). Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis. *Computational Materials Science*, 68, 314-319. DOI: 10.1016/j.commatsci.2012.10.028

Additional methodology references:

- **DFT elastic properties**: de Jong, M. et al. (2015). Charting the complete elastic properties of inorganic crystalline compounds. *Scientific Data*, 2, 150009.

- **Voigt-Reuss-Hill averaging**: Hill, R. (1952). The elastic behaviour of a crystalline aggregate. *Proceedings of the Physical Society A*, 65, 349.

- **Metastable materials**: Sun, W. et al. (2016). The thermodynamic scale of inorganic crystalline metastability. *Science Advances*, 2(11), e1600225.

- **Metastability and synthesis**: Aykol, M. et al. (2018). Thermodynamic limit for synthesis of metastable inorganic materials. *Science Advances*, 4(4), eaaq0148.

Notes and Best Practices
-------------------------

**Property Units:**

- **Energies**: eV/atom (formation energy, energy above hull)
- **Moduli**: GPa (bulk modulus, shear modulus)
- **Band gaps**: eV
- **Magnetization**: Bohr magnetons (μ_B) per atom, formula unit, or volume
- **Density**: g/cm³
- **Volume**: Å³ (cubic Ångström)

**Stability Interpretation:**

- ``energy_above_hull = 0``: **Stable** - lies on convex hull
- ``0 < Ehull ≤ 0.010 eV/atom``: **Marginally stable** - numerical tolerance, effectively on hull
- ``0.010 < Ehull ≤ 0.050 eV/atom``: **Metastable** - may be kinetically stable
- ``0.050 < Ehull ≤ 0.100 eV/atom``: **Metastable** - most known synthesizable metastables fall in this range
- ``0.100 < Ehull ≤ 0.200 eV/atom``: **Metastable** - approaching practical synthesis limits, non-equilibrium processing required
- ``Ehull > 0.200 eV/atom``: **Highly metastable** - synthesis increasingly unlikely

Literature (Sun 2016, Aykol 2018) shows most experimentally known metastable materials have Ehull ≤ 0.10 eV/atom; values above 0.20 eV/atom are rare and synthesis becomes impractical.

**Elastic Moduli Interpretation:**

- **Bulk modulus (K)**: Resistance to uniform compression (volume change)
  
  - High K → incompressible (diamond: ~440 GPa)
  - Low K → compressible (lead: ~46 GPa)

- **Shear modulus (G)**: Resistance to shear deformation (shape change)
  
  - High G → stiff against shear (tungsten: ~161 GPa)
  - Low G → compliant (aluminum: ~26 GPa)

- **Poisson's ratio (ν)**: Lateral strain / axial strain under uniaxial stress
  
  - ν ≈ 0.5 → incompressible (rubber-like, volume preserving)
  - ν ≈ 0.3 → typical metals
  - ν → 0 → cork-like (no lateral expansion)

- **Universal anisotropy**: Deviation from isotropic elasticity
  
  - A = 0 → isotropic
  - A > 1 → anisotropic (directionally dependent)

**VRH Bounds (Mixture Models):**

- **Voigt bound**: Assumes iso-strain (uniform strain, upper bound)
- **Reuss bound**: Assumes iso-stress (uniform stress, lower bound)
- **VRH average**: Arithmetic mean of Voigt and Reuss (recommended for polycrystals)
- True polycrystalline modulus lies between Reuss and Voigt
- For single crystals, use full elastic tensor (not VRH)

**Composition Matching Best Practices:**

**Note**: Composition tolerance and closest-match logic are **handler-side features**, not native MP API capabilities. The handler retrieves all materials in a chemical system and performs client-side filtering by composition.

- Use **tolerance = 0.05** (±5 at.%) for general alloy searches
- Tighten to **0.02** (±2 at.%) for precise composition requirements
- Relax to **0.10** (±10 at.%) for exploratory searches
- Enable **metastable search** (``is_stable=False``, ``ehull_max=0.20``) for non-equilibrium alloys
- Check **closest_match_used** flag in results to identify tolerance violations

**Database Coverage Limitations:**

- Emphasis on **inorganic crystalline materials** (not organic, not amorphous)
- **Binary and ternary** systems well-covered; quaternary+ coverage patchy
- **Stable phases** comprehensively included; metastable coverage incomplete
- **High-temperature phases** may be missing (DFT at 0 K)
- **Solid solutions** represented by ordered supercells (not continuous composition ranges)

**Pagination Guidelines:**

- Default: ``per_page=10`` for interactive queries
- Large exports: ``per_page=100`` (max) with pagination loop
- Always check ``total_count`` to determine number of pages
- Use ``page`` parameter to iterate: ``page=1, 2, 3, ...``

**Error Handling:**

- ``ErrorType.NOT_FOUND``: Material ID or composition not in database
- ``ErrorType.API_ERROR``: Materials Project API failure (network, rate limit)
- ``ErrorType.INVALID_INPUT``: Parameter validation error (e.g., range needs [min, max])
- ``ErrorType.COMPUTATION_ERROR``: Internal processing error

**Performance Considerations:**

- **Field selection**: Specify ``fields`` parameter to reduce data transfer
- **all_fields=False**: Use when only IDs and formulas needed
- **Caching**: Results not cached by handler; implement external caching if needed
- **Batch queries**: Use ``mp_get_material_details()`` with multiple IDs (up to 100)
- **Rate limits**: Free tier = 1000 req/day; authenticated = unlimited

**DFT Accuracy Caveats:**

- **Band gaps underestimated** by PBE (use experimental values when available)
- **Strongly correlated materials** (e.g., transition metal oxides) may have errors
- **Magnetic ordering** in some materials sensitive to exchange-correlation functional
- **Phonon/thermal properties** not included (static 0 K calculations)
- **Surface energies** are Wulff-construction weighted averages (not single facets)