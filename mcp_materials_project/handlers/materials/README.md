# Materials Handlers

This module provides AI-accessible functions for searching and retrieving material information from the Materials Project database.

## Overview

The Materials Project is a comprehensive database of computed materials properties. These handlers provide convenient access to search for materials by composition, elements, or properties, and retrieve detailed information about specific materials.

## Available Functions

### 1. `get_material`

Search for materials by chemical system, formula, or elements.

**Purpose**: Find materials based on their composition.

**Parameters**:
- `chemsys` (optional): Chemical system(s) or comma-separated list (e.g., "Li-Fe-O", "Si-*")
- `formula` (optional): Chemical formula(s), anonymized formula, or wildcard(s) (e.g., "Li2FeO3", "Fe2O3", "Fe*O*")
- `element` (optional): Element(s) or comma-separated list (e.g., "Li,Fe,O")
- `page` (optional): Page number (default 1)
- `per_page` (optional): Items per page (max 10; default 10)

**Returns**: Dictionary containing:
- `total_count`: Total number of materials found
- `page`, `per_page`, `total_pages`: Pagination info
- `data`: List of materials with material_id and formula_pretty

**Example Usage**:
```python
# Find all lithium-iron-oxide materials
result = await handler.get_material(chemsys="Li-Fe-O")

# Find materials containing aluminum
result = await handler.get_material(element="Al")

# Search by formula pattern
result = await handler.get_material(formula="Fe*O*")
```

**Use Cases**:
- Exploring materials in a specific chemical system
- Finding all materials containing certain elements
- Searching for materials matching a formula pattern
- Building a database of materials for further analysis

---

### 2. `get_material_by_char`

Find materials by their characteristics (properties).

**Purpose**: Search for materials with specific property values or ranges.

**Key Parameters**:
- `band_gap`: Min,max range in eV (e.g., [1.2, 3.0])
- `crystal_system`: One of 'Triclinic', 'Monoclinic', 'Orthorhombic', 'Tetragonal', 'Trigonal', 'Hexagonal', 'Cubic'
- `density`: Min,max density range
- `formation_energy`: Min,max formation energy in eV/atom
- `energy_above_hull`: Min,max energy above hull in eV/atom
- `is_stable`: Boolean - whether material is on the convex hull
- `is_metal`: Boolean - whether material is metallic
- `is_gap_direct`: Boolean - whether band gap is direct
- `elements`: List of elements (e.g., ['Si', 'Ge'])
- `magnetic_ordering`: One of 'paramagnetic', 'ferromagnetic', 'antiferromagnetic', 'ferrimagnetic'
- `spacegroup_number`: Integer spacegroup number
- `spacegroup_symbol`: Spacegroup symbol
- `page`, `per_page`: Pagination

**Returns**: Dictionary containing:
- `total_count`: Number of materials matching criteria
- `page`, `per_page`, `total_pages`: Pagination
- `data`: List of materials with requested properties

**Example Usage**:
```python
# Find semiconductors with band gap 1-3 eV
result = await handler.get_material_by_char(
    band_gap=[1.0, 3.0],
    is_stable=True
)

# Find stable cubic silicon-germanium alloys
result = await handler.get_material_by_char(
    elements=['Si', 'Ge'],
    crystal_system='Cubic',
    is_stable=True
)

# Find materials with low formation energy
result = await handler.get_material_by_char(
    formation_energy=[-2.0, -0.5],
    is_metal=False
)
```

**Use Cases**:
- Materials discovery for specific applications (solar cells, batteries, etc.)
- Finding materials with targeted properties
- Screening materials databases for candidates
- Identifying stable phases in a composition range
- Searching for materials with specific magnetic or electronic properties

---

### 3. `get_material_details_by_ids`

Retrieve detailed information for specific materials by their Material IDs.

**Purpose**: Get comprehensive data about known materials.

**Parameters**:
- `material_ids`: List of material IDs (e.g., ['mp-149', 'mp-150', 'mp-151'])
- `fields` (optional): List of specific fields to retrieve. If not provided, returns common fields.
- `all_fields` (optional): Boolean - return all available fields (default True)
- `page`, `per_page`: Pagination

**Available Fields** (partial list):
- **Identity**: material_id, formula_pretty, formula_anonymous, elements, chemsys
- **Structure**: structure, nsites, symmetry, volume, density, density_atomic
- **Electronic**: band_gap, cbm, vbm, efermi, is_gap_direct, is_metal, bandstructure, dos
- **Magnetic**: is_magnetic, ordering, total_magnetization, num_magnetic_sites
- **Thermodynamic**: formation_energy_per_atom, energy_above_hull, is_stable, energy_per_atom
- **Mechanical**: bulk_modulus, shear_modulus, elastic properties
- **Dielectric**: e_total, e_ionic, e_electronic
- **Surface**: weighted_surface_energy, surface_anisotropy, weighted_work_function

**Returns**: Dictionary containing:
- `total_count`: Number of materials
- `page`, `per_page`, `total_pages`: Pagination
- `data`: List of materials with requested fields

**Example Usage**:
```python
# Get all information about a material
result = await handler.get_material_details_by_ids(
    material_ids=['mp-149'],
    all_fields=True
)

# Get specific fields for multiple materials
result = await handler.get_material_details_by_ids(
    material_ids=['mp-149', 'mp-150', 'mp-151'],
    fields=['material_id', 'formula_pretty', 'band_gap', 'formation_energy_per_atom'],
    all_fields=False
)
```

**Use Cases**:
- Getting crystal structures for computational modeling
- Retrieving electronic band structures for analysis
- Accessing thermodynamic stability data
- Fetching magnetic properties for spintronics applications
- Obtaining mechanical properties for materials design

---

## Typical Workflows

### Workflow 1: Find and Analyze Semiconductors
```python
# Step 1: Search for stable semiconductors
materials = await handler.get_material_by_char(
    band_gap=[1.0, 3.0],
    is_stable=True,
    is_metal=False,
    per_page=10
)

# Step 2: Get detailed info for promising candidates
material_ids = [m['material_id'] for m in materials['data']]
details = await handler.get_material_details_by_ids(
    material_ids=material_ids,
    fields=['material_id', 'formula_pretty', 'band_gap', 'structure', 'bandstructure']
)
```

### Workflow 2: Explore a Chemical System
```python
# Step 1: Find all materials in Li-Mn-O system
materials = await handler.get_material(chemsys="Li-Mn-O")

# Step 2: Filter for stable phases
stable_materials = await handler.get_material_by_char(
    elements=['Li', 'Mn', 'O'],
    is_stable=True
)

# Step 3: Get details for stable phases
material_ids = [m['material_id'] for m in stable_materials['data']]
details = await handler.get_material_details_by_ids(material_ids=material_ids)
```

### Workflow 3: Property-Based Materials Discovery
```python
# Find materials with specific combinations of properties
candidates = await handler.get_material_by_char(
    band_gap=[2.0, 2.5],              # Target band gap for solar cells
    energy_above_hull=[0.0, 0.02],    # Nearly stable
    crystal_system='Cubic',            # High symmetry
    is_gap_direct=True                 # Direct band gap for efficiency
)
```

---

## Implementation Details

### Classes
- **MaterialSearchHandler**: Implements get_material and get_material_by_char
- **MaterialDetailsHandler**: Implements get_material_details_by_ids

Both inherit from `BaseHandler` which provides:
- Materials Project API client (mpr)
- Pagination utilities
- Parameter parsing and validation
- Range filter handling

### Data Sources
All data comes from the Materials Project database, which uses:
- DFT calculations (mostly GGA/GGA+U)
- High-throughput computational framework
- Standardized calculation protocols
- Peer-reviewed thermodynamic corrections

### Rate Limits
The Materials Project API has rate limits:
- Default pagination: 10 items per page (max)
- Use pagination parameters for large result sets
- Consider caching results for repeated queries

---

## Notes and Best Practices

1. **Pagination**: Always use pagination for large result sets. The API limits per_page to 10.

2. **Range Filters**: For properties like band_gap, formation_energy, etc., provide both min and max values as a list [min, max].

3. **Chemical Symbols**: Always use standard chemical symbols (e.g., 'Li' not 'lithium').

4. **Wildcards**: The formula parameter supports wildcards (e.g., 'Fe*O*' finds all iron oxides).

5. **Stability**: Materials with energy_above_hull = 0 are thermodynamically stable. Values < 0.1 eV/atom may be metastable.

6. **Fields Selection**: When retrieving details, specify only needed fields to reduce response size and improve performance.

7. **Error Handling**: Functions return error information in the response dictionary if queries fail.

---

## Related Modules

- **electrochemistry/**: Battery and electrode calculations
- **search/**: Web and literature search capabilities  
- **calphad/**: Phase diagram calculations and thermodynamic analysis

For questions about thermodynamic phase stability, see the CALPHAD handlers.
For battery-specific properties (voltage, capacity), see the electrochemistry handlers.

