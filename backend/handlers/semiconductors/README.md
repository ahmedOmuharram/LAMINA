# Semiconductor and Defect Analysis Handler

This handler provides comprehensive tools for analyzing semiconductors, crystal structures, defects, and doping effects in materials.

## Features

### 1. Structural Analysis
- **Octahedral Distortion Analysis**: Analyze VO₆ octahedra and other coordination environments
  - Bond length distributions and deviations
  - Bond angle analysis
  - Distortion parameters
  - Temperature-dependent structural changes

### 2. Magnetic Properties
- **Magnetic Property Retrieval**: Get detailed magnetic properties
  - Magnetic ordering (ferromagnetic, antiferromagnetic, etc.)
  - Total magnetization
  - Magnetization per volume and per formula unit
  - Magnetic site information

- **Magnetic Property Comparison**: Compare magnetism between materials
  - Analyze doping effects on magnetism (e.g., Al-doped Fe₂O₃)
  - Quantify magnetic enhancement

### 3. Defect Formation Energy
- **Defect Stability Analysis**: Compare different defect configurations
  - Substitutional vs interstitial doping
  - Formation energy calculations
  - Stability comparisons (e.g., P in Si)

### 4. Doping Site Preference
- **Site Preference Analysis**: Determine preferred substitution sites
  - Analyze compound semiconductors (e.g., N in GaAs)
  - Compare energies of dopant at different sublattice sites
  - Temperature and pressure dependence

### 5. Phase Transition Analysis
- **Polymorph Comparison**: Analyze structural differences across phases
  - Temperature-dependent phase transitions
  - Ground state vs excited state structures
  - Structural distortions across polymorphs

## AI Functions

### `analyze_octahedral_distortion_in_material`
Analyze octahedral distortions in a crystal structure.

**Use Case**: "At low temperatures there are no distortions in the VO₆ octahedra in VO₂"

**Parameters**:
- `material_id`: Material ID (e.g., 'mp-19094' for VO₂)
- `central_element`: Element at center of octahedra (e.g., 'V')
- `neighbor_element`: Element at corners (e.g., 'O')

### `get_magnetic_properties`
Get comprehensive magnetic properties for a material.

**Use Case**: Analyzing magnetic properties of Fe₂O₃

**Parameters**:
- `material_id`: Material ID

### `compare_magnetic_materials`
Compare magnetic properties between two materials.

**Use Case**: "Someone made a stronger magnet by doping α iron oxide (Fe₂O₃ trigonal) with aluminum"

**Parameters**:
- `material_id_1`: First material ID (undoped)
- `material_id_2`: Second material ID (doped)

### `analyze_defect_stability`
Analyze defect formation energy and stability.

**Use Case**: "Phosphorus interstitials in Si are unstable compared to substitutional doping"

**Parameters**:
- `host_material_id`: Host material ID (e.g., 'mp-149' for Si)
- `defect_composition`: Composition with defect (e.g., {'Si': 31, 'P': 1})
- `defect_type`: 'substitutional' or 'interstitial'

### `analyze_doping_site_preference`
Analyze which sublattice site a dopant prefers.

**Use Case**: "At STP, nitrogen doping of GaAs at Ga sites is more stable than at As sites"

**Parameters**:
- `host_formula`: Host material formula (e.g., 'GaAs')
- `dopant_element`: Dopant element (e.g., 'N')
- `site_a_element`: First potential site (e.g., 'Ga')
- `site_b_element`: Second potential site (e.g., 'As')
- `temperature`: Temperature in K (default: 298.15)
- `pressure`: Pressure in atm (default: 1.0)

### `analyze_phase_transition_structures`
Analyze structural differences across polymorphs/phases.

**Use Case**: Understanding temperature-dependent phase transitions in VO₂

**Parameters**:
- `formula`: Chemical formula (e.g., 'VO₂')
- `element_of_interest`: Element to analyze (e.g., 'V')
- `neighbor_element`: Neighboring element (e.g., 'O')

### `search_doped_materials`
Search for materials containing both host elements and dopant.

**Use Case**: Finding Al-doped Fe₂O₃ materials

**Parameters**:
- `host_elements`: List of host elements (e.g., ['Fe', 'O'])
- `dopant_element`: Dopant element (e.g., 'Al')
- `max_results`: Maximum results to return (default: 10)

## Example Questions

### 1. VO₂ Octahedral Distortions
**Question**: "At low temperatures there are no distortions in the VO₆ octahedra VO₂"

**Approach**:
1. Search for VO₂ polymorphs using `get_material`
2. Analyze ground state structure with `analyze_octahedral_distortion_in_material`
3. Compare with higher energy polymorphs using `analyze_phase_transition_structures`

### 2. Al-Doped Fe₂O₃ Magnetism
**Question**: "Someone made a stronger magnet by doping α iron oxide (Fe₂O₃ trigonal) with aluminum"

**Approach**:
1. Find pure Fe₂O₃ (trigonal) using `get_material_by_char` with crystal_system='Trigonal'
2. Find Al-doped Fe₂O₃ using `search_doped_materials` with host_elements=['Fe', 'O'], dopant_element='Al'
3. Compare magnetic properties using `compare_magnetic_materials`

### 3. P in Si: Interstitial vs Substitutional
**Question**: "Phosphorus interstitials in Si are unstable compared to substitutional doping"

**Approach**:
1. Find pure Si using `get_material` with formula='Si'
2. Analyze substitutional P using `analyze_defect_stability` with defect_type='substitutional'
3. Analyze interstitial P using `analyze_defect_stability` with defect_type='interstitial'
4. Compare formation energies

### 4. N Doping Site Preference in GaAs
**Question**: "At STP, nitrogen doping of GaAs at Ga sites is more stable than at As sites"

**Approach**:
1. Use `analyze_doping_site_preference` with:
   - host_formula='GaAs'
   - dopant_element='N'
   - site_a_element='Ga'
   - site_b_element='As'
   - temperature=298.15 (STP)
   - pressure=1.0 (STP)

## Technical Details

### Structure Analysis
Uses pymatgen's `CrystalNN` for coordination environment analysis and calculates:
- Bond length distributions
- Distortion parameters (σ/μ for bond lengths)
- Bond angles and deviations from ideal geometry
- Octahedral regularity metrics

### Magnetic Properties
Retrieves from Materials Project:
- Total magnetization (μB)
- Magnetization per volume (μB/Bohr³)
- Magnetization per formula unit (μB/f.u.)
- Magnetic ordering type
- Number and types of magnetic sites

### Defect Formation Energy
Approximates defect formation energy using:
- E_formation ≈ E(defect structure) - E(host)
- Chemical potential considerations
- Energy above hull for stability assessment

**Note**: Full defect formation energies require charged defect calculations and proper chemical potential references. This implementation provides estimates based on Materials Project database entries.

### Doping Site Preference
Compares energies of ternary compounds where dopant occupies different sublattice sites:
- Searches for materials with host + dopant elements
- Identifies substitution sites based on composition stoichiometry
- Compares energy above hull for stability

## Dependencies

- **pymatgen**: Structure analysis, coordination environments
- **mp-api**: Materials Project database access
- **numpy**: Numerical calculations
- **kani**: AI function decorators

## Data Sources

All data retrieved from:
- **Materials Project**: Crystal structures, energies, magnetic properties
- **pymatgen**: Structure analysis algorithms

## Citations

When using this handler, cite:
1. Materials Project: Jain, A. et al. APL Mater. 1, 011002 (2013)
2. pymatgen: Ong, S.P. et al. Comput. Mater. Sci. 68, 314-319 (2013)

