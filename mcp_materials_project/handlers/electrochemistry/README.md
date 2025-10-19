# Electrochemistry Handlers

This module provides AI-accessible functions for battery electrode analysis, voltage calculations, and electrochemical properties using Materials Project DFT data.

## Overview

The electrochemistry handlers enable comprehensive analysis of battery electrode materials, including:

- Voltage profile calculations from DFT energies
- Capacity and energy density predictions
- Thermodynamic stability analysis
- Lithiation mechanism studies
- Multi-material comparisons
- Phase evolution during charge/discharge

All calculations use Materials Project DFT data and thermodynamically rigorous convex hull analysis.

## Module Structure

The electrochemistry module is organized into a clean, modular architecture:

### Core Files

- **`ai_functions.py`**: Contains all AI-accessible functions decorated with `@ai_function`. These are the functions that the AI model can call directly. The functions are kept clean and focused on parameter handling and result formatting.

- **`battery_handler.py`**: Main handler class that combines the AI functions mixin and provides the interface for external use.

### Utility Modules

The implementation logic is split into focused utility modules:

- **`utils.py`**: Re-exports all utilities for backward compatibility. Import from here to access any utility function.

- **`constants.py`**: Shared constants (Faraday constant, PyMatGen availability).

- **`api_utils.py`**: Materials Project API interaction:
  - `build_electrode_query_params()` - Build search queries
  - `process_electrode_documents()` - Process API responses
  - `extract_voltage_profile()` - Extract voltage profiles

- **`voltage_utils.py`**: Voltage calculations via convex hull:
  - `compute_alloy_voltage_via_hull()` - Hull scan method
  - `calculate_voltage_from_insertion_electrode()` - PyMatGen InsertionElectrode method

- **`comparison_utils.py`**: Material comparison functions:
  - `generate_comparison_summary()` - Compare voltages, capacities, energy densities

- **`stability_utils.py`**: Thermodynamic stability analysis:
  - `check_composition_stability_detailed()` - Stability checks
  - `_calculate_decomposition_coefficients()` - Decomposition analysis

- **`lithiation_utils.py`**: Lithiation mechanism analysis:
  - `analyze_lithiation_mechanism_detailed()` - Full mechanism analysis
  - `_compute_lithiation_steps()` - Convex hull steps
  - Helper functions for phase analysis

### Benefits of This Structure

✓ **Clear separation of concerns** - Each module has a single responsibility  
✓ **Easy to navigate** - Find functionality by topic  
✓ **Backward compatible** - Import from `utils.py` still works  
✓ **Maintainable** - Isolated changes don't affect other modules  
✓ **Testable** - Each module can be tested independently  
✓ **Stable AI interface** - AI functions unchanged regardless of implementation

## Available Functions

### 1. `search_battery_electrodes`

Search Materials Project's insertion electrode database for battery materials with computed voltage profiles.

**Purpose**: Find electrode materials with pre-computed electrochemical properties.

**Parameters**:
- `formula` (optional): Chemical formula (e.g., 'AlMg', 'Al2Mg3', 'LiCoO2')
- `elements` (optional): Comma-separated elements (e.g., 'Al,Mg' or 'Li,Co,O')
- `working_ion` (optional): Working ion ('Li', 'Na', 'Mg'). Default: 'Li'
- `min_capacity`, `max_capacity` (optional): Capacity range in mAh/g
- `min_voltage`, `max_voltage` (optional): Voltage range in V vs working ion
- `max_entries` (optional): Max results. Default: 10

**Returns**: Dictionary with:
- `electrodes`: List of electrode data including:
  - `battery_id`, `material_id`: Identifiers
  - `formula`, `formula_charge`, `formula_discharge`: Compositions
  - `average_voltage`: Average voltage vs working ion (V)
  - `max_voltage_step`: Maximum voltage in profile (V)
  - `capacity_grav`: Gravimetric capacity (mAh/g)
  - `capacity_vol`: Volumetric capacity (mAh/cm³)
  - `energy_grav`: Gravimetric energy density (Wh/kg)
  - `energy_vol`: Volumetric energy density (Wh/L)
- `count`: Number of results found
- `query`: Search parameters used

**Example Usage**:
```python
# Search for Al-Mg alloy anodes
result = await handler.search_battery_electrodes(
    formula='AlMg',
    working_ion='Li',
    max_entries=5
)

# Find high-capacity materials
result = await handler.search_battery_electrodes(
    elements='Al,Li',
    min_capacity=1000,  # mAh/g
    working_ion='Li'
)

# Search sodium battery cathodes
result = await handler.search_battery_electrodes(
    elements='Na,Mn,O',
    working_ion='Na',
    min_voltage=2.0,
    max_voltage=4.0
)
```

**Use Cases**:
- Finding voltage profiles for specific compositions
- Screening high-capacity anode materials
- Comparing different electrode chemistries
- Identifying promising battery materials

**Notes**:
- Database contains pre-computed data for common compositions
- If not found, automatically falls back to convex hull calculation
- Voltages are vs. working ion reference (e.g., Li/Li+)

---

### 2. `calculate_voltage_from_formation_energy`

Calculate electrode voltage from DFT formation energies using convex hull analysis.

**Purpose**: Compute voltage for any composition using thermodynamic principles.

**Parameters**:
- `electrode_formula` (required): Electrode composition (e.g., 'Al3Mg2', 'AlMg')
- `working_ion` (optional): Working ion ('Li', 'Na', 'Mg'). Default: 'Li'
- `temperature` (optional): Temperature in K (currently unused). Default: 298.15

**Returns**: Dictionary with:
- `calculated_voltage`: Average voltage (V)
- `voltage_range`: Min, max, average voltage (V)
- `capacity_grav`: Gravimetric capacity (mAh/g)
- `energy_grav`: Gravimetric energy density (Wh/kg)
- `chemical_system`: Elements in calculation
- `framework_formula`: Host framework composition
- `num_entries_used`: Number of DFT entries in calculation
- `calculation_method`: Method used (pymatgen_insertion_electrode or phase_diagram_line_scan)
- `notes`: Important methodological details

**Example Usage**:
```python
# Calculate voltage for Al-Mg alloy
result = await handler.calculate_voltage_from_formation_energy(
    electrode_formula='Al3Mg2',
    working_ion='Li'
)

# Sodium battery anode
result = await handler.calculate_voltage_from_formation_energy(
    electrode_formula='Sn',
    working_ion='Na'
)
```

**Methodology**:
1. Retrieves all DFT entries for chemical system from Materials Project
2. Builds convex hull using PyMatGen PhaseDiagram
3. Uses InsertionElectrode to compute two-phase equilibria
4. Derives voltages from hull segment slopes: V = -ΔG/Δx
5. Returns physically valid voltage ranges only

**Use Cases**:
- Computing voltages for alloy anodes
- Theoretical voltage predictions for new materials
- Understanding thermodynamic voltage limits
- Verifying experimental voltage measurements

**Important Notes**:
- Uses 0 K convex hull (DFT energies)
- All data from consistent Materials Project entry set
- Voltages from thermodynamically rigorous two-phase equilibria
- Returns error if voltage is unphysical (<-0.1 V or >6 V)
- Automatically falls back to hull line scan if needed

---

### 3. `compare_electrode_materials`

Compare multiple electrode materials side-by-side.

**Purpose**: Direct comparison of voltages, capacities, and energy densities.

**Parameters**:
- `formulas` (required): Comma-separated formulas (e.g., 'Al,AlMg,Al3Mg2')
- `working_ion` (optional): Working ion. Default: 'Li'

**Returns**: Dictionary with:
- `comparison`: List of results for each material
- `summary`: Statistical summary including:
  - `voltages`: Voltage for each material
  - `highest_voltage`, `lowest_voltage`: Rankings
  - `voltage_comparison`: Difference statement
  - `capacities`: Capacity rankings
  - `energy_densities`: Energy density rankings
- `notes`: Data sources and methodology

**Example Usage**:
```python
# Compare Al-based anodes
result = await handler.compare_electrode_materials(
    formulas='Al,AlMg,Al3Mg2',
    working_ion='Li'
)
# Output includes: "AlMg has 0.1234 V higher voltage than Al"

# Compare cathode materials
result = await handler.compare_electrode_materials(
    formulas='LiCoO2,LiFePO4,LiMn2O4',
    working_ion='Li'
)
```

**Use Cases**:
- Answering "does X increase/decrease voltage vs Y?"
- Comparing alloying effects
- Materials selection for battery design
- Understanding composition-voltage relationships
- Evaluating trade-offs between voltage, capacity, energy density

**Essential for questions like**:
- "Does alloying Al with Mg increase voltage?"
- "Which has higher capacity: A or B?"
- "Compare Al vs AlMg as anode"

---

### 4. `check_composition_stability`

Check if a composition is thermodynamically stable on the convex hull.

**Purpose**: Determine if a phase can exist as a stable compound.

**Parameters**:
- `composition` (required): Composition to check (e.g., 'Cu8LiAl', 'Li3Al2', 'Cu80Li10Al10')

**Returns**: Dictionary with:
- `is_stable`: Boolean - true if on convex hull
- `energy_above_hull`: eV/atom above hull (0 = stable, None if no entry exists)
- `decomposition`: List of equilibrium phases with:
  - `formula`: Phase formula
  - `amount`: Formula-unit coefficient (atom-balanced)
  - `phase_fraction`: Barycentric weight (composition-space)
  - `material_id`: Materials Project ID
- `reduced_formula`: Normalized composition
- `chemical_system`: Elements involved
- `notes`: Interpretation and context

**Example Usage**:
```python
# Check if Cu8LiAl is stable
result = await handler.check_composition_stability(
    composition='Cu8LiAl'
)
# Returns: is_stable, energy_above_hull, decomposition products

# Check Li3Al2 stability
result = await handler.check_composition_stability(
    composition='Li3Al2'
)
```

**Interpretation**:
- `energy_above_hull = 0`: Stable phase (on convex hull)
- `0 < energy_above_hull < 0.1`: Possibly metastable
- `energy_above_hull > 0.1`: Likely decomposes
- `energy_above_hull = None`: No DFT entry exists for this formula

**Use Cases**:
- Determining if a composition can form
- Understanding decomposition reactions
- Evaluating metastability
- Predicting equilibrium phases
- Checking synthesis feasibility

---

### 5. `analyze_anode_viability`

Comprehensive analysis of a composition as a potential battery anode.

**Purpose**: Combined stability + voltage analysis for anode evaluation.

**Parameters**:
- `composition` (required): Composition (e.g., 'Cu8LiAl', 'AlMg', 'Li3Al2')
- `working_ion` (optional): Working ion. Default: 'Li'

**Returns**: Dictionary with:
- `stability_analysis`: Full output from check_composition_stability
- `voltage_analysis`: Voltage data if viable
- `viability_assessment`:
  - `can_form_stable_anode`: Boolean
  - `reasoning`: List of assessment points

**Example Usage**:
```python
# Evaluate AlMg as anode
result = await handler.analyze_anode_viability(
    composition='AlMg',
    working_ion='Li'
)
```

**Assessment Logic**:
1. Checks thermodynamic stability
2. If stable and no working ion: calculates voltage (good anode candidate)
3. If stable and contains working ion: identifies as lithiated phase
4. If unstable: analyzes decomposition products
5. Provides overall viability recommendation

**Use Cases**:
- Screening potential anode materials
- Understanding synthesis requirements
- Evaluating novel compositions
- Assessing thermodynamic feasibility

---

### 6. `get_voltage_profile`

Get detailed voltage profile for a specific electrode material.

**Purpose**: Full charge/discharge curve with phase evolution.

**Parameters**:
- `material_id` (required): Materials Project ID or battery ID
- `working_ion` (optional): Working ion. Default: 'Li'

**Returns**: Dictionary with:
- `voltage_profile`: List of voltage steps with:
  - `voltage`: Step voltage
  - `capacity`: Capacity at this step
  - `x_charge`, `x_discharge`: Composition limits
- `average_voltage`, `max_voltage`, `min_voltage`
- `capacity_grav`: Total capacity

**Example Usage**:
```python
result = await handler.get_voltage_profile(
    material_id='mp-12345',
    working_ion='Li'
)
```

**Use Cases**:
- Plotting voltage curves
- Analyzing voltage plateaus
- Understanding phase transitions
- Evaluating hysteresis

---

### 7. `analyze_lithiation_mechanism`

**MOST DETAILED**: Analyze the lithiation mechanism including all phases and reaction pathways.

**Purpose**: Understand complete phase evolution during lithiation.

**Parameters**:
- `host_composition` (required): Host material (e.g., 'AlCu', 'CuAl', 'Al'). **Do NOT include Li.**
- `working_ion` (optional): Working ion. Default: 'Li'
- `max_x` (optional): Max Li per host atom. Default: 3.0
- `room_temp` (optional): Filter high-energy phases. Default: True

**Returns**: Dictionary with:
- `initial_reaction`: First lithiation step with:
  - `reaction_type`: "two-phase plateau" or "multi-phase plateau"
  - `voltage`: Initial voltage
  - `phases_in_microstructure`: List of phases present
  - `is_constant_mu_plateau`: Always True (thermodynamic two-state)
  - `is_two_phase_microstructure`: True only if exactly 2 phases
  - `explanation`: Detailed mechanism description
- `lithiation_steps`: All voltage plateaus with:
  - `step_number`: Sequential numbering
  - `x_host_range`: Li per host atom range
  - `x_mix_range`: Mole fraction of Li range
  - `voltage`: Plateau voltage
  - `equilibrium_phases`: All phases present
  - `num_phases_in_microstructure`: Phase count
  - `reaction_type`: Classification
  - `microstructure_note`: Detailed description
- `num_plateau_steps`: Total plateaus detected
- `average_voltage`, `voltage_range`
- `plating_starts_at_x_host`: Li plating onset (if any)
- `methodology`: Detailed algorithm description

**Example Usage**:
```python
# Analyze Al-Cu alloy lithiation
result = await handler.analyze_lithiation_mechanism(
    host_composition='AlCu',
    working_ion='Li',
    max_x=3.0,
    room_temp=True
)
```

**Methodology**:
1. Samples G(x) along lithiation path (x = Li per host atom)
2. Builds lower convex hull using Andrew's monotone chain
3. Identifies voltage plateaus as hull segments
4. Probes 5 points per segment to find all equilibrium phases
5. Classifies reactions by phase count
6. Reports full phase set (not just endpoints)
7. Stops at Li plating (pure Li metal in equilibrium)

**Key Outputs**:

**Composition Variables**:
- `x_host`: Li per host atom (normalized so host sums to 1)
- `x_mix`: Mole fraction of Li in mixture

**Reaction Classification**:
- `is_constant_mu_plateau`: True for ALL hull segments (constant chemical potential)
- `is_two_phase_microstructure`: True ONLY if exactly 2 phases across segment
- Distinguishes thermodynamic two-state reaction from microstructure phase count

**Phase Reporting**:
- Full phase set from interior sampling
- Not limited to endpoints (captures multi-phase regions)
- Material IDs for each phase

**Use Cases**:
- Understanding initial lithiation reactions
- Identifying two-phase vs multi-phase plateaus
- Analyzing voltage steps
- Determining which phases form during cycling
- Studying alloying mechanisms
- Answering "what phases form?" questions
- Essential for mechanistic understanding

**Critical for Questions**:
- "What is the initial reaction?"
- "Is this a two-phase reaction?"
- "What phases form during lithiation?"
- "Does it follow a single-phase or multi-phase path?"

**Room Temperature Filter**:
- When `room_temp=True`: Excludes phases >30 meV/atom above hull
- Rationale: High-energy phases unlikely to form at RT
- Improves prediction of actual battery behavior
- Can disable for theoretical analysis

---

## Key Concepts

### Voltage Calculation

**Thermodynamic Definition**:
```
V = -ΔG / (n·F)
```
- V: Voltage vs working ion (V)
- ΔG: Gibbs free energy change (J/mol)
- n: Moles of electrons
- F: Faraday constant (96485 C/mol)

**For insertion electrodes** (at 0 K, DFT):
```
V = -ΔG_formation / Δx
```
- x: Li content (per formula unit or per host atom)
- Derived from convex hull segment slopes

**Two-Phase Equilibrium**:
- Constant voltage plateau
- Two phases coexist (lever rule)
- Corresponds to tie-line on convex hull
- Thermodynamically rigorous

### Capacity

**Gravimetric Capacity** (mAh/g):
```
C_grav = (26801 · n_Li) / M_host
```
- n_Li: Moles of Li per host formula unit
- M_host: Molar mass of host (g/mol)
- 26801 = F / 3600 (Faraday constant in Ah/mol)

**Volumetric Capacity** (mAh/cm³):
```
C_vol = C_grav · ρ
```
- ρ: Density of electrode material (g/cm³)

### Energy Density

**Gravimetric Energy** (Wh/kg):
```
E_grav = V_avg · C_grav
```

**Volumetric Energy** (Wh/L):
```
E_vol = V_avg · C_vol
```

### Convex Hull Analysis

The **convex hull** is the lower envelope of formation energies in composition space.

**Properties**:
- Points on hull = stable phases (energy_above_hull = 0)
- Points above hull = unstable (decompose to hull phases)
- Hull segments = two-phase equilibria
- Segment slope = voltage: V = -dG/dx

**For batteries**:
- Voltage plateaus correspond to hull segments
- Phase fractions follow lever rule
- Multiple plateaus = multi-step lithiation
- Physically rigorous (thermodynamic equilibrium)

### Working Ion

The **working ion** shuttles between electrodes:
- **Li**: Most common, lightest metal (high capacity)
- **Na**: More abundant than Li, lower voltage
- **Mg**: Divalent (2e⁻ per ion), higher volumetric capacity
- **Zn**, **Al**: Alternative chemistries

Voltage is **always reported vs. working ion reference**:
- Li: vs. Li/Li⁺ (Li metal reference)
- Na: vs. Na/Na⁺
- Mg: vs. Mg/Mg²⁺

### Composition Variables

Two conventions for Li content:

1. **x_host**: Li per host atom
   - Host atoms normalized to sum = 1
   - Total atoms = 1 + x_host
   - Example: At x_host = 1, composition is Li₁Host₁

2. **x_mix**: Mole fraction of Li
   - x_mix = n_Li / (n_Li + n_host)
   - For binary: x_mix = x_host / (1 + x_host)
   - More intuitive for phase diagrams

**Both are provided** in `analyze_lithiation_mechanism`.

---

## Typical Workflows

### Workflow 1: Evaluate New Anode Material

```python
# Step 1: Check stability
stability = await handler.check_composition_stability(
    composition='AlMg'
)

# Step 2: If stable, calculate voltage
if stability['is_stable']:
    voltage = await handler.calculate_voltage_from_formation_energy(
        electrode_formula='AlMg',
        working_ion='Li'
    )
    
# Step 3: Compare with alternatives
comparison = await handler.compare_electrode_materials(
    formulas='Al,AlMg,Al3Mg2',
    working_ion='Li'
)
```

### Workflow 2: Understand Lithiation Mechanism

```python
# Detailed phase analysis
mechanism = await handler.analyze_lithiation_mechanism(
    host_composition='AlCu',
    working_ion='Li',
    max_x=3.0
)

# Check: Initial reaction details
initial = mechanism['initial_reaction']
print(f"Initial voltage: {initial['voltage']:.3f} V")
print(f"Phases: {initial['phases_in_microstructure']}")
print(f"Is two-phase? {initial['is_two_phase_microstructure']}")
```

### Workflow 3: Materials Screening

```python
# Find high-capacity anodes
electrodes = await handler.search_battery_electrodes(
    working_ion='Li',
    min_capacity=1000,  # mAh/g
    max_voltage=1.0,    # Anode range
    max_entries=20
)

# Compare top candidates
formulas = [e['formula'] for e in electrodes['electrodes'][:5]]
comparison = await handler.compare_electrode_materials(
    formulas=','.join(formulas),
    working_ion='Li'
)
```

---

## Data Sources and Accuracy

### Materials Project DFT Database
- **Method**: Density Functional Theory (mostly GGA/GGA+U)
- **Software**: VASP with standardized settings
- **Accuracy**: Typically ±0.1-0.2 V for voltages
- **Coverage**: >100,000 inorganic materials
- **Thermodynamic corrections**: For certain chemistries

### Voltage Accuracy
- **Systematic errors**: GGA typically underestimates band gaps
- **Temperature**: Calculations at 0 K (room temp effects not included)
- **Kinetics**: Not considered (thermodynamic equilibrium only)
- **Uncertainty**: ~0.15 V typical for alloy anodes (stored in results)

### When to Trust Results
✓ Well-studied systems (Li-Al, Li-Si, etc.)
✓ Trends and comparisons (more reliable than absolute values)
✓ Thermodynamic feasibility
✓ Phase stability predictions

### When to Be Cautious
⚠ New chemistries without experimental data
⚠ Absolute voltage values (systematic DFT errors)
⚠ Kinetic effects (rate capability, nucleation barriers)
⚠ Temperature effects (entropy contributions)
⚠ Metastable phases (may exist despite thermodynamic instability)

---

## Best Practices

1. **Always specify working_ion**: Voltages are reference-dependent

2. **Use compare_ for comparisons**: More reliable than absolute values

3. **Check stability first**: Use `check_composition_stability` before assuming a phase exists

4. **Room temp filter**: Keep `room_temp=True` for practical predictions

5. **Understand composition variables**:
   - x_host: Li per host atom (good for capacity)
   - x_mix: Mole fraction (good for phase diagrams)

6. **Interpret plateaus carefully**:
   - ALL hull segments are constant chemical potential
   - Only exactly-2-phase segments are "two-phase microstructure"
   - Multi-phase plateaus are physically valid

7. **Consider voltage uncertainty**: ±0.15 V typical for GGA

8. **Cross-check with experiments**: DFT predictions should be verified

---

## Limitations

1. **0 K only**: Temperature effects (entropy) not included
2. **No kinetics**: Nucleation barriers, diffusion not considered
3. **Perfect crystallinity**: Defects, grain boundaries ignored
4. **Equilibrium only**: Non-equilibrium phases not predicted
5. **DFT errors**: Systematic errors in GGA (especially for band gaps)
6. **No SEI**: Solid-electrolyte interphase not modeled
7. **No electrolyte**: Solvation effects not included

Despite limitations, provides excellent guidance for:
- Thermodynamic feasibility
- Theoretical voltage limits
- Phase stability
- Comparative analysis
- Materials screening

---

## Related Modules

- **materials/**: Materials Project database search and properties
- **calphad/**: CALPHAD phase diagrams (temperature-dependent)
- **search/**: Literature search for experimental data

**When to use electrochemistry vs. CALPHAD**:
- Use **electrochemistry** for: Battery-specific properties (voltage, capacity, energy density)
- Use **CALPHAD** for: Temperature-dependent phase diagrams, full composition range

Both use thermodynamic principles, but:
- Electrochemistry: DFT-based, 0 K, battery-specific outputs
- CALPHAD: Fitted databases, temperature-dependent, general phase equilibria

---

## Troubleshooting

### "No entries found for chemical system"
- System not in Materials Project database
- Try broader search (fewer elements)
- Check element symbols (case-sensitive)

### "No suitable framework found"
- Composition may not support insertion/alloying
- Try different host ratios
- Check if composition is stable first

### "Voltage out of plausible range"
- DFT calculation may have issues
- System may not be suitable for this working ion
- Check input composition

### Low capacity predictions
- May be accurate (not all materials are high-capacity)
- Check if host mass is very high
- Compare with similar materials

### Many phases in mechanism
- Normal for complex systems
- Room temp filter may help reduce to accessible phases
- Focus on phases with high fractions

---

## Example Q&A

**Q: Does alloying Al with Mg increase the voltage?**
```python
result = await handler.compare_electrode_materials(
    formulas='Al,AlMg',
    working_ion='Li'
)
# Check: result['summary']['voltage_comparison']
```

**Q: What is the initial lithiation reaction for AlCu?**
```python
result = await handler.analyze_lithiation_mechanism(
    host_composition='AlCu',
    working_ion='Li'
)
# Check: result['initial_reaction']
```

**Q: Is Cu8LiAl thermodynamically stable?**
```python
result = await handler.check_composition_stability(
    composition='Cu8LiAl'
)
# Check: result['is_stable'] and result['energy_above_hull']
```

**Q: What is the voltage of an AlMg anode vs Li?**
```python
result = await handler.calculate_voltage_from_formation_energy(
    electrode_formula='AlMg',
    working_ion='Li'
)
# Check: result['calculated_voltage']
```

