# CALPHAD Phase Diagram Handlers

This module provides AI-accessible functions for calculating and analyzing phase diagrams using CALPHAD (CALculation of PHAse Diagrams) thermodynamic databases.

## Overview

CALPHAD is a computational thermodynamics approach that uses thermodynamic databases to calculate phase equilibria, phase diagrams, and thermodynamic properties as a function of temperature, composition, and pressure.

This module enables:
- Binary phase diagram generation
- Composition-specific phase stability analysis
- Temperature-dependent phase fraction calculations
- Visual and thermodynamic analysis of phase equilibria
- Interactive and static plotting capabilities

## Module Structure

The phase diagrams module is organized into focused, modular components:

### Core Files

- **`phase_diagrams.py`** (423 lines): Main handler class (`CalPhadHandler`) that combines all mixins
- **`ai_functions.py`** (861 lines): AI-accessible functions decorated with `@ai_function`

### Mixin Classes

- **`analysis.py`** (790 lines): Analysis utilities for phase diagrams
  - Liquidus/solidus extraction
  - Eutectic point detection
  - Phase fraction analysis
  - Composition-temperature analysis

- **`plotting.py`** (169 lines): Plotting utilities
  - Plot saving and formatting
  - Visual content analysis
  - Interactive HTML generation

### Utility Modules

- **`database_utils.py`** (155 lines): Database management
  - `load_database()` - Load thermodynamic databases
  - `get_db_elements()` - Extract elements from database
  - `pick_tdb_path()` - Select appropriate database
  - `is_excluded_phase()` - Filter non-equilibrium phases
  - `compose_alias_map()` - Element alias handling

- **`diagram_utils.py`** (181 lines): Phase diagram generation
  - `generate_binary_phase_diagram()` - Create binary phase diagrams
  - `format_success_message()` - Format output messages
  - `validate_elements_in_database()` - Element validation

- **`equilibrium_utils.py`** (198 lines): Equilibrium calculations
  - `calculate_equilibrium_at_point()` - Single-point equilibrium
  - `calculate_coarse_equilibrium_grid()` - Grid calculations
  - `extract_stable_phases()` - Phase extraction
  - `calculate_phase_fractions_at_temperature()` - Phase fraction calculation

- **`consts.py`** (57 lines): Constants and static data
  - Element aliases
  - Phase exclusion patterns

- **`utils.py`** (26 lines): Re-exports for backward compatibility

### Benefits of This Structure

✓ **Modular organization** - Each module has a focused responsibility  
✓ **Easy to navigate** - Find functionality by category  
✓ **Backward compatible** - Existing imports still work via utils.py  
✓ **Maintainable** - Changes isolated to specific modules  
✓ **Testable** - Each module can be tested independently  
✓ **Extensible** - Easy to add new utilities without bloating files

## Available Functions

### 1. `plot_binary_phase_diagram`

**PREFERRED for phase diagram questions**. Generate a complete binary phase diagram showing all phases across the full composition range.

**Purpose**: Visualize phase equilibria for a binary system across all compositions and temperatures.

**Parameters**:
- `system` (required): Chemical system as 'A-B', 'AB', or 'element1-element2' (e.g., 'Al-Zn', 'AlZn', 'aluminum-zinc')
- `min_temperature` (optional): Minimum temperature in Kelvin. Default: auto-scaled
- `max_temperature` (optional): Maximum temperature in Kelvin. Default: auto-scaled
- `composition_step` (optional): Composition resolution (0-1). Default: 0.02
- `figure_width` (optional): Width in inches. Default: 9
- `figure_height` (optional): Height in inches. Default: 6

**Returns**: String with success message and key findings including:
- System identified (A-B)
- Phases present
- Temperature range used
- Pure element melting points
- Eutectic points (if detected)

**Additional Outputs** (handled by stream state):
- High-resolution PNG image with phase boundaries
- Phase labels and legend
- Grid and axis labels
- Marked eutectic points
- Combined visual + thermodynamic analysis

**Example Usage**:
```python
# Basic phase diagram
result = await handler.plot_binary_phase_diagram(
    system='Al-Zn'
)

# With specific temperature range
result = await handler.plot_binary_phase_diagram(
    system='Al-Si',
    min_temperature=500,
    max_temperature=1500
)

# High resolution
result = await handler.plot_binary_phase_diagram(
    system='Cu-Zn',
    composition_step=0.01,
    figure_width=12,
    figure_height=8
)
```

**Visual Features**:
- **Phase regions**: Color-coded stable phases
- **Phase boundaries**: Solidus, liquidus, solvus lines
- **Eutectic points**: Marked with red dots and annotations
- **Pure endpoints**: Melting points at x=0 and x=1
- **Legend**: Phase names and colors
- **Grid**: For easy reading of values

**Use Cases**:
- Answering general system queries ("Al-Zn phase diagram")
- Understanding phase transformations
- Identifying liquidus and solidus temperatures
- Finding eutectic compositions
- Studying phase stability regions
- Alloy design and processing

**Important Notes**:
- Auto-scales temperature range if not specified
- Detects and marks eutectic points automatically
- Provides both visual representation and text analysis
- Full composition range (0-100% of component B)

---

### 2. `plot_composition_temperature`

**PREFERRED for composition-specific questions**. Plot phase stability vs temperature for a single composition.

**Purpose**: Show which phases are stable and their fractions at a specific composition across a temperature range.

**Parameters**:
- `composition` (required): Specific composition like:
  - 'Al20Zn80': 20 at% Al, 80 at% Zn
  - 'Al80Zn20': 80 at% Al, 20 at% Zn
  - 'Zn' or 'Al': Pure element
  - Element-number format (e.g., 'Zn30Al70')
- `min_temperature` (optional): Min temperature (K). Default: 300
- `max_temperature` (optional): Max temperature (K). Default: 1000
- `composition_type` (optional): 'atomic' for at% or 'weight' for wt%. Default: 'atomic'
- `figure_width`, `figure_height` (optional): Figure dimensions
- `interactive` (optional): 'html' for interactive Plotly plot. Default: 'html'

**Returns**: String with success message

**Additional Outputs**:
- Interactive HTML plot (Plotly) or static PNG
- Phase fraction curves vs temperature
- Phase transition temperatures
- Analysis of phase stability

**Example Usage**:
```python
# Specific composition
result = await handler.plot_composition_temperature(
    composition='Al20Zn80',
    min_temperature=300,
    max_temperature=900
)

# Pure element (melting point)
result = await handler.plot_composition_temperature(
    composition='Zn',
    min_temperature=600,
    max_temperature=800
)

# Weight percent composition
result = await handler.plot_composition_temperature(
    composition='Al30Zn70',
    composition_type='weight',
    min_temperature=400,
    max_temperature=1000
)
```

**Visual Features** (Interactive Mode):
- Hover tooltips with exact values
- Zoom and pan capabilities
- Phase fraction curves
- Transition temperature markers
- Legend with phase names

**Use Cases**:
- Finding melting points of specific compositions
- Identifying phase transitions (e.g., solid → liquid)
- Understanding heat treatment temperatures
- Analyzing phase fractions during cooling/heating
- Answering "what phases are stable at composition X?"

**Important for Questions**:
- "What is the melting point of Al20Zn80?"
- "What phases are stable at 500K for Al50Zn50?"
- "When does Al80Zn20 fully melt?"

---

### 3. `analyze_last_generated_plot`

Analyze and interpret the most recently generated phase diagram or composition plot.

**Purpose**: Get detailed analysis of the last plot's visual and thermodynamic features.

**Parameters**: None

**Returns**: Detailed analysis string including:
- **Visual analysis**: What's visible in the plot
  - Phase regions and boundaries
  - Color coding
  - Key features (eutectics, melting points)
- **Thermodynamic analysis**: Physical interpretation
  - Pure element melting points
  - Eutectic reactions
  - Phase equilibria
  - Composition ranges

**Example Usage**:
```python
# Generate a diagram
await handler.plot_binary_phase_diagram(system='Al-Zn')

# Get detailed analysis
analysis = await handler.analyze_last_generated_plot()
```

**Use Cases**:
- Understanding what a plot shows
- Getting interpretation of features
- Extracting quantitative information
- Explaining phase diagram features

---

### 4. `list_available_systems`

List all binary systems supported by the thermodynamic database.

**Purpose**: Check which element pairs can be calculated.

**Parameters**: None

**Returns**: Dictionary with:
- `systems`: List of available binary systems
- `total_systems`: Count
- `database_file`: Name of .tdb file used
- `tdb_directory`: Path to database directory

**Example Usage**:
```python
systems = await handler.list_available_systems()
```

**Use Cases**:
- Checking if a system is available
- Exploring database coverage
- Understanding supported chemistries

---

### 5. `calculate_equilibrium_at_point`

Calculate thermodynamic equilibrium at a specific temperature and composition.

**Purpose**: Get exact phase fractions and compositions at a single point.

**Parameters**:
- `composition` (required): Multi-component composition (e.g., 'Al30Si55C15', 'Al80Zn20')
- `temperature` (required): Temperature in Kelvin
- `composition_type` (optional): 'atomic' or 'weight'. Default: 'atomic'

**Returns**: Formatted string with:
- Temperature (K and °C)
- Composition
- Stable phases with:
  - Phase name
  - Phase fraction (%)
  - Composition of each phase

**Example Usage**:
```python
result = await handler.calculate_equilibrium_at_point(
    composition='Al80Zn20',
    temperature=600
)
```

**Use Cases**:
- Verifying phase fractions at specific conditions
- Understanding multi-phase equilibria
- Checking calculation accuracy
- Extracting exact numerical values

---

### 6. `calculate_phase_fractions_vs_temperature`

Calculate how phase fractions change with temperature for a fixed composition.

**Purpose**: Understand precipitation, dissolution, and phase transformations.

**Parameters**:
- `composition` (required): Composition (e.g., 'Al30Si55C15', 'Al80Zn20')
- `min_temperature` (required): Min temperature (K)
- `max_temperature` (required): Max temperature (K)
- `temperature_step` (optional): Step size in K. Default: 10
- `composition_type` (optional): 'atomic' or 'weight'. Default: 'atomic'

**Returns**: Formatted analysis with:
- Temperature range
- Composition
- Phase evolution for each phase:
  - Starting and ending fractions
  - Trend (increasing, decreasing, stable)
  - Change amount

**Example Usage**:
```python
result = await handler.calculate_phase_fractions_vs_temperature(
    composition='Al30Si55C15',
    min_temperature=500,
    max_temperature=1500,
    temperature_step=10
)
```

**Use Cases**:
- Understanding precipitation behavior
- Analyzing dissolution temperatures
- Phase transformation studies
- Heat treatment design
- Solvus boundary determination

**Essential for Questions**:
- "Does phase X increase or decrease with temperature?"
- "At what temperature does phase Y precipitate?"
- "How does phase Z dissolve upon heating?"

---

### 7. `analyze_phase_fraction_trend`

Analyze whether a specific phase increases or decreases with temperature.

**Purpose**: Verify statements about precipitation or dissolution behavior.

**Parameters**:
- `composition` (required): Composition (e.g., 'Al30Si55C15')
- `phase_name` (required): Name of phase (e.g., 'AL4C3', 'SIC', 'FCC_A1')
- `min_temperature` (required): Min temperature (K)
- `max_temperature` (required): Max temperature (K)
- `expected_trend` (optional): Expected behavior ('increase', 'decrease', 'stable')

**Returns**: Detailed analysis with:
- Phase fraction at min and max temperature
- Change amount and percentage
- Trend description
- Maximum and minimum fractions
- Verification of expected trend (if provided)

**Example Usage**:
```python
result = await handler.analyze_phase_fraction_trend(
    composition='Al30Si55C15',
    phase_name='AL4C3',
    min_temperature=500,
    max_temperature=1500,
    expected_trend='increases with decreasing temperature'
)
```

**Use Cases**:
- Verifying precipitation behavior
- Checking dissolution trends
- Validating expected behavior
- Understanding specific phase evolution

**Verification**:
- ✅ or ❌ indicator if expected_trend matches calculation
- Handles various trend descriptions:
  - "increasing/decreasing temperature"
  - "upon cooling/heating"
  - "with temperature"

---

## Key Concepts

### Phase Diagrams

A **phase diagram** shows which phases are thermodynamically stable as a function of temperature, composition, and sometimes pressure.

**Features**:
- **Phase regions**: Areas where specific phases are stable
- **Phase boundaries**: Lines separating regions (solidus, liquidus, solvus)
- **Eutectic point**: Composition and temperature where liquid transforms to two solid phases
- **Liquidus**: Boundary above which material is completely liquid
- **Solidus**: Boundary below which material is completely solid
- **Two-phase regions**: Areas where two phases coexist

### CALPHAD Method

**CALPHAD** = CALculation of PHAse Diagrams

**Approach**:
1. Thermodynamic models for each phase (Gibbs free energy functions)
2. Model parameters fitted to experimental data
3. Numerical minimization of Gibbs free energy
4. Prediction of phase equilibria

**Advantages**:
- Temperature-dependent (unlike DFT at 0 K)
- Fast calculations
- Validated against experiments
- Extrapolation to unmeasured regions
- Covers wide temperature ranges

**Limitations**:
- Requires fitted database (not all systems available)
- Accuracy depends on database quality
- Limited to systems in database
- May not include metastable phases

### Temperature Dependence

Unlike DFT (0 K), CALPHAD includes:
- **Enthalpy** (H): Heat content
- **Entropy** (S): Disorder
- **Gibbs free energy**: G = H - TS

As temperature changes:
- Entropy contribution (TS) becomes more important
- Phase stability changes
- High-T: Entropy favors disordered phases (liquid, FCC)
- Low-T: Enthalpy favors ordered phases (intermetallics)

### Phase Fractions

When multiple phases coexist:
- **Lever rule**: Determines phase fractions from composition
- **Tie-lines**: Connect coexisting phase compositions
- **Phase fraction**: Amount of each phase (0-1 or 0-100%)

Example: In a two-phase region at 60% B:
- Phase α at 40% B (fraction: 0.67)
- Phase β at 100% B (fraction: 0.33)
- Lever rule: fraction_α = (100-60)/(100-40)

### Eutectic Reactions

**Eutectic**: Special composition where:
```
Liquid → Solid₁ + Solid₂
```

**Characteristics**:
- Lowest melting point in the system
- Simultaneous formation of two solid phases
- Important for casting alloys
- Marked on phase diagrams

---

## Typical Workflows

### Workflow 1: Understanding a Binary System

```python
# Step 1: Generate full phase diagram
diagram = await handler.plot_binary_phase_diagram(
    system='Al-Zn'
)

# Step 2: Get detailed analysis
analysis = await handler.analyze_last_generated_plot()

# Step 3: Check specific composition
composition_plot = await handler.plot_composition_temperature(
    composition='Al20Zn80',
    min_temperature=300,
    max_temperature=900
)
```

### Workflow 2: Melting Point Determination

```python
# For a specific composition
result = await handler.plot_composition_temperature(
    composition='Al50Zn50',
    min_temperature=600,
    max_temperature=900
)
# Look for phase transition from solid to liquid
```

### Workflow 3: Phase Transformation Analysis

```python
# Analyze phase fractions
fractions = await handler.calculate_phase_fractions_vs_temperature(
    composition='Al30Si55C15',
    min_temperature=500,
    max_temperature=1500
)

# Verify specific phase behavior
trend = await handler.analyze_phase_fraction_trend(
    composition='Al30Si55C15',
    phase_name='AL4C3',
    min_temperature=500,
    max_temperature=1500,
    expected_trend='increases upon cooling'
)
```

---

## Current Database Support

The module uses thermodynamic databases (TDB files) for calculations.

**Currently Available**:
- **Al-based systems**: Comprehensive aluminum alloy database
  - Al-Zn, Al-Cu, Al-Mg, Al-Si, and others
  - Multi-component: Al-Zn-Mg, Al-Cu-Mg, etc.

**Database Location**: `tdbs/` directory in project root

**To check availability**:
```python
systems = await handler.list_available_systems()
```

---

## Comparison: CALPHAD vs DFT (Materials Project)

| Feature | CALPHAD (this module) | DFT (Materials Project) |
|---------|----------------------|-------------------------|
| **Temperature** | Full range (0-3000 K) | 0 K only |
| **Speed** | Fast (seconds) | Pre-computed (database) |
| **Accuracy** | Fitted to experiments | First principles |
| **Coverage** | Limited systems | >100,000 materials |
| **Phase diagrams** | Yes (T vs composition) | No (0 K only) |
| **Voltage** | No | Yes (battery module) |
| **Validation** | Experimental data | Benchmarked calculations |

**Use CALPHAD for**:
- Temperature-dependent phase equilibria
- Full phase diagrams
- Melting points
- Phase transformations
- Alloy processing

**Use DFT/Materials Project for**:
- Electronic properties (band gap, DOS)
- 0 K thermodynamic stability
- Crystal structures
- Battery voltages
- Large materials screening

---

## Best Practices

1. **Choose the right function**:
   - General system → `plot_binary_phase_diagram`
   - Specific composition → `plot_composition_temperature`
   - Melting points → `plot_composition_temperature` with temperature range around melting
   - Phase fractions → `calculate_phase_fractions_vs_temperature`

2. **Temperature ranges**:
   - Let auto-scaling work for full diagrams
   - Specify range for focused analysis
   - Include safety margin (±100 K) around region of interest

3. **Composition format**:
   - Use Element-Number format: 'Al20Zn80'
   - Specify composition_type if using weight percent
   - For pure elements, just use symbol: 'Al', 'Zn'

4. **Interactive plots**:
   - Default HTML mode for exploration
   - Hover for exact values
   - Zoom for detailed regions

5. **Analysis**:
   - Always use `analyze_last_generated_plot` after generating a diagram
   - Provides quantitative information
   - Helps interpret visual features

---

## Troubleshooting

### "No thermodynamic database found"
- System not in current database
- Check available systems: `list_available_systems()`
- May need additional .tdb files

### "Elements not found in database"
- Typo in element symbols (case-sensitive)
- Element not in this database
- Use chemical symbols: 'Al' not 'aluminum'

### Empty or strange phase diagram
- Temperature range may be too narrow
- Try auto-scaling (don't specify temperatures)
- Check composition step (try 0.01 for higher resolution)

### Phase fraction calculation fails
- Composition outside valid range
- Temperature outside database limits
- Check that all elements are in database

### Plot not displaying
- Image URL provided in metadata
- Check stream state handling
- Verify file saving permissions

---

## Implementation Details

### PyCalphad Backend
- **Library**: PyCalphad
- **Engine**: Thermodynamic equilibrium solver
- **Method**: Gibbs free energy minimization
- **Models**: Sublattice models, compound energy formalism

### Database Format
- **TDB files**: Thermodynamic database format
- **Standard**: CALPHAD format (widely used)
- **Content**: Gibbs energy functions, phase models, parameters

### Calculation Process
1. Parse composition and temperature
2. Load appropriate .tdb file
3. Filter applicable phases
4. Set up thermodynamic conditions
5. Minimize Gibbs free energy
6. Extract equilibrium phases and fractions
7. Generate visualizations

### Performance
- Typical binary diagram: 5-15 seconds
- Single point equilibrium: <1 second
- Multi-temperature scan: 2-10 seconds
- Resolution vs speed trade-off

---

## Related Modules

- **electrochemistry/**: Battery voltage calculations (0 K DFT-based)
- **materials/**: Materials Project database (crystal structures, properties)
- **search/**: Literature search for experimental phase diagrams

**When to use CALPHAD vs electrochemistry**:
- **CALPHAD**: Temperature-dependent phase diagrams, melting points, phase fractions
- **Electrochemistry**: Battery voltages, capacities, energy densities (0 K thermodynamics)

Both are thermodynamic, but:
- CALPHAD: T-dependent, fitted to experiments, general alloys
- Electrochemistry: 0 K, DFT-based, battery-specific

---

## Example Q&A

**Q: What is the Al-Zn phase diagram?**
```python
result = await handler.plot_binary_phase_diagram(system='Al-Zn')
analysis = await handler.analyze_last_generated_plot()
```

**Q: What is the melting point of Al20Zn80?**
```python
result = await handler.plot_composition_temperature(
    composition='Al20Zn80',
    min_temperature=600,
    max_temperature=900
)
# Look for solid→liquid transition
```

**Q: Does the AL4C3 phase increase with decreasing temperature?**
```python
result = await handler.analyze_phase_fraction_trend(
    composition='Al30Si55C15',
    phase_name='AL4C3',
    min_temperature=500,
    max_temperature=1500,
    expected_trend='increases with decreasing temperature'
)
```

**Q: What phases are stable at 600K for Al80Zn20?**
```python
result = await handler.calculate_equilibrium_at_point(
    composition='Al80Zn20',
    temperature=600
)
```

