# Handlers Module

This directory contains all handler modules for the MCP Materials Project system. Each subdirectory focuses on a specific domain with its own AI functions and comprehensive documentation.

## Directory Structure

```
handlers/
├── README.md                    # This file
├── __init__.py                  # Main package exports
├── base.py                      # Base handler class
│
├── materials/                   # Materials Project database handlers
│   ├── __init__.py
│   ├── ai_functions.py         # AI function documentation
│   ├── README.md               # Complete usage guide
│   ├── material_search.py      # Search by composition/elements
│   └── material_details.py     # Detailed material properties
│
├── search/                      # Web and literature search handlers
│   ├── __init__.py
│   ├── ai_functions.py         # AI function documentation
│   ├── README.md               # Complete usage guide
│   └── searxng_search.py       # SearXNG metasearch integration
│
├── electrochemistry/            # Battery and electrode handlers
│   ├── __init__.py
│   ├── ai_functions.py         # AI function documentation
│   ├── README.md               # Complete usage guide
│   └── battery_handler.py      # Voltage, capacity, stability calculations
│
└── calphad/                     # CALPHAD thermodynamics handlers
    └── phase_diagrams/
        ├── __init__.py
        ├── ai_functions.py     # AI function documentation
        ├── README.md           # Complete usage guide
        ├── phase_diagrams.py   # Phase diagram calculations
        ├── analysis.py         # Thermodynamic analysis
        ├── plotting.py         # Visualization
        └── utils.py            # Utilities
```

## Module Overview

### Materials (`materials/`)

**Purpose**: Search and retrieve material information from the Materials Project database.

**AI Functions**:
- `get_material` - Search by chemical system, formula, or elements
- `get_material_by_char` - Search by properties (band gap, stability, etc.)
- `get_material_details_by_ids` - Get detailed material information

**Data Source**: Materials Project DFT database (>100,000 materials)

**Use For**: Crystal structures, electronic properties, thermodynamic data, material discovery

📖 [Read the full Materials documentation](materials/README.md)

---

### Search (`search/`)

**Purpose**: Search the web and scientific literature using SearXNG metasearch engine.

**AI Functions**:
- `search_web` - Universal search (general, scientific, materials science)
- `get_search_engines` - Check available search engines

**Data Sources**: Multiple search engines (Google Scholar, arXiv, PubMed, etc.)

**Use For**: Research papers, literature reviews, technical documentation, fact verification

📖 [Read the full Search documentation](search/README.md)

---

### Electrochemistry (`electrochemistry/`)

**Purpose**: Battery electrode analysis, voltage calculations, and electrochemical properties.

**AI Functions**:
- `search_battery_electrodes` - Find electrode materials with voltage profiles
- `calculate_voltage_from_formation_energy` - Compute voltage from DFT energies
- `compare_electrode_materials` - Side-by-side material comparison
- `check_composition_stability` - Thermodynamic stability analysis
- `analyze_anode_viability` - Comprehensive anode evaluation
- `get_voltage_profile` - Detailed voltage curves
- `analyze_lithiation_mechanism` - Phase evolution during lithiation

**Data Source**: Materials Project DFT database with convex hull analysis

**Use For**: Battery design, voltage predictions, capacity calculations, stability assessment

📖 [Read the full Electrochemistry documentation](electrochemistry/README.md)

---

### CALPHAD (`calphad/phase_diagrams/`)

**Purpose**: Phase diagram calculations using CALPHAD thermodynamic databases.

**AI Functions**:
- `plot_binary_phase_diagram` - Full binary phase diagrams
- `plot_composition_temperature` - Phase stability for specific composition
- `analyze_last_generated_plot` - Detailed plot analysis
- `list_available_systems` - Check supported systems
- `calculate_equilibrium_at_point` - Single point equilibrium
- `calculate_phase_fractions_vs_temperature` - Phase evolution
- `analyze_phase_fraction_trend` - Verify precipitation/dissolution

**Data Source**: CALPHAD thermodynamic databases (TDB files)

**Use For**: Temperature-dependent phase diagrams, melting points, phase transformations, alloy processing

📖 [Read the full CALPHAD documentation](calphad/phase_diagrams/README.md)

---

## Choosing the Right Handler

### For material properties and structures:
→ Use **materials/** handlers
- Crystal structures
- Electronic properties (band gaps, DOS)
- Formation energies
- Material discovery

### For literature and research:
→ Use **search/** handlers
- Finding research papers
- Technical documentation
- Verifying facts
- Literature reviews

### For battery applications:
→ Use **electrochemistry/** handlers
- Electrode voltages
- Battery capacity
- Energy density
- Lithiation mechanisms

### For phase diagrams and temperature effects:
→ Use **calphad/** handlers
- Phase diagrams
- Melting points
- Phase transformations
- Temperature-dependent equilibria

---

## Common Workflows

### Workflow: New Battery Material Discovery

```python
# 1. Search Materials Project for candidates
materials = await materials_handler.get_material_by_char(
    elements=['Al', 'Mg'],
    is_stable=True
)

# 2. Check stability
stability = await battery_handler.check_composition_stability(
    composition='AlMg'
)

# 3. Calculate voltage
voltage = await battery_handler.calculate_voltage_from_formation_energy(
    electrode_formula='AlMg',
    working_ion='Li'
)

# 4. Search literature
papers = await search_handler.search_web(
    query='AlMg lithium battery anode',
    search_type='scientific'
)
```

### Workflow: Alloy Design

```python
# 1. Generate phase diagram
diagram = await calphad_handler.plot_binary_phase_diagram(
    system='Al-Zn'
)

# 2. Analyze specific composition
composition = await calphad_handler.plot_composition_temperature(
    composition='Al20Zn80',
    min_temperature=300,
    max_temperature=900
)

# 3. Get material properties
materials = await materials_handler.get_material(
    chemsys='Al-Zn'
)

# 4. Search for experimental data
papers = await search_handler.search_web(
    query='Al-Zn alloy properties',
    search_type='materials_science'
)
```

### Workflow: Understanding Phase Stability

```python
# 1. Check thermodynamic stability (DFT, 0 K)
stability_0K = await battery_handler.check_composition_stability(
    composition='Al3Mg2'
)

# 2. Check temperature-dependent stability
phase_diagram = await calphad_handler.plot_binary_phase_diagram(
    system='Al-Mg'
)

# 3. Analyze phase evolution
fractions = await calphad_handler.calculate_phase_fractions_vs_temperature(
    composition='Al60Mg40',
    min_temperature=300,
    max_temperature=900
)
```

---

## Data Sources Comparison

| Handler | Data Source | Temperature | Validation |
|---------|-------------|-------------|------------|
| **materials** | Materials Project DFT | 0 K | Benchmarked |
| **search** | Multiple search engines | N/A | Literature |
| **electrochemistry** | Materials Project DFT | 0 K | Computed |
| **calphad** | CALPHAD databases | 0-3000 K | Experimental |

**Key Differences**:
- **materials** & **electrochemistry**: First-principles DFT (0 K only)
- **calphad**: Fitted to experiments (full temperature range)
- **search**: External literature and data sources

**Complementary Use**:
- Use DFT for 0 K predictions and discovery
- Use CALPHAD for temperature-dependent behavior
- Use search for experimental validation

---

## AI Function Documentation

Each handler module includes an `ai_functions.py` file that documents all AI-accessible functions. These files provide:

- Function descriptions
- Parameter definitions
- Return value formats
- Use cases and examples
- Important notes and limitations

**Example**:
```python
from mcp_materials_project.handlers.materials import ai_functions

# Get info about a specific function
info = ai_functions.get_function_info('get_material')

# List all functions
all_funcs = ai_functions.list_all_functions()
```

---

## README Documentation

Each handler module includes a comprehensive README with:

1. **Overview**: Module purpose and capabilities
2. **Available Functions**: Detailed function documentation
3. **Parameters**: Complete parameter descriptions
4. **Returns**: Output format and contents
5. **Examples**: Usage examples
6. **Use Cases**: When to use each function
7. **Workflows**: Common task sequences
8. **Best Practices**: Tips and recommendations
9. **Troubleshooting**: Common issues and solutions

---

## Base Handler Class

All handlers inherit from `BaseHandler` (`base.py`), which provides:

- Materials Project API client (`mpr`)
- Pagination utilities
- Parameter parsing and validation
- Range filter handling
- Common helper methods

---

## Design Principles

1. **Modular Organization**: Each domain in its own directory
2. **Consistent Interface**: All handlers follow similar patterns
3. **Comprehensive Documentation**: README + ai_functions.py for each module
4. **Self-Contained**: Each module can be understood independently
5. **Cross-Module Workflows**: Handlers work together seamlessly

---

## Development Guidelines

### Adding a New Handler

1. Create subdirectory: `handlers/new_module/`
2. Implement handler class inheriting from `BaseHandler`
3. Add `@ai_function` decorators to AI-accessible methods
4. Create `ai_functions.py` documenting all functions
5. Write comprehensive `README.md`
6. Update `handlers/__init__.py` with imports
7. Add examples and workflows

### Documentation Standards

Each README should include:
- Module overview
- Function-by-function documentation
- Parameters and returns
- Usage examples
- Typical workflows
- Best practices
- Troubleshooting
- Related modules

---

## Getting Started

1. **Explore the documentation**:
   - Read module READMEs to understand capabilities
   - Check `ai_functions.py` for quick reference

2. **Try basic examples**:
   - Start with simple searches or calculations
   - Experiment with parameters

3. **Combine modules**:
   - Use multiple handlers for complex analyses
   - Follow workflow examples

4. **Read the notes**:
   - Understand data sources and limitations
   - Learn best practices

---

## Support and Resources

- **Module READMEs**: Comprehensive documentation for each handler
- **AI Functions**: Quick reference in `ai_functions.py` files
- **Examples**: Provided in each README
- **Base Handler**: `base.py` for common functionality

---

## Module Status

✅ **materials/**: Complete with documentation  
✅ **search/**: Complete with documentation  
✅ **electrochemistry/**: Complete with documentation  
✅ **calphad/**: Complete with documentation  

All modules are fully documented with:
- Comprehensive READMEs
- AI function documentation
- Usage examples
- Best practices guides

