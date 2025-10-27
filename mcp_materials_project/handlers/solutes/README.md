# Solutes Handler

Handler for analyzing lattice parameter effects of substitutional solute atoms in fcc matrices.

## Purpose

This handler provides tools to evaluate and rank how different solute elements affect the lattice parameter of an fcc matrix material in the dilute substitutional limit. It's designed to support or refute claims like:

- "Mg in Al causes the largest lattice expansion"
- "Cu in Al causes a moderate lattice contraction"
- "Zn addition has negligible effect on Al lattice parameter"

## Physics Basis

### Core Principles

1. **FCC Geometry**: For face-centered cubic structures, the lattice parameter relates to atomic radius by:
   ```
   a = 2√2 × r
   ```

2. **Vegard's Law**: In the dilute limit, lattice parameter changes linearly with composition:
   ```
   Δa/a₀ ≈ δ_size × x
   ```
   where x is the solute mole fraction and δ_size is the size misfit parameter.

3. **Size Misfit**: The substitutional size mismatch is defined as:
   ```
   δ_size = (r_solute - r_matrix) / r_matrix
   ```

4. **Hume-Rothery Rules**: Size mismatch > ~15% significantly reduces solubility in substitutional solid solutions.

### Reference Data

- **Lattice Parameters** (at ~300 K):
  - Al: 4.046 Å
  - Ni: 3.52 Å
  - Cu: 3.60 Å

- **Metallic Radii** (12-fold coordination, in pm):
  - Al: 143, Mg: 160, Cu: 128, Zn: 134
  - Ni: 124, Fe: 126, Co: 125, Cr: 128
  - And more...

## AI Functions

### 1. `calculate_solute_lattice_effect`

Calculate how a single substitutional solute changes the lattice parameter.

**Inputs:**
- `matrix_element`: Matrix element (e.g., "Al")
- `solute_element`: Solute element (e.g., "Mg")
- `solute_atpct`: Nominal solute concentration (at.%)
- `temperature_K`: Temperature in Kelvin
- `matrix_phase_name`: CALPHAD phase name (must be "FCC_A1")
- `matrix_phase_composition`: Element mole fractions from CALPHAD equilibrium
- `min_required_solute_atpct`: Threshold for valid dissolution (default: 0.1 at.%)

**Returns:**
- Lattice parameter change prediction (Δa/a₀)
- Size misfit parameters
- Classification (expands/contracts, large/moderate/negligible)
- Hume-Rothery analysis
- Applicability validation

**Example:**
```python
result = await calculate_solute_lattice_effect(
    matrix_element="Al",
    solute_element="Mg",
    solute_atpct=1.0,
    temperature_K=300,
    matrix_phase_name="FCC_A1",
    matrix_phase_composition={"AL": 0.995, "MG": 0.005}
)
# Returns: expands_lattice (moderate), Δa/a₀ ≈ +0.12%, size misfit ~11.9%
```

### 2. `rank_solutes_by_lattice_expansion`

Compare multiple solutes and rank by lattice expansion effect.

**Inputs:**
- `matrix_element`: Matrix element (e.g., "Al")
- `solute_elements`: List of solutes to compare (e.g., ["Mg", "Cu", "Zn"])
- `solute_atpct`: Nominal concentration for all solutes
- `temperature_K`: Temperature
- `matrix_phase_name`: CALPHAD phase name
- `calphad_matrix_compositions`: Per-solute equilibrium compositions

**Returns:**
- Detailed results for each solute
- Sorted ranking (largest expander first)
- Largest expander identification
- Human-readable commentary

**Example:**
```python
result = await rank_solutes_by_lattice_expansion(
    matrix_element="Al",
    solute_elements=["Mg", "Cu", "Zn"],
    solute_atpct=1.0,
    temperature_K=300,
    matrix_phase_name="FCC_A1",
    calphad_matrix_compositions={
        "MG": {"AL": 0.995, "MG": 0.005},
        "CU": {"AL": 0.999, "CU": 0.001},
        "ZN": {"AL": 0.998, "ZN": 0.002}
    }
)
# Returns: Ranking with Mg as largest expander
```

### 3. `get_solute_reference_data`

Get available reference data for supported elements.

**Returns:**
- Supported fcc matrices
- Available metallic radii
- Reference temperature
- Data sources

## Workflow Integration

This handler is designed to work with the CALPHAD handler:

1. **First**: Use CALPHAD to get equilibrium phase compositions at your temperature:
   ```python
   calphad_result = await calculate_equilibrium_at_point(...)
   ```

2. **Extract** the FCC_A1 phase composition:
   ```python
   fcc_composition = calphad_result["phases"]["FCC_A1"]["composition"]
   ```

3. **Then**: Use this handler to analyze lattice effects:
   ```python
   lattice_result = await calculate_solute_lattice_effect(
       matrix_phase_composition=fcc_composition,
       ...
   )
   ```

## Important Notes

### Applicability

This model is ONLY valid when:
- The matrix phase is fcc (FCC_A1)
- The solute is substitutionally dissolved in the matrix
- The concentration is in the dilute limit (typically < 5 at.%)
- CALPHAD confirms non-zero solute presence in the matrix phase

### Limitations

1. **Dilute Limit Only**: Vegard's law breaks down at higher concentrations
2. **Room Temperature Data**: Reference lattice parameters are at ~300 K
3. **Ideal Substitution**: Assumes no clustering, ordering, or interstitial effects
4. **Database Coverage**: Only elements in METALLIC_RADII_PM are supported

### When It Says "Not Applicable"

The handler will refuse to make predictions if:
- Matrix phase is not FCC_A1
- Solute concentration in matrix < threshold (default 0.1 at.%)
- Missing reference data for matrix or solute
- CALPHAD shows solute precipitates out rather than dissolving

## Example Use Cases

### Claim: "Mg causes the largest expansion in Al"

1. Run CALPHAD equilibrium for Al-Mg, Al-Cu, Al-Zn at 300K, 1 at.%
2. Extract FCC_A1 compositions for each
3. Call `rank_solutes_by_lattice_expansion`
4. Check if Mg is indeed the largest_expander

### Claim: "Cu contracts the Al lattice"

1. Run CALPHAD for Al-Cu at 300K, 1 at.%
2. Extract FCC_A1 composition
3. Call `calculate_solute_lattice_effect`
4. Verify classification is "contracts_lattice"

## Citations

When using this handler, cite:

- Vegard's law for dilute substitutional alloys
- Hume-Rothery rules for solid solutions  
- Standard crystallographic data for fcc lattice parameters
- Metallic radii compilations for 12-fold coordination

## Future Extensions

Potential improvements:
- Add BCC and HCP matrix support
- Temperature-dependent lattice parameters
- Non-dilute corrections (higher-order terms)
- Integration with elastic modulus data
- Experimental lattice parameter validation

