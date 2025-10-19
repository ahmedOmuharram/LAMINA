# Test Questions to Handler Functions Mapping

This document maps each test question to the specific handler functions being called.

## CALPHAD Phase Diagram Tests

### Q1: Al20Zn80 at 870K is a solid at equilibrium

**Expected Answer**: TRUE

**Handler Functions Used**:
- `calculate_equilibrium_at_point(composition="Al20Zn80", temperature=870.0)`
  - Returns phase fractions and phase names at specific T
  - Analysis checks for presence/absence of LIQUID phase
  
- `plot_composition_temperature(composition="Al20Zn80", min_temp=300, max_temp=1200)`
  - Generates phase stability plot showing phase fractions vs T
  - Visual confirmation of phase state at 870K

**Analysis Method**: 
- Parse equilibrium output for phase names
- Check if only solid phases (FCC, HCP, BCC) are present
- Verify no LIQUID phase

---

### Q2: Al and Zn form an eutectic at ~Al15Zn75

**Expected Answer**: TRUE

**Handler Functions Used**:
- `plot_binary_phase_diagram(system="Al-Zn", min_temp=300, max_temp=1000)`
  - Generates full binary phase diagram
  - Automatically detects eutectic points using `_find_eutectic_points()`
  - Returns analysis with eutectic composition and temperature

**Analysis Method**:
- Extract analysis text from metadata
- Search for "eutectic" mentions
- Parse composition values (e.g., "75 at% Zn")
- Verify composition is near 75 at% Zn (±5%)

---

### Q3: Al50Zn50 forms a single solid phase <700K

**Expected Answer**: TRUE (or FALSE depending on actual phase diagram)

**Handler Functions Used**:
- `calculate_equilibrium_at_point(composition="Al50Zn50", temperature=T)` for T ∈ {300, 500, 650}K
  - Multiple point calculations below 700K
  - Counts number of stable solid phases at each temperature
  
- `plot_composition_temperature(composition="Al50Zn50", min_temp=300, max_temp=900)`
  - Visual confirmation of phase evolution

**Analysis Method**:
- Parse equilibrium results to count stable phases
- Exclude LIQUID from count
- Check if exactly 1 solid phase at all temperatures
- Look for phase boundaries in plot

---

### Q4: Al30Si55C15 precipitates increase with decreasing temperature (500K→300K)

**Expected Answer**: TRUE

**Handler Functions Used**:
- `calculate_phase_fractions_vs_temperature(composition="Al30Si55C15", min_temp=300, max_temp=500, step=10)`
  - Calculates phase fractions at many temperature points
  - Returns trend analysis for each phase
  
- `plot_composition_temperature(composition="Al30Si55C15", min_temp=300, max_temp=1600)`
  - Visual plot showing precipitate behavior

**Analysis Method**:
- Identify carbide phases (containing 'C' or 'CARBIDE' in name)
- Compare phase fraction at 300K vs 500K
- Check if fraction_300K > fraction_500K (increase upon cooling)
- Look for "increasing" trend in analysis

---

### Q5: Heating Al30Si55C15 to 1500K is insufficient to dissolve all carbides

**Expected Answer**: TRUE (or FALSE depending on solvus temperature)

**Handler Functions Used**:
- `calculate_equilibrium_at_point(composition="Al30Si55C15", temperature=1500.0)`
  - Single point calculation at 1500K
  - Checks for carbide phase presence
  
- `calculate_phase_fractions_vs_temperature(composition="Al30Si55C15", min_temp=300, max_temp=1600, step=50)`
  - Phase evolution to see dissolution behavior
  
- `plot_composition_temperature(composition="Al30Si55C15", min_temp=300, max_temp=1600)`
  - Visual confirmation

**Analysis Method**:
- Parse equilibrium output at 1500K
- Look for carbide phases (AL4C3, SIC, etc.)
- If carbide fraction > 0, then not fully dissolved
- Check phase fraction trends approaching 1500K

---

## Battery/Electrochemistry Tests

### Q6: AlMg anode has ~0.5 V vs. Li/Li+

**Expected Answer**: TRUE

**Handler Functions Used**:
- `search_battery_electrodes(formula="AlMg", working_ion="Li")`
  - Searches Materials Project electrode database
  - Returns voltage profiles and average voltage
  
- `calculate_voltage_from_formation_energy(electrode_formula="AlMg", working_ion="Li")` (fallback)
  - Calculates voltage from convex hull if database unavailable
  - Uses InsertionElectrode or phase diagram scan

**Analysis Method**:
- Extract average_voltage from electrode data
- Check if voltage ∈ [0.3, 0.7] V (±0.2V tolerance around 0.5V)
- If multiple results, average them

**Data Source**: Materials Project DFT calculations + convex hull analysis

---

### Q7: Alloying Mg into Al anode increases voltage

**Expected Answer**: TRUE (or FALSE depending on actual thermodynamics)

**Handler Functions Used**:
- `compare_electrode_materials(formulas="Al,AlMg", working_ion="Li")`
  - Side-by-side comparison of Al and AlMg
  - Calculates voltage for both materials
  - Returns comparison summary

**Analysis Method**:
- Extract voltage_Al and voltage_AlMg
- Calculate Δ_voltage = voltage_AlMg - voltage_Al
- If Δ_voltage > 0: Mg increases voltage (TRUE)
- If Δ_voltage ≤ 0: Mg does not increase voltage (FALSE)

**Data Source**: Materials Project electrode database or convex hull calculations

---

### Q8: Cu80Li10Al10 can form a thermodynamically stable anode

**Expected Answer**: FALSE (composition contains Li, likely unstable)

**Handler Functions Used**:
- `check_composition_stability(composition="Cu80Li10Al10")`
  - Checks if composition is on convex hull
  - Returns energy_above_hull and decomposition products
  
- `analyze_anode_viability(composition="Cu80Li10Al10", working_ion="Li")`
  - Comprehensive stability + voltage analysis
  - Assessment of anode suitability

**Analysis Method**:
- Check if is_stable == True (E_hull ≈ 0)
- If E_hull > 0.03 eV/atom: unstable at room temperature
- Check if composition contains Li (problematic for anode)
- Review decomposition products if unstable

**Data Source**: Materials Project thermodynamic data (formation energies, phase diagram)

---

### Q9: For (CuAl)_{1-x}Li_x electrode, max capacity is x=0.4

**Expected Answer**: TRUE (or FALSE depending on lithiation limit)

**Handler Functions Used**:
- `analyze_lithiation_mechanism(host_composition="CuAl", working_ion="Li", max_x=1.0, room_temp=True)`
  - Calculates convex hull of G(x) vs x
  - Identifies voltage plateaus and phase equilibria
  - Determines maximum practical x value
  
- `search_battery_electrodes(formula="CuAl", working_ion="Li")` (baseline)
  - Gets pure CuAl electrode data

**Analysis Method**:
- Extract max_practical_x from lithiation analysis
- Check if max_x ∈ [0.3, 0.5] (±0.1 tolerance around 0.4)
- Look for voltage plateau endpoints
- Check when E_hull > 0.03 eV/atom (room temp limit)

**Data Source**: Materials Project phase diagram data, lithiation pathways

---

## Function Summary

### CALPHAD Handler Functions

| Function | Purpose | Returns |
|----------|---------|---------|
| `calculate_equilibrium_at_point()` | Single T,X point | Phase fractions, compositions |
| `plot_binary_phase_diagram()` | Full binary diagram | Phase boundaries, eutectics, plot |
| `plot_composition_temperature()` | Fixed X, variable T | Phase stability plot, fractions vs T |
| `calculate_phase_fractions_vs_temperature()` | Phase evolution data | Numerical phase fraction arrays |
| `analyze_phase_fraction_trend()` | Trend analysis for specific phase | Increasing/decreasing/stable |

### Battery Handler Functions

| Function | Purpose | Returns |
|----------|---------|---------|
| `search_battery_electrodes()` | Find electrode materials | Voltage profiles, capacity, energy |
| `calculate_voltage_from_formation_energy()` | Calculate voltage | Voltage from convex hull |
| `compare_electrode_materials()` | Side-by-side comparison | Relative voltages, capacities |
| `check_composition_stability()` | Thermodynamic stability | E_hull, decomposition |
| `analyze_anode_viability()` | Comprehensive anode analysis | Stability + voltage + assessment |
| `analyze_lithiation_mechanism()` | Lithiation pathway | Voltage plateaus, max Li content |

---

## Test Output Interpretation

### Success Criteria

Each test determines success based on:

1. **Data Retrieval**: Handler returns valid data (not error)
2. **Analysis Extraction**: Can parse relevant information from results
3. **Comparison to Expected**: Result matches expected answer

### Status Codes

- ✅ **PASSED**: Handler output confirms expected answer
- ❌ **FAILED**: Handler output contradicts expected answer
- ❓ **INCONCLUSIVE**: Handler succeeded but answer unclear from output
- **ERROR**: Handler call failed or threw exception

### Common Reasons for INCONCLUSIVE

- Calculation succeeded but output format differs from expected
- Data available but needs manual interpretation
- Edge case or boundary condition
- Database doesn't contain expected entry

### Common Reasons for ERROR

- Missing thermodynamic database (CALPHAD)
- API key not set (Battery)
- Network issues (Battery - Materials Project API)
- Composition not in database
- Calculation convergence failure

---

## Expected Behavior Notes

### CALPHAD Tests

- Tests rely on thermodynamic databases in `tdbs/` directory
- Different databases may give slightly different results
- Temperature ranges are automatically adjusted if not specified
- Plots are generated in `interactive_plots/` directory
- Analysis includes both visual and thermodynamic components

### Battery Tests

- Tests require Materials Project API access
- Some calculations use curated electrode database
- Other calculations use convex hull analysis (more general)
- Voltages are always reported vs. working ion (e.g., Li/Li+)
- Stability checks use 0K DFT energies
- Room temperature filter: E_hull < 0.03 eV/atom

---

## Validation Methodology

### How Tests Work

1. **Call Handler Function**: Execute the AI function with appropriate parameters
2. **Extract Results**: Parse returned string/dict for relevant data
3. **Apply Analysis Logic**: Use pattern matching, numerical comparison, etc.
4. **Compare to Expected**: Determine if result matches expected answer
5. **Record Evidence**: Store all handler calls and intermediate results

### Why This Approach?

- Tests the actual AI functions as they would be called by an LLM
- Validates both calculation correctness and output formatting
- Ensures analysis text is parseable and informative
- Checks that plots contain expected features
- Verifies error handling and edge cases

### What's NOT Tested

These tests do NOT directly test:
- Internal calculation accuracy (assumes pycalphad/pymatgen are correct)
- Plot rendering quality
- Performance/speed
- Concurrent access
- Edge cases beyond the specific questions

For those aspects, separate unit tests of individual functions would be needed.

---

## Extending the Test Suite

To add new questions:

1. **Identify the question type**:
   - Phase equilibrium → CALPHAD test
   - Battery voltage/stability → Battery test

2. **Select appropriate handler functions**:
   - Refer to function table above
   - May need multiple functions for comprehensive answer

3. **Define expected answer**:
   - TRUE/FALSE for yes/no questions
   - Numerical range for quantitative questions
   - Qualitative description for open-ended questions

4. **Implement analysis logic**:
   - Parse handler output
   - Extract relevant data
   - Compare to expected
   - Handle edge cases

5. **Add to test file**:
   - Follow existing test function pattern
   - Add to `run_all_tests()` list
   - Update documentation

6. **Test the test**:
   - Run on known systems
   - Verify correct pass/fail/inconclusive
   - Check error handling

---

## Question Difficulty Assessment

| Question | Difficulty | Why |
|----------|-----------|-----|
| Q1 | Easy | Single point calculation, clear criteria |
| Q2 | Medium | Requires eutectic detection algorithm |
| Q3 | Medium | Multiple calculations, phase counting |
| Q4 | Hard | Multicomponent system, trend analysis |
| Q5 | Hard | Multicomponent, high temperature, dissolution |
| Q6 | Easy | Database lookup, simple comparison |
| Q7 | Easy | Two materials, voltage comparison |
| Q8 | Medium | Stability check + Li content consideration |
| Q9 | Hard | Lithiation mechanism, capacity limit |

**Easy**: Single calculation or database lookup, clear pass/fail criteria

**Medium**: Multiple calculations or complex parsing, some ambiguity

**Hard**: Advanced analysis, multicomponent systems, or requires sophisticated algorithms

