# Handler Test Suites

Comprehensive test suites for CALPHAD phase diagram and battery/electrochemistry handlers.

## Overview

This test suite validates the handlers' ability to answer specific materials science questions, based on the following test cases:

### CALPHAD Phase Diagram Tests (`test_calphad_questions.py`)

1. **Q1**: Al20Zn80 at 870K is a solid at equilibrium
2. **Q2**: Al and Zn form an eutectic at ~Al15Zn75
3. **Q3**: Al50Zn50 forms a single solid phase <700K
4. **Q4**: The phase fraction of Aluminum Silicon Carbide precipitates in Al30Si55C15 increases with decreasing temperature from 500K to 300K
5. **Q5**: Heating a sample of Al30Si55C15 to 1500K is insufficient to dissolve all carbide precipitates

### Battery/Electrochemistry Tests (`test_battery_questions.py`)

6. **Q6**: The equilibrium open-circuit potential for a lithium battery with an AlMg anode is ~0.5 V vs. Li/Li+
7. **Q7**: Alloying Mg into an Al anode increases the voltage of a lithium battery
8. **Q8**: Cu80Li10Al10 can form a thermodynamically stable anode for a lithium battery
9. **Q9**: For a (CuAl)_{1-x}Li_{x} electrode, the maximum practical capacity is x=0.4

## Installation

### Prerequisites

```bash
# Python 3.8+
python3 --version

# Install required packages
pip install pycalphad matplotlib plotly numpy
pip install mp-api pymatgen
```

### Environment Setup

For battery tests, you need a Materials Project API key:

```bash
# Get your API key from https://materialsproject.org/api
export MP_API_KEY='your-api-key-here'
```

## Running Tests

### Run All Tests

```bash
# Run comprehensive test suite
python3 run_all_handler_tests.py
```

This will:
- Run all CALPHAD tests
- Run all battery/electrochemistry tests
- Generate JSON results files
- Create an HTML report

### Run Individual Test Suites

```bash
# Run only CALPHAD tests
python3 test_calphad_questions.py

# Run only battery tests
python3 test_battery_questions.py
```

## Output Files

### JSON Results

- `test_results_calphad.json` - CALPHAD test results
- `test_results_battery.json` - Battery test results
- `test_results_comprehensive.json` - Combined results

### HTML Report

- `test_results_report.html` - Interactive HTML report with:
  - Overall statistics
  - Per-suite breakdown
  - Individual test results
  - Handler call details
  - Visual status indicators

### Generated Plots

Phase diagrams and plots are saved to `interactive_plots/`:
- HTML interactive plots
- PNG static images

## Understanding Results

### Test Status

- **✅ PASSED**: Handler correctly answered the question
- **❌ FAILED**: Handler's answer doesn't match expected result
- **❓ INCONCLUSIVE**: Could not determine if answer is correct (needs manual review)
- **ERROR**: Test encountered an error during execution

### Result Structure

Each test result includes:

```json
{
  "test_id": "Q1",
  "question": "Al20Zn80 at 870K is a solid at equilibrium",
  "expected": "TRUE - Should show only solid phases",
  "handler_calls": [
    {
      "function": "calculate_equilibrium_at_point",
      "parameters": {"composition": "Al20Zn80", "temperature": 870.0},
      "result": "...",
      "analysis": "✅ Only solid phases present at 870K"
    }
  ],
  "analysis": "✅ Only solid phases present at 870K",
  "conclusion": "TRUE - Material is solid at equilibrium",
  "passed": true,
  "metadata": {}
}
```

## Test Architecture

### CALPHAD Tests

Tests use the following handler functions:

- `calculate_equilibrium_at_point()` - Point calculations
- `plot_binary_phase_diagram()` - Full phase diagrams
- `plot_composition_temperature()` - Composition-temperature plots
- `calculate_phase_fractions_vs_temperature()` - Phase evolution
- `analyze_phase_fraction_trend()` - Precipitation analysis

### Battery Tests

Tests use the following handler functions:

- `search_battery_electrodes()` - Find electrode materials
- `calculate_voltage_from_formation_energy()` - Voltage calculations
- `compare_electrode_materials()` - Side-by-side comparison
- `check_composition_stability()` - Thermodynamic stability
- `analyze_anode_viability()` - Anode suitability
- `analyze_lithiation_mechanism()` - Lithiation behavior

## Interpreting Analysis

### CALPHAD Tests

The tests analyze:
- Phase presence/absence at specific conditions
- Phase boundaries and critical points (eutectic, etc.)
- Phase fraction trends with temperature
- Precipitation/dissolution behavior

### Battery Tests

The tests analyze:
- Voltage values vs. Li/Li+
- Voltage comparisons between materials
- Thermodynamic stability (energy above hull)
- Lithiation limits and capacity

## Troubleshooting

### CALPHAD Tests Failing

1. **Check TDB files**: Ensure thermodynamic database files are in `tdbs/` directory
2. **Check elements**: Not all systems may be in the available databases
3. **Temperature range**: Some calculations may fail at extreme temperatures

### Battery Tests Failing

1. **API Key**: Ensure `MP_API_KEY` is set correctly
2. **Network**: Materials Project API requires internet access
3. **Data availability**: Not all compositions exist in Materials Project database
4. **Rate limits**: MP API has rate limits, wait between test runs if needed

### Inconclusive Results

Inconclusive results typically mean:
- Calculation succeeded but output format unexpected
- Data available but needs manual interpretation
- Edge cases or ambiguous conditions

Review the detailed handler call results in the JSON files for these cases.

## Customization

### Adding New Tests

1. Create a new test function following the pattern:

```python
async def test_qN_description(handler: HandlerType) -> TestResult:
    test = TestResult(
        test_id="QN",
        question="Your question here",
        expected="Expected outcome"
    )
    
    # Call handler functions
    result = await handler.some_function(...)
    
    # Analyze results
    test.add_handler_call("some_function", params, result, analysis)
    
    # Set conclusion
    test.analysis = "..."
    test.conclusion = "..."
    test.passed = True/False/None
    
    return test
```

2. Add to `run_all_tests()` function
3. Update README with new test description

### Adjusting Tolerances

Tests use tolerances for numerical comparisons:
- Voltage: ±0.2V for "approximately" comparisons
- Composition: ±5 at% for eutectic positions
- Capacity: ±0.1 for x values

Modify these in the test functions as needed.

## Contributing

When adding new tests:
1. Follow existing naming conventions
2. Include detailed docstrings
3. Add appropriate analysis and conclusions
4. Update this README
5. Test on multiple systems if possible

## License

Part of the MCP Materials Project handler test suite.

