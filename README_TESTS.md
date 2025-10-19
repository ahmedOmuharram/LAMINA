# Handler Test Suite

Comprehensive automated testing framework for CALPHAD phase diagram and battery/electrochemistry handlers.

## 🎯 Purpose

Validate handler functionality by testing their ability to answer specific materials science questions. Tests ensure:

- ✅ Handlers execute without errors
- ✅ Output is consistent and parseable
- ✅ Analysis matches expected physical behavior
- ✅ Plots are generated correctly

## 📦 What's Included

### Test Scripts (4 files, 108KB total)

- **`verify_test_setup.py`** - Pre-flight checks for dependencies and setup
- **`test_calphad_questions.py`** - 5 CALPHAD phase diagram tests
- **`test_battery_questions.py`** - 4 battery/electrochemistry tests
- **`run_all_handler_tests.py`** - Master runner with HTML report generation

### Documentation (5 files, 45KB total)

- **`TEST_INDEX.md`** - Navigation guide (start here for documentation)
- **`QUICKSTART_TESTS.md`** - 5-minute quick start guide
- **`TEST_SUITE_SUMMARY.md`** - Executive overview
- **`TEST_SUITE_README.md`** - Complete documentation
- **`TEST_QUESTIONS_MAPPING.md`** - Technical reference

## ⚡ Quick Start

### 1. Verify Setup

```bash
./verify_test_setup.py
```

### 2. Run Tests

```bash
# All tests (recommended)
./run_all_handler_tests.py

# Or individually:
./test_calphad_questions.py  # No API key needed
./test_battery_questions.py  # Requires MP_API_KEY
```

### 3. View Results

```bash
open test_results_report.html
```

## 🧪 Test Questions

### CALPHAD Tests (Phase Diagrams)

1. ✓ Al20Zn80 at 870K is solid at equilibrium
2. ✓ Al and Zn form eutectic at ~Al15Zn75
3. ? Al50Zn50 forms single solid phase <700K
4. ✓ Al30Si55C15 precipitates increase with cooling
5. ? Al30Si55C15 carbides remain at 1500K

### Battery Tests (Electrochemistry)

6. ✓ AlMg anode ~0.5V vs Li/Li+
7. ? Mg alloying increases Al anode voltage
8. ✗ Cu80Li10Al10 forms stable anode
9. ? (CuAl)Li_x max capacity at x=0.4

Legend: ✓ = Expected TRUE, ✗ = Expected FALSE, ? = To be determined

## 📊 Output

### Generated Files

- `test_results_comprehensive.json` - All results (machine-readable)
- `test_results_report.html` - Visual report (human-readable)
- `test_results_calphad.json` - CALPHAD results only
- `test_results_battery.json` - Battery results only
- `interactive_plots/*.html` - Interactive Plotly plots
- `interactive_plots/*.png` - Static plot images

### Example Output

```
📊 TEST SUMMARY
================================================================================

✅ Q1: Al20Zn80 at 870K is a solid at equilibrium
   Result: TRUE - Material is solid at equilibrium
   Analysis: ✅ Only solid phases present at 870K

❌ Q8: Cu80Li10Al10 can form a thermodynamically stable anode
   Result: FALSE - Not thermodynamically stable
   Analysis: ❌ Not stable (E_hull: 0.15 eV/atom)

📈 OVERALL STATISTICS
Total Tests: 9
✅ Passed: 6 (66.7%)
❌ Failed: 2 (22.2%)
❓ Inconclusive: 1 (11.1%)
```

## 🔧 Requirements

### Prerequisites

```bash
# Python 3.8+
python3 --version

# CALPHAD dependencies
pip install pycalphad matplotlib plotly numpy kaleido

# Battery dependencies
pip install mp-api pymatgen

# API key (battery tests only)
export MP_API_KEY='your-key-from-materialsproject.org'
```

### System Requirements

- Python 3.8 or higher
- 100MB+ disk space (for plots)
- Internet connection (for battery tests)
- TDB files in `tdbs/` directory

## 📚 Documentation

**Quick Navigation**: See [`TEST_INDEX.md`](TEST_INDEX.md)

### For Users

1. **[QUICKSTART_TESTS.md](QUICKSTART_TESTS.md)** - Setup and run tests in 5 minutes
2. **[TEST_SUITE_SUMMARY.md](TEST_SUITE_SUMMARY.md)** - Overview and statistics
3. **[TEST_SUITE_README.md](TEST_SUITE_README.md)** - Complete documentation

### For Developers

4. **[TEST_QUESTIONS_MAPPING.md](TEST_QUESTIONS_MAPPING.md)** - Technical details and function mapping

## 🎓 Test Architecture

### CALPHAD Handler Functions

- `calculate_equilibrium_at_point()` - Single T,X calculations
- `plot_binary_phase_diagram()` - Full phase diagrams with eutectic detection
- `plot_composition_temperature()` - Phase stability vs temperature
- `calculate_phase_fractions_vs_temperature()` - Phase evolution
- `analyze_phase_fraction_trend()` - Precipitation analysis

### Battery Handler Functions

- `search_battery_electrodes()` - Electrode database search
- `calculate_voltage_from_formation_energy()` - Voltage calculations
- `compare_electrode_materials()` - Material comparison
- `check_composition_stability()` - Stability on convex hull
- `analyze_anode_viability()` - Comprehensive anode analysis
- `analyze_lithiation_mechanism()` - Lithiation pathways

## 📈 Test Methodology

Each test:

1. **Calls handler function(s)** with specific parameters
2. **Parses output** to extract relevant data
3. **Applies analysis logic** to determine if answer is correct
4. **Records all evidence** including intermediate results
5. **Generates conclusion** (PASSED/FAILED/INCONCLUSIVE)

### Status Codes

- ✅ **PASSED** - Handler correctly answered the question
- ❌ **FAILED** - Handler's answer doesn't match expected (may be intentional)
- ❓ **INCONCLUSIVE** - Calculation succeeded but needs manual review
- **ERROR** - Test encountered an error during execution

## 🚀 Usage Examples

### Basic Usage

```bash
# Verify everything is set up
./verify_test_setup.py

# Run all tests
./run_all_handler_tests.py

# View HTML report
open test_results_report.html

# View JSON results
cat test_results_comprehensive.json | python3 -m json.tool
```

### Advanced Usage

```bash
# Run only CALPHAD tests (faster, no API needed)
./test_calphad_questions.py

# Run only battery tests
./test_battery_questions.py

# Check specific question output
cat test_results_calphad.json | jq '.tests[] | select(.test_id=="Q1")'

# List all generated plots
ls -lh interactive_plots/
```

## 🔍 Understanding Results

### Passed Tests

Indicates handler correctly answered the question:
- Calculation executed successfully
- Output contained expected information
- Analysis matched predicted behavior

### Failed Tests

May indicate:
- Handler incorrectly analyzed the system (actual bug)
- Question was intentionally FALSE (testing detection)
- Expected answer was wrong (update test)

### Inconclusive Tests

Requires manual review:
- Calculation succeeded but output format unexpected
- Data available but interpretation ambiguous
- Edge case or boundary condition

Check JSON files for detailed handler output.

## 🛠️ Troubleshooting

### Tests Won't Run

```bash
# Run diagnostic
./verify_test_setup.py

# Common fixes:
pip install pycalphad matplotlib plotly mp-api pymatgen
export MP_API_KEY='your-key'
cd /Users/ahmedmuharram/thesis
```

### Tests Fail or Error

1. Check Python version (need 3.8+)
2. Verify all dependencies installed
3. Confirm TDB files exist in `tdbs/`
4. For battery tests: verify `MP_API_KEY` is set
5. Check internet connection
6. Review error messages in console output

### INCONCLUSIVE Results

1. Open JSON result file for the test
2. Review `handler_calls` section for detailed output
3. Check `metadata` for additional context
4. Manually verify the analysis
5. May need to adjust test parsing logic

See [`QUICKSTART_TESTS.md`](QUICKSTART_TESTS.md) troubleshooting section for more details.

## 🎯 Expected Behavior

### Normal Run

- **Runtime**: 10-15 minutes total
  - CALPHAD: 5-10 minutes
  - Battery: 2-5 minutes
- **Success rate**: 60-100% (depends on databases)
- **Output files**: 4 JSON files + 1 HTML file + multiple plots

### Common Results

- Most tests should PASS or FAIL (clear conclusion)
- Few INCONCLUSIVE (needs review)
- Minimal ERRORs (indicates setup issues)

## 🔄 Maintenance

### Adding New Tests

1. Identify question type (CALPHAD or Battery)
2. Determine which handler functions to call
3. Define expected answer
4. Implement test function in appropriate file
5. Add to `run_all_tests()` function
6. Update documentation

See [`TEST_QUESTIONS_MAPPING.md`](TEST_QUESTIONS_MAPPING.md) for details.

### Updating Tests

- Adjust tolerance values if needed
- Improve parsing logic for new output formats
- Add intermediate validation steps
- Enhance error messages

## 📊 Statistics

**Test Suite Size**:
- Total files: 9 (4 scripts + 5 docs)
- Code: ~2,135 lines
- Test cases: 9 questions
- Handler functions tested: 11

**Coverage**:
- CALPHAD: 5/5 main functions tested
- Battery: 6/6 main functions tested

**Runtime**:
- Setup verification: <1 minute
- CALPHAD tests: 5-10 minutes
- Battery tests: 2-5 minutes
- Total: 10-15 minutes

## 🎉 Benefits

✅ **Automated Validation** - No manual checking needed
✅ **Comprehensive** - Tests all major handler functions
✅ **Well-Documented** - 5 documentation files
✅ **Visual Reports** - HTML output with charts
✅ **Evidence Trail** - All handler calls recorded
✅ **Reproducible** - Same inputs → same results
✅ **Extensible** - Easy to add new tests

## 📝 License

Part of the thesis project - handler validation suite.

## 🤝 Contributing

To add new test questions:

1. Follow existing test function patterns
2. Update documentation files
3. Test on multiple systems
4. Submit with example results

## 📞 Support

### Quick Links

- **Start Here**: [`QUICKSTART_TESTS.md`](QUICKSTART_TESTS.md)
- **Navigation**: [`TEST_INDEX.md`](TEST_INDEX.md)
- **Overview**: [`TEST_SUITE_SUMMARY.md`](TEST_SUITE_SUMMARY.md)
- **Full Docs**: [`TEST_SUITE_README.md`](TEST_SUITE_README.md)
- **Technical**: [`TEST_QUESTIONS_MAPPING.md`](TEST_QUESTIONS_MAPPING.md)

### Getting Help

1. Run `./verify_test_setup.py`
2. Check documentation files above
3. Review JSON output for details
4. Examine handler READMEs in `mcp_materials_project/handlers/`

---

**Created**: October 2025  
**Version**: 1.0  
**Purpose**: Handler validation and regression testing

**Ready to start?** → [`QUICKSTART_TESTS.md`](QUICKSTART_TESTS.md)

