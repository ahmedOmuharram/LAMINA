# Test Suite Documentation Index

Quick navigation guide for all test suite documentation.

## 🚀 Getting Started

**→ Start here**: [`QUICKSTART_TESTS.md`](QUICKSTART_TESTS.md)
- 5-minute setup guide
- Quick commands
- Basic troubleshooting

**→ Verify your setup**: Run `./verify_test_setup.py`
- Checks Python version
- Verifies dependencies
- Tests API access

## 📚 Main Documentation

### For Users

1. **[QUICKSTART_TESTS.md](QUICKSTART_TESTS.md)** - Fast setup and running
   - Installation steps
   - Run commands
   - View results
   - Common issues

2. **[TEST_SUITE_SUMMARY.md](TEST_SUITE_SUMMARY.md)** - Executive overview
   - What tests exist
   - What's tested
   - Expected results
   - Statistics

3. **[TEST_SUITE_README.md](TEST_SUITE_README.md)** - Complete documentation
   - Detailed installation
   - All features
   - Advanced usage
   - Full troubleshooting

### For Developers

4. **[TEST_QUESTIONS_MAPPING.md](TEST_QUESTIONS_MAPPING.md)** - Technical details
   - Question → Function mapping
   - Analysis methodology
   - Data sources
   - Adding new tests

## 🧪 Test Files

### Executable Scripts

| File | Purpose | Runtime | Requires API |
|------|---------|---------|--------------|
| `verify_test_setup.py` | Pre-flight checks | <1 min | No |
| `test_calphad_questions.py` | CALPHAD tests (5 questions) | 5-10 min | No |
| `test_battery_questions.py` | Battery tests (4 questions) | 2-5 min | Yes |
| `run_all_handler_tests.py` | Master test runner | 10-15 min | Partial* |

*Battery tests require API key, CALPHAD tests don't

### Test Questions

#### CALPHAD (Phase Diagrams)

1. **Q1**: Al20Zn80 at 870K is solid ✓
2. **Q2**: Al-Zn eutectic at ~Al15Zn75 ✓
3. **Q3**: Al50Zn50 single phase <700K ?
4. **Q4**: Al30Si55C15 precipitates increase on cooling ✓
5. **Q5**: Al30Si55C15 carbides remain at 1500K ?

#### Battery (Electrochemistry)

6. **Q6**: AlMg anode ~0.5V vs Li/Li+ ✓
7. **Q7**: Mg alloying increases voltage ?
8. **Q8**: Cu80Li10Al10 stable anode ✗
9. **Q9**: (CuAl)Li_x max capacity x=0.4 ?

Legend: ✓ = Expected TRUE, ✗ = Expected FALSE, ? = To be determined

## 📊 Output Files

### Results

Generated after running tests:

- `test_results_calphad.json` - CALPHAD test results
- `test_results_battery.json` - Battery test results  
- `test_results_comprehensive.json` - Combined results
- `test_results_report.html` - **Visual report (open this!)**

### Plots

Generated in `interactive_plots/` directory:

- Phase diagrams (Al-Zn, etc.)
- Composition-temperature plots
- Both PNG (static) and HTML (interactive) formats
- Timestamped filenames

## 🎯 Quick Commands

### First Time Setup

```bash
# 1. Verify setup
./verify_test_setup.py

# 2. Set API key (for battery tests)
export MP_API_KEY='your-api-key'

# 3. Run all tests
./run_all_handler_tests.py

# 4. View results
open test_results_report.html
```

### Run Specific Tests

```bash
# Only CALPHAD tests (no API needed)
./test_calphad_questions.py

# Only battery tests (API required)
./test_battery_questions.py

# Re-run all tests
./run_all_handler_tests.py
```

### View Results

```bash
# HTML report (recommended)
open test_results_report.html

# JSON results (programmatic)
cat test_results_comprehensive.json | python3 -m json.tool

# Individual suite results
cat test_results_calphad.json | python3 -m json.tool
cat test_results_battery.json | python3 -m json.tool
```

## 🗺️ Documentation Map

```
TEST_INDEX.md (you are here)
├── QUICKSTART_TESTS.md ← Start here for quick setup
│   ├── Installation
│   ├── Running tests
│   ├── Viewing results
│   └── Troubleshooting
│
├── TEST_SUITE_SUMMARY.md ← Executive overview
│   ├── Overview
│   ├── Test questions
│   ├── Expected results
│   └── Statistics
│
├── TEST_SUITE_README.md ← Full documentation
│   ├── Detailed installation
│   ├── Complete usage guide
│   ├── Result interpretation
│   ├── Advanced features
│   └── Contributing
│
└── TEST_QUESTIONS_MAPPING.md ← Technical reference
    ├── Question → Function mapping
    ├── Analysis methods
    ├── Data sources
    └── Extending tests
```

## 📋 Checklists

### Before Running Tests

- [ ] Python 3.8+ installed
- [ ] All dependencies installed
- [ ] TDB files in `tdbs/` directory
- [ ] Running from thesis directory
- [ ] For battery tests: `MP_API_KEY` set
- [ ] Disk space available (~100MB for plots)

### After Running Tests

- [ ] Check console output for errors
- [ ] Open HTML report
- [ ] Review passed/failed/inconclusive tests
- [ ] Examine generated plots
- [ ] Check JSON files for details

### Troubleshooting Checklist

If tests fail:

1. [ ] Run `./verify_test_setup.py`
2. [ ] Check Python version
3. [ ] Verify all packages installed
4. [ ] Confirm TDB files present
5. [ ] Test API key (for battery tests)
6. [ ] Check internet connection
7. [ ] Review error messages
8. [ ] Check disk space

## 🎓 Learning Path

### Beginner

1. Read `QUICKSTART_TESTS.md`
2. Run `./verify_test_setup.py`
3. Run `./test_calphad_questions.py` (simpler, no API)
4. Open HTML report
5. Examine a few plots in `interactive_plots/`

### Intermediate

1. Set up API key for battery tests
2. Run full test suite with `./run_all_handler_tests.py`
3. Review JSON results files
4. Read `TEST_SUITE_SUMMARY.md`
5. Understand test status codes

### Advanced

1. Read `TEST_QUESTIONS_MAPPING.md`
2. Understand handler function mappings
3. Read `TEST_SUITE_README.md` completely
4. Try modifying a test
5. Add a custom test question

## 🔗 Related Documentation

### Handler Documentation

- **CALPHAD Handler**: `mcp_materials_project/handlers/calphad/phase_diagrams/README.md`
- **Battery Handler**: `mcp_materials_project/handlers/electrochemistry/README.md`

### Example Scripts

- **Basic test example**: `test_plot_composition_temperature.py`

### Source Code

- **CALPHAD Handler**: `mcp_materials_project/handlers/calphad/phase_diagrams/`
- **Battery Handler**: `mcp_materials_project/handlers/electrochemistry/`

## 📞 Getting Help

### Documentation to Check

1. **Quick issues**: `QUICKSTART_TESTS.md` → Troubleshooting section
2. **Setup problems**: Run `./verify_test_setup.py`
3. **Test failures**: `TEST_SUITE_README.md` → Troubleshooting section
4. **Technical details**: `TEST_QUESTIONS_MAPPING.md`
5. **Handler issues**: Handler-specific READMEs

### Common Questions

**Q: Where do I start?**
A: Read `QUICKSTART_TESTS.md` and run `./verify_test_setup.py`

**Q: Tests are failing, what do I do?**
A: Run `./verify_test_setup.py` first, then check troubleshooting sections

**Q: How do I add a new test?**
A: See `TEST_QUESTIONS_MAPPING.md` → "Extending the Test Suite"

**Q: What do the status codes mean?**
A: ✅ = Passed, ❌ = Failed, ❓ = Inconclusive, ERROR = Test crashed

**Q: Where are the plots?**
A: In `interactive_plots/` directory

**Q: Can I run tests in parallel?**
A: Not recommended - may cause database/API conflicts

**Q: How long should tests take?**
A: CALPHAD ~5-10min, Battery ~2-5min, Total ~10-15min

## 🎯 Quick Reference

### File Purposes

| File | What It Does |
|------|--------------|
| `verify_test_setup.py` | Checks if you're ready to run tests |
| `test_calphad_questions.py` | Tests phase diagram calculations |
| `test_battery_questions.py` | Tests battery/electrochemistry |
| `run_all_handler_tests.py` | Runs everything, generates reports |
| `QUICKSTART_TESTS.md` | Quick setup guide |
| `TEST_SUITE_SUMMARY.md` | Overview and statistics |
| `TEST_SUITE_README.md` | Complete documentation |
| `TEST_QUESTIONS_MAPPING.md` | Technical reference |
| `TEST_INDEX.md` | This navigation guide |

### Result Files

| File | What It Contains |
|------|------------------|
| `test_results_calphad.json` | CALPHAD results (machine-readable) |
| `test_results_battery.json` | Battery results (machine-readable) |
| `test_results_comprehensive.json` | All results (machine-readable) |
| `test_results_report.html` | Visual report (human-readable) |
| `interactive_plots/*.html` | Interactive plots (open in browser) |
| `interactive_plots/*.png` | Static plots (for reports) |

## ✨ Tips

💡 **Tip 1**: Always run `./verify_test_setup.py` first
💡 **Tip 2**: Start with CALPHAD tests (no API needed)
💡 **Tip 3**: Open HTML report for best visualization
💡 **Tip 4**: Check JSON for detailed analysis
💡 **Tip 5**: INCONCLUSIVE ≠ FAILED (may need review)
💡 **Tip 6**: Generated plots are timestamped (safe to re-run)

## 🎉 Success Indicators

You know tests are working when you see:

✅ `./verify_test_setup.py` → All checks pass
✅ Tests run without errors
✅ HTML report opens successfully
✅ Plots appear in `interactive_plots/`
✅ JSON files have `"success": true`
✅ Some tests show ✅ PASSED

## 📅 Last Updated

October 2025

---

**Happy Testing!** 🧪🔬🔋

**Need help?** Start with [`QUICKSTART_TESTS.md`](QUICKSTART_TESTS.md)

