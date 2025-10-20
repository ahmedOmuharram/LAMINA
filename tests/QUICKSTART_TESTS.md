# Quick Start Guide - Handler Test Suites

Get up and running with the handler test suites in 5 minutes.

## 🚀 Quick Setup

### 1. Check Prerequisites

```bash
# Check Python version (need 3.8+)
python3 --version

# Check if you're in the thesis directory
pwd  # Should show: /Users/ahmedmuharram/thesis
```

### 2. Install Dependencies

```bash
# CALPHAD dependencies
pip install pycalphad matplotlib plotly numpy kaleido

# Battery/electrochemistry dependencies  
pip install mp-api pymatgen
```

### 3. Set API Key (for battery tests only)

```bash
# Get your free API key from: https://materialsproject.org/api
export MP_API_KEY='your-api-key-here'

# Verify it's set
echo $MP_API_KEY
```

## ▶️ Run Tests

### Option 1: Run Everything (Recommended)

```bash
# Run all tests and generate HTML report
./run_all_handler_tests.py

# Or if not executable:
python3 run_all_handler_tests.py
```

**Output**:
- Console output with test results
- `test_results_comprehensive.json` - Full results
- `test_results_report.html` - Visual report (open in browser)
- Phase diagrams in `interactive_plots/`

### Option 2: Run Individual Suites

```bash
# CALPHAD tests only (no API key needed)
./test_calphad_questions.py

# Battery tests only (requires API key)
./test_battery_questions.py
```

## 📊 View Results

### Console Output

Tests print results in real-time:

```
🧪 Q1: Al20Zn80 at 870K is a solid at equilibrium
✅ Only solid phases present at 870K
TRUE - Material is solid at equilibrium

🧪 Q2: Al and Zn form an eutectic at ~Al15Zn75
✅ Eutectic detected with composition(s): [75.2]. Near 75 at% Zn
TRUE - Eutectic found near expected composition
```

### HTML Report

Open the HTML report for a visual overview:

```bash
# macOS
open test_results_report.html

# Linux
xdg-open test_results_report.html

# Windows
start test_results_report.html
```

### JSON Results

For programmatic analysis:

```bash
# View comprehensive results
cat test_results_comprehensive.json | python3 -m json.tool

# View just CALPHAD results
cat test_results_calphad.json | python3 -m json.tool

# View just battery results
cat test_results_battery.json | python3 -m json.tool
```

## 🎯 What Gets Tested

### CALPHAD Tests (5 questions)

1. ✓ Al20Zn80 solid state at 870K
2. ✓ Al-Zn eutectic location
3. ✓ Al50Zn50 single phase behavior
4. ✓ Al-Si-C precipitate trends
5. ✓ Carbide dissolution temperature

**Time**: ~5-10 minutes (generates multiple plots)

### Battery Tests (4 questions)

6. ✓ AlMg anode voltage
7. ✓ Mg alloying effect on voltage
8. ✓ Cu-Li-Al stability
9. ✓ Cu-Al-Li capacity limits

**Time**: ~2-5 minutes (depends on API response time)

## 🔍 Understanding Results

### Status Icons

- ✅ **PASSED** - Handler correctly answered the question
- ❌ **FAILED** - Handler's answer doesn't match expected
- ❓ **INCONCLUSIVE** - Needs manual review
- ⚠️ **ERROR** - Test encountered an error

### Example Output

```
📊 TEST SUMMARY
================================================================================

✅ Q1: Al20Zn80 at 870K is a solid at equilibrium
   Expected: TRUE - Should show only solid phases, no liquid at 870K
   Result: TRUE - Material is solid at equilibrium
   Analysis: ✅ Only solid phases present at 870K

❌ Q3: Al50Zn50 forms a single solid phase <700K
   Expected: TRUE - Should show only one solid phase below 700K
   Result: FALSE - Not single phase below 700K
   Analysis: ❌ Multiple phases detected: {300: 2, 500: 2, 650: 2}
```

## 🛠️ Troubleshooting

### "No module named 'pycalphad'"

```bash
pip install pycalphad
```

### "No module named 'mp_api'"

```bash
pip install mp-api
```

### "MPRester client not initialized"

You forgot to set the API key:

```bash
export MP_API_KEY='your-api-key-here'
```

### "TDB directory not found"

Make sure you're running from the thesis directory:

```bash
cd /Users/ahmedmuharram/thesis
./run_all_handler_tests.py
```

### "Could not export static PNG"

Install kaleido for Plotly image export:

```bash
pip install kaleido
```

### Battery tests fail with network errors

- Check internet connection
- Verify API key is valid
- Materials Project may be rate-limiting (wait a few minutes)

## 📈 What Success Looks Like

### Good Run (Example)

```
🧪 COMPREHENSIVE HANDLER TEST SUITE
================================================================================

🔬 RUNNING CALPHAD TESTS...
  ✅ Q1: PASSED
  ✅ Q2: PASSED  
  ❌ Q3: FAILED
  ✅ Q4: PASSED
  ✅ Q5: PASSED

🔋 RUNNING BATTERY/ELECTROCHEMISTRY TESTS...
  ✅ Q6: PASSED
  ✅ Q7: PASSED
  ❌ Q8: FAILED
  ❓ Q9: INCONCLUSIVE

📈 OVERALL STATISTICS
Total Tests: 9
✅ Passed: 6 (66.7%)
❌ Failed: 2 (22.2%)
❓ Inconclusive: 1 (11.1%)
```

### Typical Issues

**Failed tests** don't always mean bugs:
- Some questions are intentionally FALSE statements
- Tests verify the handler can detect incorrect statements
- Check the "Expected" field in results

**Inconclusive tests** may indicate:
- Output format changed
- Edge case behavior
- Manual review needed

## 🎓 Advanced Usage

### Run Specific Test

```python
# Edit test file to run only one test
python3 -c "
import asyncio
from test_calphad_questions import CalPhadHandler, test_q1_al20zn80_solid_at_870k

handler = CalPhadHandler()
asyncio.run(test_q1_al20zn80_solid_at_870k(handler))
"
```

### Generate Only HTML Report

```python
from run_all_handler_tests import generate_html_report
import json

with open('test_results_comprehensive.json') as f:
    results = json.load(f)

html = generate_html_report(results)

with open('custom_report.html', 'w') as f:
    f.write(html)
```

### Custom Test Timeout

```python
# In test file, add timeout to async calls
import asyncio

result = await asyncio.wait_for(
    handler.plot_binary_phase_diagram(system="Al-Zn"),
    timeout=300.0  # 5 minutes
)
```

## 📚 Next Steps

1. ✅ Run the tests
2. 📄 Review `test_results_report.html`
3. 🔍 Check failed/inconclusive tests
4. 📊 Examine generated plots in `interactive_plots/`
5. 📖 Read `TEST_SUITE_README.md` for detailed docs
6. 🗺️ See `TEST_QUESTIONS_MAPPING.md` for function mapping

## 🤝 Getting Help

### Files to Check

- `TEST_SUITE_README.md` - Full documentation
- `TEST_QUESTIONS_MAPPING.md` - Question → function mapping
- `test_results_*.json` - Detailed results
- Handler READMEs:
  - `mcp_materials_project/handlers/calphad/phase_diagrams/README.md`
  - `mcp_materials_project/handlers/electrochemistry/README.md`

### Common Questions

**Q: How long should tests take?**
A: CALPHAD ~5-10 min, Battery ~2-5 min, Total ~10-15 min

**Q: Can I run tests in parallel?**
A: Not recommended - may cause database/API issues

**Q: Are the questions designed to pass or fail?**
A: Mixed - some are TRUE, some are FALSE. Tests verify correct analysis.

**Q: What if all tests are inconclusive?**
A: Check prerequisites, especially TDB files and API key

**Q: Can I add my own questions?**
A: Yes! See TEST_SUITE_README.md "Customization" section

## ✅ Checklist

Before reporting issues, verify:

- [ ] Python 3.8+ installed
- [ ] All dependencies installed (`pip list | grep pycalphad`)
- [ ] Running from thesis directory
- [ ] TDB files present in `tdbs/` directory
- [ ] API key set (for battery tests): `echo $MP_API_KEY`
- [ ] Internet connection working
- [ ] Disk space available for plots

## 🎉 Success!

If you see output like:

```
✅ Test suite completed at: 2025-10-10 12:34:56
💾 Results saved to: test_results_comprehensive.json
📄 HTML report saved to: test_results_report.html
```

You're done! Open the HTML report and explore the results.

---

**Happy Testing!** 🧪🔬🔋

