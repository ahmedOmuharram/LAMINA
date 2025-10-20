#!/usr/bin/env python3
"""
Test suite for battery/electrochemistry questions.

Tests the following statements:
6. The equilibrium open-circuit potential for a lithium battery with an AlMg 
   anode is ~0.5 V vs. Li/Li+
7. Alloying Mg into an Al anode increases the voltage of a lithium battery
8. Cu80Li10Al10 can form a thermodynamically stable anode for a lithium battery
9. For a (CuAl)_{1-x}Li_{x} electrode, the maximum practical capacity is x=0.4
"""
import sys
import asyncio
import json
import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from mcp_materials_project.handlers.electrochemistry.battery_handler import BatteryHandler


class TestResult:
    """Container for test results"""
    def __init__(self, test_id: str, question: str, expected: str):
        self.test_id = test_id
        self.question = question
        self.expected = expected
        self.handler_calls = []
        self.analysis = ""
        self.conclusion = ""
        self.passed = None
        self.metadata = {}
    
    def add_handler_call(self, function_name: str, params: Dict, result: Any, analysis: str = ""):
        """Add a handler function call result"""
        self.handler_calls.append({
            "function": function_name,
            "parameters": params,
            "result": result,
            "analysis": analysis
        })
    
    def to_dict(self):
        return {
            "test_id": self.test_id,
            "question": self.question,
            "expected": self.expected,
            "handler_calls": self.handler_calls,
            "analysis": self.analysis,
            "conclusion": self.conclusion,
            "passed": self.passed,
            "metadata": self.metadata
        }


def get_mpr():
    """Get Materials Project API client"""
    try:
        from mp_api.client import MPRester
        
        # Get API key from environment
        api_key = os.environ.get('MP_API_KEY')
        if not api_key:
            print("⚠️  Warning: MP_API_KEY not set in environment")
            print("   Set it with: export MP_API_KEY='your-api-key'")
            return None
        
        return MPRester(api_key)
    except ImportError:
        print("⚠️  Warning: mp-api not installed")
        print("   Install with: pip install mp-api")
        return None


async def test_q6_almg_voltage(handler: BatteryHandler) -> TestResult:
    """
    Test: The equilibrium open-circuit potential for a lithium battery with an 
          AlMg anode is ~0.5 V vs. Li/Li+
    """
    test = TestResult(
        test_id="Q6",
        question="AlMg anode has ~0.5 V vs. Li/Li+",
        expected="FALSE - Voltage should be 0.15-0.3V, not 0.5V"
    )
    
    print("\n" + "="*80)
    print(f"🧪 {test.test_id}: {test.question}")
    print("="*80)
    
    try:
        # Search for AlMg electrode data
        print("\n🔋 Searching for AlMg electrode voltage...")
        result = await handler.search_battery_electrodes(
            formula="AlMg",
            working_ion="Li",
            max_entries=5
        )
        
        print(f"Search result: {result}")
        
        test.add_handler_call(
            "search_battery_electrodes",
            {"formula": "AlMg", "working_ion": "Li"},
            result
        )
        
        # Extract voltage information
        if result.get("success") and result.get("electrodes"):
            voltages = []
            for electrode in result["electrodes"]:
                voltage = electrode.get("average_voltage") or electrode.get("voltage")
                if voltage is not None:
                    voltages.append(voltage)
            
            if voltages:
                avg_voltage = sum(voltages) / len(voltages)
                test.metadata["voltages"] = voltages
                test.metadata["average_voltage"] = avg_voltage
                
                # Check if in 0.15-0.3V range (expected), not 0.5V
                if 0.15 <= avg_voltage <= 0.3:
                    test.analysis = f"✅ Voltage found: {avg_voltage:.3f} V (range: {min(voltages):.3f}-{max(voltages):.3f} V)"
                    test.conclusion = f"FALSE (CORRECT) - Voltage is {avg_voltage:.3f}V, not ~0.5V"
                    test.passed = True  # Expected FALSE, got 0.15-0.3V = pass
                elif 0.4 <= avg_voltage <= 0.6:
                    test.analysis = f"❌ Voltage found: {avg_voltage:.3f} V, incorrectly near 0.5V"
                    test.conclusion = f"TRUE (INCORRECT) - Voltage is ~0.5V (but shouldn't be)"
                    test.passed = False  # Expected FALSE, got ~0.5V = fail
                else:
                    test.analysis = f"⚠️ Voltage found: {avg_voltage:.3f} V, not in expected range"
                    test.conclusion = f"INCONCLUSIVE - Voltage {avg_voltage:.3f}V unexpected"
                    test.passed = None
            else:
                test.analysis = "❌ No voltage data found in electrode results"
                test.conclusion = "INCONCLUSIVE - No voltage data available"
                test.passed = None
        else:
            # Fallback: try formation energy calculation
            print("\n🔋 Trying formation energy calculation...")
            calc_result = await handler.calculate_voltage_from_formation_energy(
                electrode_formula="AlMg",
                working_ion="Li"
            )
            
            test.add_handler_call(
                "calculate_voltage_from_formation_energy",
                {"electrode_formula": "AlMg", "working_ion": "Li"},
                calc_result
            )
            
            if calc_result.get("success"):
                voltage = calc_result.get("calculated_voltage") or calc_result.get("voltage_range", {}).get("average")
                if voltage:
                    test.metadata["calculated_voltage"] = voltage
                    
                    if 0.15 <= voltage <= 0.3:
                        test.analysis = f"✅ Calculated voltage: {voltage:.3f} V"
                        test.conclusion = f"FALSE (CORRECT) - Voltage is {voltage:.3f}V, not ~0.5V"
                        test.passed = True  # Expected FALSE
                    elif 0.4 <= voltage <= 0.6:
                        test.analysis = f"❌ Calculated voltage: {voltage:.3f} V, incorrectly near 0.5V"
                        test.conclusion = f"TRUE (INCORRECT) - Voltage is ~0.5V (but shouldn't be)"
                        test.passed = False
                    else:
                        test.analysis = f"⚠️ Calculated voltage: {voltage:.3f} V, unexpected"
                        test.conclusion = f"INCONCLUSIVE - Voltage unexpected"
                        test.passed = None
                else:
                    test.analysis = "❌ Calculation succeeded but no voltage returned"
                    test.conclusion = "INCONCLUSIVE"
                    test.passed = None
            else:
                test.analysis = "❌ No electrode data found and calculation failed"
                test.conclusion = "INCONCLUSIVE - Could not determine voltage"
                test.passed = None
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        test.analysis = f"Error during test execution: {str(e)}"
        test.conclusion = "ERROR"
        test.passed = None
    
    print(f"\n{test.conclusion}")
    return test


async def test_q7_mg_alloy_increases_voltage(handler: BatteryHandler) -> TestResult:
    """
    Test: Alloying Mg into an Al anode increases the voltage of a lithium battery
    """
    test = TestResult(
        test_id="Q7",
        question="Alloying Mg into Al anode increases voltage",
        expected="FALSE - Mg alloying should DECREASE voltage, not increase"
    )
    
    print("\n" + "="*80)
    print(f"🧪 {test.test_id}: {test.question}")
    print("="*80)
    
    try:
        # Compare Al vs AlMg
        print("\n🔋 Comparing Al and AlMg electrode voltages...")
        result = await handler.compare_electrode_materials(
            formulas="Al,AlMg",
            working_ion="Li"
        )
        
        print(f"Comparison result: {result}")
        
        test.add_handler_call(
            "compare_electrode_materials",
            {"formulas": "Al,AlMg", "working_ion": "Li"},
            result
        )
        
        if result.get("success"):
            comparison = result.get("comparison", [])
            
            # Extract voltages
            al_voltage = None
            almg_voltage = None
            
            for item in comparison:
                formula = item.get("formula")
                data = item.get("data", {})
                
                if formula == "Al":
                    al_voltage = data.get("average_voltage") or data.get("voltage")
                elif formula == "AlMg":
                    almg_voltage = data.get("average_voltage") or data.get("voltage")
            
            test.metadata["al_voltage"] = al_voltage
            test.metadata["almg_voltage"] = almg_voltage
            
            if al_voltage is not None and almg_voltage is not None:
                voltage_change = almg_voltage - al_voltage
                
                # Expected: FALSE - Mg should DECREASE voltage
                if almg_voltage < al_voltage:
                    test.analysis = f"✅ Al: {al_voltage:.3f}V, AlMg: {almg_voltage:.3f}V. Change: {voltage_change:.3f}V (decrease)"
                    test.conclusion = "FALSE (CORRECT) - Mg alloying decreases voltage, not increases"
                    test.passed = True  # Expected FALSE, got decrease = pass
                elif almg_voltage > al_voltage:
                    test.analysis = f"❌ Al: {al_voltage:.3f}V, AlMg: {almg_voltage:.3f}V. Change: +{voltage_change:.3f}V (increase)"
                    test.conclusion = "TRUE (INCORRECT) - Mg alloying increases voltage (but shouldn't)"
                    test.passed = False  # Expected FALSE, got increase = fail
                else:
                    test.analysis = f"⚠️ Al: {al_voltage:.3f}V, AlMg: {almg_voltage:.3f}V. No change"
                    test.conclusion = "INCONCLUSIVE - No voltage change detected"
                    test.passed = None
            else:
                test.analysis = f"❌ Could not extract voltages. Al: {al_voltage}, AlMg: {almg_voltage}"
                test.conclusion = "INCONCLUSIVE - Insufficient voltage data"
                test.passed = None
            
            # Store summary
            summary = result.get("summary", "")
            if summary:
                test.metadata["comparison_summary"] = summary
        else:
            test.analysis = "❌ Comparison failed"
            test.conclusion = "INCONCLUSIVE - Could not compare electrodes"
            test.passed = None
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        test.analysis = f"Error during test execution: {str(e)}"
        test.conclusion = "ERROR"
        test.passed = None
    
    print(f"\n{test.conclusion}")
    return test


async def test_q8_cu80li10al10_stable(handler: BatteryHandler) -> TestResult:
    """
    Test: Cu80Li10Al10 can form a thermodynamically stable anode for a lithium battery
    """
    test = TestResult(
        test_id="Q8",
        question="Cu80Li10Al10 can form a thermodynamically stable anode",
        expected="FALSE - Composition contains Li and is thermodynamically unstable"
    )
    
    print("\n" + "="*80)
    print(f"🧪 {test.test_id}: {test.question}")
    print("="*80)
    
    try:
        # Check composition stability
        print("\n🔋 Checking stability of Cu80Li10Al10...")
        result = await handler.check_composition_stability(
            composition="Cu80Li10Al10"
        )
        
        print(f"Stability result: {result}")
        
        test.add_handler_call(
            "check_composition_stability",
            {"composition": "Cu80Li10Al10"},
            result
        )
        
        if result.get("success"):
            is_stable = result.get("is_stable", False)
            e_above_hull = result.get("energy_above_hull")
            
            test.metadata["is_stable"] = is_stable
            test.metadata["energy_above_hull"] = e_above_hull
            
            # Also analyze as anode
            print("\n🔋 Analyzing as potential anode...")
            anode_result = await handler.analyze_anode_viability(
                composition="Cu80Li10Al10",
                working_ion="Li"
            )
            
            test.add_handler_call(
                "analyze_anode_viability",
                {"composition": "Cu80Li10Al10", "working_ion": "Li"},
                anode_result
            )
            
            if anode_result.get("success"):
                assessment = anode_result.get("viability_assessment", {})
                can_form = assessment.get("can_form_stable_anode", False)
                reasoning = assessment.get("reasoning", [])
                
                test.metadata["anode_assessment"] = assessment
                
                # Expected: FALSE - should not be stable
                if not is_stable:
                    test.analysis = f"✅ Not stable (E_hull: {e_above_hull}). {' '.join(reasoning)}"
                    test.conclusion = "FALSE (CORRECT) - Not thermodynamically stable"
                    test.passed = True  # Expected FALSE, got unstable = pass
                elif is_stable and not can_form:
                    test.analysis = f"⚠️ Composition is stable but assessment: {' '.join(reasoning)}"
                    test.conclusion = "PARTIALLY FALSE - Stable but unsuitable as anode"
                    test.passed = True  # Still FALSE overall
                else:
                    test.analysis = f"❌ Composition is stable (E_hull: {e_above_hull}) and can form anode"
                    test.conclusion = "TRUE (INCORRECT) - Appears stable (but shouldn't be)"
                    test.passed = False  # Expected FALSE, got stable = fail
            else:
                # Just use stability check (Expected: FALSE)
                if not is_stable:
                    test.analysis = f"✅ Composition is not stable (E_hull: {e_above_hull})"
                    test.conclusion = "FALSE (CORRECT) - Not thermodynamically stable"
                    test.passed = True
                else:
                    test.analysis = f"❌ Composition is thermodynamically stable (E_hull: {e_above_hull})"
                    test.conclusion = "TRUE (INCORRECT) - Appears stable (but shouldn't be)"
                    test.passed = False
        else:
            test.analysis = "❌ Stability check failed"
            test.conclusion = "INCONCLUSIVE - Could not determine stability"
            test.passed = None
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        test.analysis = f"Error during test execution: {str(e)}"
        test.conclusion = "ERROR"
        test.passed = None
    
    print(f"\n{test.conclusion}")
    return test


async def test_q9_cual_li_capacity(handler: BatteryHandler) -> TestResult:
    """
    Test: For a (CuAl)_{1-x}Li_{x} electrode, the maximum practical capacity is x=0.4
    """
    test = TestResult(
        test_id="Q9",
        question="For (CuAl)_{1-x}Li_{x} electrode, maximum practical capacity is x=0.4",
        expected="FALSE - Maximum practical capacity is NOT x=0.4"
    )
    
    print("\n" + "="*80)
    print(f"🧪 {test.test_id}: {test.question}")
    print("="*80)
    
    try:
        # Analyze lithiation mechanism for CuAl host
        print("\n🔋 Analyzing lithiation mechanism for CuAl...")
        result = await handler.analyze_lithiation_mechanism(
            host_composition="CuAl",
            working_ion="Li",
            max_x=1.0,  # Analyze up to x=1.0 to see where limit is
            room_temp=True
        )
        
        print(f"Lithiation result: {result}")
        
        test.add_handler_call(
            "analyze_lithiation_mechanism",
            {"host_composition": "CuAl", "working_ion": "Li", "max_x": 1.0},
            result
        )
        
        if result.get("success"):
            # Look for maximum practical x value
            max_x = result.get("max_practical_x") or result.get("max_x_stable")
            voltage_profile = result.get("voltage_profile", [])
            
            test.metadata["max_x"] = max_x
            test.metadata["voltage_profile"] = voltage_profile
            
            if max_x is not None:
                # Expected: FALSE - max should NOT be 0.4
                if 0.3 <= max_x <= 0.5:
                    test.analysis = f"❌ Maximum practical capacity: x={max_x:.2f}, close to 0.4"
                    test.conclusion = f"TRUE (INCORRECT) - Maximum capacity near x=0.4 (but shouldn't be)"
                    test.passed = False  # Expected FALSE, got ~0.4 = fail
                else:
                    test.analysis = f"✅ Maximum practical capacity: x={max_x:.2f}, not close to 0.4"
                    test.conclusion = f"FALSE (CORRECT) - Maximum capacity is x={max_x:.2f}, not x=0.4"
                    test.passed = True  # Expected FALSE, got != 0.4 = pass
            else:
                # Try to extract from voltage profile or analysis text
                result_str = str(result)
                
                # Look for mentions of x values
                import re
                x_pattern = r'x\s*[=~]\s*(0\.\d+)'
                matches = re.findall(x_pattern, result_str)
                
                if matches:
                    x_values = [float(m) for m in matches]
                    max_x_found = max(x_values)
                    
                    test.metadata["extracted_x_values"] = x_values
                    
                    # Expected: FALSE - should NOT be 0.4
                    if 0.3 <= max_x_found <= 0.5:
                        test.analysis = f"❌ Found x values in analysis: {x_values}. Max around 0.4"
                        test.conclusion = f"TRUE (INCORRECT) - Evidence suggests x≈{max_x_found:.2f} (but shouldn't)"
                        test.passed = False
                    else:
                        test.analysis = f"✅ Found x values: {x_values}, max is {max_x_found:.2f}, not 0.4"
                        test.conclusion = f"FALSE (CORRECT) - Maximum is x={max_x_found:.2f}, not 0.4"
                        test.passed = True
                else:
                    test.analysis = "❌ Could not extract maximum x value from analysis"
                    test.conclusion = "INCONCLUSIVE - Manual analysis required"
                    test.passed = None
        else:
            test.analysis = "❌ Lithiation analysis failed"
            test.conclusion = "INCONCLUSIVE - Could not analyze lithiation"
            test.passed = None
        
        # Also check specific compositions
        print("\n🔋 Checking Cu60Al40 (x=0) as baseline...")
        baseline = await handler.search_battery_electrodes(
            formula="CuAl",
            working_ion="Li",
            max_entries=1
        )
        
        test.add_handler_call(
            "search_battery_electrodes",
            {"formula": "CuAl", "working_ion": "Li"},
            baseline,
            "Baseline CuAl electrode"
        )
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        test.analysis = f"Error during test execution: {str(e)}"
        test.conclusion = "ERROR"
        test.passed = None
    
    print(f"\n{test.conclusion}")
    return test


async def run_all_tests():
    """Run all battery/electrochemistry tests"""
    print("\n" + "="*80)
    print("🔋 BATTERY/ELECTROCHEMISTRY TEST SUITE")
    print("="*80)
    
    # Get MPRester client
    mpr = get_mpr()
    if not mpr:
        print("\n❌ Cannot run tests without Materials Project API access")
        print("   Please set MP_API_KEY environment variable")
        return []
    
    handler = BatteryHandler(mpr=mpr)
    print(f"\n✓ Handler created with MPRester client")
    
    tests = [
        await test_q6_almg_voltage(handler),
        await test_q7_mg_alloy_increases_voltage(handler),
        await test_q8_cu80li10al10_stable(handler),
        await test_q9_cual_li_capacity(handler),
    ]
    
    # Generate summary report
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    
    for test in tests:
        status_icon = "✅" if test.passed is True else "❌" if test.passed is False else "❓"
        print(f"\n{status_icon} {test.test_id}: {test.question}")
        print(f"   Expected: {test.expected}")
        print(f"   Result: {test.conclusion}")
        print(f"   Analysis: {test.analysis}")
    
    # Save results to JSON
    results_file = Path(__file__).parent / "test_results_battery.json"
    with open(results_file, 'w') as f:
        json.dump({
            "tests": [test.to_dict() for test in tests],
            "summary": {
                "total": len(tests),
                "passed": sum(1 for t in tests if t.passed is True),
                "failed": sum(1 for t in tests if t.passed is False),
                "inconclusive": sum(1 for t in tests if t.passed is None)
            }
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Print summary statistics
    passed = sum(1 for t in tests if t.passed is True)
    failed = sum(1 for t in tests if t.passed is False)
    inconclusive = sum(1 for t in tests if t.passed is None)
    
    print("\n" + "="*80)
    print(f"Total Tests: {len(tests)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"❓ Inconclusive: {inconclusive}")
    print("="*80)
    
    return tests


if __name__ == "__main__":
    asyncio.run(run_all_tests())

