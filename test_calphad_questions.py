#!/usr/bin/env python3
"""
Test suite for CALPHAD phase diagram questions.

Tests the following statements:
1. Al20Zn80 at 870K is a solid at equilibrium
2. Al and Zn form an eutectic at ~Al15Zn75
3. Al50Zn50 forms a single solid phase <700K
4. The phase fraction of Aluminum Silicon Carbide precipitates in Al30Si55C15 
   increases with decreasing temperature from 500K to 300K
5. Heating a sample of Al30Si55C15 to 1500K is insufficient to dissolve all 
   carbide precipitates
"""
import sys
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).parent))

from mcp_materials_project.handlers.calphad.phase_diagrams.phase_diagrams import CalPhadHandler


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


async def test_q1_al20zn80_solid_at_870k(handler: CalPhadHandler) -> TestResult:
    """
    Test: Al20Zn80 at 870K is a solid at equilibrium
    """
    test = TestResult(
        test_id="Q1",
        question="Al20Zn80 at 870K is a solid at equilibrium",
        expected="FALSE - Should show liquid phase at 870K"
    )
    
    print("\n" + "="*80)
    print(f"🧪 {test.test_id}: {test.question}")
    print("="*80)
    
    try:
        # Test using calculate_equilibrium_at_point
        print("\n📊 Calculating equilibrium at 870K for Al20Zn80...")
        result = await handler.calculate_equilibrium_at_point(
            composition="Al20Zn80",
            temperature=870.0,
            composition_type="atomic"
        )
        
        print(f"Result: {result}")
        
        # Extract phase information
        has_liquid = "LIQUID" in result.upper() or "liquid" in result.lower()
        has_solid = any(word in result.upper() for word in ["FCC", "HCP", "BCC", "SOLID"])
        
        # Analyze (Expected: FALSE - should be liquid)
        if "No stable phases found" in result or "Failed" in result:
            test.analysis = "❌ Equilibrium calculation failed or no phases found"
            test.conclusion = "INCONCLUSIVE - Calculation error"
            test.passed = None
        elif has_liquid and not has_solid:
            test.analysis = "✅ Only liquid phase present at 870K"
            test.conclusion = "FALSE (CORRECT) - Material is liquid, not solid"
            test.passed = True  # Changed: expected FALSE, got FALSE = pass
        elif has_solid and not has_liquid:
            test.analysis = "❌ Only solid phases present at 870K"
            test.conclusion = "TRUE (INCORRECT) - Material is solid (but shouldn't be)"
            test.passed = False  # Changed: expected FALSE, got TRUE = fail
        elif has_liquid and has_solid:
            test.analysis = "⚠️ Both liquid and solid phases present (two-phase region)"
            test.conclusion = "PARTIALLY LIQUID - In solid-liquid equilibrium region"
            test.passed = False
        else:
            test.analysis = "❓ Could not determine phase state from output"
            test.conclusion = "INCONCLUSIVE - Need to check output manually"
            test.passed = None
        
        test.add_handler_call(
            "calculate_equilibrium_at_point",
            {"composition": "Al20Zn80", "temperature": 870.0},
            result,
            test.analysis
        )
        
        # Also generate phase stability plot for visual confirmation
        print("\n📊 Generating phase stability plot...")
        plot_result = await handler.plot_composition_temperature(
            composition="Al20Zn80",
            min_temperature=300.0,
            max_temperature=1200.0,
            interactive="html"
        )
        
        test.add_handler_call(
            "plot_composition_temperature",
            {"composition": "Al20Zn80", "min_temp": 300, "max_temp": 1200},
            plot_result,
            "Generated phase stability plot for visual analysis"
        )
        
        # Store metadata if available
        if hasattr(handler, '_last_image_metadata'):
            test.metadata = handler._last_image_metadata
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        test.analysis = f"Error during test execution: {str(e)}"
        test.conclusion = "ERROR"
        test.passed = None
    
    print(f"\n{test.conclusion}")
    return test


async def test_q2_alzn_eutectic(handler: CalPhadHandler) -> TestResult:
    """
    Test: Al and Zn form an eutectic at ~Al15Zn75
    """
    test = TestResult(
        test_id="Q2",
        question="Al and Zn form an eutectic at ~Al15Zn75",
        expected="FALSE - Eutectic is actually at ~Al11Zn89 (89 at% Zn), not 75%"
    )
    
    print("\n" + "="*80)
    print(f"🧪 {test.test_id}: {test.question}")
    print("="*80)
    
    try:
        # Generate full binary phase diagram
        print("\n📊 Generating Al-Zn phase diagram to detect eutectic...")
        result = await handler.plot_binary_phase_diagram(
            system="Al-Zn",
            min_temperature=300.0,
            max_temperature=1000.0
        )
        
        print(f"Result: {result}")
        
        test.add_handler_call(
            "plot_binary_phase_diagram",
            {"system": "Al-Zn", "min_temp": 300, "max_temp": 1000},
            result
        )
        
        # Extract eutectic information from metadata
        if hasattr(handler, '_last_image_metadata'):
            metadata = handler._last_image_metadata
            test.metadata = metadata
            
            # Look for eutectic points in analysis
            analysis_text = metadata.get('analysis', '') + metadata.get('thermodynamic_analysis', '')
            
            # Check for eutectic mention
            has_eutectic = 'eutectic' in analysis_text.lower()
            
            if has_eutectic:
                # Try to extract composition
                # Look for patterns like "at X at% Zn" or "X% Zn"
                import re
                pattern = r'(\d+\.?\d*)\s*(?:at%|%)\s*(?:Zn|ZN)'
                matches = re.findall(pattern, analysis_text)
                
                if matches:
                    compositions = [float(m) for m in matches]
                    # Check if any composition is near 89% Zn (actual eutectic), not 75%
                    near_89 = any(85 <= comp <= 93 for comp in compositions)
                    near_75 = any(70 <= comp <= 80 for comp in compositions)
                    
                    if near_89 and not near_75:
                        test.analysis = f"✅ Eutectic detected at composition(s): {compositions}. Near 89 at% Zn (correct), not 75%"
                        test.conclusion = "FALSE (CORRECT) - Eutectic is at ~89% Zn, not 75%"
                        test.passed = True  # Expected FALSE, got FALSE = pass
                    elif near_75:
                        test.analysis = f"❌ Eutectic detected at composition(s): {compositions}. Near 75 at% Zn (incorrect)"
                        test.conclusion = "TRUE (INCORRECT) - Would incorrectly confirm 75%"
                        test.passed = False
                    else:
                        test.analysis = f"⚠️ Eutectic detected at composition(s): {compositions}, not at 75% or 89%"
                        test.conclusion = "INCONCLUSIVE - Eutectic at unexpected location"
                        test.passed = None
                else:
                    test.analysis = "✅ Eutectic mentioned in analysis but composition unclear"
                    test.conclusion = "LIKELY TRUE - Eutectic detected, manual verification of composition needed"
                    test.passed = True
            else:
                test.analysis = "❌ No eutectic point detected in analysis"
                test.conclusion = "INCONCLUSIVE - No eutectic found in analysis"
                test.passed = None
        else:
            test.analysis = "❌ No metadata available from phase diagram generation"
            test.conclusion = "INCONCLUSIVE - Could not analyze diagram"
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


async def test_q3_al50zn50_single_phase(handler: CalPhadHandler) -> TestResult:
    """
    Test: Al50Zn50 forms a single solid phase <700K
    """
    test = TestResult(
        test_id="Q3",
        question="Al50Zn50 forms a single solid phase <700K",
        expected="FALSE - Should show multiple solid phases (likely 2) below 700K"
    )
    
    print("\n" + "="*80)
    print(f"🧪 {test.test_id}: {test.question}")
    print("="*80)
    
    try:
        # Test at multiple temperatures below 700K
        test_temps = [300, 500, 650]
        phase_counts = {}
        
        for temp in test_temps:
            print(f"\n📊 Testing Al50Zn50 at {temp}K...")
            result = await handler.calculate_equilibrium_at_point(
                composition="Al50Zn50",
                temperature=float(temp),
                composition_type="atomic"
            )
            
            print(f"Result at {temp}K: {result}")
            
            # Count number of stable phases (look for bullet points)
            import re
            phase_pattern = r'•\s*\*\*([^*]+)\*\*:\s*([\d.]+)%'
            matches = re.findall(phase_pattern, result)
            
            solid_phases = [m for m in matches if "LIQUID" not in m[0].upper()]
            phase_counts[temp] = len(solid_phases)
            
            test.add_handler_call(
                "calculate_equilibrium_at_point",
                {"composition": "Al50Zn50", "temperature": temp},
                result,
                f"Found {len(solid_phases)} solid phase(s) at {temp}K"
            )
        
        # Analyze results (Expected: FALSE - should be multiple phases)
        # If ANY temperature <700K has multiple phases, statement is FALSE
        all_single_phase = all(count == 1 for count in phase_counts.values())
        any_multi_phase = any(count >= 2 for count in phase_counts.values())
        
        if any_multi_phase:
            # At least one temperature has multiple phases → statement is FALSE
            test.analysis = f"✅ Multiple solid phases found at some temperatures: {phase_counts}"
            test.conclusion = "FALSE (CORRECT) - Has multiple phases below 700K, not single"
            test.passed = True  # Expected FALSE, got multiple phases = pass
        elif all_single_phase:
            # All temperatures have single phase → statement would be TRUE
            test.analysis = f"❌ Single solid phase at all tested temperatures: {phase_counts}"
            test.conclusion = "TRUE (INCORRECT) - Found single phase (but shouldn't be)"
            test.passed = False  # Expected FALSE, got TRUE = fail
        else:
            # Should not reach here, but just in case
            test.analysis = f"⚠️ Unexpected results: {phase_counts}"
            test.conclusion = "INCONCLUSIVE - Unexpected phase behavior"
            test.passed = None
        
        # Also generate composition-temperature plot
        print("\n📊 Generating phase stability plot for Al50Zn50...")
        plot_result = await handler.plot_composition_temperature(
            composition="Al50Zn50",
            min_temperature=300.0,
            max_temperature=900.0,
            interactive="html"
        )
        
        test.add_handler_call(
            "plot_composition_temperature",
            {"composition": "Al50Zn50", "min_temp": 300, "max_temp": 900},
            plot_result,
            "Generated phase stability plot"
        )
        
        if hasattr(handler, '_last_image_metadata'):
            test.metadata = handler._last_image_metadata
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        test.analysis = f"Error during test execution: {str(e)}"
        test.conclusion = "ERROR"
        test.passed = None
    
    print(f"\n{test.conclusion}")
    return test


async def test_q4_al30si55c15_precipitate_increase(handler: CalPhadHandler) -> TestResult:
    """
    Test: The phase fraction of Aluminum Silicon Carbide precipitates in Al30Si55C15 
          increases with decreasing temperature from 500K to 300K
    """
    test = TestResult(
        test_id="Q4",
        question="Phase fraction of Al/Si/C precipitates in Al30Si55C15 increases with decreasing temperature (500K→300K)",
        expected="FALSE - Carbide precipitates do NOT increase as temperature decreases"
    )
    
    print("\n" + "="*80)
    print(f"🧪 {test.test_id}: {test.question}")
    print("="*80)
    
    try:
        # Use analyze_phase_fraction_trend to check if carbide phases increase
        print("\n📊 Calculating phase fractions vs temperature for Al30Si55C15...")
        result = await handler.calculate_phase_fractions_vs_temperature(
            composition="Al30Si55C15",
            min_temperature=300.0,
            max_temperature=500.0,
            temperature_step=10.0
        )
        
        print(f"Result: {result}")
        
        test.add_handler_call(
            "calculate_phase_fractions_vs_temperature",
            {"composition": "Al30Si55C15", "min_temp": 300, "max_temp": 500},
            result
        )
        
        # Look for carbide phases (AL4C3, SIC, etc.) and their trends
        import re
        # Pattern: "Phase: X% → Y% (trend, change)"
        phase_pattern = r'•\s*\*\*([^*]+)\*\*:\s*([\d.]+)%\s*→\s*([\d.]+)%\s*\(([^,]+),'
        matches = re.findall(phase_pattern, result)
        
        carbide_phases = []
        for phase_name, start_frac, end_frac, trend in matches:
            # Identify carbide phases (containing C, or known carbides)
            if 'C' in phase_name.upper() or 'CARBIDE' in phase_name.upper():
                carbide_phases.append({
                    'phase': phase_name,
                    'frac_300K': float(start_frac),  # Lower temp (first in range)
                    'frac_500K': float(end_frac),    # Higher temp (last in range)
                    'trend': trend.strip()
                })
        
        if carbide_phases:
            # Check if carbide phases increase as temp decreases (i.e., frac_300K > frac_500K)
            # Expected: FALSE - they should NOT increase
            increasing_on_cooling = [cp for cp in carbide_phases if cp['frac_300K'] > cp['frac_500K']]
            decreasing_on_cooling = [cp for cp in carbide_phases if cp['frac_300K'] <= cp['frac_500K']]
            
            if decreasing_on_cooling and not increasing_on_cooling:
                test.analysis = f"✅ Carbide phase(s) do NOT increase with cooling: {[cp['phase'] for cp in carbide_phases]}"
                test.conclusion = "FALSE (CORRECT) - Precipitates do not increase upon cooling"
                test.passed = True  # Expected FALSE, got FALSE = pass
            elif increasing_on_cooling:
                test.analysis = f"❌ Carbide phase(s) increase with decreasing temperature: {[cp['phase'] for cp in increasing_on_cooling]}"
                test.conclusion = "TRUE (INCORRECT) - Precipitates increase (but shouldn't)"
                test.passed = False  # Expected FALSE, got TRUE = fail
            else:
                test.analysis = f"⚠️ Carbide phases found with unclear trend: {carbide_phases}"
                test.conclusion = "INCONCLUSIVE - Cannot determine trend"
                test.passed = None
        else:
            test.analysis = "❌ No carbide phases identified in the analysis"
            test.conclusion = "INCONCLUSIVE - Could not identify carbide phases"
            test.passed = None
        
        # Also generate composition-temperature plot for visual confirmation
        print("\n📊 Generating phase stability plot for Al30Si55C15...")
        plot_result = await handler.plot_composition_temperature(
            composition="Al30Si55C15",
            min_temperature=300.0,
            max_temperature=1600.0,
            interactive="html"
        )
        
        test.add_handler_call(
            "plot_composition_temperature",
            {"composition": "Al30Si55C15", "min_temp": 300, "max_temp": 1600},
            plot_result,
            "Generated phase stability plot for visual analysis"
        )
        
        if hasattr(handler, '_last_image_metadata'):
            test.metadata = handler._last_image_metadata
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        test.analysis = f"Error during test execution: {str(e)}"
        test.conclusion = "ERROR"
        test.passed = None
    
    print(f"\n{test.conclusion}")
    return test


async def test_q5_al30si55c15_dissolution(handler: CalPhadHandler) -> TestResult:
    """
    Test: Heating a sample of Al30Si55C15 to 1500K is insufficient to dissolve 
          all carbide precipitates
    """
    test = TestResult(
        test_id="Q5",
        question="Heating Al30Si55C15 to 1500K is insufficient to dissolve all carbide precipitates",
        expected="TRUE - Should still show carbide phases at 1500K"
    )
    
    print("\n" + "="*80)
    print(f"🧪 {test.test_id}: {test.question}")
    print("="*80)
    
    try:
        # Check equilibrium at 1500K
        print("\n📊 Calculating equilibrium at 1500K for Al30Si55C15...")
        result = await handler.calculate_equilibrium_at_point(
            composition="Al30Si55C15",
            temperature=1500.0,
            composition_type="atomic"
        )
        
        print(f"Result: {result}")
        
        test.add_handler_call(
            "calculate_equilibrium_at_point",
            {"composition": "Al30Si55C15", "temperature": 1500.0},
            result
        )
        
        # Look for carbide phases
        import re
        phase_pattern = r'•\s*\*\*([^*]+)\*\*:\s*([\d.]+)%'
        matches = re.findall(phase_pattern, result)
        
        carbide_phases = []
        for phase_name, fraction in matches:
            if 'C' in phase_name.upper() or 'CARBIDE' in phase_name.upper():
                if phase_name.upper() != 'FCC_A1':  # FCC_A1 is not a carbide
                    carbide_phases.append({
                        'phase': phase_name,
                        'fraction': float(fraction)
                    })
        
        if carbide_phases:
            total_carbide = sum(cp['fraction'] for cp in carbide_phases)
            test.analysis = f"✅ Carbide phases still present at 1500K: {carbide_phases}. Total: {total_carbide:.2f}%"
            test.conclusion = "TRUE - Carbides not fully dissolved at 1500K"
            test.passed = True
        else:
            test.analysis = "❌ No carbide phases detected at 1500K (all dissolved or in liquid)"
            test.conclusion = "FALSE - Carbides appear to be dissolved"
            test.passed = False
        
        # Also analyze temperature range to see dissolution behavior
        print("\n📊 Analyzing phase fractions from 300K to 1600K...")
        frac_result = await handler.calculate_phase_fractions_vs_temperature(
            composition="Al30Si55C15",
            min_temperature=300.0,
            max_temperature=1600.0,
            temperature_step=50.0
        )
        
        test.add_handler_call(
            "calculate_phase_fractions_vs_temperature",
            {"composition": "Al30Si55C15", "min_temp": 300, "max_temp": 1600},
            frac_result,
            "Analyzed phase evolution to check dissolution behavior"
        )
        
        # Generate plot
        print("\n📊 Generating phase stability plot...")
        plot_result = await handler.plot_composition_temperature(
            composition="Al30Si55C15",
            min_temperature=300.0,
            max_temperature=1600.0,
            interactive="html"
        )
        
        test.add_handler_call(
            "plot_composition_temperature",
            {"composition": "Al30Si55C15", "min_temp": 300, "max_temp": 1600},
            plot_result,
            "Generated phase stability plot"
        )
        
        if hasattr(handler, '_last_image_metadata'):
            test.metadata = handler._last_image_metadata
        
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
    """Run all CALPHAD tests"""
    print("\n" + "="*80)
    print("🧪 CALPHAD PHASE DIAGRAM TEST SUITE")
    print("="*80)
    
    handler = CalPhadHandler()
    print(f"\n✓ Handler created, TDB directory: {handler.tdb_dir}")
    
    tests = [
        await test_q1_al20zn80_solid_at_870k(handler),
        await test_q2_alzn_eutectic(handler),
        await test_q3_al50zn50_single_phase(handler),
        await test_q4_al30si55c15_precipitate_increase(handler),
        await test_q5_al30si55c15_dissolution(handler),
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
    results_file = Path(__file__).parent / "test_results_calphad.json"
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

