#!/usr/bin/env python3
"""
Master test runner for all handler test suites.

Runs both CALPHAD and battery/electrochemistry test suites and generates
a comprehensive report.
"""
import sys
import asyncio
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

# Import test modules
import test_calphad_questions
import test_battery_questions


async def run_all_test_suites():
    """Run all test suites and generate comprehensive report"""
    
    print("\n" + "="*80)
    print("🧪 COMPREHENSIVE HANDLER TEST SUITE")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "suites": {}
    }
    
    # Run CALPHAD tests
    print("\n\n" + "🔬 RUNNING CALPHAD TESTS...")
    print("="*80)
    try:
        calphad_tests = await test_calphad_questions.run_all_tests()
        all_results["suites"]["calphad"] = {
            "tests": [t.to_dict() for t in calphad_tests],
            "summary": {
                "total": len(calphad_tests),
                "passed": sum(1 for t in calphad_tests if t.passed is True),
                "failed": sum(1 for t in calphad_tests if t.passed is False),
                "inconclusive": sum(1 for t in calphad_tests if t.passed is None)
            }
        }
    except Exception as e:
        print(f"\n❌ CALPHAD tests failed with error: {e}")
        import traceback
        traceback.print_exc()
        all_results["suites"]["calphad"] = {"error": str(e)}
    
    # Run battery tests
    print("\n\n" + "🔋 RUNNING BATTERY/ELECTROCHEMISTRY TESTS...")
    print("="*80)
    try:
        battery_tests = await test_battery_questions.run_all_tests()
        all_results["suites"]["battery"] = {
            "tests": [t.to_dict() for t in battery_tests],
            "summary": {
                "total": len(battery_tests),
                "passed": sum(1 for t in battery_tests if t.passed is True),
                "failed": sum(1 for t in battery_tests if t.passed is False),
                "inconclusive": sum(1 for t in battery_tests if t.passed is None)
            }
        }
    except Exception as e:
        print(f"\n❌ Battery tests failed with error: {e}")
        import traceback
        traceback.print_exc()
        all_results["suites"]["battery"] = {"error": str(e)}
    
    # Generate comprehensive report
    print("\n\n" + "="*80)
    print("📊 COMPREHENSIVE TEST REPORT")
    print("="*80)
    
    total_tests = 0
    total_passed = 0
    total_failed = 0
    total_inconclusive = 0
    
    for suite_name, suite_data in all_results["suites"].items():
        if "error" in suite_data:
            print(f"\n❌ {suite_name.upper()}: ERROR")
            print(f"   {suite_data['error']}")
            continue
        
        summary = suite_data.get("summary", {})
        print(f"\n{'🔬' if suite_name == 'calphad' else '🔋'} {suite_name.upper()}:")
        print(f"   Total: {summary.get('total', 0)}")
        print(f"   ✅ Passed: {summary.get('passed', 0)}")
        print(f"   ❌ Failed: {summary.get('failed', 0)}")
        print(f"   ❓ Inconclusive: {summary.get('inconclusive', 0)}")
        
        total_tests += summary.get('total', 0)
        total_passed += summary.get('passed', 0)
        total_failed += summary.get('failed', 0)
        total_inconclusive += summary.get('inconclusive', 0)
    
    print("\n" + "="*80)
    print("📈 OVERALL STATISTICS")
    print("="*80)
    print(f"Total Tests: {total_tests}")
    print(f"✅ Passed: {total_passed} ({total_passed/total_tests*100:.1f}%)" if total_tests > 0 else "✅ Passed: 0")
    print(f"❌ Failed: {total_failed} ({total_failed/total_tests*100:.1f}%)" if total_tests > 0 else "❌ Failed: 0")
    print(f"❓ Inconclusive: {total_inconclusive} ({total_inconclusive/total_tests*100:.1f}%)" if total_tests > 0 else "❓ Inconclusive: 0")
    print("="*80)
    
    # Save comprehensive results
    results_file = Path(__file__).parent / "test_results_comprehensive.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Comprehensive results saved to: {results_file}")
    
    # Generate HTML report
    html_report = generate_html_report(all_results)
    html_file = Path(__file__).parent / "test_results_report.html"
    with open(html_file, 'w') as f:
        f.write(html_report)
    
    print(f"📄 HTML report saved to: {html_file}")
    
    print(f"\n✅ Test suite completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return all_results


def generate_html_report(results: dict) -> str:
    """Generate HTML report from test results"""
    
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Handler Test Results</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        h1, h2, h3 {
            color: #333;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        .suite {
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .test {
            background: #f9f9f9;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #ddd;
            border-radius: 4px;
        }
        .test.passed {
            border-left-color: #10b981;
            background: #f0fdf4;
        }
        .test.failed {
            border-left-color: #ef4444;
            background: #fef2f2;
        }
        .test.inconclusive {
            border-left-color: #f59e0b;
            background: #fffbeb;
        }
        .test-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .test-id {
            font-weight: bold;
            font-size: 1.1em;
        }
        .badge {
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: bold;
        }
        .badge.passed {
            background: #10b981;
            color: white;
        }
        .badge.failed {
            background: #ef4444;
            color: white;
        }
        .badge.inconclusive {
            background: #f59e0b;
            color: white;
        }
        .test-question {
            font-size: 1.05em;
            margin: 10px 0;
            color: #555;
        }
        .test-analysis {
            margin: 10px 0;
            padding: 10px;
            background: white;
            border-radius: 4px;
            font-size: 0.95em;
        }
        .test-conclusion {
            margin-top: 10px;
            font-weight: bold;
            color: #333;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stat-number {
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }
        .stat-label {
            color: #666;
            font-size: 0.9em;
        }
        .handler-calls {
            margin-top: 15px;
            padding: 10px;
            background: #f5f5f5;
            border-radius: 4px;
            font-size: 0.9em;
        }
        .handler-call {
            margin: 8px 0;
            padding: 8px;
            background: white;
            border-radius: 3px;
        }
        .timestamp {
            color: rgba(255,255,255,0.9);
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧪 Handler Test Results</h1>
        <p class="timestamp">Generated: """ + results.get('timestamp', 'N/A') + """</p>
    </div>
"""
    
    # Overall statistics
    total_tests = 0
    total_passed = 0
    total_failed = 0
    total_inconclusive = 0
    
    for suite_data in results.get("suites", {}).values():
        if "summary" in suite_data:
            summary = suite_data["summary"]
            total_tests += summary.get('total', 0)
            total_passed += summary.get('passed', 0)
            total_failed += summary.get('failed', 0)
            total_inconclusive += summary.get('inconclusive', 0)
    
    html += f"""
    <div class="stats">
        <div class="stat-card">
            <div class="stat-label">Total Tests</div>
            <div class="stat-number">{total_tests}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">✅ Passed</div>
            <div class="stat-number" style="color: #10b981;">{total_passed}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">❌ Failed</div>
            <div class="stat-number" style="color: #ef4444;">{total_failed}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">❓ Inconclusive</div>
            <div class="stat-number" style="color: #f59e0b;">{total_inconclusive}</div>
        </div>
    </div>
"""
    
    # Test suites
    for suite_name, suite_data in results.get("suites", {}).items():
        if "error" in suite_data:
            html += f"""
    <div class="suite">
        <h2>❌ {suite_name.upper()} Suite - ERROR</h2>
        <p style="color: #ef4444;">{suite_data['error']}</p>
    </div>
"""
            continue
        
        icon = "🔬" if suite_name == "calphad" else "🔋"
        html += f"""
    <div class="suite">
        <h2>{icon} {suite_name.upper()} Suite</h2>
"""
        
        # Suite statistics
        summary = suite_data.get("summary", {})
        html += f"""
        <div class="stats">
            <div class="stat-card">
                <div class="stat-label">Total</div>
                <div class="stat-number">{summary.get('total', 0)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Passed</div>
                <div class="stat-number" style="color: #10b981;">{summary.get('passed', 0)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Failed</div>
                <div class="stat-number" style="color: #ef4444;">{summary.get('failed', 0)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Inconclusive</div>
                <div class="stat-number" style="color: #f59e0b;">{summary.get('inconclusive', 0)}</div>
            </div>
        </div>
"""
        
        # Individual tests
        for test in suite_data.get("tests", []):
            status = "passed" if test['passed'] is True else "failed" if test['passed'] is False else "inconclusive"
            badge_text = "✅ PASSED" if test['passed'] is True else "❌ FAILED" if test['passed'] is False else "❓ INCONCLUSIVE"
            
            html += f"""
        <div class="test {status}">
            <div class="test-header">
                <span class="test-id">{test['test_id']}</span>
                <span class="badge {status}">{badge_text}</span>
            </div>
            <div class="test-question"><strong>Question:</strong> {test['question']}</div>
            <div class="test-analysis"><strong>Analysis:</strong> {test['analysis']}</div>
            <div class="test-conclusion"><strong>Conclusion:</strong> {test['conclusion']}</div>
"""
            
            # Handler calls
            if test.get('handler_calls'):
                html += """
            <div class="handler-calls">
                <strong>Handler Calls:</strong>
"""
                for call in test['handler_calls']:
                    html += f"""
                <div class="handler-call">
                    <strong>{call['function']}()</strong><br>
                    Parameters: {call['parameters']}<br>
                    {call.get('analysis', '')}
                </div>
"""
                html += """
            </div>
"""
            
            html += """
        </div>
"""
        
        html += """
    </div>
"""
    
    html += """
</body>
</html>
"""
    
    return html


if __name__ == "__main__":
    asyncio.run(run_all_test_suites())

