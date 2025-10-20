#!/usr/bin/env python3
"""
Verify that the test environment is properly set up.

Checks all prerequisites before running tests.
"""
import sys
import os
from pathlib import Path

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python version: {version.major}.{version.minor}.{version.micro} (need 3.8+)")
        return False

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✅ {package_name} installed")
        return True
    except ImportError:
        print(f"❌ {package_name} not installed (pip install {package_name})")
        return False

def check_tdb_directory():
    """Check if TDB directory exists and has files"""
    tdb_dir = Path(__file__).parent / "tdbs"
    
    if not tdb_dir.exists():
        print(f"❌ TDB directory not found: {tdb_dir}")
        return False
    
    tdb_files = list(tdb_dir.glob("*.tdb"))
    if not tdb_files:
        print(f"❌ No .tdb files found in {tdb_dir}")
        return False
    
    print(f"✅ TDB directory found with {len(tdb_files)} files:")
    for tdb_file in tdb_files:
        print(f"   - {tdb_file.name}")
    return True

def check_api_key():
    """Check if Materials Project API key is set"""
    api_key = os.environ.get('MP_API_KEY')
    
    if not api_key:
        print("⚠️  MP_API_KEY not set (required for battery tests)")
        print("   Get your key from: https://materialsproject.org/api")
        print("   Set with: export MP_API_KEY='your-api-key'")
        return False
    
    if len(api_key) < 10:
        print(f"⚠️  MP_API_KEY seems too short: {len(api_key)} chars")
        return False
    
    print(f"✅ MP_API_KEY set ({len(api_key)} chars)")
    return True

def check_output_directory():
    """Check if output directory exists"""
    plots_dir = Path(__file__).parent / "interactive_plots"
    
    if not plots_dir.exists():
        print(f"⚠️  Creating output directory: {plots_dir}")
        plots_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created {plots_dir}")
    else:
        print(f"✅ Output directory exists: {plots_dir}")
    
    return True

def main():
    """Run all checks"""
    print("\n" + "="*80)
    print("🔍 VERIFYING TEST SETUP")
    print("="*80 + "\n")
    
    checks = {
        "Python Version": check_python_version(),
        "TDB Directory": check_tdb_directory(),
        "Output Directory": check_output_directory(),
    }
    
    print("\n" + "-"*80)
    print("📦 Checking Required Packages (CALPHAD)")
    print("-"*80 + "\n")
    
    calphad_packages = [
        ("pycalphad", "pycalphad"),
        ("matplotlib", "matplotlib"),
        ("plotly", "plotly"),
        ("numpy", "numpy"),
    ]
    
    for pkg_name, import_name in calphad_packages:
        checks[pkg_name] = check_package(pkg_name, import_name)
    
    print("\n" + "-"*80)
    print("📦 Checking Required Packages (Battery)")
    print("-"*80 + "\n")
    
    battery_packages = [
        ("mp-api", "mp_api"),
        ("pymatgen", "pymatgen"),
    ]
    
    for pkg_name, import_name in battery_packages:
        checks[pkg_name] = check_package(pkg_name, import_name)
    
    print("\n" + "-"*80)
    print("🔑 Checking API Access (Battery)")
    print("-"*80 + "\n")
    
    checks["API Key"] = check_api_key()
    
    print("\n" + "-"*80)
    print("📦 Checking Optional Packages")
    print("-"*80 + "\n")
    
    optional_packages = [
        ("kaleido", "kaleido"),  # For plotly image export
    ]
    
    for pkg_name, import_name in optional_packages:
        check_package(pkg_name, import_name)
    
    # Summary
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80 + "\n")
    
    calphad_ready = all([
        checks["Python Version"],
        checks["TDB Directory"],
        checks["pycalphad"],
        checks["matplotlib"],
        checks["plotly"],
        checks["numpy"],
    ])
    
    battery_ready = all([
        checks["Python Version"],
        checks["mp-api"],
        checks["pymatgen"],
        checks["API Key"],
    ])
    
    overall_ready = calphad_ready and battery_ready
    
    if calphad_ready:
        print("✅ CALPHAD tests ready to run")
    else:
        print("❌ CALPHAD tests NOT ready (see missing items above)")
    
    if battery_ready:
        print("✅ Battery tests ready to run")
    else:
        print("⚠️  Battery tests NOT ready (see missing items above)")
    
    if overall_ready:
        print("\n🎉 All tests ready to run!")
        print("\nRun tests with:")
        print("  ./run_all_handler_tests.py")
    else:
        print("\n⚠️  Some tests are not ready to run")
        print("\nTo run CALPHAD tests only:")
        print("  ./test_calphad_questions.py")
        print("\nTo fix missing items:")
        print("  - Install packages: pip install <package-name>")
        print("  - Set API key: export MP_API_KEY='your-key'")
    
    print("\n" + "="*80)
    
    return 0 if overall_ready else 1

if __name__ == "__main__":
    sys.exit(main())

