#!/usr/bin/env python3
"""
Test runner script for materials handler tests.

This script provides convenient commands to run different test suites
with appropriate options and reporting.
"""

import sys
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n{'=' * 70}")
    print(f"{description}")
    print(f"{'=' * 70}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n❌ {description} failed!")
        return False
    else:
        print(f"\n✅ {description} passed!")
        return True


def main():
    """Main test runner."""
    if len(sys.argv) > 1:
        test_target = sys.argv[1]
    else:
        test_target = "all"
    
    # Change to tests directory
    tests_dir = Path(__file__).parent
    
    commands = {
        "utils": (
            f"cd {tests_dir} && pytest materials/test_utils.py -v",
            "Running utility function tests"
        ),
        "ai": (
            f"cd {tests_dir} && pytest materials/test_ai_functions.py -v",
            "Running AI function tests"
        ),
        "handler": (
            f"cd {tests_dir} && pytest materials/test_handler.py -v",
            "Running handler method tests"
        ),
        "materials": (
            f"cd {tests_dir} && pytest materials/ -v",
            "Running all materials handler tests"
        ),
        "all": (
            f"cd {tests_dir} && pytest -v",
            "Running all tests"
        ),
        "coverage": (
            f"cd {tests_dir} && pytest --cov=backend.handlers.materials --cov-report=html --cov-report=term",
            "Running tests with coverage report"
        ),
        "quick": (
            f"cd {tests_dir} && pytest -x",
            "Running tests (stop on first failure)"
        ),
    }
    
    if test_target not in commands:
        print(f"Unknown test target: {test_target}")
        print(f"Available targets: {', '.join(commands.keys())}")
        sys.exit(1)
    
    cmd, description = commands[test_target]
    success = run_command(cmd, description)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

