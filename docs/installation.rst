Installation Guide
==================

This guide will help you set up the Kani-Enhanced Materials Science system on your machine.

Prerequisites
=============

System Requirements
-------------------

- **Python**: 3.8 or higher (3.10+ recommendedadd)
- **Operating System**: Linux, macOS, or Windows
- **Memory**: 4GB RAM minimum (8GB+ recommended)
- **Disk Space**: 2GB for dependencies and data
- **Internet**: Required for Materials Project API and SearXNG search

Required Accounts
-----------------

1. **Materials Project API Key**
   
   - Register at https://next-gen.materialsproject.org/
   - Navigate to your account settings
   - Generate an API key
   - Save the key for later use

Installation Steps
==================

1. Clone the Repository
-----------------------

.. code-block:: bash

   git clone https://github.com/yourusername/thesis.git
   cd thesis

2. Create Virtual Environment
------------------------------

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

3. Install Dependencies
-----------------------

.. code-block:: bash

   pip install -r requirements.txt

Core dependencies include:

- **kani**: AI framework for LLM integration
- **mp-api**: Materials Project API client
- **pycalphad**: CALPHAD thermodynamic calculations
- **pymatgen**: Materials analysis toolkit
- **plotly**: Interactive plotting
- **matplotlib**: Static plotting
- **numpy**: Numerical computing
- **pandas**: Data manipulation

4. Configure Environment Variables
-----------------------------------

Create a `.env` file in the project root:

.. code-block:: bash

   # Materials Project API Key
   MP_API_KEY=your_materials_project_api_key_here
   
   # OpenAI API Key (for Kani)
   OPENAI_API_KEY=your_openai_api_key_here
   
   # Optional: SearXNG instance URL
   SEARXNG_URL=http://localhost:8080

5. Verify Installation
-----------------------

Run the verification script:

.. code-block:: bash

   python verify_test_setup.py

This will check:

- Python version
- Required packages
- API key configuration
- TDB files availability
- Output directories

Expected output:

.. code-block:: text

   ✓ Python version: 3.10.0
   ✓ All required packages installed
   ✓ MP_API_KEY configured
   ✓ OPENAI_API_KEY configured
   ✓ TDB files found: 2
   ✓ Output directories created
   
   All checks passed! System is ready.

Optional Components
===================

SearXNG Search Engine
---------------------

For local scientific literature search:

.. code-block:: bash

   cd searxng-docker
   docker-compose up -d

This will start a local SearXNG instance at http://localhost:8080

CALPHAD Databases
-----------------

The system includes sample TDB files in the `tdbs/` directory:

- `COST507.tdb`: Al-Zn and other binary systems
- `mc_al_v2037_pycal.tdb`: Aluminum-based systems

Additional TDB files can be added to this directory.

Testing the Installation
=========================

Quick Test
----------

Run a simple CALPHAD test (no API key needed):

.. code-block:: bash

   python test_calphad_questions.py

This should generate phase diagrams in the `interactive_plots/` directory.

Full Test Suite
---------------

Run all tests:

.. code-block:: bash

   python run_all_handler_tests.py

This will:

1. Test CALPHAD phase diagram calculations
2. Test battery/electrochemistry analysis
3. Generate comprehensive HTML report
4. Create interactive plots

View results:

.. code-block:: bash

   open test_results_report.html

Troubleshooting
===============

Common Issues
-------------

**Issue**: "MP_API_KEY not found"

**Solution**: 
   - Ensure `.env` file exists in project root
   - Verify API key is correct
   - Try: `export MP_API_KEY="your_key"` (temporary)

**Issue**: "Module not found: pycalphad"

**Solution**:
   - Activate virtual environment: `source .venv/bin/activate`
   - Reinstall: `pip install -r requirements.txt`

**Issue**: "TDB file not found"

**Solution**:
   - Verify `tdbs/` directory exists
   - Check TDB files are present: `ls tdbs/`
   - Download missing TDB files if needed

**Issue**: "SearXNG connection failed"

**Solution**:
   - Start SearXNG: `cd searxng-docker && docker-compose up -d`
   - Check status: `docker-compose ps`
   - Verify URL in `.env` file

**Issue**: "Kaleido not found" (for plot export)

**Solution**:
   - Install Kaleido: `pip install kaleido`
   - Or use browser-based export instead

Platform-Specific Notes
=======================

macOS
-----

- May need to install Xcode Command Line Tools: `xcode-select --install`
- For M1/M2 Macs, use native Python 3.10+

Linux
-----

- May need development packages: `sudo apt-get install python3-dev build-essential`
- For Ubuntu/Debian: `sudo apt-get install libgraphviz-dev`

Windows
-------

- Use PowerShell or Git Bash
- Activate venv: `.venv\Scripts\activate`
- May need Visual C++ Build Tools for some packages

Development Setup
=================

For development work:

.. code-block:: bash

   # Install development dependencies
   pip install -r requirements-dev.txt
   
   # Install pre-commit hooks
   pre-commit install
   
   # Run tests with coverage
   pytest --cov=mcp_materials_project tests/

IDE Configuration
-----------------

**VS Code**:

1. Install Python extension
2. Select interpreter: `.venv/bin/python`
3. Enable type checking in settings

**PyCharm**:

1. Set project interpreter to `.venv`
2. Mark `mcp_materials_project` as sources root
3. Enable type hints

Next Steps
==========

After installation:

1. Read the :doc:`quickstart` guide
2. Explore :doc:`examples` for usage patterns
3. Review :doc:`api/index` for detailed API documentation
4. Run the test suite to verify functionality

For questions or issues, see the project README or open an issue on GitHub.
