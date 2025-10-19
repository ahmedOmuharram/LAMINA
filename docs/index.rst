Welcome to Kani-Enhanced Materials Science Documentation
========================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   overview
   installation
   quickstart
   handlers/index
   api/index
   examples
   testing
   benchmark
   contributing

Overview
========

This project demonstrates the integration of the Kani framework with Large Language Models (LLMs) to enhance materials science research capabilities. The system provides AI-accessible tools for:

- **Materials Discovery**: Search and analyze materials from the Materials Project database
- **Phase Diagrams**: Generate CALPHAD-based thermodynamic phase diagrams
- **Battery Research**: Analyze electrode materials and electrochemical properties
- **Literature Search**: Access scientific literature through SearXNG metasearch
- **Data Integration**: Combine multiple data sources for comprehensive analysis

Key Features
============

- 🤖 **AI-First Design**: All tools are designed for LLM consumption via Kani
- 🔬 **Materials Science Focus**: Specialized handlers for materials research
- 📊 **Comprehensive Testing**: Automated validation against scientific benchmarks
- 🌐 **Multi-Source Data**: Integration of DFT, CALPHAD, and literature data
- 📈 **Visualization**: Interactive plots and phase diagrams
- 🔍 **Verification**: Built-in web search for fact-checking and validation

Architecture
============

The system is built around the Kani framework, which provides:

- **MCP Server**: Model Context Protocol server for tool integration
- **Handler Modules**: Specialized modules for different research domains
- **AI Functions**: Decorated functions accessible to LLMs
- **Data Sources**: Materials Project API, CALPHAD databases, SearXNG search

.. image:: _static/architecture_diagram.png
   :alt: System Architecture
   :width: 600px
   :align: center

Quick Start
===========

1. **Install Dependencies**:
   .. code-block:: bash
   
      pip install -r requirements.txt
      export MP_API_KEY="your_materials_project_key"

2. **Run the MCP Server**:
   .. code-block:: bash
   
      python -m mcp_materials_project

3. **Test the System**:
   .. code-block:: bash
   
      python run_all_handler_tests.py

4. **View Results**:
   .. code-block:: bash
   
      open test_results_report.html

Benchmark Results
=================

The system has been validated against scientific benchmarks with the following results:

- **CALPHAD Tests**: 80-100% accuracy on phase diagram calculations
- **Battery Tests**: 60-80% accuracy on electrochemical predictions
- **Overall Performance**: 70-90% success rate across all test cases

See the :doc:`benchmark` section for detailed results and analysis.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
