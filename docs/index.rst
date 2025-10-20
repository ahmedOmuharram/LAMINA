LAMINA
======================================================
LAMINA (**L**LM-**A**ssisted **M**aterial **IN**formatics and **A**nalysis) is a system that uses Large Language Models (LLMs) to assist in material science research.

Project Description
===================

This thesis project explores the integration of the Kani framework with Large Language Models (LLMs) to enhance materials science research capabilities. The work demonstrates how AI-accessible tools can be used by LLMs to perform complex scientific calculations, data analysis, and research tasks.

System Architecture
===================

The project implements a modular architecture centered around the Kani framework with four main handler modules:

Handler Modules
---------------

1. **Materials Handler**: Materials Project database integration
2. **CALPHAD Handler**: Thermodynamic phase diagram calculations  
3. **Electrochemistry Handler**: Battery and electrode analysis
4. **Search Handler**: Scientific literature and web search

Data Sources
============

- **Materials Project Database**: >200,000 DFT-calculated materials
- **CALPHAD Databases**: TDB files for thermodynamic calculations
- **SearXNG Search Engine**: Scientific literature metasearch

For detailed function documentation, see :doc:`api/index`.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api/index
   references

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
