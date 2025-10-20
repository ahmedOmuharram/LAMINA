API Documentation
=================

This section documents all AI-accessible functions in the system.

.. toctree::
   :maxdepth: 2
   
   materials
   calphad
   electrochemistry
   search

Handler Modules
===============

Materials Handler
-----------------

Functions for searching and retrieving material information from the Materials Project database.

- :ref:`get_material`
- :ref:`get_material_by_char`
- :ref:`get_material_details_by_ids`

CALPHAD Handler
---------------

Functions for generating phase diagrams and performing thermodynamic calculations.

- :ref:`plot_binary_phase_diagram`
- :ref:`plot_composition_temperature`
- :ref:`calculate_equilibrium_at_point`
- :ref:`calculate_phase_fractions_vs_temperature`
- :ref:`analyze_phase_fraction_trend`
- :ref:`analyze_last_generated_plot`
- :ref:`list_available_systems`

Electrochemistry Handler
-------------------------

Functions for battery electrode analysis and electrochemical properties.

- :ref:`search_battery_electrodes`
- :ref:`calculate_voltage_from_formation_energy`
- :ref:`compare_electrode_materials`
- :ref:`check_composition_stability`
- :ref:`analyze_anode_viability`
- :ref:`get_voltage_profile`
- :ref:`analyze_lithiation_mechanism`

Search Handler
--------------

Functions for searching the web and scientific literature.

- :ref:`search_web`
- :ref:`get_search_engines`
