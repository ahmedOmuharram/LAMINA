API Documentation
=================

This section documents all AI-accessible functions in the system.

.. toctree::
   :maxdepth: 2
   
   materials
   calphad
   electrochemistry
   search
   magnets
   semiconductors
   solutes
   superconductors
   alloys

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

Functions for generating phase diagrams, performing thermodynamic calculations, and verifying metallurgical claims.

**Visualization Functions:**

- :ref:`plot_binary_phase_diagram`
- :ref:`plot_composition_temperature`
- :ref:`analyze_last_generated_plot`

**Calculation Functions:**

- :ref:`calculate_equilibrium_at_point`
- :ref:`calculate_phase_fractions_vs_temperature`
- :ref:`analyze_phase_fraction_trend`

**Verification Functions:**

- :ref:`verify_phase_formation_across_composition`
- :ref:`sweep_microstructure_claim_over_region`
- :ref:`fact_check_microstructure_claim`

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

Magnets Handler
---------------

Functions for analyzing magnetic materials and permanent magnets.

- :ref:`assess_magnet_strength_with_doping`
- :ref:`get_phase_and_magnetic_ordering`
- :ref:`estimate_permanent_magnet_properties`
- :ref:`calculate_magnet_pull_force`
- :ref:`assess_doping_effect_on_saturation_magnetization`
- :ref:`compare_dopants_for_saturation_magnetization`
- :ref:`get_saturation_magnetization_detailed`
- :ref:`search_doped_magnetic_materials`

Semiconductors Handler
----------------------

Functions for analyzing semiconductor materials, defects, and doping.

- :ref:`analyze_octahedral_distortion_in_material`
- :ref:`get_magnetic_properties`
- :ref:`compare_magnetic_materials`
- :ref:`analyze_defect_stability`
- :ref:`analyze_doping_site_preference`
- :ref:`analyze_phase_transition_structures`
- :ref:`search_same_phase_doped_variants`
- :ref:`predict_defect_site_preference`
- :ref:`evaluate_dopant_effect_on_Ms`

Solutes Handler
---------------

Functions for analyzing solute effects on lattice parameters and solid solution behavior.

- :ref:`analyze_solute_lattice_effect`
- :ref:`compare_solute_lattice_effects`
- :ref:`calculate_solute_lattice_effect`
- :ref:`get_solute_reference_data`

Superconductors Handler
-----------------------

Functions for analyzing superconducting materials, particularly cuprates.

- :ref:`analyze_cuprate_octahedral_stability`

Alloys Handler
--------------

Functions for analyzing metallic alloys, surface diffusion, and mechanical properties.

- :ref:`estimate_surface_diffusion_barrier`
- :ref:`assess_phase_strength_and_stiffness_claims`
