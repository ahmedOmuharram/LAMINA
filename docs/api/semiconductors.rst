Semiconductors Handler
======================

The Semiconductors handler provides AI functions for analyzing semiconductor materials, octahedral distortions, magnetic properties, defect stability, doping effects, and phase transitions.

Overview
--------

This handler enables analysis of:

1. **Octahedral Distortions**: Analyze geometric distortions in octahedral coordination
2. **Magnetic Properties**: Retrieve and analyze magnetic properties of semiconductors
3. **Magnetic Material Comparison**: Compare magnetic properties between materials
4. **Defect Stability**: Analyze stability of defects in crystal structures
5. **Doping Site Preference**: Determine preferred doping sites in crystal structures
6. **Phase Transitions**: Analyze structural phase transitions
7. **Doped Variants**: Search for doped variants of materials in same phase
8. **Defect Site Prediction**: Predict preferred defect and dopant sites
9. **Magnetization Effects**: Evaluate dopant effects on saturation magnetization

Functions
---------

.. _analyze_octahedral_distortion_in_material:

analyze_octahedral_distortion_in_material
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Analyze octahedral distortions in a material's crystal structure. Useful for understanding structural distortions in perovskites and related materials.

.. _get_magnetic_properties:

get_magnetic_properties
^^^^^^^^^^^^^^^^^^^^^^^

Retrieve magnetic properties (magnetization, ordering) for a semiconductor material from Materials Project.

.. _compare_magnetic_materials:

compare_magnetic_materials
^^^^^^^^^^^^^^^^^^^^^^^^^^

Compare magnetic properties between two semiconductor materials.

.. _analyze_defect_stability:

analyze_defect_stability
^^^^^^^^^^^^^^^^^^^^^^^^

Analyze thermodynamic stability of defects in a crystal structure.

.. _analyze_doping_site_preference:

analyze_doping_site_preference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Determine preferred doping sites for a dopant element in a host crystal structure.

.. _analyze_phase_transition_structures:

analyze_phase_transition_structures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Analyze crystal structures before and after phase transitions.

.. _search_same_phase_doped_variants:

search_same_phase_doped_variants
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Search for doped variants of a material that maintain the same crystal phase/spacegroup.

.. _predict_defect_site_preference:

predict_defect_site_preference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Predict preferred sites for defects and dopants in crystal structures using geometric and chemical criteria.

.. _evaluate_dopant_effect_on_Ms:

evaluate_dopant_effect_on_Ms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Evaluate the effect of dopant elements on saturation magnetization (Ms).

Citations
---------

All Semiconductors functions cite Materials Project DFT data and crystallographic databases.

