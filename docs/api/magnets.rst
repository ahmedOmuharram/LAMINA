Magnets Handler
===============

The Magnets handler provides AI functions for analyzing magnetic materials, permanent magnets, and the effects of doping on magnetic properties using Materials Project DFT data.

Overview
--------

This handler enables:

1. **Magnet Strength Assessment**: Analyze permanent magnet properties with doping effects
2. **Phase and Magnetic Ordering**: Determine magnetic ordering and phases
3. **Permanent Magnet Properties**: Estimate energy product and magnet performance
4. **Pull Force Calculation**: Calculate magnetic pull force for practical applications
5. **Doping Analysis**: Assess effects of doping on saturation magnetization
6. **Material Search**: Find doped magnetic variants

Functions
---------

.. _assess_magnet_strength_with_doping:

assess_magnet_strength_with_doping
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Analyze permanent magnet strength considering doping effects. Evaluates magnetization, anisotropy, and estimates maximum energy product (BH)max.

.. _get_phase_and_magnetic_ordering:

get_phase_and_magnetic_ordering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Determine the phase and magnetic ordering of a material. Returns crystal structure, space group, and magnetic ordering (ferromagnetic, antiferromagnetic, ferrimagnetic, or paramagnetic).

.. _estimate_permanent_magnet_properties:

estimate_permanent_magnet_properties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Estimate permanent magnet properties including maximum energy product, coercivity, and remanence. Provides comprehensive assessment of magnet performance.

.. _calculate_magnet_pull_force:

calculate_magnet_pull_force
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Calculate the pull force of a permanent magnet for practical applications. Estimates force based on magnet geometry, material properties, and gap distance.

.. _assess_doping_effect_on_saturation_magnetization:

assess_doping_effect_on_saturation_magnetization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Assess the effect of doping on saturation magnetization. Compares undoped and doped materials to quantify magnetization changes.

.. _compare_dopants_for_saturation_magnetization:

compare_dopants_for_saturation_magnetization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Compare multiple dopants and their effects on saturation magnetization. Ranks dopants by their impact on magnetic properties.

.. _get_saturation_magnetization_detailed:

get_saturation_magnetization_detailed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Get detailed saturation magnetization data for a material including temperature dependence and magnetic moment contributions.

.. _search_doped_magnetic_materials:

search_doped_magnetic_materials
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Search for magnetically doped materials in the Materials Project database. Find materials with specific magnetic properties and doping configurations.

Citations
---------

All Magnets functions cite Materials Project DFT data and relevant magnetic property databases.

