Superconductors Handler
=======================

The Superconductors handler provides AI functions for analyzing superconducting materials, focusing on cuprate superconductors and octahedral stability.

Overview
--------

This handler provides specialized analysis for:

1. **Cuprate Superconductors**: Analyze octahedral stability in cuprate materials
2. **Structural Distortions**: Evaluate geometric distortions affecting superconductivity
3. **Oxygen Coordination**: Analyze copper-oxygen octahedral environments

Function
--------

.. _analyze_cuprate_octahedral_stability:

analyze_cuprate_octahedral_stability
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Analyze octahedral stability and distortions in cuprate superconductors. Evaluates copper-oxygen coordination geometry and its impact on superconducting properties.

**When to Use:**

- Analyzing cuprate superconductor structures
- Understanding octahedral distortions in high-Tc materials
- Evaluating structural factors affecting superconductivity

**Parameters:**

- ``material_id`` (str, required): Materials Project ID of cuprate material
- ``cu_site_index`` (int, optional): Specific copper site index to analyze. If not provided, analyzes all Cu sites.

**Returns:**

Dictionary containing:

- ``material_id``: Materials Project ID
- ``formula``: Chemical formula
- ``cu_sites_analyzed``: Number of Cu sites analyzed
- ``octahedral_distortions``: List of distortion analyses for each site with:
  
  - ``site_index``: Site index in structure
  - ``coordination_number``: Number of coordinating oxygens
  - ``distortion_magnitude``: Quantitative distortion measure
  - ``bond_lengths``: Cu-O bond lengths (Å)
  - ``bond_angle_variance``: Variance in O-Cu-O angles
  - ``stability_assessment``: Qualitative stability assessment

- ``average_distortion``: Average distortion across all sites
- ``most_distorted_site``: Site with maximum distortion
- ``structural_quality``: Overall structural quality assessment

**Example:**

.. code-block:: python

   # Analyze octahedral stability in YBCO
   result = await handler.analyze_cuprate_octahedral_stability(
       material_id="mp-12345"  # YBa2Cu3O7
   )

**Technical Details:**

- Analyzes copper-oxygen octahedral coordination
- Calculates bond length variations
- Evaluates bond angle distortions
- Assesses impact on superconducting properties

**Physical Significance:**

- Octahedral distortions affect hole carrier density
- Cu-O bond lengths influence electronic structure
- Distortions can suppress or enhance Tc
- Critical for understanding cuprate superconductivity

Citations
---------

Superconductor functions cite Materials Project DFT data and superconductivity literature.

Notes
-----

- Specialized for cuprate high-temperature superconductors
- Analyzes octahedral CuO6 coordination environments
- Bond lengths in Ångströms (Å)
- Distortion measures dimensionless
- Structural quality correlates with superconducting properties

