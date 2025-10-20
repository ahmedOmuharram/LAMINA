CALPHAD Handler
===============

Functions for generating phase diagrams and performing thermodynamic calculations using CALPHAD databases.

.. _plot_binary_phase_diagram:

plot_binary_phase_diagram
--------------------------

Generate a binary phase diagram for a chemical system using CALPHAD data.

**Parameters:**

- ``system`` (str): Chemical system (e.g., 'Al-Zn', 'AlZn', 'aluminum-zinc')
- ``min_temperature`` (float, optional): Minimum temperature in Kelvin (default: auto)
- ``max_temperature`` (float, optional): Maximum temperature in Kelvin (default: auto)
- ``composition_step`` (float, optional): Composition step size 0-1 (default: 0.02)
- ``figure_width`` (float, optional): Figure width in inches (default: 9)
- ``figure_height`` (float, optional): Figure height in inches (default: 6)

**Returns:** Success message with key findings (phases, eutectic points, melting points)

**Side Effects:** Saves PNG image to interactive_plots/ directory

.. _plot_composition_temperature:

plot_composition_temperature
-----------------------------

Plot phase stability vs temperature for a specific composition.

**Parameters:**

- ``composition`` (str): Composition (e.g., 'Al20Zn80', 'Al0.2Zn0.8', 'pure Al')
- ``min_temperature`` (float, optional): Minimum temperature in Kelvin (default: 300)
- ``max_temperature`` (float, optional): Maximum temperature in Kelvin (default: 1000)
- ``temperature_step`` (float, optional): Temperature step in Kelvin (default: 5)
- ``figure_width`` (float, optional): Figure width in inches (default: 10)
- ``figure_height`` (float, optional): Figure height in inches (default: 6)

**Returns:** Success message with phase stability analysis

**Side Effects:** Saves interactive HTML and PNG plots to interactive_plots/ directory

.. _calculate_equilibrium_at_point:

calculate_equilibrium_at_point
-------------------------------

Calculate thermodynamic equilibrium at a single temperature and composition.

**Parameters:**

- ``composition`` (str): Composition (e.g., 'Al20Zn80', 'Al0.2Zn0.8')
- ``temperature`` (float): Temperature in Kelvin
- ``pressure`` (float, optional): Pressure in Pa (default: 101325)

**Returns:** Dictionary containing stable phases, phase fractions, and thermodynamic properties

.. _calculate_phase_fractions_vs_temperature:

calculate_phase_fractions_vs_temperature
-----------------------------------------

Calculate how phase fractions change with temperature for a specific composition.

**Parameters:**

- ``composition`` (str): Composition (e.g., 'Al30Si55C15')
- ``min_temperature`` (float, optional): Minimum temperature in Kelvin (default: 300)
- ``max_temperature`` (float, optional): Maximum temperature in Kelvin (default: 1500)
- ``temperature_step`` (float, optional): Temperature step in Kelvin (default: 10)

**Returns:** Dictionary containing temperature points and phase fractions at each temperature

.. _analyze_phase_fraction_trend:

analyze_phase_fraction_trend
-----------------------------

Analyze if a phase increases or decreases with cooling.

**Parameters:**

- ``composition`` (str): Composition (e.g., 'Al30Si55C15')
- ``phase_name`` (str): Phase name to analyze (e.g., 'carbide', 'precipitate')
- ``min_temperature`` (float, optional): Minimum temperature in Kelvin (default: 300)
- ``max_temperature`` (float, optional): Maximum temperature in Kelvin (default: 1500)

**Returns:** Dictionary containing trend analysis (increasing/decreasing/stable)

.. _analyze_last_generated_plot:

analyze_last_generated_plot
----------------------------

Analyze the most recently generated phase diagram or composition-temperature plot.

**Parameters:** None

**Returns:** Detailed analysis of the last generated plot including phases, transitions, and key features

.. _list_available_systems:

list_available_systems
----------------------

List all chemical systems available in the loaded CALPHAD databases.

**Parameters:** None

**Returns:** List of available chemical systems and their elements
