"""
API utility functions for handling Materials Project API errors and formatting.

This module provides utilities for working with Materials Project API responses
and error handling.
"""

import re
from typing import Dict


def format_field_error(error: Exception) -> str:
    """
    Format API error messages about invalid fields to include available fields and mappings.
    
    This function parses Materials Project API error messages about invalid field names
    and provides helpful suggestions including:
    - Lists of invalid fields requested
    - Available fields grouped by category with descriptions
    - Common field name corrections
    
    Args:
        error: Exception raised by the Materials Project API
        
    Returns:
        Formatted error string with helpful field information
    """
    error_str = str(error)
    
    # Field descriptions for better AI understanding
    field_descriptions = {
        # Basic
        'material_id': 'Materials Project ID',
        'formula_pretty': 'Chemical formula',
        'formula_anonymous': 'Stoichiometry pattern',
        'chemsys': 'Chemical system',
        'elements': 'List of element symbols',
        'num_elements': 'Number of elements',
        'composition': 'Full composition dict',
        'composition_reduced': 'Reduced composition',
        'nsites': 'Number of sites in unit cell',
        # Structural
        'structure': 'Crystal structure',
        'volume': 'Unit cell volume in Å³',
        'density': 'Density in g/cm³',
        'density_atomic': 'Atomic density',
        'symmetry': 'Symmetry information',
        'crystal_system': 'Crystal system',
        'spacegroup_number': 'Spacegroup number',
        'spacegroup_symbol': 'Spacegroup symbol',
        # Energetic
        'energy_per_atom': 'Total energy in eV/atom',
        'formation_energy_per_atom': 'Formation energy in eV/atom',
        'energy_above_hull': 'Stability indicator in eV/atom',
        'is_stable': 'On convex hull (boolean)',
        'uncorrected_energy_per_atom': 'Uncorrected energy in eV/atom',
        'equilibrium_reaction_energy_per_atom': 'Equilibrium reaction energy',
        'decomposes_to': 'Products of decomposition',
        # Electronic
        'band_gap': 'Band gap in eV',
        'cbm': 'Conduction band minimum in eV',
        'vbm': 'Valence band maximum in eV',
        'efermi': 'Fermi energy in eV',
        'is_gap_direct': 'Direct vs indirect gap (boolean)',
        'is_metal': 'Metallic behavior (boolean)',
        'bandstructure': 'Full band structure data',
        'dos': 'Density of states data',
        'dos_energy_up': 'Spin-up DOS',
        'dos_energy_down': 'Spin-down DOS',
        # Magnetic
        'is_magnetic': 'Magnetic ordering (boolean)',
        'ordering': 'Magnetic ordering type',
        'total_magnetization': 'Total magnetization in μ_B/atom',
        'total_magnetization_normalized_vol': 'Magnetization per volume',
        'total_magnetization_normalized_formula_units': 'Magnetization per formula unit',
        'num_magnetic_sites': 'Number of magnetic sites',
        'num_unique_magnetic_sites': 'Number of unique magnetic sites',
        'types_of_magnetic_species': 'List of magnetic element types',
        # Mechanical
        'bulk_modulus': 'Bulk modulus in GPa',
        'shear_modulus': 'Shear modulus in GPa',
        'universal_anisotropy': 'Universal elastic anisotropy index',
        'homogeneous_poisson': 'Poisson\'s ratio',
        # Dielectric
        'e_total': 'Total dielectric constant',
        'e_ionic': 'Ionic dielectric constant',
        'e_electronic': 'Electronic dielectric constant',
        'n': 'Refractive index',
        'piezoelectric_modulus': 'Piezoelectric modulus in C/m²',
        'e_ij_max': 'Maximum dielectric tensor component',
        # Surface
        'weighted_surface_energy': 'Weighted surface energy in J/m²',
        'weighted_surface_energy_EV_PER_ANG2': 'Surface energy in eV/Å²',
        'weighted_work_function': 'Weighted work function in eV',
        'surface_anisotropy': 'Surface energy anisotropy',
        'shape_factor': 'Wulff shape factor',
        'has_reconstructed': 'Surface reconstruction (boolean)',
        # Metadata
        'builder_meta': 'Builder metadata',
        'deprecated': 'Deprecated status (boolean)',
        'deprecation_reasons': 'Reasons for deprecation',
        'last_updated': 'Last update timestamp',
        'origins': 'Data origin information',
        'warnings': 'Calculation warnings',
        'task_ids': 'Associated task IDs',
        'theoretical': 'Theoretical vs experimental (boolean)',
        'has_props': 'Available calculated properties',
        'possible_species': 'Possible species in material',
        'database_Ids': 'External database IDs',
        'property_name': 'Property name',
        'xas': 'X-ray absorption spectroscopy data',
        'grain_boundaries': 'Grain boundary data',
        'es_source_calc_id': 'Electronic structure source calc ID'
    }
    
    # Check if this is an invalid fields error
    if "invalid fields" in error_str.lower() or "invalid field" in error_str.lower():
        # Try to extract available fields from the error message
        available_fields_match = re.search(r'Available fields?:\s*\[(.*?)\]', error_str)
        invalid_fields_match = re.search(r'invalid fields? requested:\s*\[(.*?)\]', error_str)
        
        formatted_error = "Invalid field names requested. "
        
        if invalid_fields_match:
            invalid_fields = [f.strip().strip("'\"") for f in invalid_fields_match.group(1).split(",")]
            formatted_error += f"Requested invalid fields: {invalid_fields}. "
        
        # Common field name corrections - users should use the actual API field names
        field_corrections = {
            "refractive_index": "n",
            "formula": "formula_pretty",
            "magnetic_ordering": "ordering"
        }
        
        if invalid_fields_match:
            invalid_fields = [f.strip().strip("'\"").strip() for f in invalid_fields_match.group(1).split(",")]
            suggestions = []
            for field in invalid_fields:
                if field in field_corrections:
                    suggestions.append(f"'{field}' should be '{field_corrections[field]}'")
            if suggestions:
                formatted_error += "Field name corrections: " + "; ".join(suggestions) + ". "
        
        if available_fields_match:
            available_fields_str = available_fields_match.group(1)
            available_fields = [f.strip().strip("'\"") for f in available_fields_str.split(",")]
            
            # Group fields by category for better readability
            basic_fields = ['material_id', 'formula_pretty', 'formula_anonymous', 'chemsys', 'elements', 'num_elements', 'composition', 'composition_reduced', 'nsites']
            structural = ['structure', 'volume', 'density', 'density_atomic', 'symmetry', 'crystal_system', 'spacegroup_number', 'spacegroup_symbol']
            energetic = ['energy_per_atom', 'formation_energy_per_atom', 'energy_above_hull', 'is_stable', 'uncorrected_energy_per_atom', 'equilibrium_reaction_energy_per_atom', 'decomposes_to']
            electronic = ['band_gap', 'cbm', 'vbm', 'efermi', 'is_gap_direct', 'is_metal', 'bandstructure', 'dos', 'dos_energy_up', 'dos_energy_down']
            magnetic = ['is_magnetic', 'ordering', 'total_magnetization', 'total_magnetization_normalized_vol', 'total_magnetization_normalized_formula_units', 'num_magnetic_sites', 'num_unique_magnetic_sites', 'types_of_magnetic_species']
            mechanical = ['bulk_modulus', 'shear_modulus', 'universal_anisotropy', 'homogeneous_poisson']
            dielectric = ['e_total', 'e_ionic', 'e_electronic', 'n', 'piezoelectric_modulus', 'e_ij_max']
            surface = ['weighted_surface_energy', 'weighted_surface_energy_EV_PER_ANG2', 'weighted_work_function', 'surface_anisotropy', 'shape_factor', 'has_reconstructed']
            metadata = ['builder_meta', 'deprecated', 'deprecation_reasons', 'last_updated', 'origins', 'warnings', 'task_ids', 'theoretical', 'has_props', 'possible_species', 'database_Ids', 'property_name', 'xas', 'grain_boundaries', 'es_source_calc_id']
            
            all_categories = {
                'Basic': [f for f in available_fields if f in basic_fields],
                'Structural': [f for f in available_fields if f in structural],
                'Energetic': [f for f in available_fields if f in energetic],
                'Electronic': [f for f in available_fields if f in electronic],
                'Magnetic': [f for f in available_fields if f in magnetic],
                'Mechanical': [f for f in available_fields if f in mechanical],
                'Dielectric': [f for f in available_fields if f in dielectric],
                'Surface': [f for f in available_fields if f in surface],
                'Metadata': [f for f in available_fields if f in metadata]
            }
            
            # Add any fields not in categories
            categorized_fields = set()
            for cat_fields in all_categories.values():
                categorized_fields.update(cat_fields)
            uncategorized = [f for f in available_fields if f not in categorized_fields]
            
            formatted_error += "\nAvailable fields by category:\n"
            for category, fields_list in all_categories.items():
                if fields_list:
                    # Format each field with its description
                    field_list_with_desc = []
                    for field in fields_list:
                        desc = field_descriptions.get(field, '')
                        if desc:
                            field_list_with_desc.append(f"'{field}' ({desc})")
                        else:
                            field_list_with_desc.append(f"'{field}'")
                    formatted_error += f"- {category}: {', '.join(field_list_with_desc)}\n"
            
            if uncategorized:
                field_list_with_desc = []
                for field in uncategorized:
                    desc = field_descriptions.get(field, '')
                    if desc:
                        field_list_with_desc.append(f"'{field}' ({desc})")
                    else:
                        field_list_with_desc.append(f"'{field}'")
                formatted_error += f"- Other: {', '.join(field_list_with_desc)}\n"
            
            formatted_error += "\nNote: Use actual API field names:\n"
            formatted_error += "- For refractive index, use 'n' (not 'refractive_index')\n"
            formatted_error += "- For formula, use 'formula_pretty' (not 'formula')\n"
            formatted_error += "- For magnetic ordering, use 'ordering' (not 'magnetic_ordering')\n"
        
        return formatted_error
    
    # Not a field error, return original
    return error_str

