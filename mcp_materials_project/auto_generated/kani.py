from __future__ import annotations

from typing import Any, Optional, List, Tuple, Dict, Annotated
from kani import ai_function, AIParam

class GeneratedKaniTools:
    @ai_function(desc="Convert an english worded chemical name to a list of symbols. The name can be a chemical formula (e.g. Iron Oxide 2 -> Fe2O3), an element (e.g. Oxygen -> O), or a chemical system (e.g. Lithium-Iron-* -> Li-Fe-*). Wildcards are supported. It must be worded and not a chemical formula. If it is a chemical formula, use the other tools directly.", auto_truncate=128000)
    async def convert_name_to_symbols(self, name: Annotated[str, AIParam(desc="The english worded chemical name to convert. The name can be a chemical formula (e.g. Iron Oxide 2 -> Fe2O3), an element (e.g. Oxygen -> O), or a chemical system (e.g. Lithium-Iron-* -> Li-Fe-*). Wildcards are supported. It must be worded and not a chemical formula. If it is a chemical formula, use the other tools directly.")]) -> str:
        """Convert an english worded chemical name to a list of symbols. The name can be a chemical formula (e.g. Iron Oxide 2 -> Fe2O3), an element (e.g. Oxygen -> O), or a chemical system (e.g. Lithium-Iron-* -> Li-Fe-*). Wildcards are supported. It must be worded and not a chemical formula. If it is a chemical formula, use the other tools directly."""
        _args = {}
        _args["name"] = name
        _result = await self._proxy.call_tool("convert_name_to_symbols", _args)
        try:
            self.recent_tool_outputs.append({"tool": "convert_name_to_symbols", "args": _args, "result": _result})
        except Exception:
            pass
        try:
            # Prefer pretty JSON if possible; fall back to str()
            import json as _json
            return _json.dumps(_result, ensure_ascii=False)
        except Exception:
            return str(_result)

    @ai_function(desc="Query materials by their chemical system and return their material IDs and formula. At least one of chemsys, formula, or element must be provided. Must be in symbols, so use the convert_name_to_symbols tool to convert the name to symbols.", auto_truncate=128000)
    async def get_material(self, chemsys: Annotated[str, AIParam(desc="Chemical system(s) or comma-separated list (e.g., Li-Fe-O,Si-*). Must be in symbols, so use the convert_name_to_symbols tool to convert the name to symbols. Example:")] = None, formula: Annotated[str, AIParam(desc="Formula(s), anonymized formula, or wildcard(s) (e.g., Li2FeO3,Fe2O3,Fe*O*). Must be in symbols, so use the convert_name_to_symbols tool to convert the name to symbols.")] = None, element: Annotated[str, AIParam(desc="Element(s) or comma-separated list (e.g., Li,Fe,O). Must be in symbols, so use the convert_name_to_symbols tool to convert the name to symbols.")] = None, page: Annotated[int, AIParam(desc="Page number (default 1).")] = 1, per_page: Annotated[int, AIParam(desc="Items per page (max 10; default 10).")] = 10) -> str:
        """Query materials by their chemical system and return their material IDs and formula. At least one of chemsys, formula, or element must be provided. Must be in symbols, so use the convert_name_to_symbols tool to convert the name to symbols."""
        _args = {}
        if chemsys is not None:
            _args["chemsys"] = chemsys
        if formula is not None:
            _args["formula"] = formula
        if element is not None:
            _args["element"] = element
        _args["page"] = page if page is not None else 1
        _args["per_page"] = per_page if per_page is not None else 10
        _result = await self._proxy.call_tool("get_material", _args)
        try:
            self.recent_tool_outputs.append({"tool": "get_material", "args": _args, "result": _result})
        except Exception:
            pass
        try:
            # Prefer pretty JSON if possible; fall back to str()
            import json as _json
            return _json.dumps(_result, ensure_ascii=False)
        except Exception:
            return str(_result)

    @ai_function(desc="Fetch a material id and formula by a characteristic of the material", auto_truncate=128000)
    async def get_material_by_char(self, band_gap: Annotated[List[float], AIParam(desc="Min,max range of band gap in eV (e.g., [1.2, 3.0]). Tuple of floats, and both values are required.")] = None, crystal_system: Annotated[str, AIParam(desc="Crystal system of material. Available options are 'Triclinic', 'Monoclinic', 'Orthorhombic', 'Tetragonal', 'Trigonal', 'Hexagonal' or 'Cubic'.")] = None, density: Annotated[List[float], AIParam(desc="Min,max density range. Tuple of floats, and both values are required.")] = None, e_electronic: Annotated[List[float], AIParam(desc="Min,max electronic dielectric constant. Tuple of floats, and both values are required.")] = None, e_ionic: Annotated[List[float], AIParam(desc="Min,max ionic dielectric constant. Tuple of floats, and both values are required.")] = None, e_total: Annotated[List[float], AIParam(desc="Min,max total dielectric constant. Tuple of floats, and both values are required.")] = None, efermi: Annotated[List[float], AIParam(desc="Min,max fermi energy in eV. Tuple of floats, and both values are required.")] = None, elastic_anisotropy: Annotated[List[float], AIParam(desc="Min,max elastic anisotropy. Tuple of floats, and both values are required.")] = None, elements: Annotated[List[str], AIParam(desc="List of elements (e.g., ['Li', 'Fe', 'O']).")] = None, energy_above_hull: Annotated[List[float], AIParam(desc="Min,max energy above hull in eV/atom. Tuple of floats, and both values are required.")] = None, equilibrium_reaction_energy: Annotated[List[float], AIParam(desc="Min,max equilibrium reaction energy in eV/atom. Tuple of floats, and both values are required.")] = None, formation_energy: Annotated[List[float], AIParam(desc="Min,max formation energy in eV/atom. Tuple of floats, and both values are required.")] = None, g_reuss: Annotated[List[float], AIParam(desc="Min,max Reuss grain boundary energy in eV/atom. Tuple of floats, and both values are required.")] = None, g_voigt: Annotated[List[float], AIParam(desc="Min,max Voigt grain boundary energy in eV/atom. Tuple of floats, and both values are required.")] = None, g_vrh: Annotated[List[float], AIParam(desc="Min,max Voigt-Reuss-Hill grain boundary energy in eV/atom. Tuple of floats, and both values are required.")] = None, k_reuss: Annotated[List[float], AIParam(desc="Min,max Reuss bulk modulus in GPa. Tuple of floats, and both values are required.")] = None, k_voigt: Annotated[List[float], AIParam(desc="Min,max Voigt bulk modulus in GPa. Tuple of floats, and both values are required.")] = None, k_vrh: Annotated[List[float], AIParam(desc="Min,max Voigt-Reuss-Hill bulk modulus in GPa. Tuple of floats, and both values are required.")] = None, n: Annotated[List[int], AIParam(desc="Min,max number of atoms. Tuple of ints, and both values are required.")] = None, nelements: Annotated[List[int], AIParam(desc="Min,max number of elements. Tuple of ints, and both values are required.")] = None, num_sites: Annotated[List[int], AIParam(desc="Min,max number of sites. Tuple of ints, and both values are required.")] = None, num_magnetic_sites: Annotated[List[int], AIParam(desc="Min,max number of magnetic sites.")] = None, num_unique_magnetic_sites: Annotated[List[int], AIParam(desc="Min,max number of unique magnetic sites. Tuple of ints, and both values are required.")] = None, piezoelectric_modulus: Annotated[List[float], AIParam(desc="Min,max piezoelectric modulus in C/m^2. Tuple of floats, and both values are required.")] = None, poisson_ratio: Annotated[List[float], AIParam(desc="Min,max Poisson's ratio. Tuple of floats, and both values are required.")] = None, shape_factor: Annotated[List[float], AIParam(desc="Min,max shape factor. Tuple of floats, and both values are required.")] = None, surface_energy_anisotropy: Annotated[List[float], AIParam(desc="Min,max surface energy anisotropy. Tuple of floats, and both values are required.")] = None, total_energy: Annotated[List[float], AIParam(desc="Min,max total energy in eV/atom.")] = None, total_magnetization: Annotated[List[float], AIParam(desc="Min,max total magnetization in Bohr magnetons/atom.")] = None, total_magnetization_normalized_formula_units: Annotated[List[float], AIParam(desc="Min,max total magnetization normalized to formula units in Bohr magnetons/formula unit.")] = None, total_magnetization_normalized_vol: Annotated[List[float], AIParam(desc="Min,max total magnetization normalized to volume in Bohr magnetons/bohr^3.")] = None, uncorrected_energy: Annotated[List[float], AIParam(desc="Min,max uncorrected energy in eV/atom. Tuple of floats, and both values are required.")] = None, volume: Annotated[List[float], AIParam(desc="Min,max volume in bohr^3. Tuple of floats, and both values are required.")] = None, weighted_surface_energy: Annotated[List[float], AIParam(desc="Min,max weighted surface energy in eV/ang^2. Tuple of floats, and both values are required.")] = None, weighted_work_function: Annotated[List[float], AIParam(desc="Min,max weighted work function in eV. Tuple of floats, and both values are required.")] = None, surface_anisotropy: Annotated[List[float], AIParam(desc="Min,max surface anisotropy. Tuple of floats, and both values are required.")] = None, has_reconstructed: Annotated[bool, AIParam(desc="Whether the entry has reconstructed surfaces.")] = None, is_gap_direct: Annotated[bool, AIParam(desc="Whether the material has a direct band gap.")] = None, is_metal: Annotated[bool, AIParam(desc="Whether the material is considered a metal.")] = None, is_stable: Annotated[bool, AIParam(desc="Whether the material lies on the convex energy hull.")] = None, magnetic_ordering: Annotated[str, AIParam(desc="Magnetic ordering of material. Available options are 'paramagnetic', 'ferromagnetic', 'antiferromagnetic', and 'ferrimagnetic'.")] = None, spacegroup_number: Annotated[int, AIParam(desc="Spacegroup number of material.")] = None, spacegroup_symbol: Annotated[str, AIParam(desc="Spacegroup symbol of material.")] = None, exclude_elements: Annotated[str, AIParam(desc="Elements to exclude (e.g., Li,Fe,O).")] = None, possible_species: Annotated[str, AIParam(desc="Possible species of material (e.g., Li,Fe,O).")] = None, has_props: Annotated[str, AIParam(desc="Calculated properties available (list of HasProps or strings).")] = None, theoretical: Annotated[bool, AIParam(desc="Whether the entry is theoretical (true) or experimental/experimentally observed (false).")] = None, temperature: Annotated[Optional[float], AIParam(desc="Temperature in Kelvin (optional).")] = None, pressure: Annotated[Optional[float], AIParam(desc="Pressure in GPa (optional).")] = None, page: Annotated[int, AIParam(desc="Page number (default 1).")] = 1, per_page: Annotated[int, AIParam(desc="Items per page (max 10; default 10).")] = 10) -> str:
        """Fetch a material id and formula by a characteristic of the material"""
        _args = {}
        if band_gap is not None:
            _args["band_gap"] = band_gap
        if crystal_system is not None:
            _args["crystal_system"] = crystal_system
        if density is not None:
            _args["density"] = density
        if e_electronic is not None:
            _args["e_electronic"] = e_electronic
        if e_ionic is not None:
            _args["e_ionic"] = e_ionic
        if e_total is not None:
            _args["e_total"] = e_total
        if efermi is not None:
            _args["efermi"] = efermi
        if elastic_anisotropy is not None:
            _args["elastic_anisotropy"] = elastic_anisotropy
        if elements is not None:
            _args["elements"] = elements
        if energy_above_hull is not None:
            _args["energy_above_hull"] = energy_above_hull
        if equilibrium_reaction_energy is not None:
            _args["equilibrium_reaction_energy"] = equilibrium_reaction_energy
        if formation_energy is not None:
            _args["formation_energy"] = formation_energy
        if g_reuss is not None:
            _args["g_reuss"] = g_reuss
        if g_voigt is not None:
            _args["g_voigt"] = g_voigt
        if g_vrh is not None:
            _args["g_vrh"] = g_vrh
        if k_reuss is not None:
            _args["k_reuss"] = k_reuss
        if k_voigt is not None:
            _args["k_voigt"] = k_voigt
        if k_vrh is not None:
            _args["k_vrh"] = k_vrh
        if n is not None:
            _args["n"] = n
        if nelements is not None:
            _args["nelements"] = nelements
        if num_sites is not None:
            _args["num_sites"] = num_sites
        if num_magnetic_sites is not None:
            _args["num_magnetic_sites"] = num_magnetic_sites
        if num_unique_magnetic_sites is not None:
            _args["num_unique_magnetic_sites"] = num_unique_magnetic_sites
        if piezoelectric_modulus is not None:
            _args["piezoelectric_modulus"] = piezoelectric_modulus
        if poisson_ratio is not None:
            _args["poisson_ratio"] = poisson_ratio
        if shape_factor is not None:
            _args["shape_factor"] = shape_factor
        if surface_energy_anisotropy is not None:
            _args["surface_energy_anisotropy"] = surface_energy_anisotropy
        if total_energy is not None:
            _args["total_energy"] = total_energy
        if total_magnetization is not None:
            _args["total_magnetization"] = total_magnetization
        if total_magnetization_normalized_formula_units is not None:
            _args["total_magnetization_normalized_formula_units"] = total_magnetization_normalized_formula_units
        if total_magnetization_normalized_vol is not None:
            _args["total_magnetization_normalized_vol"] = total_magnetization_normalized_vol
        if uncorrected_energy is not None:
            _args["uncorrected_energy"] = uncorrected_energy
        if volume is not None:
            _args["volume"] = volume
        if weighted_surface_energy is not None:
            _args["weighted_surface_energy"] = weighted_surface_energy
        if weighted_work_function is not None:
            _args["weighted_work_function"] = weighted_work_function
        if surface_anisotropy is not None:
            _args["surface_anisotropy"] = surface_anisotropy
        if has_reconstructed is not None:
            _args["has_reconstructed"] = has_reconstructed
        if is_gap_direct is not None:
            _args["is_gap_direct"] = is_gap_direct
        if is_metal is not None:
            _args["is_metal"] = is_metal
        if is_stable is not None:
            _args["is_stable"] = is_stable
        if magnetic_ordering is not None:
            _args["magnetic_ordering"] = magnetic_ordering
        if spacegroup_number is not None:
            _args["spacegroup_number"] = spacegroup_number
        if spacegroup_symbol is not None:
            _args["spacegroup_symbol"] = spacegroup_symbol
        if exclude_elements is not None:
            _args["exclude_elements"] = exclude_elements
        if possible_species is not None:
            _args["possible_species"] = possible_species
        if has_props is not None:
            _args["has_props"] = has_props
        if theoretical is not None:
            _args["theoretical"] = theoretical
        if temperature is not None:
            _args["temperature"] = temperature
        if pressure is not None:
            _args["pressure"] = pressure
        _args["page"] = page if page is not None else 1
        _args["per_page"] = per_page if per_page is not None else 10
        _result = await self._proxy.call_tool("get_material_by_char", _args)
        try:
            self.recent_tool_outputs.append({"tool": "get_material_by_char", "args": _args, "result": _result})
        except Exception:
            pass
        try:
            # Prefer pretty JSON if possible; fall back to str()
            import json as _json
            return _json.dumps(_result, ensure_ascii=False)
        except Exception:
            return str(_result)

    @ai_function(desc="Fetch a one or more materials by id(s).", auto_truncate=128000)
    async def get_material_details_by_ids(self, material_ids: Annotated[List[str], AIParam(desc="List of material ids, e.g., ['mp-149', 'mp-150', 'mp-151'].")], fields: Annotated[List[str], AIParam(desc="List of fields to include. Values include 'builder_meta', 'nsites', 'elements', 'nelements', 'composition', 'composition_reduced', 'formula_pretty', 'formula_anonymous', 'chemsys', 'volume', 'density', 'density_atomic', 'symmetry', 'property_name', 'material_id', 'deprecated', 'deprecation_reasons', 'last_updated', 'origins', 'warnings', 'structure', 'task_ids', 'uncorrected_energy_per_atom', 'energy_per_atom', 'formation_energy_per_atom', 'energy_above_hull', 'is_stable', 'equilibrium_reaction_energy_per_atom', 'decomposes_to', 'xas', 'grain_boundaries', 'band_gap', 'cbm', 'vbm', 'efermi', 'is_gap_direct', 'is_metal', 'es_source_calc_id', 'bandstructure', 'dos', 'dos_energy_up', 'dos_energy_down', 'is_magnetic', 'ordering', 'total_magnetization', 'total_magnetization_normalized_vol', 'total_magnetization_normalized_formula_units', 'num_magnetic_sites', 'num_unique_magnetic_sites', 'types_of_magnetic_species', 'bulk_modulus', 'shear_modulus', 'universal_anisotropy', 'homogeneous_poisson', 'e_total', 'e_ionic', 'e_electronic', 'n', 'e_ij_max', 'weighted_surface_energy_EV_PER_ANG2', 'weighted_surface_energy', 'weighted_work_function', 'surface_anisotropy', 'shape_factor', 'has_reconstructed', 'possible_species', 'has_props', 'theoretical', 'database_Ids'")] = None, all_fields: Annotated[bool, AIParam(desc="Whether to return all document fields. Useful if the user wants to know about the material without explicitly asking for certain fields (default True).")] = True, page: Annotated[int, AIParam(desc="Page number (default 1).")] = 1, per_page: Annotated[int, AIParam(desc="Items per page (max 10; default 10).")] = 10) -> str:
        """Fetch a one or more materials by id(s)."""
        _args = {}
        _args["material_ids"] = material_ids
        if fields is not None:
            _args["fields"] = fields
        _args["all_fields"] = all_fields if all_fields is not None else True
        _args["page"] = page if page is not None else 1
        _args["per_page"] = per_page if per_page is not None else 10
        _result = await self._proxy.call_tool("get_material_details_by_ids", _args)
        try:
            self.recent_tool_outputs.append({"tool": "get_material_details_by_ids", "args": _args, "result": _result})
        except Exception:
            pass
        try:
            # Prefer pretty JSON if possible; fall back to str()
            import json as _json
            return _json.dumps(_result, ensure_ascii=False)
        except Exception:
            return str(_result)

