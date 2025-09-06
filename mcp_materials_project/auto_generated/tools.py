from __future__ import annotations

import json as _json
from typing import Any, Optional, List, Tuple, Dict
import requests
from mcp.server.fastmcp import FastMCP
from mcp_materials_project.client import MaterialsProjectClient

def _serialize_param(value: Any) -> Any:
    # For complex types, default to JSON string; primitives pass through
    if isinstance(value, (list, tuple, dict)):
        try: return _json.dumps(value, ensure_ascii=False)
        except Exception: return str(value)
    return value

def _coerce_arg(value: Any) -> Any:
    # Try to turn JSON-like strings into real Python objects
    if isinstance(value, str):
        s = value.strip()
        # JSON arrays/objects like "[0,1]" or "{...}"
        if (s.startswith('[') and s.endswith(']')) or (s.startswith('{') and s.endswith('}')):
            try: return _json.loads(s)
            except Exception: pass
        # Also accept simple comma-separated scalars: "a,b,c" or "1,2"
        if ',' in s and '[' not in s and ']' not in s and '{' not in s and '}' not in s:
            parts = [p.strip() for p in s.split(',') if p.strip()]
            # Best-effort numeric cast
            def _maybe_num(x: str):
                try:
                    if x.lower() in {'true','false'}: return x.lower() == 'true'
                    if '.' in x or 'e' in x.lower(): return float(x)
                    return int(x)
                except Exception:
                    return x
            return [_maybe_num(p) for p in parts]
    return value

def _client() -> MaterialsProjectClient:
    return MaterialsProjectClient()

def register_generated_tools(mcp: FastMCP) -> None:
    @mcp.tool(name="convert_name_to_symbols", description="Convert a chemical name to a list of symbols. The name can be a chemical formula (e.g. Iron Oxide 2 -> Fe2O3), an element (e.g. Oxygen -> O), or a chemical system (e.g. Lithium-Iron-* -> Li-Fe-*). Wildcards are supported.")
    async def convert_name_to_symbols(name: str) -> Any:
        """Convert a chemical name to a list of symbols. The name can be a chemical formula (e.g. Iron Oxide 2 -> Fe2O3), an element (e.g. Oxygen -> O), or a chemical system (e.g. Lithium-Iron-* -> Li-Fe-*). Wildcards are supported."""
        client = _client()
        try:
            _params = {}
            _val = _coerce_arg(name)
            _params["name"] = _serialize_param(_val)
            _select_keys = {}
            _force_keys = set(_select_keys) | {'material_id'}
            _af = _params.get('all_fields')
            _all_fields = False
            if _af is not None:
                s = str(_af).strip().lower()
                _all_fields = s in {'1','true','t','yes','y'}
            if not _all_fields and _force_keys:
                def _as_list(v):
                    if v is None: return []
                    if isinstance(v, (list, tuple)): return list(v)
                    s = str(v).strip()
                    if s.startswith('[') and s.endswith(']'):
                        try:
                            dec = _json.loads(s)
                            if isinstance(dec, list): return dec
                        except Exception: pass
                    return [p.strip() for p in s.split(',') if p.strip()]
                _cur_fields = _as_list(_params.get('fields'))
                for _k in _force_keys:
                    if _k not in _cur_fields:
                        _cur_fields.append(_k)
                if _cur_fields:
                    _params['fields'] = _cur_fields
            _resp = client.get("materials/convert/name_to_symbols", params=_params)
            _has_server_env = isinstance(_resp, dict) and 'data' in _resp
            _items = _resp['data'] if _has_server_env else _resp
            if not _has_server_env:
                _resp = _items
            if _has_server_env:
                _resp['data'] = _items
            return _resp
        except requests.HTTPError as e:
            status = getattr(getattr(e, 'response', None), 'status_code', None)
            body = None
            try:
                if e.response is not None:
                    body = e.response.json()
            except Exception:
                body = getattr(getattr(e, 'response', None), 'text', None)
            return {"error": {"type": "http_error", "status": status, "message": str(e), "body": body}}
        except Exception as e:
            return {"error": {"type": "unexpected_error", "message": str(e)}}

    @mcp.tool(name="get_material", description="Query materials by their chemical system and return their material IDs and formula. At least one of chemsys, formula, or element must be provided. Must be in symbols, so use the convert_name_to_symbols tool to convert the name to symbols.")
    async def get_material(chemsys: str = None, formula: str = None, element: str = None, page: int = 1, per_page: int = 10) -> Any:
        """Query materials by their chemical system and return their material IDs and formula. At least one of chemsys, formula, or element must be provided. Must be in symbols, so use the convert_name_to_symbols tool to convert the name to symbols."""
        client = _client()
        try:
            _params = {}
            if chemsys is not None:
                _val = _coerce_arg(chemsys)
                _params["chemsys"] = _serialize_param(_val)
            if formula is not None:
                _val = _coerce_arg(formula)
                _params["formula"] = _serialize_param(_val)
            if element is not None:
                _val = _coerce_arg(element)
                _params["element"] = _serialize_param(_val)
            if page is not None:
                _val = _coerce_arg(page)
                _params["page"] = _serialize_param(_val)
            if per_page is not None:
                _val = _coerce_arg(per_page)
                _params["per_page"] = _serialize_param(_val)
            _select_keys = {'material_id', 'formula_pretty'}
            _force_keys = set(_select_keys) | {'material_id'}
            _af = _params.get('all_fields')
            _all_fields = False
            if _af is not None:
                s = str(_af).strip().lower()
                _all_fields = s in {'1','true','t','yes','y'}
            if not _all_fields and _force_keys:
                def _as_list(v):
                    if v is None: return []
                    if isinstance(v, (list, tuple)): return list(v)
                    s = str(v).strip()
                    if s.startswith('[') and s.endswith(']'):
                        try:
                            dec = _json.loads(s)
                            if isinstance(dec, list): return dec
                        except Exception: pass
                    return [p.strip() for p in s.split(',') if p.strip()]
                _cur_fields = _as_list(_params.get('fields'))
                for _k in _force_keys:
                    if _k not in _cur_fields:
                        _cur_fields.append(_k)
                if _cur_fields:
                    _params['fields'] = _cur_fields
            _resp = client.get("materials/summary/get_material", params=_params)
            _has_server_env = isinstance(_resp, dict) and 'data' in _resp
            _items = _resp['data'] if _has_server_env else _resp
            def _coerce_value(key, value):
                _t = {

                }
                typ = _t.get(key)
                try:
                    if typ == 'int': return int(value) if value is not None else None
                    if typ == 'float': return float(value) if value is not None else None
                    if typ == 'bool':
                        if isinstance(value, bool): return value
                        s = str(value).strip().lower() if value is not None else ''
                        return True if s in {'1','true','t','yes','y'} else (False if s in {'0','false','f','no','n'} else None)
                    if typ == 'str': return '' if value is None else str(value)
                except Exception:
                    return value
                return value
            if isinstance(_items, list):
                _proj = []
                for _it in _items:
                    if isinstance(_it, dict):
                        _obj = {}
                        _obj["material_id"] = _coerce_value("material_id", _it.get("material_id"))
                        _obj["formula_pretty"] = _coerce_value("formula_pretty", _it.get("formula_pretty"))
                        _proj.append(_obj)
                    else:
                        _proj.append(_it)
                _items = _proj
            elif isinstance(_items, dict):
                _obj = {}
                _obj["material_id"] = _coerce_value("material_id", _items.get("material_id"))
                _obj["formula_pretty"] = _coerce_value("formula_pretty", _items.get("formula_pretty"))
                _items = _obj
            if not _has_server_env:
                _resp = _items
            if _has_server_env:
                _resp['data'] = _items
            return _resp
        except requests.HTTPError as e:
            status = getattr(getattr(e, 'response', None), 'status_code', None)
            body = None
            try:
                if e.response is not None:
                    body = e.response.json()
            except Exception:
                body = getattr(getattr(e, 'response', None), 'text', None)
            return {"error": {"type": "http_error", "status": status, "message": str(e), "body": body}}
        except Exception as e:
            return {"error": {"type": "unexpected_error", "message": str(e)}}

    @mcp.tool(name="get_material_by_char", description="Fetch a material id and formula by a characteristic of the material")
    async def get_material_by_char(band_gap: List[float] = None, crystal_system: str = None, density: List[float] = None, e_electronic: List[float] = None, e_ionic: List[float] = None, e_total: List[float] = None, efermi: List[float] = None, elastic_anisotropy: List[float] = None, elements: List[str] = None, energy_above_hull: List[float] = None, equilibrium_reaction_energy: List[float] = None, formation_energy: List[float] = None, g_reuss: List[float] = None, g_voigt: List[float] = None, g_vrh: List[float] = None, k_reuss: List[float] = None, k_voigt: List[float] = None, k_vrh: List[float] = None, n: List[int] = None, nelements: List[int] = None, num_sites: List[int] = None, num_magnetic_sites: List[int] = None, num_unique_magnetic_sites: List[int] = None, piezoelectric_modulus: List[float] = None, poisson_ratio: List[float] = None, shape_factor: List[float] = None, surface_energy_anisotropy: List[float] = None, total_energy: List[float] = None, total_magnetization: List[float] = None, total_magnetization_normalized_formula_units: List[float] = None, total_magnetization_normalized_vol: List[float] = None, uncorrected_energy: List[float] = None, volume: List[float] = None, weighted_surface_energy: List[float] = None, weighted_work_function: List[float] = None, surface_anisotropy: List[float] = None, has_reconstructed: bool = None, is_gap_direct: bool = None, is_metal: bool = None, is_stable: bool = None, magnetic_ordering: str = None, spacegroup_number: int = None, spacegroup_symbol: str = None, exclude_elements: str = None, possible_species: str = None, has_props: str = None, theoretical: bool = None, temperature: float = None, pressure: float = None, page: int = 1, per_page: int = 10) -> Any:
        """Fetch a material id and formula by a characteristic of the material"""
        client = _client()
        try:
            _params = {}
            if band_gap is not None:
                _val = _coerce_arg(band_gap)
                _params["band_gap"] = _serialize_param(_val)
            if crystal_system is not None:
                _val = _coerce_arg(crystal_system)
                _params["crystal_system"] = _serialize_param(_val)
            if density is not None:
                _val = _coerce_arg(density)
                _params["density"] = _serialize_param(_val)
            if e_electronic is not None:
                _val = _coerce_arg(e_electronic)
                _params["e_electronic"] = _serialize_param(_val)
            if e_ionic is not None:
                _val = _coerce_arg(e_ionic)
                _params["e_ionic"] = _serialize_param(_val)
            if e_total is not None:
                _val = _coerce_arg(e_total)
                _params["e_total"] = _serialize_param(_val)
            if efermi is not None:
                _val = _coerce_arg(efermi)
                _params["efermi"] = _serialize_param(_val)
            if elastic_anisotropy is not None:
                _val = _coerce_arg(elastic_anisotropy)
                _params["elastic_anisotropy"] = _serialize_param(_val)
            if elements is not None:
                _val = _coerce_arg(elements)
                if isinstance(_val, (list, tuple)):
                    _params["elements"] = list(_val)
                elif isinstance(_val, dict):
                    for _k, _v in _val.items():
                        _params[f"elements[{_k}]"] = _serialize_param(_v)
                else:
                    _params["elements"] = _serialize_param(_val)
            if energy_above_hull is not None:
                _val = _coerce_arg(energy_above_hull)
                _params["energy_above_hull"] = _serialize_param(_val)
            if equilibrium_reaction_energy is not None:
                _val = _coerce_arg(equilibrium_reaction_energy)
                _params["equilibrium_reaction_energy"] = _serialize_param(_val)
            if formation_energy is not None:
                _val = _coerce_arg(formation_energy)
                _params["formation_energy"] = _serialize_param(_val)
            if g_reuss is not None:
                _val = _coerce_arg(g_reuss)
                _params["g_reuss"] = _serialize_param(_val)
            if g_voigt is not None:
                _val = _coerce_arg(g_voigt)
                _params["g_voigt"] = _serialize_param(_val)
            if g_vrh is not None:
                _val = _coerce_arg(g_vrh)
                _params["g_vrh"] = _serialize_param(_val)
            if k_reuss is not None:
                _val = _coerce_arg(k_reuss)
                _params["k_reuss"] = _serialize_param(_val)
            if k_voigt is not None:
                _val = _coerce_arg(k_voigt)
                _params["k_voigt"] = _serialize_param(_val)
            if k_vrh is not None:
                _val = _coerce_arg(k_vrh)
                _params["k_vrh"] = _serialize_param(_val)
            if n is not None:
                _val = _coerce_arg(n)
                _params["n"] = _serialize_param(_val)
            if nelements is not None:
                _val = _coerce_arg(nelements)
                _params["nelements"] = _serialize_param(_val)
            if num_sites is not None:
                _val = _coerce_arg(num_sites)
                _params["num_sites"] = _serialize_param(_val)
            if num_magnetic_sites is not None:
                _val = _coerce_arg(num_magnetic_sites)
                _params["num_magnetic_sites"] = _serialize_param(_val)
            if num_unique_magnetic_sites is not None:
                _val = _coerce_arg(num_unique_magnetic_sites)
                _params["num_unique_magnetic_sites"] = _serialize_param(_val)
            if piezoelectric_modulus is not None:
                _val = _coerce_arg(piezoelectric_modulus)
                _params["piezoelectric_modulus"] = _serialize_param(_val)
            if poisson_ratio is not None:
                _val = _coerce_arg(poisson_ratio)
                _params["poisson_ratio"] = _serialize_param(_val)
            if shape_factor is not None:
                _val = _coerce_arg(shape_factor)
                _params["shape_factor"] = _serialize_param(_val)
            if surface_energy_anisotropy is not None:
                _val = _coerce_arg(surface_energy_anisotropy)
                _params["surface_energy_anisotropy"] = _serialize_param(_val)
            if total_energy is not None:
                _val = _coerce_arg(total_energy)
                _params["total_energy"] = _serialize_param(_val)
            if total_magnetization is not None:
                _val = _coerce_arg(total_magnetization)
                _params["total_magnetization"] = _serialize_param(_val)
            if total_magnetization_normalized_formula_units is not None:
                _val = _coerce_arg(total_magnetization_normalized_formula_units)
                _params["total_magnetization_normalized_formula_units"] = _serialize_param(_val)
            if total_magnetization_normalized_vol is not None:
                _val = _coerce_arg(total_magnetization_normalized_vol)
                _params["total_magnetization_normalized_vol"] = _serialize_param(_val)
            if uncorrected_energy is not None:
                _val = _coerce_arg(uncorrected_energy)
                _params["uncorrected_energy"] = _serialize_param(_val)
            if volume is not None:
                _val = _coerce_arg(volume)
                _params["volume"] = _serialize_param(_val)
            if weighted_surface_energy is not None:
                _val = _coerce_arg(weighted_surface_energy)
                _params["weighted_surface_energy"] = _serialize_param(_val)
            if weighted_work_function is not None:
                _val = _coerce_arg(weighted_work_function)
                _params["weighted_work_function"] = _serialize_param(_val)
            if surface_anisotropy is not None:
                _val = _coerce_arg(surface_anisotropy)
                _params["surface_anisotropy"] = _serialize_param(_val)
            if has_reconstructed is not None:
                _val = _coerce_arg(has_reconstructed)
                _params["has_reconstructed"] = _serialize_param(_val)
            if is_gap_direct is not None:
                _val = _coerce_arg(is_gap_direct)
                _params["is_gap_direct"] = _serialize_param(_val)
            if is_metal is not None:
                _val = _coerce_arg(is_metal)
                _params["is_metal"] = _serialize_param(_val)
            if is_stable is not None:
                _val = _coerce_arg(is_stable)
                _params["is_stable"] = _serialize_param(_val)
            if magnetic_ordering is not None:
                _val = _coerce_arg(magnetic_ordering)
                _params["magnetic_ordering"] = _serialize_param(_val)
            if spacegroup_number is not None:
                _val = _coerce_arg(spacegroup_number)
                _params["spacegroup_number"] = _serialize_param(_val)
            if spacegroup_symbol is not None:
                _val = _coerce_arg(spacegroup_symbol)
                _params["spacegroup_symbol"] = _serialize_param(_val)
            if exclude_elements is not None:
                _val = _coerce_arg(exclude_elements)
                _params["exclude_elements"] = _serialize_param(_val)
            if possible_species is not None:
                _val = _coerce_arg(possible_species)
                _params["possible_species"] = _serialize_param(_val)
            if has_props is not None:
                _val = _coerce_arg(has_props)
                _params["has_props"] = _serialize_param(_val)
            if theoretical is not None:
                _val = _coerce_arg(theoretical)
                _params["theoretical"] = _serialize_param(_val)
            if temperature is not None:
                _val = _coerce_arg(temperature)
                _params["temperature"] = _serialize_param(_val)
            if pressure is not None:
                _val = _coerce_arg(pressure)
                _params["pressure"] = _serialize_param(_val)
            if page is not None:
                _val = _coerce_arg(page)
                _params["page"] = _serialize_param(_val)
            if per_page is not None:
                _val = _coerce_arg(per_page)
                _params["per_page"] = _serialize_param(_val)
            _select_keys = {'material_id', 'formula_pretty'}
            _force_keys = set(_select_keys) | {'material_id'}
            _af = _params.get('all_fields')
            _all_fields = False
            if _af is not None:
                s = str(_af).strip().lower()
                _all_fields = s in {'1','true','t','yes','y'}
            if not _all_fields and _force_keys:
                def _as_list(v):
                    if v is None: return []
                    if isinstance(v, (list, tuple)): return list(v)
                    s = str(v).strip()
                    if s.startswith('[') and s.endswith(']'):
                        try:
                            dec = _json.loads(s)
                            if isinstance(dec, list): return dec
                        except Exception: pass
                    return [p.strip() for p in s.split(',') if p.strip()]
                _cur_fields = _as_list(_params.get('fields'))
                for _k in _force_keys:
                    if _k not in _cur_fields:
                        _cur_fields.append(_k)
                if _cur_fields:
                    _params['fields'] = _cur_fields
            _resp = client.get("materials/summary/get_material_by_char", params=_params)
            _has_server_env = isinstance(_resp, dict) and 'data' in _resp
            _items = _resp['data'] if _has_server_env else _resp
            def _coerce_value(key, value):
                _t = {

                }
                typ = _t.get(key)
                try:
                    if typ == 'int': return int(value) if value is not None else None
                    if typ == 'float': return float(value) if value is not None else None
                    if typ == 'bool':
                        if isinstance(value, bool): return value
                        s = str(value).strip().lower() if value is not None else ''
                        return True if s in {'1','true','t','yes','y'} else (False if s in {'0','false','f','no','n'} else None)
                    if typ == 'str': return '' if value is None else str(value)
                except Exception:
                    return value
                return value
            if isinstance(_items, list):
                _proj = []
                for _it in _items:
                    if isinstance(_it, dict):
                        _obj = {}
                        _obj["material_id"] = _coerce_value("material_id", _it.get("material_id"))
                        _obj["formula_pretty"] = _coerce_value("formula_pretty", _it.get("formula_pretty"))
                        _proj.append(_obj)
                    else:
                        _proj.append(_it)
                _items = _proj
            elif isinstance(_items, dict):
                _obj = {}
                _obj["material_id"] = _coerce_value("material_id", _items.get("material_id"))
                _obj["formula_pretty"] = _coerce_value("formula_pretty", _items.get("formula_pretty"))
                _items = _obj
            if not _has_server_env:
                _resp = _items
            if _has_server_env:
                _resp['data'] = _items
            return _resp
        except requests.HTTPError as e:
            status = getattr(getattr(e, 'response', None), 'status_code', None)
            body = None
            try:
                if e.response is not None:
                    body = e.response.json()
            except Exception:
                body = getattr(getattr(e, 'response', None), 'text', None)
            return {"error": {"type": "http_error", "status": status, "message": str(e), "body": body}}
        except Exception as e:
            return {"error": {"type": "unexpected_error", "message": str(e)}}

    @mcp.tool(name="get_material_details_by_ids", description="Fetch a one or more materials by id(s).")
    async def get_material_details_by_ids(material_ids: List[str], fields: List[str] = None, all_fields: bool = True, page: int = 1, per_page: int = 10) -> Any:
        """Fetch a one or more materials by id(s)."""
        client = _client()
        try:
            _params = {}
            _val = _coerce_arg(material_ids)
            _params["material_ids"] = _serialize_param(_val)
            if fields is not None:
                _val = _coerce_arg(fields)
                if isinstance(_val, (list, tuple)):
                    _params["fields"] = list(_val)
                elif isinstance(_val, dict):
                    for _k, _v in _val.items():
                        _params[f"fields[{_k}]"] = _serialize_param(_v)
                else:
                    _params["fields"] = _serialize_param(_val)
            if all_fields is not None:
                _val = _coerce_arg(all_fields)
                _params["all_fields"] = _serialize_param(_val)
            if page is not None:
                _val = _coerce_arg(page)
                _params["page"] = _serialize_param(_val)
            if per_page is not None:
                _val = _coerce_arg(per_page)
                _params["per_page"] = _serialize_param(_val)
            _select_keys = {}
            _force_keys = set(_select_keys) | {'material_id'}
            _af = _params.get('all_fields')
            _all_fields = False
            if _af is not None:
                s = str(_af).strip().lower()
                _all_fields = s in {'1','true','t','yes','y'}
            if not _all_fields and _force_keys:
                def _as_list(v):
                    if v is None: return []
                    if isinstance(v, (list, tuple)): return list(v)
                    s = str(v).strip()
                    if s.startswith('[') and s.endswith(']'):
                        try:
                            dec = _json.loads(s)
                            if isinstance(dec, list): return dec
                        except Exception: pass
                    return [p.strip() for p in s.split(',') if p.strip()]
                _cur_fields = _as_list(_params.get('fields'))
                for _k in _force_keys:
                    if _k not in _cur_fields:
                        _cur_fields.append(_k)
                if _cur_fields:
                    _params['fields'] = _cur_fields
            _resp = client.get("materials/summary/get_material_details_by_ids", params=_params)
            _has_server_env = isinstance(_resp, dict) and 'data' in _resp
            _items = _resp['data'] if _has_server_env else _resp
            if not _has_server_env:
                _resp = _items
            if _has_server_env:
                _resp['data'] = _items
            return _resp
        except requests.HTTPError as e:
            status = getattr(getattr(e, 'response', None), 'status_code', None)
            body = None
            try:
                if e.response is not None:
                    body = e.response.json()
            except Exception:
                body = getattr(getattr(e, 'response', None), 'text', None)
            return {"error": {"type": "http_error", "status": status, "message": str(e), "body": body}}
        except Exception as e:
            return {"error": {"type": "unexpected_error", "message": str(e)}}

