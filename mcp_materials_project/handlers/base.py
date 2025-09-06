"""
Base handler class and common utilities for Material Project API endpoints.
"""

import json
import logging
import re
from typing import Any, Dict, List, Mapping, Optional
from mp_api.client import MPRester
import os

_log = logging.getLogger(__name__)


class InvalidRangeError(Exception):
    """Raised when a range parameter has only one bound or is otherwise invalid."""
    pass


# Which query params are treated as ranges (min,max)
RANGE_KEYS = [
    "band_gap", "density", "e_electronic", "e_ionic", "e_total", "efermi",
    "elastic_anisotropy", "energy_above_hull", "equilibrium_reaction_energy",
    "formation_energy", "g_reuss", "g_voigt", "g_vrh", "k_reuss", "k_voigt",
    "k_vrh", "n", "nelements", "num_sites", "num_magnetic_sites",
    "num_unique_magnetic_sites", "piezoelectric_modulus", "poisson_ratio",
    "shape_factor", "surface_energy_anisotropy", "total_energy",
    "total_magnetization", "total_magnetization_normalized_formula_units",
    "total_magnetization_normalized_vol", "uncorrected_energy", "volume",
    "weighted_surface_energy", "weighted_work_function",
]


class BaseHandler:
    """Base class for all endpoint handlers."""
    
    def __init__(self, mpr: MPRester):
        self.mpr = mpr
    
    def _get_pagination(self, params: Mapping[str, Any]) -> tuple[int, int]:
        """Return (page, per_page) with defaults and safety caps.

        - Default page=1 when missing/invalid
        - Default per_page=10 when missing/invalid
        """
        page = self._parse_int(params.get("page")) or 1
        per_page = self._parse_int(params.get("per_page")) or 10
        if page < 1:
            page = 1
        # Enforce maximum page size of 10 items
        if per_page is None or per_page < 1:
            per_page = 10
        return page, per_page
    
    def _slice_for_page(self, items: List[Any], page: int, per_page: int) -> List[Any]:
        """Return the sublist for the requested page and per_page.

        If page is beyond range, return an empty list.
        """
        try:
            start = (page - 1) * per_page
            end = start + per_page
            return list(items[start:end])
        except Exception:
            # Be defensive; never raise during pagination
            return list(items)[:per_page]
    
    def _is_singleton_range_value(self, value: Any) -> bool:
        """True iff 'value' looks like a one-bound range."""
        if value is None:
            return False

        if isinstance(value, (list, tuple)):
            return len(value) == 1 and str(value[0]).strip() != ""

        text = str(value).strip()
        if not text:
            return False

        # Try JSON-like list/tuple
        if (text.startswith("[") and text.endswith("]")) or (text.startswith("(") and text.endswith(")")):
            try:
                decoded = json.loads(text.replace("(", "[").replace(")", "]"))
                return isinstance(decoded, list) and len(decoded) == 1 and str(decoded[0]).strip() != ""
            except Exception:
                # fall through
                pass

        # Comma variants like "2.0", "2.0,", ",2.0"
        parts = [p.strip() for p in text.split(",")]
        non_empty = [p for p in parts if p]
        return len(non_empty) == 1
    
    def _parse_csv_list(self, value: Any) -> Optional[List[str]]:
        """Parse CSV list values."""
        if value is None:
            return None
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        text = str(value).strip()
        if not text:
            return None
        return [s.strip() for s in text.split(",") if s.strip()]
    
    def _parse_bool(self, value: Any) -> Optional[bool]:
        """Parse boolean values."""
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "t", "yes", "y"}:
            return True
        if text in {"0", "false", "f", "no", "n"}:
            return False
        return None
    
    def _parse_int(self, value: Any) -> Optional[int]:
        """Parse integer values."""
        if value is None or value == "":
            return None
        try:
            return int(value)
        except Exception:
            return None
    
    def _parse_range(self, value: Any, *, require_two: bool = False) -> Optional[tuple[float, float]]:
        """Parse a (min, max) range from various formats."""
        if value is None:
            return None

        # Direct list/tuple
        if isinstance(value, (list, tuple)):
            if len(value) == 1:
                if require_two and str(value[0]).strip():
                    raise InvalidRangeError("Expected two numbers (min,max), got one.")
                return None
            if len(value) >= 2:
                try:
                    return float(value[0]), float(value[1])
                except Exception:
                    return None
            return None

        text = str(value).strip()
        if not text:
            return None

        # Try JSON decode first if it looks like JSON array
        if text.startswith("[") and text.endswith("]"):
            try:
                arr = json.loads(text)
                if isinstance(arr, list):
                    if len(arr) == 1:
                        if require_two and str(arr[0]).strip():
                            raise InvalidRangeError("Expected two numbers (min,max), got one.")
                        return None
                    if len(arr) >= 2:
                        return float(arr[0]), float(arr[1])
            except Exception:
                pass  # fall through to flexible parsing

        # Strip surrounding brackets/parentheses/braces
        if (text[0], text[-1]) in {("(", ")"), ("[", "]"), ("{", "}")}:
            text = text[1:-1].strip()

        parts = [p.strip() for p in text.split(",")]
        non_empty = [p for p in parts if p]

        if len(non_empty) == 1:
            if require_two:
                raise InvalidRangeError("Expected two numbers (min,max), got one.")
            return None

        if len(non_empty) >= 2:
            try:
                return float(non_empty[0]), float(non_empty[1])
            except Exception:
                # Last resort: extract first two numbers anywhere in the string
                nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
                if len(nums) >= 2:
                    try:
                        return float(nums[0]), float(nums[1])
                    except Exception:
                        return None
        return None
    
    def _build_summary_search_kwargs(self, _params: Mapping[str, Any]) -> Dict[str, Any]:
        """Build kwargs for summary.search with proper coercions."""
        import inspect as _inspect
        sig = _inspect.signature(self.mpr.materials.summary.search)
        allowed = set(sig.parameters.keys())

        raw_kwargs: Dict[str, Any] = {}
        errors: List[Dict[str, Any]] = []

        # ---- Ranges (min,max floats) ----
        for key in RANGE_KEYS:
            try:
                rng = self._parse_range(_params.get(key), require_two=True)
            except InvalidRangeError as e:
                errors.append({"param": key, "value": _params.get(key), "reason": str(e)})
                continue
            if rng is not None:
                raw_kwargs[key] = rng

        # ---- Lists / list-or-str ----
        def _list_or_str(name: str) -> Any:
            vals = self._parse_csv_list(_params.get(name))
            if vals is None:
                return None
            return vals if len(vals) > 1 else (vals[0] if vals else None)

        raw_kwargs["chemsys"] = _list_or_str("chemsys")
        # Always send lists for list-typed params to avoid character-splitting of strings
        def _as_list(name: str) -> Any:
            vals = self._parse_csv_list(_params.get(name))
            return vals if vals is not None else None
        raw_kwargs["elements"] = _as_list("elements")
        raw_kwargs["exclude_elements"] = _as_list("exclude_elements")
        raw_kwargs["formula"] = _list_or_str("formula")
        raw_kwargs["material_ids"] = self._parse_csv_list(_params.get("material_ids"))
        raw_kwargs["possible_species"] = _as_list("possible_species")

        # ---- Booleans ----
        for key in [
            "deprecated", "has_reconstructed", "is_gap_direct", "is_metal",
            "is_stable", "theoretical", "include_gnome", "all_fields",
        ]:
            b = self._parse_bool(_params.get(key))
            if b is not None:
                raw_kwargs[key] = b

        # ---- Ints ----
        for key in ["spacegroup_number", "num_chunks", "chunk_size"]:
            iv = self._parse_int(_params.get(key))
            if iv is not None:
                raw_kwargs[key] = iv

        # ---- Strings ----
        for key in ["crystal_system", "magnetic_ordering", "spacegroup_symbol"]:
            if _params.get(key):
                raw_kwargs[key] = str(_params.get(key))

        # ---- Fields (list of strings) ----
        fields = self._parse_csv_list(_params.get("fields"))
        if fields is not None:
            raw_kwargs["fields"] = fields

        # ---- Aliases / Renames ----
        if "num_elements" in _params and _params.get("num_elements"):
            try:
                rng = self._parse_range(_params.get("num_elements"), require_two=True)
            except InvalidRangeError as e:
                errors.append({"param": "num_elements", "value": _params.get("num_elements"), "reason": str(e)})
            else:
                if rng is not None:
                    raw_kwargs["nelements"] = rng

        if errors:
            return {"__errors__": errors}

        kwargs = {k: v for k, v in raw_kwargs.items() if k in allowed and v is not None}
        _log.info(f"summary.search kwargs: {kwargs}")
        return kwargs
    
    def _convert_docs_to_dicts(self, docs) -> List[Dict[str, Any]]:
        """Convert MPRester documents to plain dictionaries."""
        out: List[Any] = []
        try:
            for doc in docs:
                converted = None
                for attr in ("model_dump", "dict"):
                    fn = getattr(doc, attr, None)
                    if callable(fn):
                        try:
                            converted = fn()
                            break
                        except Exception:
                            converted = None
                if converted is None:
                    try:
                        converted = json.loads(json.dumps(doc, default=str))
                    except Exception:
                        converted = str(doc)
                out.append(self._to_jsonable(converted))
            _log.info(f"Converted {len(out)} docs to dicts.")
        except Exception as exc:
            _log.warning(f"Failed converting docs to dicts: {exc}")
            out = docs if isinstance(docs, list) else [str(docs)]

        def _remove_nones(obj: Any) -> Any:
            if obj is None or isinstance(obj, (str, int, float, bool)):
                return obj
            if isinstance(obj, dict):
                return {k: _remove_nones(v) for k, v in obj.items() if v is not None}
            if isinstance(obj, (list, tuple, set)):
                return [_remove_nones(v) for v in obj if v is not None]
            return obj

        out = _remove_nones(out)

        for item in out:
            if isinstance(item, dict) and "fields_not_requested" in item:
                item.pop("fields_not_requested", None)

        return out
    
    def _to_jsonable(self, obj: Any) -> Any:
        """Recursively convert objects to JSON-serializable forms."""
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, dict):
            return {str(k): self._to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [self._to_jsonable(v) for v in obj]
        for attr in ("model_dump", "dict"):
            fn = getattr(obj, attr, None)
            if callable(fn):
                try:
                    return self._to_jsonable(fn())
                except Exception:
                    pass
        try:
            return json.loads(json.dumps(obj, default=str))
        except Exception:
            return str(obj)
    
    def _call_summary_count(self, count_criteria: Dict[str, Any]) -> Optional[int]:
        """Call summary count with the given criteria."""
        return self._call_summary_count_impl(count_criteria)
    
    def _call_summary_count_impl(self, filters: Dict[str, Any]) -> Optional[int]:
        """Implementation of summary count call."""
        norm = self._expand_range_criteria(filters)

        try:
            out = self.mpr.materials.summary.count(**norm)  # newer style
        except TypeError:
            try:
                out = self.mpr.materials.summary.count(criteria=norm)  # older style
            except Exception as exc:
                _log.warning(f"count(criteria=...) failed: {exc}")
                return None
        except Exception as exc:
            _log.warning(f"count(**filters) failed: {exc}")
            try:
                out = self.mpr.materials.summary.count(criteria=norm)
            except Exception as exc2:
                _log.warning(f"count(criteria=...) fallback failed: {exc2}")
                return None

        try:
            if isinstance(out, int):
                return out
            if isinstance(out, dict) and "count" in out:
                return int(out["count"])
            return int(out)
        except Exception:
            return None
    
    def _expand_range_criteria(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Convert (min,max) tuples into *_min/*_max for older mp_api implementations."""
        out: Dict[str, Any] = {}
        for k, v in filters.items():
            if k in RANGE_KEYS and isinstance(v, (list, tuple)) and len(v) >= 2:
                try:
                    lo, hi = float(v[0]), float(v[1])
                    out[f"{k}_min"] = lo
                    out[f"{k}_max"] = hi
                except Exception:
                    out[k] = v  # fallback to original
            else:
                out[k] = v
        return out
    
    def _total_count_for_summary(self, kwargs: Dict[str, Any]) -> Optional[int]:
        """Compute total count for summary search."""
        # Extract count criteria (drop non-filter args)
        count_criteria = {
            k: v for k, v in kwargs.items()
            if k not in {"fields", "num_chunks", "chunk_size", "all_fields", "limit"}
               and v is not None
        }
        return self._call_summary_count_impl(count_criteria)
    
    @property
    def RANGE_KEYS(self):
        """Access to RANGE_KEYS constant."""
        return RANGE_KEYS
