from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import json as _json

import logging as _log

def _load_template(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_templates(templates_dir: Path) -> List[Dict[str, Any]]:
    if not templates_dir.exists():
        return []
    templates: List[Dict[str, Any]] = []
    for entry in sorted(templates_dir.iterdir()):
        if entry.is_file() and entry.suffix == ".json":
            try:
                templates.append(_load_template(entry))
            except Exception:
                # Skip malformed templates rather than failing the server
                _log.warning(f"Skipping malformed template: {entry}")
                continue
    return templates


def _normalize_primitive(t: str) -> str:
    tok = t.strip().lower()
    # Known primitives
    if tok in {"string", "str"}: return "str"
    if tok in {"int", "integer"}: return "int"
    if tok in {"float", "number"}: return "float"
    if tok in {"bool", "boolean"}: return "bool"
    if tok in {"any"}: return "str"
    # Bare containers → safe concrete typing
    if tok in {"list", "array"}: return "List[str]"       # or "List[float]" if you prefer
    if tok in {"tuple"}: return "List[str]"               # tuples not supported; degrade to list
    if tok in {"dict", "object"}: return "Dict[str, str]" # ensure object schema has string props
    # Fallback: be safe
    return "str"

def _strip_optional(t: str) -> str:
    if t.startswith("Optional[") and t.endswith("]"):
        return t[len("Optional["):-1]
    return t

def _split_top_level(s: str) -> List[str]:
    """Split by commas not nested inside []"""
    out, depth, buf = [], 0, []
    for ch in s:
        if ch == '[':
            depth += 1
            buf.append(ch)
        elif ch == ']':
            depth -= 1
            buf.append(ch)
        elif ch == ',' and depth == 0:
            out.append(''.join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        out.append(''.join(buf))
    return out

def _parse_generic(type_str: str) -> str:
    s = type_str.strip()
    s = re.sub(r"\s+", "", s)
    s = (s
         .replace("Optional[", "optional[")
         .replace("List[", "list[")
         .replace("Tuple[", "tuple[")
         .replace("Dict[", "dict["))

    def parse(s: str) -> str:
        # optional[X]
        if s.startswith("optional[") and s.endswith("]"):
            inner = parse(s[len("optional["):-1])
            return f"Optional[{inner}]"

        # list[X]
        if s.startswith("list[") and s.endswith("]"):
            inner = parse(s[len("list["):-1])
            # If the inner failed to resolve to a typing param (e.g., still 'List[str]'),
            # fall back to a simple scalar to ensure JSON Schema 'items' is concrete.
            if inner in {"List[str]", "List[float]", "Dict[str, str]"}:
                inner = "str"
            return f"List[{inner}]"

        # tuple[A,B,...]  →  List[common] or List[str]
        if s.startswith("tuple[") and s.endswith("]"):
            inner = s[len("tuple["):-1]
            parts = _split_top_level(inner)
            parsed = [parse(p) for p in parts] if parts else []
            # If all parsed types are identical and scalar, keep that; else default to str
            scalars = {"str", "int", "float", "bool"}
            if parsed and all(p in scalars for p in parsed) and len(set(parsed)) == 1:
                return f"List[{parsed[0]}]"
            # Special-case common numeric range tuples like (float,float)
            if parsed and all(p in {"int", "float"} for p in parsed):
                return "List[float]"
            return "List[str]"

        # dict[K,V]
        if s.startswith("dict[") and s.endswith("]"):
            inner = s[len("dict["):-1]
            parts = _split_top_level(inner)
            if len(parts) != 2:
                return "Dict[str, str]"
            k, v = parse(parts[0]), parse(parts[1])
            # Keys must be stringy for JSON object schemas
            k = "str"
            # Guard V against nested containers without concrete leaf types
            if v in {"List[str]", "List[float]", "Dict[str, str]"}:
                v = "str"
            return f"Dict[{k}, {v}]"

        # primitive / bare tokens
        return _normalize_primitive(s)

    return parse(s)

def _py_type(t: Optional[str]) -> str:
    if not t:
        return "str"
    return _parse_generic(t)


# ----------------------------
# GENERATE: MCP tools module
# ----------------------------
def render_tools_module(templates: List[Dict[str, Any]]) -> str:
    # Default param aliasing used when templates don't provide query_name
    DEFAULT_PARAM_ALIASES: Dict[str, str] = {
        "element": "elements",
        "fields_csv": "fields",
    }

    lines: List[str] = []
    lines.append("from __future__ import annotations")
    lines.append("")
    lines.append("import json as _json")
    lines.append("from typing import Any, Optional, List, Tuple, Dict")
    lines.append("import requests")
    lines.append("from mcp.server.fastmcp import FastMCP")
    # Use absolute import so generated module works inside subpackage (auto_generated)
    lines.append("from mcp_materials_project.client import MaterialsProjectClient")
    lines.append("")
    # Add convenient serializer for complex query params
    lines.append("def _serialize_param(value: Any) -> Any:")
    lines.append("    # For complex types, default to JSON string; primitives pass through")
    lines.append("    if isinstance(value, (list, tuple, dict)):")
    lines.append("        try: return _json.dumps(value, ensure_ascii=False)")
    lines.append("        except Exception: return str(value)")
    lines.append("    return value")
    lines.append("")
    # Coerce JSON-looking strings into real lists/dicts before explode/serialization
    lines.append("def _coerce_arg(value: Any) -> Any:")
    lines.append("    # Try to turn JSON-like strings into real Python objects")
    lines.append("    if isinstance(value, str):")
    lines.append("        s = value.strip()")
    lines.append("        # JSON arrays/objects like \"[0,1]\" or \"{...}\"")
    lines.append("        if (s.startswith('[') and s.endswith(']')) or (s.startswith('{') and s.endswith('}')):")
    lines.append("            try: return _json.loads(s)")
    lines.append("            except Exception: pass")
    lines.append("        # Also accept simple comma-separated scalars: \"a,b,c\" or \"1,2\"")
    lines.append("        if ',' in s and '[' not in s and ']' not in s and '{' not in s and '}' not in s:")
    lines.append("            parts = [p.strip() for p in s.split(',') if p.strip()]")
    lines.append("            # Best-effort numeric cast")
    lines.append("            def _maybe_num(x: str):")
    lines.append("                try:")
    lines.append("                    if x.lower() in {'true','false'}: return x.lower() == 'true'")
    lines.append("                    if '.' in x or 'e' in x.lower(): return float(x)")
    lines.append("                    return int(x)")
    lines.append("                except Exception:")
    lines.append("                    return x")
    lines.append("            return [_maybe_num(p) for p in parts]")
    lines.append("    return value")
    lines.append("")
    lines.append("def _client() -> MaterialsProjectClient:")
    lines.append("    return MaterialsProjectClient()")
    lines.append("")
    lines.append("def register_generated_tools(mcp: FastMCP) -> None:")
    if not templates:
        lines.append("    # No templates found; nothing to register.")
        return "\n".join(lines) + "\n"

    for tpl in templates:
        name = tpl.get("name") or "unnamed_tool"
        method = (tpl.get("method") or "GET").upper()
        path = tpl.get("path") or ""
        description = tpl.get("description") or f"Auto-generated tool for {method} {path}"
        params = tpl.get("params") or []
        response_cfg = tpl.get("response") or {}
        response_mode = str(response_cfg.get("mode") or "json").lower()
        response_pretty = bool(response_cfg.get("pretty", False))
        # Support both snake_case and camelCase in templates
        select_keys = response_cfg.get("select_keys") or response_cfg.get("selectKeys") or []
        select_all = False
        try:
            if isinstance(select_keys, str) and select_keys.strip() == "*":
                select_all = True
            elif isinstance(select_keys, list) and any(str(k).strip() == "*" for k in select_keys):
                select_all = True
        except Exception:
            select_all = False
        coerce_types: Dict[str, str] = response_cfg.get("coerce_types") or response_cfg.get("coerceTypes") or {}
        exclude_keys: List[str] = response_cfg.get("exclude_keys") or response_cfg.get("excludeKeys") or []
        envelope_cfg = response_cfg.get("envelope") or response_cfg.get("envelopeConfig") or False
        single_doc = bool(response_cfg.get("single", response_cfg.get("singleDoc", False)))
        extras = response_cfg.get("extras") or {}

        # Build function signature pieces with enhanced type handling
        arg_defs: List[str] = []
        for p in params:
            p_name = p.get("name")
            if not p_name:
                continue
            p_type = _py_type(p.get("type"))
            required = bool(p.get("required", False))
            template_default = p.get("default")

            # If not required and no default => use concrete type with None default
            if not required and template_default is None:
                # Avoid Optional[...] in type hints to keep JSON Schema with a top-level "type"
                ann = _strip_optional(p_type)
                arg_defs.append(f"{p_name}: {ann} = None")
                continue

            # If there's an explicit default, try to render it
            if not required and template_default is not None:
                if isinstance(template_default, str):
                    default_val = f'\"{template_default}\"'
                elif isinstance(template_default, bool):
                    default_val = "True" if template_default else "False"
                else:
                    default_val = str(template_default)
                arg_defs.append(f"{p_name}: {p_type} = {default_val}")
                continue

            # Required
            arg_defs.append(f"{p_name}: {p_type}")

        # Assemble query param dict (only include those provided)
        lines.append(f"    @mcp.tool(name=\"{name}\", description=\"{description}\")")
        lines.append(f"    async def {name}({', '.join(arg_defs)}) -> Any:")
        lines.append(f'        """{description}"""')
        lines.append("        client = _client()")
        lines.append("        try:")
        # Build params dict with enhanced coercion + serialization
        lines.append("            _params = {}")
        for p in params:
            p_name = p.get("name")
            if not p_name:
                continue
            # Use template-provided query_name, or default alias if present, or p_name
            qn = p.get("query_name") or p.get("queryName") or DEFAULT_PARAM_ALIASES.get(p_name, p_name)
            required = bool(p.get("required", False))
            explode = bool(p.get("explode", False))  # optional template flag

            if required:
                # Always include; coerce JSON-looking strings, then serialize
                lines.append(f"            _val = _coerce_arg({p_name})")
                lines.append(f"            _params[\"{qn}\"] = _serialize_param(_val)")
                continue

            # Optional
            lines.append(f"            if {p_name} is not None:")
            lines.append(f"                _val = _coerce_arg({p_name})")
            if explode:
                # Explode lists/tuples to multiple params: ?{name}=a&{name}=b...
                lines.append(f"                if isinstance(_val, (list, tuple)):")
                lines.append(f"                    _params[\"{qn}\"] = list(_val)")
                lines.append(f"                elif isinstance(_val, dict):")
                lines.append(f"                    for _k, _v in _val.items():")
                lines.append(f"                        _params[f\"{qn}[{{_k}}]\"] = _serialize_param(_v)")
                lines.append(f"                else:")
                lines.append(f"                    _params[\"{qn}\"] = _serialize_param(_val)")
            else:
                # Single param; JSON encode complex values
                lines.append(f"                _params[\"{qn}\"] = _serialize_param(_val)")

        # ----------- ALWAYS rewrite outgoing fields to include what we'll project -----------
        # Merge template select_keys (if any) plus 'material_id' into the outgoing 'fields'
        # unless the caller explicitly asked for all_fields=True.
        sel_literal = "{" + ", ".join([f"'{str(k)}'" for k in (select_keys if isinstance(select_keys, list) else [])]) + "}"
        lines.append(f"            _select_keys = {sel_literal}")
        lines.append("            _force_keys = set(_select_keys) | {'material_id'}")
        lines.append("            _af = _params.get('all_fields')")
        lines.append("            _all_fields = False")
        lines.append("            if _af is not None:")
        lines.append("                s = str(_af).strip().lower()")
        lines.append("                _all_fields = s in {'1','true','t','yes','y'}")
        lines.append("            if not _all_fields and _force_keys:")
        lines.append("                def _as_list(v):")
        lines.append("                    if v is None: return []")
        lines.append("                    if isinstance(v, (list, tuple)): return list(v)")
        lines.append("                    s = str(v).strip()")
        lines.append("                    if s.startswith('[') and s.endswith(']'):")
        lines.append("                        try:")
        lines.append("                            dec = _json.loads(s)")
        lines.append("                            if isinstance(dec, list): return dec")
        lines.append("                        except Exception: pass")
        lines.append("                    return [p.strip() for p in s.split(',') if p.strip()]")
        lines.append("                _cur_fields = _as_list(_params.get('fields'))")
        lines.append("                for _k in _force_keys:")
        lines.append("                    if _k not in _cur_fields:")
        lines.append("                        _cur_fields.append(_k)")
        lines.append("                if _cur_fields:")
        lines.append("                    _params['fields'] = _cur_fields")

        # Call server
        if method == "GET":
            lines.append(f"            _resp = client.get(\"{path}\", params=_params)")
        else:
            # Only GET supported initially
            lines.append("            raise RuntimeError(\"Only GET method is supported by the generator currently.\")")

        # ----------- normalize: detect server envelope {total_count, data} -----------
        lines.append("            _has_server_env = isinstance(_resp, dict) and 'data' in _resp")
        lines.append("            _items = _resp['data'] if _has_server_env else _resp")

        # ----------- single doc shaping (applies inside data) -----------
        if single_doc:
            lines.append("            if isinstance(_items, list):")
            lines.append("                _items = _items[0] if _items else None")

        # ----------- selection/projection (inside data) -----------
        if select_keys and not select_all:
            # Build coerce helper inline to avoid imports
            lines.append("            def _coerce_value(key, value):")
            lines.append("                _t = {\n" + ",\n".join([f"                    \"{k}\": \"{v}\"" for k, v in coerce_types.items()]) + "\n                }")
            lines.append("                typ = _t.get(key)")
            lines.append("                try:")
            lines.append("                    if typ == 'int': return int(value) if value is not None else None")
            lines.append("                    if typ == 'float': return float(value) if value is not None else None")
            lines.append("                    if typ == 'bool':")
            lines.append("                        if isinstance(value, bool): return value")
            lines.append("                        s = str(value).strip().lower() if value is not None else ''")
            lines.append("                        return True if s in {'1','true','t','yes','y'} else (False if s in {'0','false','f','no','n'} else None)")
            lines.append("                    if typ == 'str': return '' if value is None else str(value)")
            lines.append("                except Exception:")
            lines.append("                    return value")
            lines.append("                return value")
            # Apply projection
            lines.append("            if isinstance(_items, list):")
            lines.append("                _proj = []")
            lines.append("                for _it in _items:")
            lines.append("                    if isinstance(_it, dict):")
            lines.append("                        _obj = {}")
            for k in select_keys:
                lines.append(f"                        _obj[\"{k}\"] = _coerce_value(\"{k}\", _it.get(\"{k}\"))")
            lines.append("                        _proj.append(_obj)")
            lines.append("                    else:")
            lines.append("                        _proj.append(_it)")
            lines.append("                _items = _proj")
            lines.append("            elif isinstance(_items, dict):")
            lines.append("                _obj = {}")
            for k in select_keys:
                lines.append(f"                _obj[\"{k}\"] = _coerce_value(\"{k}\", _items.get(\"{k}\"))")
            lines.append("                _items = _obj")
        elif exclude_keys:
            # Exclude specified keys from dicts
            lines.append("            _ex_keys = {" + ", ".join([f'\"{k}\"' for k in exclude_keys]) + "}")
            lines.append("            if isinstance(_items, list):")
            lines.append("                _proj = []")
            lines.append("                for _it in _items:")
            lines.append("                    if isinstance(_it, dict):")
            lines.append("                        _obj = {k: v for k, v in _it.items() if k not in _ex_keys}")
            lines.append("                        _proj.append(_obj)")
            lines.append("                    else:")
            lines.append("                        _proj.append(_it)")
            lines.append("                _items = _proj")
            lines.append("            elif isinstance(_items, dict):")
            lines.append("                _items = {k: v for k, v in _items.items() if k not in _ex_keys}")

        # ----------- optional custom envelope (skip if server already enveloped) -----------
        lines.append("            if not _has_server_env:")
        if envelope_cfg:
            lines.append("                _items_key = 'items'")
            lines.append("                _count_key = 'count'")
            lines.append("                if isinstance(envelope_cfg := " + _json.dumps(envelope_cfg) + ", dict):")
            lines.append("                    _items_key = envelope_cfg.get('items_key', _items_key)")
            lines.append("                    _count_key = envelope_cfg.get('count_key', _count_key)")
            lines.append("                _out = {}")
            for ex_k, ex_v in extras.items():
                lines.append(f"                _out[\"{ex_k}\"] = { _json.dumps(ex_v) }")
            lines.append("                if isinstance(_items, list):")
            lines.append("                    _out[_items_key] = _items")
            lines.append("                    try:")
            lines.append("                        _out[_count_key] = len(_items)")
            lines.append("                    except Exception:")
            lines.append("                        _out[_count_key] = None")
            lines.append("                else:")
            lines.append("                    _out[_items_key] = _items")
            lines.append("                    _out[_count_key] = 1 if _items is not None else 0")
            lines.append("                _resp = _out")
        else:
            # no template envelope; just assign shaped items back
            lines.append("                _resp = _items")

        # ----------- if server envelope exists, put items back into data -----------
        lines.append("            if _has_server_env:")
        lines.append("                _resp['data'] = _items")

        # Final return according to mode
        if response_mode == "text":
            if response_pretty:
                lines.append("            try:")
                lines.append("                return _json.dumps(_resp, indent=2, ensure_ascii=False)")
                lines.append("            except Exception:")
                lines.append("                return str(_resp)")
            else:
                lines.append("            return str(_resp)")
        else:
            lines.append("            return _resp")
        lines.append("        except requests.HTTPError as e:")
        lines.append("            status = getattr(getattr(e, 'response', None), 'status_code', None)")
        lines.append("            body = None")
        lines.append("            try:")
        lines.append("                if e.response is not None:")
        lines.append("                    body = e.response.json()")
        lines.append("            except Exception:")
        lines.append("                body = getattr(getattr(e, 'response', None), 'text', None)")
        lines.append("            return {\"error\": {\"type\": \"http_error\", \"status\": status, \"message\": str(e), \"body\": body}}")
        lines.append("        except Exception as e:")
        lines.append("            return {\"error\": {\"type\": \"unexpected_error\", \"message\": str(e)}}")
        lines.append("")

    return "\n".join(lines) + "\n"


def generate_tools(templates_dir: os.PathLike[str] | str, output_file: os.PathLike[str] | str) -> Optional[Path]:
    tdir = Path(templates_dir)
    ofile = Path(output_file)
    templates = load_templates(tdir)
    module_src = render_tools_module(templates)
    try:
        ofile.write_text(module_src, encoding="utf-8")
        return ofile
    except Exception:
        return None


# ----------------------------
# GENERATE: Kani wrapper module
# ----------------------------
def render_kani_module(templates: List[Dict[str, Any]]) -> str:
    # Default param aliasing used when templates don't provide query_name
    DEFAULT_PARAM_ALIASES: Dict[str, str] = {
        "element": "elements",
        "fields_csv": "fields",
    }

    lines: List[str] = []
    lines.append("from __future__ import annotations")
    lines.append("")
    lines.append("from typing import Any, Optional, List, Tuple, Dict, Annotated")
    lines.append("from kani import ai_function, AIParam")
    lines.append("")
    lines.append("class GeneratedKaniTools:")
    if not templates:
        lines.append("    pass")
        return "\n".join(lines) + "\n"

    for tpl in templates:
        name = tpl.get("name") or "unnamed_tool"
        description = tpl.get("description") or f"Auto-generated Kani function for {name}"
        params = tpl.get("params") or []

        # Build argument signature with enhanced types
        arg_defs: List[str] = ["self"]
        for p in params:
            p_name = p.get("name")
            if not p_name:
                continue
            base_t = _py_type(p.get("type"))
            p_desc = (p.get("description") or p.get("desc") or "").replace('"', '\\"')
            p_example = p.get("example")  # optional
            if p_desc or p_example is not None:
                ai_param_bits = [f'desc="{p_desc}"'] if p_desc else []
                if p_example is not None:
                    # stringify safely
                    ex = _json.dumps(p_example, ensure_ascii=False)
                    ai_param_bits.append(f"example={ex}")
                ai_param = ", ".join(ai_param_bits) or ""
                p_type = f"Annotated[{base_t}, AIParam({ai_param})]"
            else:
                p_type = base_t
            required = bool(p.get("required", False))
            template_default = p.get("default")
            
            if required:
                arg_defs.append(f"{p_name}: {p_type}")
            else:
                # Use template default if available, otherwise None for Optional types
                if template_default is not None:
                    if isinstance(template_default, bool):
                        default_val = str(template_default)
                    elif isinstance(template_default, (int, float)):
                        default_val = str(template_default)
                    else:
                        default_val = f'"{template_default}"'
                    arg_defs.append(f"{p_name}: {p_type} = {default_val}")
                else:
                    # Avoid Optional[...] in type hints to keep JSON Schema with a top-level "type"
                    ann = _strip_optional(p_type)
                    arg_defs.append(f"{p_name}: {ann} = None")

        fn_desc = (description or f"Auto-generated Kani function for {name}").replace('"', '\\"')
        lines.append(f'    @ai_function(desc="{fn_desc}", auto_truncate=128000)')
        lines.append(f"    async def {name}({', '.join(arg_defs)}) -> str:")
        lines.append(f'        """{fn_desc}"""')
        lines.append("        _args = {}")
        for p in params:
            p_name = p.get("name")
            if not p_name:
                continue
            required = bool(p.get("required", False))
            template_default = p.get("default")
            # Use template-provided query_name or our default aliases
            qn = p.get("query_name") or p.get("queryName") or DEFAULT_PARAM_ALIASES.get(p_name, p_name)
            
            if required:
                lines.append(f"        _args[\"{qn}\"] = {p_name}")
            else:
                if template_default is not None:
                    # Always include the parameter with its default value
                    if isinstance(template_default, bool):
                        lines.append(f"        _args[\"{qn}\"] = {p_name} if {p_name} is not None else {template_default}")
                    elif isinstance(template_default, (int, float)):
                        lines.append(f"        _args[\"{qn}\"] = {p_name} if {p_name} is not None else {template_default}")
                    else:
                        lines.append(f"        _args[\"{qn}\"] = {p_name} if {p_name} is not None else \"{template_default}\"")
                else:
                    lines.append(f"        if {p_name} is not None:")
                    lines.append(f"            _args[\"{qn}\"] = {p_name}")
        lines.append(f"        _result = await self._proxy.call_tool(\"{name}\", _args)")
        lines.append("        try:")
        lines.append(f"            self.recent_tool_outputs.append({{\"tool\": \"{name}\", \"args\": _args, \"result\": _result}})")
        lines.append("        except Exception:")
        lines.append("            pass")
        lines.append("        try:")
        lines.append("            # Prefer pretty JSON if possible; fall back to str()")
        lines.append("            import json as _json")
        lines.append("            return _json.dumps(_result, ensure_ascii=False)")
        lines.append("        except Exception:")
        lines.append("            return str(_result)")
        lines.append("")

    return "\n".join(lines) + "\n"


def generate_kani(templates_dir: os.PathLike[str] | str, output_file: os.PathLike[str] | str) -> Optional[Path]:
    tdir = Path(templates_dir)
    ofile = Path(output_file)
    templates = load_templates(tdir)
    module_src = render_kani_module(templates)
    try:
        ofile.write_text(module_src, encoding="utf-8")
        return ofile
    except Exception:
        return None
