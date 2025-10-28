"""
Standardized Result Wrappers for AI Functions

This module provides utilities for creating consistent return structures across all handlers.
All @ai_function decorated methods should use these wrappers to ensure uniform output format
for both the frontend (status indication) and the LLM (informative tool outputs).

Schema Version: 0.0.`
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import time
from functools import wraps
import logging

_log = logging.getLogger(__name__)

SCHEMA_VERSION = "0.0.1"


def success_result(
    handler: str,
    function: str,
    data: Dict[str, Any],
    citations: Optional[List[str]] = None,
    has_image: bool = False,
    image_url: Optional[str] = None,
    has_html: bool = False,
    html_url: Optional[str] = None,
    notes: Optional[List[str]] = None,
    warnings: Optional[List[str]] = None,
    suggestions: Optional[List[str]] = None,
    confidence: Optional[str] = None,
    caveats: Optional[List[str]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
    duration_ms: Optional[float] = None,
    **extra_fields
) -> Dict[str, Any]:
    """
    Create a standardized success response.
    
    Args:
        handler: Handler name (e.g., "magnets", "calphad")
        function: Function name (e.g., "assess_magnet_strength_with_doping")
        data: Dictionary containing all domain-specific results
        citations: List of citation strings (default: [])
        has_image: Whether visual output exists (default: False)
        image_url: Full URL to image if has_image=True
        has_html: Whether interactive HTML exists (default: False)
        html_url: Full URL to HTML if has_html=True
        notes: List of human-readable context for interpreting results
        warnings: List of cautions about result validity
        suggestions: List of hints on what to try next
        confidence: Quality indicator ("high" | "medium" | "low" | None)
        caveats: List of limitations and assumptions
        diagnostics: Debug information (not shown to end users)
        duration_ms: Execution time in milliseconds
        **extra_fields: Any additional fields to include at top level
    
    Returns:
        Standardized success dictionary
    """
    result = {
        "success": True,
        "error": None,
        "error_type": None,
        "metadata": {
            "handler": handler,
            "function": function,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "version": SCHEMA_VERSION
        },
        "citations": citations or [],
        "data": data
    }
    
    # Add duration if provided
    if duration_ms is not None:
        result["metadata"]["duration_ms"] = duration_ms
    
    # Add presentation layer if applicable
    if has_image:
        result["has_image"] = True
        if image_url:
            result["image_url"] = image_url
    else:
        result["has_image"] = False
    
    if has_html:
        result["has_html"] = True
        if html_url:
            result["html_url"] = html_url
    else:
        result["has_html"] = False
    
    # Add guidance layer if provided
    if notes is not None:
        result["notes"] = notes if isinstance(notes, list) else [notes]
    
    if warnings is not None:
        result["warnings"] = warnings if isinstance(warnings, list) else [warnings]
    
    if suggestions is not None:
        result["suggestions"] = suggestions if isinstance(suggestions, list) else [suggestions]
    
    if confidence is not None:
        result["confidence"] = confidence
    
    if caveats is not None:
        result["caveats"] = caveats if isinstance(caveats, list) else [caveats]
    
    # Add diagnostics if provided
    if diagnostics is not None:
        result["diagnostics"] = diagnostics
    
    # Add any extra fields
    result.update(extra_fields)
    
    return result


def error_result(
    handler: str,
    function: str,
    error: Union[str, Exception],
    error_type: str = "computation_error",
    citations: Optional[List[str]] = None,
    suggestions: Optional[List[str]] = None,
    notes: Optional[List[str]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
    duration_ms: Optional[float] = None,
    **extra_fields
) -> Dict[str, Any]:
    """
    Create a standardized error response.
    
    Args:
        handler: Handler name (e.g., "magnets", "calphad")
        function: Function name
        error: Error message or exception
        error_type: Error category ("not_found" | "invalid_input" | "api_error" | 
                   "computation_error" | "timeout" | "permission_denied")
        citations: List of citation strings (even on error)
        suggestions: List of hints on what to try next
        notes: List of additional context
        diagnostics: Debug information
        duration_ms: Execution time in milliseconds
        **extra_fields: Any additional fields to include at top level
    
    Returns:
        Standardized error dictionary
    """
    error_message = str(error) if isinstance(error, Exception) else error
    
    result = {
        "success": False,
        "error": error_message,
        "error_type": error_type,
        "metadata": {
            "handler": handler,
            "function": function,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "version": SCHEMA_VERSION
        },
        "citations": citations or []
    }
    
    # Add duration if provided
    if duration_ms is not None:
        result["metadata"]["duration_ms"] = duration_ms
    
    # Add guidance if provided
    if suggestions is not None:
        result["suggestions"] = suggestions if isinstance(suggestions, list) else [suggestions]
    
    if notes is not None:
        result["notes"] = notes if isinstance(notes, list) else [notes]
    
    # Add diagnostics if provided
    if diagnostics is not None:
        result["diagnostics"] = diagnostics
    
    # Add any extra fields
    result.update(extra_fields)
    
    return result


def simple_success(data: Dict[str, Any], citations: Optional[List[str]] = None, **optional) -> Dict[str, Any]:
    """
    Create a simple success result without metadata (for backward compatibility or simple cases).
    
    Args:
        data: Domain-specific results
        citations: List of citation strings
        **optional: Any optional fields (notes, warnings, etc.)
    
    Returns:
        Simple success dictionary with data at top level
    """
    result = {
        "success": True,
        "citations": citations or [],
        **data,
        **optional
    }
    return result


def simple_error(error: Union[str, Exception], citations: Optional[List[str]] = None, **optional) -> Dict[str, Any]:
    """
    Create a simple error result without metadata (for backward compatibility).
    
    Args:
        error: Error message or exception
        citations: List of citation strings
        **optional: Any optional fields (suggestions, notes, etc.)
    
    Returns:
        Simple error dictionary
    """
    error_message = str(error) if isinstance(error, Exception) else error
    result = {
        "success": False,
        "error": error_message,
        "citations": citations or [],
        **optional
    }
    return result


def with_timing(handler: str):
    """
    Decorator that automatically adds timing metadata to function results.
    Use this for functions that already return dict results but need timing added.
    
    Usage:
        @ai_function(...)
        @with_timing("magnets")
        async def some_function(self, ...):
            return {"success": True, "data": {...}, ...}
    
    Args:
        handler: Handler name
    
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            duration_ms = (time.time() - start_time) * 1000
            
            # If result is a dict with success field, add metadata
            if isinstance(result, dict) and "success" in result:
                if "metadata" not in result:
                    result["metadata"] = {}
                result["metadata"].update({
                    "handler": handler,
                    "function": func.__name__,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "duration_ms": duration_ms,
                    "version": SCHEMA_VERSION
                })
            
            return result
        return wrapper
    return decorator


# Common error types for consistency
class ErrorType:
    """Standard error type constants"""
    NOT_FOUND = "not_found"
    INVALID_INPUT = "invalid_input"
    API_ERROR = "api_error"
    COMPUTATION_ERROR = "computation_error"
    TIMEOUT = "timeout"
    PERMISSION_DENIED = "permission_denied"


# Common confidence levels
class Confidence:
    """Standard confidence level constants"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


def ensure_list(value: Optional[Union[str, List[str]]]) -> List[str]:
    """
    Helper to ensure a value is a list.
    
    Args:
        value: String or list of strings or None
    
    Returns:
        List of strings (empty list if None)
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return value


def merge_citations(*citation_lists: Optional[List[str]]) -> List[str]:
    """
    Merge multiple citation lists, removing duplicates while preserving order.
    
    Args:
        *citation_lists: Variable number of citation lists
    
    Returns:
        Merged list with no duplicates
    """
    seen = set()
    result = []
    for citation_list in citation_lists:
        if citation_list:
            for citation in citation_list:
                if citation not in seen:
                    seen.add(citation)
                    result.append(citation)
    return result

