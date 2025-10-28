"""
Decorators for handler methods.

This module provides decorators that can be used with AI functions to add
common functionality like result tracking, timing, and error handling.
"""

from functools import wraps
from typing import Callable, Any
import logging

_log = logging.getLogger(__name__)


def track_tool_output(func: Callable) -> Callable:
    """
    Decorator that automatically tracks tool output after AI function execution.
    
    This decorator should be applied to @ai_function methods. It will automatically
    call self._track_tool_output() with the function name and result after execution.
    
    Usage:
        @ai_function(desc="...")
        @track_tool_output
        async def my_ai_function(self, ...):
            # ... function implementation ...
            return result
    
    Note:
        - The decorated function must be a method (have 'self' as first arg)
        - The 'self' object must have a _track_tool_output method (provided by BaseHandler)
        - Works with both sync and async functions
    
    Args:
        func: The function to decorate
        
    Returns:
        Wrapped function that tracks its output
    """
    # Handle async functions
    if func.__name__.startswith('async') or hasattr(func, '__wrapped__'):
        @wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            result = await func(self, *args, **kwargs)
            
            # Track the output if the handler has the tracking method
            if hasattr(self, '_track_tool_output'):
                try:
                    self._track_tool_output(func.__name__, result)
                except Exception as e:
                    _log.warning(f"Failed to track tool output for {func.__name__}: {e}")
            
            return result
        return async_wrapper
    
    # Handle sync functions
    @wraps(func)
    def sync_wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        
        # Track the output if the handler has the tracking method
        if hasattr(self, '_track_tool_output'):
            try:
                self._track_tool_output(func.__name__, result)
            except Exception as e:
                _log.warning(f"Failed to track tool output for {func.__name__}: {e}")
        
        return result
    
    return sync_wrapper


def auto_track(tool_name: str = None) -> Callable:
    """
    Decorator factory that tracks tool output with a custom tool name.
    
    This is useful when you want to specify a different name than the function name
    for tracking purposes.
    
    Usage:
        @ai_function(desc="...")
        @auto_track(tool_name="custom_tool_name")
        async def my_function(self, ...):
            return result
    
    Args:
        tool_name: Optional custom name to use for tracking (defaults to function name)
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        # Determine if async
        is_async = func.__name__.startswith('async') or hasattr(func, '__wrapped__')
        
        if is_async:
            @wraps(func)
            async def async_wrapper(self, *args, **kwargs):
                result = await func(self, *args, **kwargs)
                
                # Use custom tool name or function name
                name = tool_name or func.__name__
                
                if hasattr(self, '_track_tool_output'):
                    try:
                        self._track_tool_output(name, result)
                    except Exception as e:
                        _log.warning(f"Failed to track tool output for {name}: {e}")
                
                return result
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(self, *args, **kwargs):
                result = func(self, *args, **kwargs)
                
                # Use custom tool name or function name
                name = tool_name or func.__name__
                
                if hasattr(self, '_track_tool_output'):
                    try:
                        self._track_tool_output(name, result)
                    except Exception as e:
                        _log.warning(f"Failed to track tool output for {name}: {e}")
                
                return result
            return sync_wrapper
    
    return decorator

