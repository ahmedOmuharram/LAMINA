"""
Search handlers for web and scientific literature search.

This module provides handlers for SearXNG-based search functionality.
"""

from .searxng_search import SearXNGSearchHandler, handle_searxng_search, handle_searxng_engine_stats

__all__ = [
    "SearXNGSearchHandler",
    "handle_searxng_search",
    "handle_searxng_engine_stats",
]

