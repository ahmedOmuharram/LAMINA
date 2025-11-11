"""
SearXNG API Client

This module provides the core SearXNG API interactions for search and engine stats.
"""

import json
import logging
import requests
from typing import Any, Dict, List, Optional

from ..shared import success_result, error_result, ErrorType, Confidence

_log = logging.getLogger(__name__)


class SearXNGClient:
    """Client for interacting with SearXNG API."""
    
    def __init__(self, searxng_url: str = "http://localhost:8080"):
        """
        Initialize the SearXNG client.
        
        Args:
            searxng_url: Base URL of the SearXNG instance
        """
        self.searxng_url = searxng_url.rstrip('/')
        self.search_endpoint = f"{self.searxng_url}/search"
    
    def search(
        self, 
        query: str, 
        format: str = "json",
        categories: Optional[List[str]] = None,
        engines: Optional[List[str]] = None,
        language: str = "auto",
        safesearch: int = 0,
        time_range: str = "",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform a search using SearXNG.
        
        Args:
            query: Search query string
            format: Response format (json, html, csv, rss)
            categories: List of search categories (general, images, videos, news, etc.)
            engines: List of specific search engines to use
            language: Search language (auto, en, de, etc.)
            safesearch: Safe search level (0=none, 1=moderate, 2=strict)
            time_range: Time range for results (empty, day, week, month, year)
            **kwargs: Additional search parameters
            
        Returns:
            Dictionary containing search results
        """
        try:
            # Prepare search parameters
            params = {
                'q': query,
                'format': format,
                'language': language,
                'safesearch': safesearch,
                'time_range': time_range,
            }
            
            # Add categories if specified
            if categories:
                for category in categories:
                    params[f'category_{category}'] = '1'
            
            # Add engines if specified
            if engines:
                params['engines'] = ','.join(engines)
            
            # Add any additional parameters
            params.update(kwargs)
            
            # Make the request
            response = requests.get(
                self.search_endpoint,
                params=params,
                timeout=30,
                headers={
                    'User-Agent': 'MCP-Materials-Project/1.0 (Academic Research Bot)',
                    'Accept': 'application/json' if format == 'json' else '*/*'
                }
            )
            response.raise_for_status()
            
            if format == 'json':
                data = response.json()
                return success_result(
                    handler="search",
                    function="search",
                    data=data,
                    citations=["SearXNG"],
                    confidence=Confidence.HIGH if data.get('results') else Confidence.LOW
                )
            else:
                return success_result(
                    handler="search",
                    function="search",
                    data={'content': response.text, 'format': format},
                    citations=["SearXNG"]
                )
                
        except requests.exceptions.RequestException as e:
            _log.error(f"SearXNG search request failed: {e}")
            return error_result(
                handler="search",
                function="search",
                error=f'Search request failed: {str(e)}',
                error_type=ErrorType.API_ERROR,
                citations=["SearXNG"]
            )
        except json.JSONDecodeError as e:
            _log.error(f"Failed to parse SearXNG JSON response: {e}")
            return error_result(
                handler="search",
                function="search",
                error=f'Failed to parse search results: {str(e)}',
                error_type=ErrorType.API_ERROR,
                citations=["SearXNG"]
            )
        except Exception as e:
            _log.error(f"Unexpected error during SearXNG search: {e}")
            return error_result(
                handler="search",
                function="search",
                error=f'Unexpected error: {str(e)}',
                error_type=ErrorType.COMPUTATION_ERROR,
                citations=["SearXNG"]
            )
    
    def search_scientific(
        self,
        query: str,
        include_arxiv: bool = True,
        include_pubmed: bool = True,
        include_scholar: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform a scientific literature search optimized for academic sources.
        
        Args:
            query: Scientific search query
            include_arxiv: Include arXiv preprints
            include_pubmed: Include PubMed medical literature
            include_scholar: Include Google Scholar
            **kwargs: Additional search parameters
            
        Returns:
            Dictionary containing scientific search results
        """
        # Configure for scientific search
        # Extract categories from kwargs if provided, otherwise use default
        categories = kwargs.pop('categories', ['general'])
        # Note: Not specifying engines to avoid JSON serialization issues
        # The search will use all available engines and filter results
        
        return self.search(
            query=query,
            categories=categories,
            language='en',  # Scientific literature is mostly in English
            safesearch=0,   # No filtering for scientific content
            **kwargs
        )
    
    def search_materials_science(
        self,
        query: str,
        include_phase_diagrams: bool = True,
        include_thermodynamics: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform a materials science specific search.
        
        Args:
            query: Materials science search query
            include_phase_diagrams: Include phase diagram related results
            include_thermodynamics: Include thermodynamic property results
            **kwargs: Additional search parameters
            
        Returns:
            Dictionary containing materials science search results
        """
        # Enhance query for materials science
        enhanced_query = query
        
        if include_phase_diagrams and 'phase' not in query.lower():
            enhanced_query += " phase diagram"
        
        if include_thermodynamics and any(term in query.lower() for term in ['temperature', 'pressure', 'composition']):
            enhanced_query += " thermodynamics"
        
        return self.search_scientific(
            query=enhanced_query,
            include_arxiv=True,
            include_pubmed=False,  # Less relevant for materials science
            include_scholar=True,
            **kwargs
        )

