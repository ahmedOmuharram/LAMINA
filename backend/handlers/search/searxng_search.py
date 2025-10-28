"""
SearXNG search handler for scientific literature and web search.

This handler integrates SearXNG search capabilities into the MCP Materials Project system,
allowing for scientific literature search and web search functionality.
"""

import json
import logging
import requests
from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import urlparse
import time

from ..base.base import BaseHandler
from .ai_functions import SearchAIFunctionsMixin

_log = logging.getLogger(__name__)


class SearXNGSearchHandler(SearchAIFunctionsMixin, BaseHandler):
    """Handler for SearXNG search functionality."""
    
    def __init__(self, searxng_url: str = "http://localhost:8080"):
        """
        Initialize the SearXNG search handler.
        
        Args:
            searxng_url: Base URL of the SearXNG instance
        """
        super().__init__()
        self.searxng_url = searxng_url.rstrip('/')
        self.search_endpoint = f"{self.searxng_url}/search"
    
    def search(self, 
               query: str, 
               format: str = "json",
               categories: Optional[List[str]] = None,
               engines: Optional[List[str]] = None,
               language: str = "auto",
               safesearch: int = 0,
               time_range: str = "",
               **kwargs) -> Dict[str, Any]:
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
                return response.json()
            else:
                return {'content': response.text, 'format': format}
                
        except requests.exceptions.RequestException as e:
            _log.error(f"SearXNG search request failed: {e}")
            return {
                'error': f'Search request failed: {str(e)}',
                'query': query,
                'results': []
            }
        except json.JSONDecodeError as e:
            _log.error(f"Failed to parse SearXNG JSON response: {e}")
            return {
                'error': f'Failed to parse search results: {str(e)}',
                'query': query,
                'results': []
            }
        except Exception as e:
            _log.error(f"Unexpected error during SearXNG search: {e}")
            return {
                'error': f'Unexpected error: {str(e)}',
                'query': query,
                'results': []
            }
    
    def search_scientific(self, 
                         query: str,
                         include_arxiv: bool = True,
                         include_pubmed: bool = True,
                         include_scholar: bool = True,
                         **kwargs) -> Dict[str, Any]:
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
        categories = ['general']
        # Note: Not specifying engines to avoid JSON serialization issues
        # The search will use all available engines and filter results
        
        return self.search(
            query=query,
            categories=categories,
            language='en',  # Scientific literature is mostly in English
            safesearch=0,   # No filtering for scientific content
            **kwargs
        )
    
    def search_materials_science(self, 
                                query: str,
                                include_phase_diagrams: bool = True,
                                include_thermodynamics: bool = True,
                                **kwargs) -> Dict[str, Any]:
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
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """
        Get statistics about available search engines.
        
        Returns:
            Dictionary containing engine statistics
        """
        try:
            response = requests.get(
                f"{self.searxng_url}/stats",
                timeout=10,
                headers={'User-Agent': 'MCP-Materials-Project/1.0'}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            _log.error(f"Failed to get engine stats: {e}")
            return {'error': f'Failed to get engine stats: {str(e)}'}
    
    def _extract_single_url_content(self, url: str, timeout: int = 10) -> Dict[str, Any]:
        """
        Extract content from a specific URL.
        
        Args:
            url: URL to extract content from
            timeout: Request timeout in seconds
            
        Returns:
            Dictionary containing extracted content and metadata
        """
        try:
            # Validate URL
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                return {
                    'url': url,
                    'error': 'Invalid URL format',
                    'content': '',
                    'title': '',
                    'status_code': None
                }
            
            # Make request with appropriate headers
            headers = {
                'User-Agent': 'MCP-Materials-Project/1.0 (Academic Research Bot)',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
            response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            response.raise_for_status()
            
            # Extract basic content
            content = response.text
            title = ""
            
            # Try to extract title from HTML
            if 'text/html' in response.headers.get('content-type', '').lower():
                import re
                title_match = re.search(r'<title[^>]*>(.*?)</title>', content, re.IGNORECASE | re.DOTALL)
                if title_match:
                    title = title_match.group(1).strip()
            
            # Clean up content (basic HTML stripping)
            if content:
                import re
                # Remove script and style elements
                content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.IGNORECASE | re.DOTALL)
                content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.IGNORECASE | re.DOTALL)
                # Remove HTML tags
                content = re.sub(r'<[^>]+>', ' ', content)
                # Clean up whitespace
                content = re.sub(r'\s+', ' ', content).strip()
                # Limit content length
                if len(content) > 10000:
                    content = content[:10000] + "..."
            
            return {
                'url': url,
                'title': title,
                'content': content,
                'status_code': response.status_code,
                'content_length': len(content),
                'content_type': response.headers.get('content-type', ''),
                'success': True
            }
            
        except requests.exceptions.Timeout:
            _log.warning(f"Timeout while extracting content from {url}")
            return {
                'url': url,
                'error': 'Request timeout',
                'content': '',
                'title': '',
                'status_code': None,
                'success': False
            }
        except requests.exceptions.RequestException as e:
            _log.warning(f"Request failed for {url}: {e}")
            return {
                'url': url,
                'error': f'Request failed: {str(e)}',
                'content': '',
                'title': '',
                'status_code': getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None,
                'success': False
            }
        except Exception as e:
            _log.error(f"Unexpected error extracting content from {url}: {e}")
            return {
                'url': url,
                'error': f'Unexpected error: {str(e)}',
                'content': '',
                'title': '',
                'status_code': None,
                'success': False
            }
    
    def extract_content_from_high_score_urls(self, search_results: Dict[str, Any], min_score: float = 5.0) -> Dict[str, Any]:
        """
        Extract content from URLs in search results that have a score >= min_score.
        
        Args:
            search_results: Search results from SearXNG
            min_score: Minimum score threshold for URL extraction
            
        Returns:
            Dictionary containing original results plus extracted content
        """
        if not search_results:
            return search_results
        
        # Try to find results in different possible keys
        results_list = None
        if 'results' in search_results:
            results_list = search_results['results']
        elif isinstance(search_results, list):
            results_list = search_results
        else:
            # Look for any list in the response that might contain results
            for key, value in search_results.items():
                if isinstance(value, list) and value and isinstance(value[0], dict) and 'url' in value[0]:
                    results_list = value
                    break
        
        if not results_list:
            search_results['extracted_content'] = []
            search_results['extraction_summary'] = f"No results found in search response"
            return search_results
        
        # Filter URLs with score >= min_score
        high_score_urls = []
        for result in results_list:
            if isinstance(result, dict) and result.get('score', 0) >= min_score:
                high_score_urls.append(result)
        
        if not high_score_urls:
            search_results['extracted_content'] = []
            search_results['extraction_summary'] = f"No URLs found with score >= {min_score}"
            return search_results
        
        # Extract content from high-score URLs
        extracted_content = []
        successful_extractions = 0
        
        for result in high_score_urls:
            url = result.get('url', '')
            if not url:
                continue
                
            # Add small delay to be respectful to servers
            time.sleep(0.5)
            
            content_data = self._extract_single_url_content(url)
            content_data['original_score'] = result.get('score', 0)
            content_data['original_title'] = result.get('title', '')
            
            extracted_content.append(content_data)
            
            if content_data.get('success', False):
                successful_extractions += 1
        
        # Add extracted content to results
        search_results['extracted_content'] = extracted_content
        search_results['extraction_summary'] = {
            'total_high_score_urls': len(high_score_urls),
            'successful_extractions': successful_extractions,
            'failed_extractions': len(high_score_urls) - successful_extractions,
            'min_score_threshold': min_score
        }
        
        return search_results

    def extract_url_content(
        self,
        urls: List[str],
        timeout: int = 10
    ) -> Dict[str, Any]:
        """Extract content from specific URLs."""
        if not urls:
            return {'error': 'No URLs provided', 'extracted_content': []}
        
        extracted_content = []
        successful_extractions = 0
        
        for url in urls:
            _log.info(f"Extracting content from URL: {url}")
            
            # Add small delay to be respectful to servers
            time.sleep(0.5)
            
            content_data = self._extract_single_url_content(url, timeout)
            extracted_content.append(content_data)
            
            if content_data.get('success', False):
                successful_extractions += 1
        
        result = {
            'extracted_content': extracted_content,
            'extraction_summary': {
                'total_urls': len(urls),
                'successful_extractions': successful_extractions,
                'failed_extractions': len(urls) - successful_extractions
            }
        }
        
        return result

    def handle_searxng_search(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        """Handle SearXNG search requests."""
        # Extract search parameters
        query = params.get('query', '')
        if not query:
            return {'error': 'Query parameter is required', 'results': []}
        
        search_type = params.get('search_type', 'general')
        extract_content = params.get('extract_content', True)
        min_score = params.get('min_score', 5.0)
        
        # Perform the search
        if search_type == 'scientific':
            search_results = self.search_scientific(
                query=query,
                include_arxiv=params.get('include_arxiv', True),
                include_pubmed=params.get('include_pubmed', True),
                include_scholar=params.get('include_scholar', True)
            )
        elif search_type == 'materials_science':
            search_results = self.search_materials_science(
                query=query,
                include_phase_diagrams=params.get('include_phase_diagrams', True),
                include_thermodynamics=params.get('include_thermodynamics', True)
            )
        else:
            # General search
            search_results = self.search(
                query=query,
                format=params.get('format', 'json'),
                categories=params.get('categories'),
                engines=params.get('engines'),
                language=params.get('language', 'auto'),
                safesearch=params.get('safesearch', 0),
                time_range=params.get('time_range', '')
            )
        
        # Extract content from high-score URLs if requested
        if extract_content and search_results:
            search_results = self.extract_content_from_high_score_urls(search_results, min_score)
        
        return search_results

    def handle_searxng_engine_stats(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        """Handle requests for SearXNG engine statistics."""
        return self.get_engine_stats()


def handle_searxng_search(params: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Handle SearXNG search requests.
    
    Args:
        params: Request parameters containing search query and options
        
    Returns:
        Dictionary containing search results
    """
    handler = SearXNGSearchHandler()
    return handler.handle_searxng_search(params)


def handle_searxng_engine_stats(params: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Handle requests for SearXNG engine statistics.
    
    Args:
        params: Request parameters (unused for stats)
        
    Returns:
        Dictionary containing engine statistics
    """
    handler = SearXNGSearchHandler()
    return handler.get_engine_stats()
