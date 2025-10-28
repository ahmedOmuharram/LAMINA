"""
Content Extraction

This module handles URL content extraction from search results.
"""

import logging
import time
import requests
from typing import Any, Dict, List
from urllib.parse import urlparse

from ..shared import success_result, error_result, ErrorType, Confidence

_log = logging.getLogger(__name__)


def extract_single_url_content(url: str, timeout: int = 10) -> Dict[str, Any]:
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
            return error_result(
                handler="search",
                function="extract_single_url_content",
                error='Invalid URL format',
                error_type=ErrorType.INVALID_INPUT,
                citations=[url]
            )
        
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
        
        return success_result(
            handler="search",
            function="extract_single_url_content",
            data={
                'url': url,
                'title': title,
                'content': content,
                'status_code': response.status_code,
                'content_length': len(content),
                'content_type': response.headers.get('content-type', '')
            },
            citations=[url],
            confidence=Confidence.HIGH
        )
        
    except requests.exceptions.Timeout:
        _log.warning(f"Timeout while extracting content from {url}")
        return error_result(
            handler="search",
            function="extract_single_url_content",
            error='Request timeout',
            error_type=ErrorType.TIMEOUT,
            citations=[url]
        )
    except requests.exceptions.RequestException as e:
        _log.warning(f"Request failed for {url}: {e}")
        return error_result(
            handler="search",
            function="extract_single_url_content",
            error=f'Request failed: {str(e)}',
            error_type=ErrorType.API_ERROR,
            citations=[url]
        )
    except Exception as e:
        _log.error(f"Unexpected error extracting content from {url}: {e}")
        return error_result(
            handler="search",
            function="extract_single_url_content",
            error=f'Unexpected error: {str(e)}',
            error_type=ErrorType.COMPUTATION_ERROR,
            citations=[url]
        )


def extract_content_from_high_score_urls(
    ranked_results: List[Dict[str, Any]],
    min_score: float = 5.0
) -> List[Dict[str, Any]]:
    """
    Extract content from URLs that have a score >= min_score.
    
    Args:
        ranked_results: Ranked search results
        min_score: Minimum score threshold for URL extraction
        
    Returns:
        List of extracted content entries
    """
    # Filter URLs with score >= min_score
    high_score_urls = [r for r in ranked_results if r.get('score', 0) >= min_score]
    
    if not high_score_urls:
        return []
    
    # Extract content from high-score URLs
    extracted_content = []
    
    for result in high_score_urls:
        url = result.get('url', '')
        if not url:
            continue
            
        # Add small delay to be respectful to servers
        time.sleep(0.5)
        
        extraction_result = extract_single_url_content(url)
        
        # Unwrap standardized result
        content_entry = {
            'original_score': result.get('score', 0),
            'original_title': result.get('title', ''),
            'success': extraction_result.get('success', False)
        }
        
        if extraction_result.get('success'):
            content_entry.update(extraction_result.get('data', {}))
        else:
            content_entry['error'] = extraction_result.get('error', 'Unknown error')
            content_entry['url'] = url
        
        extracted_content.append(content_entry)
    
    return extracted_content


def extract_url_content(urls: List[str], timeout: int = 10) -> Dict[str, Any]:
    """
    Extract content from specific URLs.
    
    Args:
        urls: List of URLs to extract content from
        timeout: Request timeout in seconds
        
    Returns:
        Standardized result with extracted content
    """
    if not urls:
        return error_result(
            handler="search",
            function="extract_url_content",
            error='No URLs provided',
            error_type=ErrorType.INVALID_INPUT,
            citations=["SearXNG"]
        )
    
    extracted_content = []
    successful_extractions = 0
    
    for url in urls:
        _log.info(f"Extracting content from URL: {url}")
        
        # Add small delay to be respectful to servers
        time.sleep(0.5)
        
        extraction_result = extract_single_url_content(url, timeout)
        
        # Unwrap standardized result
        content_entry = {
            'success': extraction_result.get('success', False)
        }
        
        if extraction_result.get('success'):
            content_entry.update(extraction_result.get('data', {}))
            successful_extractions += 1
        else:
            content_entry['error'] = extraction_result.get('error', 'Unknown error')
            content_entry['url'] = url
        
        extracted_content.append(content_entry)
    
    return success_result(
        handler="search",
        function="extract_url_content",
        data={
            'extracted_content': extracted_content,
            'extraction_summary': {
                'total_urls': len(urls),
                'successful_extractions': successful_extractions,
                'failed_extractions': len(urls) - successful_extractions
            }
        },
        citations=urls,
        confidence=Confidence.HIGH if successful_extractions > 0 else Confidence.LOW
    )

