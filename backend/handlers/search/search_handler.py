"""
Search Handler

Main orchestrator for the search pipeline, coordinating all search components.
"""

import logging
from typing import Any, Dict, List, Mapping

from ..base import BaseHandler
from ..shared import success_result, error_result, ErrorType

from .searxng_client import SearXNGClient
from .result_processing import normalize_results, cluster_by_canonical_url
from .ranking import rerank_results
from .content_extraction import extract_content_from_high_score_urls, extract_url_content
from .summarization import SummarizationEngine, enrich_extracted_content
from .confidence import calculate_confidence
from .ai_functions import SearchAIFunctionsMixin

_log = logging.getLogger(__name__)


class SearXNGSearchHandler(SearchAIFunctionsMixin, BaseHandler):
    """Handler for SearXNG search functionality with research agent capabilities."""
    
    def __init__(self, searxng_url: str = "http://localhost:8080", **kwargs):
        """
        Initialize the SearXNG search handler.
        
        Args:
            searxng_url: Base URL of the SearXNG instance
        """
        super().__init__(**kwargs)
        
        # Initialize components
        self.client = SearXNGClient(searxng_url)
        self.summarization_engine = SummarizationEngine()
    
    # ========================================================================
    # Pipeline Orchestration
    # ========================================================================
    
    def _do_search_dispatch(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Dispatch search based on search_type parameter.
        
        Args:
            params: Search parameters
            
        Returns:
            Raw search results from SearXNG
        """
        query = params.get('query', '')
        search_type = params.get('search_type', 'general')
        include_images = params.get('include_images', False)
        
        # Build categories list - include images if requested
        categories = params.get('categories')
        if include_images:
            if categories is None:
                categories = ['general', 'images']
            elif isinstance(categories, list) and 'images' not in categories:
                categories = categories + ['images']
        
        # Build kwargs for search methods
        search_kwargs = {
            'time_range': params.get('time_range', '')
        }
        if categories is not None:
            search_kwargs['categories'] = categories
        
        # Perform the search based on type
        if search_type == 'scientific':
            return self.client.search_scientific(
                query=query,
                include_arxiv=params.get('include_arxiv', True),
                include_pubmed=params.get('include_pubmed', True),
                include_scholar=params.get('include_scholar', True),
                **search_kwargs
            )
        elif search_type == 'materials_science':
            return self.client.search_materials_science(
                query=query,
                include_phase_diagrams=params.get('include_phase_diagrams', True),
                include_thermodynamics=params.get('include_thermodynamics', True),
                **search_kwargs
            )
        else:
            # General search
            return self.client.search(
                query=query,
                format=params.get('format', 'json'),
                categories=categories,
                engines=params.get('engines'),
                language=params.get('language', 'auto'),
                safesearch=params.get('safesearch', 0),
                time_range=params.get('time_range', '')
            )
    
    def _postprocess_results(
        self,
        raw_search_results: Dict[str, Any],
        query: str,
        time_range: str = ""
    ) -> Dict[str, Any]:
        """
        Post-process raw search results: normalize, cluster, and re-rank.
        
        Args:
            raw_search_results: Raw results from SearXNG
            query: Original search query
            time_range: Time range filter
            
        Returns:
            Dictionary with normalized, clustered, and ranked results
        """
        if not raw_search_results.get('success'):
            return raw_search_results
        
        data = raw_search_results.get('data', {})
        raw_results = data.get('results', [])
        
        if not raw_results:
            return raw_search_results
        
        # Stage 1: Normalize
        normalized_results = normalize_results(raw_results, query)
        
        # Stage 2: Cluster
        clustered_results = cluster_by_canonical_url(normalized_results)
        
        # Stage 3: Re-rank
        ranked_results = rerank_results(clustered_results, time_range)
        
        # Add processed results to data
        data['normalized_results'] = normalized_results
        data['clustered_results'] = clustered_results
        data['ranked_results'] = ranked_results
        data['raw_results'] = raw_results  # Keep original raw results for image extraction
        
        # Update the main results to show ranked version
        data['results'] = ranked_results
        
        return raw_search_results
    
    def _attach_extracted_fulltext(
        self,
        structured_results: Dict[str, Any],
        params: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract full text from high-scoring URLs and enrich with summaries.
        
        Args:
            structured_results: Structured results with ranked_results
            params: Search parameters
            
        Returns:
            Results with extracted_enriched content and high_level_summary
        """
        query = params.get('query', '')
        min_score = params.get('min_score', 5.0)
        
        data = structured_results.get('data', {})
        ranked_results = data.get('ranked_results', [])
        
        # Extract content from high-score URLs
        extracted_content = extract_content_from_high_score_urls(
            ranked_results,
            min_score=min_score
        )
        
        # Build extraction summary
        successful_extractions = sum(1 for item in extracted_content if item.get('success'))
        data['extraction_summary'] = {
            'total_high_score_urls': len([r for r in ranked_results if r.get('score', 0) >= min_score]),
            'successful_extractions': successful_extractions,
            'failed_extractions': len(extracted_content) - successful_extractions,
            'min_score_threshold': min_score
        }
        
        # Stage 4: Enrich with summaries
        enriched_docs = enrich_extracted_content(
            extracted_content,
            ranked_results,
            query,
            self.summarization_engine,
            max_enrichments=5
        )
        
        # Stage 5: Synthesize high-level summary
        high_level_summary = self.summarization_engine.synthesize_high_level_summary(
            enriched_docs,
            query,
            max_docs=5
        )
        
        # Add enriched data
        data['extracted_content'] = extracted_content
        data['extracted_enriched'] = enriched_docs
        data['high_level_summary'] = high_level_summary
        
        # Calculate better confidence
        confidence = calculate_confidence(ranked_results, enriched_docs)
        structured_results['confidence'] = confidence
        
        return structured_results
    
    # ========================================================================
    # Public API
    # ========================================================================
    
    async def handle_searxng_search(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Handle SearXNG search requests with research agent pipeline.
        
        Pipeline: fetch → enrich → organize → summarize
        
        Args:
            params: Search parameters including query, search_type, extract_content, etc.
            
        Returns:
            Comprehensive search results with normalized, ranked, and enriched data
        """
        # Extract search parameters
        query = params.get('query', '')
        if not query:
            return error_result(
                handler="search",
                function="handle_searxng_search",
                error='Query parameter is required',
                error_type=ErrorType.INVALID_INPUT,
                citations=["SearXNG"]
            )
        
        extract_content = params.get('extract_content', True)
        time_range = params.get('time_range', '')
        
        # ===================================================================
        # STAGE 1: FETCH - Run search dispatch
        # ===================================================================
        raw_results = self._do_search_dispatch(params)
        
        if not raw_results.get('success'):
            return raw_results
        
        # ===================================================================
        # STAGE 2: ENRICH METADATA - Normalize, cluster, re-rank
        # ===================================================================
        structured = self._postprocess_results(raw_results, query, time_range)
        
        if not structured.get('success'):
            return structured
        
        # ===================================================================
        # STAGE 3 (OPTIONAL): DEEP CRAWL + SUMMARIZE
        # ===================================================================
        if extract_content:
            structured = self._attach_extracted_fulltext(structured, params)
        else:
            # Calculate confidence even without content extraction
            data = structured.get('data', {})
            ranked_results = data.get('ranked_results', [])
            confidence = calculate_confidence(ranked_results, [])
            structured['confidence'] = confidence
        
        # ===================================================================
        # STAGE 4 (OPTIONAL): EXTRACT IMAGES
        # ===================================================================
        include_images = params.get('include_images', False)
        if include_images:
            structured = await self._extract_and_attach_images(structured, params)
        
        return structured
    
    async def _extract_and_attach_images(
        self,
        structured_results: Dict[str, Any],
        params: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract images from search results and make them available to the model.
        
        Args:
            structured_results: Structured results from search
            params: Search parameters including max_images
            
        Returns:
            Results with extracted images attached and described
        """
        max_images = params.get('max_images', 3)
        query = params.get('query', '')
        
        # Get raw results that might contain images
        raw_data = structured_results.get('data', {})
        raw_results = raw_data.get('raw_results', [])
        
        # Extract image results (from SearXNG image search category)
        image_results = []
        for result in raw_results:
            # Check if this is an image result
            if result.get('category') == 'images' or result.get('img_src') or result.get('template') == 'images.html':
                img_url = result.get('img_src') or result.get('thumbnail_src')
                if img_url:
                    image_results.append({
                        'url': img_url,
                        'thumbnail': result.get('thumbnail_src', img_url),
                        'source_url': result.get('url', ''),
                        'title': result.get('title', ''),
                        'resolution': result.get('resolution', 'unknown'),
                        'format': result.get('img_format', 'unknown'),
                        'source_domain': result.get('source', ''),
                        'score': result.get('score', 0)
                    })
        
        # Sort by score and limit
        image_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        selected_images = image_results[:max_images]
        
        # Attach images to the data
        if selected_images:
            raw_data['images'] = selected_images
            
            # Describe images with GPT-4o so the model can "see" them
            from ...app.utils import describe_image_with_gpt4o
            image_descriptions = []
            
            for idx, img in enumerate(selected_images[:2], 1):  # Describe top 2 images
                try:
                    description = await describe_image_with_gpt4o(
                        {"url": img['url']},
                        user_text=f"Search query: {query}\nImage title: {img['title']}"
                    )
                    image_descriptions.append(f"**Image {idx}** (from {img['source_domain']}):\n{description}")
                    _log.info(f"Described image {idx} from search results")
                except Exception as e:
                    _log.error(f"Failed to describe search image {idx}: {e}")
                    image_descriptions.append(f"**Image {idx}**: [Unable to process - {str(e)}]")
            
            # Add descriptions to the data so they're included in the tool result
            if image_descriptions:
                raw_data['image_descriptions'] = "\n\n".join(image_descriptions)
            
            # Set first image on kani instance to be streamed to client
            first_image = selected_images[0]
            self._last_image_url = first_image['url']
            self._last_image_metadata = {
                'type': 'search_result_image',
                'query': query,
                'title': first_image['title'],
                'source': first_image['source_domain'],
                'resolution': first_image['resolution'],
                'format': first_image['format'],
                'total_images': len(selected_images),
                'all_images': selected_images  # Include all for reference
            }
            
            _log.info(f"Extracted {len(selected_images)} images for query: {query}")
        
        return structured_results
    
    def handle_extract_url_content(self, urls: List[str], timeout: int = 10) -> Dict[str, Any]:
        """
        Handle URL content extraction requests.
        
        Args:
            urls: List of URLs to extract content from
            timeout: Request timeout in seconds
            
        Returns:
            Standardized result with extracted content
        """
        return extract_url_content(urls, timeout)

