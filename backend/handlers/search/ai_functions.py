"""
AI Functions for Web and Scientific Literature Search

This module contains all AI-accessible functions for searching the web and scientific literature
using SearXNG metasearch engine.
"""

import logging
import time
from typing import Any, Dict, Annotated

from kani import ai_function, AIParam
from ..shared import success_result, error_result, ErrorType, Confidence

_log = logging.getLogger(__name__)


class SearchAIFunctionsMixin:
    """Mixin class containing AI function methods for Search handlers."""
    
    @ai_function(desc="Search the web and scientific literature using SearXNG. Use this to find information on any topic - recent papers, articles, web resources, news, technical documentation, or general information. This is your primary tool for gathering current information and verifying facts. Can optionally extract images from results to provide visual context. Results are paginated - use page parameter to navigate through results.", auto_truncate=128000)
    async def search_web(
        self,
        query: Annotated[str, AIParam(desc="Search query string - can be about any topic")],
        search_type: Annotated[str, AIParam(desc="Type of search: 'general' for web search, 'scientific' for academic literature, 'materials_science' for materials-specific search")] = "general",
        page: Annotated[int, AIParam(desc="Page number to retrieve (1-indexed, starts at 1)")] = 1,
        results_per_page: Annotated[int, AIParam(desc="Number of results per page (1-20)")] = 1,
        include_arxiv: Annotated[bool, AIParam(desc="Include arXiv preprints (for scientific search)")] = True,
        include_pubmed: Annotated[bool, AIParam(desc="Include PubMed medical literature (for scientific search)")] = True,
        include_scholar: Annotated[bool, AIParam(desc="Include Google Scholar (for scientific search)")] = True,
        include_phase_diagrams: Annotated[bool, AIParam(desc="Include phase diagram related results (for materials science)")] = True,
        include_thermodynamics: Annotated[bool, AIParam(desc="Include thermodynamic property results (for materials science)")] = True,
        language: Annotated[str, AIParam(desc="Search language (auto, en, de, etc.)")] = "auto",
        time_range: Annotated[str, AIParam(desc="Time range for results (empty, day, week, month, year)")] = "",
        extract_content: Annotated[bool, AIParam(desc="Extract content from high-scoring URLs (score >= 5)")] = True,
        min_score: Annotated[float, AIParam(desc="Minimum score threshold for content extraction")] = 5.0,
        include_images: Annotated[bool, AIParam(desc="Include relevant images from search results that the model can analyze")] = True,
        max_images: Annotated[int, AIParam(desc="Maximum number of images to extract (1-5)")] = 3
    ) -> Dict[str, Any]:
        """Search the web and scientific literature using SearXNG for any topic."""
        start_time = time.time()
        
        # Validate and clamp pagination parameters
        page = max(1, page)  # Ensure page is at least 1
        results_per_page = max(1, min(20, results_per_page))  # Clamp between 1-20
        
        params = {
            'query': query,
            'search_type': search_type,
            'page': page,
            'results_per_page': results_per_page,
            'include_arxiv': include_arxiv,
            'include_pubmed': include_pubmed,
            'include_scholar': include_scholar,
            'include_phase_diagrams': include_phase_diagrams,
            'include_thermodynamics': include_thermodynamics,
            'language': language,
            'time_range': time_range,
            'extract_content': extract_content,
            'min_score': min_score,
            'include_images': include_images,
            'max_images': max(1, min(5, max_images))  # Clamp between 1-5
        }
        
        util_result = await self.handle_searxng_search(params)
        
        duration_ms = (time.time() - start_time) * 1000
        
        if not util_result.get("success"):
            result = error_result(
                handler="search",
                function="search_web",
                error=util_result.get("error", "Search failed"),
                error_type=ErrorType.API_ERROR,
                citations=["Web Search"],
                duration_ms=duration_ms
            )
        else:
            # Extract the enriched data
            raw_data = util_result.get("data", {})
            
            # Build layered response with digestible components
            high_level_summary = raw_data.get("high_level_summary", {})
            ranked_results = raw_data.get("ranked_results", [])
            extracted_enriched = raw_data.get("extracted_enriched", [])
            
            # Calculate pagination indices
            total_results = len(ranked_results)
            start_idx = (page - 1) * results_per_page
            end_idx = start_idx + results_per_page
            total_pages = (total_results + results_per_page - 1) // results_per_page if total_results > 0 else 1
            
            # Paginate the results
            paginated_results = ranked_results[start_idx:end_idx]
            paginated_enriched = extracted_enriched[start_idx:end_idx] if extracted_enriched else []
            
            # Build citation list from paginated results
            page_citations = [r.get("url") for r in paginated_results if r.get("url")]
            
            # Extract images if present (only on first page)
            images = raw_data.get("images", []) if page == 1 else []
            image_descriptions = raw_data.get("image_descriptions", "") if page == 1 else ""
            
            # Create comprehensive but organized response
            layered_data = {
                # LAYER 1: High-level synthesis (ALWAYS included on every page)
                "high_level_summary": high_level_summary,
                
                # LAYER 2: Paginated results (lightweight, ranked, ready to present)
                "results": paginated_results,
                
                # LAYER 3: Enriched content with summaries (deeper dive, paginated)
                "extracted_enriched": paginated_enriched,
                
                # LAYER 4: Images (if requested and available, only on first page)
                "image_descriptions": image_descriptions if image_descriptions else None,
                
                # LAYER 5: Pagination metadata
                "pagination": {
                    "current_page": page,
                    "results_per_page": results_per_page,
                    "total_results": total_results,
                    "total_pages": total_pages,
                    "has_next_page": page < total_pages,
                    "has_previous_page": page > 1,
                    "showing_results": f"{start_idx + 1}-{min(end_idx, total_results)} of {total_results}"
                },
                
                # LAYER 6: Metadata and stats
                "stats": {
                    "total_results": total_results,
                    "total_enriched": len(extracted_enriched),
                    "num_academic": sum(1 for r in ranked_results if r.get("is_academic", False)),
                    "unique_domains": len(set(r.get("source_domain") for r in ranked_results if r.get("source_domain"))),
                    "extraction_summary": raw_data.get("extraction_summary", {}),
                    "num_images": len(images) if images else 0
                }
            }
            
            # Use calculated confidence from pipeline
            calculated_confidence = util_result.get("confidence", Confidence.MEDIUM)
            
            result = success_result(
                handler="search",
                function="search_web",
                data=layered_data,
                citations=page_citations or ["Web Search"],
                confidence=calculated_confidence,
                notes=[n for n in [
                    f"Search type: {search_type}",
                    f"Query: {query}",
                    f"Page {page} of {total_pages} ({layered_data['pagination']['showing_results']})",
                    f"Total: {total_results} results from {layered_data['stats']['unique_domains']} domains",
                    f"Academic sources: {layered_data['stats']['num_academic']}",
                    f"Images extracted: {layered_data['stats']['num_images']}" if layered_data['stats']['num_images'] > 0 else None,
                    "Call again with page parameter to see more results" if layered_data['pagination']['has_next_page'] else None
                ] if n is not None],
                duration_ms=duration_ms
            )
        
        # Store the result for tooltip display
        self._track_tool_output("search_web", result)
        return result
