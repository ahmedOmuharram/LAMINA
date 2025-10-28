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
    
    @ai_function(desc="Search the web and scientific literature using SearXNG. Use this to find information on any topic - recent papers, articles, web resources, news, technical documentation, or general information. This is your primary tool for gathering current information and verifying facts.", auto_truncate=128000)
    async def search_web(
        self,
        query: Annotated[str, AIParam(desc="Search query string - can be about any topic")],
        search_type: Annotated[str, AIParam(desc="Type of search: 'general' for web search, 'scientific' for academic literature, 'materials_science' for materials-specific search")] = "general",
        include_arxiv: Annotated[bool, AIParam(desc="Include arXiv preprints (for scientific search)")] = True,
        include_pubmed: Annotated[bool, AIParam(desc="Include PubMed medical literature (for scientific search)")] = True,
        include_scholar: Annotated[bool, AIParam(desc="Include Google Scholar (for scientific search)")] = True,
        include_phase_diagrams: Annotated[bool, AIParam(desc="Include phase diagram related results (for materials science)")] = True,
        include_thermodynamics: Annotated[bool, AIParam(desc="Include thermodynamic property results (for materials science)")] = True,
        language: Annotated[str, AIParam(desc="Search language (auto, en, de, etc.)")] = "auto",
        time_range: Annotated[str, AIParam(desc="Time range for results (empty, day, week, month, year)")] = "",
        extract_content: Annotated[bool, AIParam(desc="Extract content from high-scoring URLs (score >= 5)")] = True,
        min_score: Annotated[float, AIParam(desc="Minimum score threshold for content extraction")] = 5.0
    ) -> Dict[str, Any]:
        """Search the web and scientific literature using SearXNG for any topic."""
        start_time = time.time()
        
        params = {
            'query': query,
            'search_type': search_type,
            'include_arxiv': include_arxiv,
            'include_pubmed': include_pubmed,
            'include_scholar': include_scholar,
            'include_phase_diagrams': include_phase_diagrams,
            'include_thermodynamics': include_thermodynamics,
            'language': language,
            'time_range': time_range,
            'extract_content': extract_content,
            'min_score': min_score
        }
        
        util_result = self.handle_searxng_search(params)
        
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
            
            # Build citation list from top ranked results
            top_citations = [r.get("url") for r in ranked_results[:5] if r.get("url")]
            
            # Create comprehensive but organized response
            layered_data = {
                # LAYER 1: High-level synthesis (what the AI should read first)
                "high_level_summary": high_level_summary,
                
                # LAYER 2: Top results (lightweight, ranked, ready to present)
                "top_results": ranked_results[:10],  # Top 10 ranked results
                
                # LAYER 3: Enriched content with summaries (deeper dive)
                "extracted_enriched": extracted_enriched[:5],  # Top 5 with full summaries
                
                # LAYER 4: Metadata and stats
                "stats": {
                    "total_results": len(ranked_results),
                    "total_enriched": len(extracted_enriched),
                    "num_academic": sum(1 for r in ranked_results if r.get("is_academic", False)),
                    "unique_domains": len(set(r.get("source_domain") for r in ranked_results if r.get("source_domain"))),
                    "extraction_summary": raw_data.get("extraction_summary", {})
                },
                
                # LAYER 5: Full raw data (for debugging/deep analysis)
                "raw": raw_data
            }
            
            # Use calculated confidence from pipeline
            calculated_confidence = util_result.get("confidence", Confidence.MEDIUM)
            
            result = success_result(
                handler="search",
                function="search_web",
                data=layered_data,
                citations=top_citations or ["Web Search"],
                confidence=calculated_confidence,
                notes=[
                    f"Search type: {search_type}",
                    f"Query: {query}",
                    f"Found {len(ranked_results)} results from {layered_data['stats']['unique_domains']} domains",
                    f"Academic sources: {layered_data['stats']['num_academic']}"
                ],
                duration_ms=duration_ms
            )
        
        # Store the result for tooltip display
        self._track_tool_output("search_web", result)
        return result

    @ai_function(desc="Get information about available search engines and their status in SearXNG.", auto_truncate=128000)
    async def get_search_engines(self) -> Dict[str, Any]:
        """Get information about available search engines and their status."""
        start_time = time.time()
        
        util_result = self.handle_searxng_engine_stats({})
        
        duration_ms = (time.time() - start_time) * 1000
        
        if not util_result.get("success"):
            result = error_result(
                handler="search",
                function="get_search_engines",
                error=util_result.get("error", "Failed to get engine stats"),
                error_type=ErrorType.API_ERROR,
                citations=["Web Search"],
                duration_ms=duration_ms
            )
        else:
            data = {k: v for k, v in util_result.items() if k != "success"}
            result = success_result(
                handler="search",
                function="get_search_engines",
                data=data,
                citations=["Web Search"],
                confidence=Confidence.HIGH,
                notes=["Engine statistics from SearXNG instance"],
                duration_ms=duration_ms
            )
        
        # Store the result for tooltip display
        self._track_tool_output("get_search_engines", result)
        return result
