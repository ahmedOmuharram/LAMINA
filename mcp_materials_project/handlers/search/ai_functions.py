"""
AI Functions for Web and Scientific Literature Search

This module contains all AI-accessible functions for searching the web and scientific literature
using SearXNG metasearch engine.
"""

import logging
from typing import Any, Dict, Annotated

from kani import ai_function, AIParam

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
        
        result = self.handle_searxng_search(params)
        result["citations"] = ["Web Search"]
        # Store the result for tooltip display
        if hasattr(self, 'recent_tool_outputs'):
            self.recent_tool_outputs.append({
                "tool_name": "search_web",
                "result": result
            })
        return result

    @ai_function(desc="Get information about available search engines and their status in SearXNG.", auto_truncate=128000)
    async def get_search_engines(self) -> Dict[str, Any]:
        """Get information about available search engines and their status."""
        result = self.handle_searxng_engine_stats({})
        result["citations"] = ["Web Search"]
        # Store the result for tooltip display
        if hasattr(self, 'recent_tool_outputs'):
            self.recent_tool_outputs.append({
                "tool_name": "get_search_engines",
                "result": result
            })
        return result
