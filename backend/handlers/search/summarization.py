"""
Summarization and Synthesis

This module handles AI-powered document summarization and cross-document synthesis.
"""

import logging
import os
from typing import Any, Dict, List, Optional

_log = logging.getLogger(__name__)


class SummarizationEngine:
    """Engine for AI-powered summarization using OpenAI."""
    
    def __init__(self):
        """Initialize the summarization engine."""
        self.openai_client = None
        self.summarization_enabled = False
        self._init_client()
    
    def _init_client(self):
        """Initialize OpenAI client for document summarization."""
        try:
            from openai import OpenAI
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.openai_client = OpenAI(api_key=api_key)
                self.summarization_enabled = True
                _log.info("OpenAI client initialized for summarization")
            else:
                self.openai_client = None
                self.summarization_enabled = False
                _log.warning("OpenAI API key not found - summarization will be disabled")
        except ImportError:
            self.openai_client = None
            self.summarization_enabled = False
            _log.warning("OpenAI library not installed - summarization will be disabled")
    
    def summarize_document(self, text: str, query: str = "", max_tokens: int = 150) -> str:
        """
        Summarize a document using GPT-4o-mini.
        
        Args:
            text: Document text to summarize
            query: Original search query for context
            max_tokens: Maximum tokens for summary
            
        Returns:
            Summary string
        """
        if not self.summarization_enabled or not text:
            # Fallback: return first few sentences
            sentences = text.split(". ")[:3]
            return ". ".join(sentences)[:600]
        
        try:
            # Truncate text if too long (keep first ~4000 chars)
            text_sample = text[:4000] if len(text) > 4000 else text
            
            prompt = f"""Summarize the following document excerpt in 2-3 sentences. Focus on the main findings and key points relevant to the query: "{query}"

Document:
{text_sample}

Summary:"""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a research assistant that creates concise, informative summaries of scientific and technical documents."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.3
            )
            
            summary = response.choices[0].message.content.strip()
            return summary
            
        except Exception as e:
            _log.error(f"Summarization failed: {e}")
            # Fallback to simple extraction
            sentences = text.split(". ")[:3]
            return ". ".join(sentences)[:600]
    
    def synthesize_high_level_summary(
        self,
        enriched_docs: List[Dict[str, Any]],
        query: str,
        max_docs: int = 5
    ) -> Dict[str, Any]:
        """
        Create cross-document synthesis with key points and citations.
        
        Args:
            enriched_docs: List of enriched documents with summaries
            query: Original search query
            max_docs: Maximum number of documents to include in synthesis
            
        Returns:
            Dictionary with query, key_points (with citations), and overall_synthesis
        """
        if not enriched_docs:
            return {
                "query": query,
                "key_points": [],
                "overall_synthesis": "No documents available for synthesis.",
                "num_sources": 0
            }
        
        # Build key points from top documents
        key_points = []
        top_docs = enriched_docs[:max_docs]
        
        for doc in top_docs:
            point = {
                "title": doc.get("title", "Untitled"),
                "summary": doc.get("summary", "No summary available"),
                "sources": [doc.get("url", "")],
                "source_domain": doc.get("source_domain", ""),
                "is_academic": doc.get("is_academic", False),
                "score": doc.get("final_score", 0)
            }
            key_points.append(point)
        
        # Create overall synthesis if summarization is enabled
        if self.summarization_enabled and len(top_docs) > 0:
            try:
                # Combine summaries for meta-analysis
                combined_text = "\n\n".join([
                    f"{i+1}. {doc['title']}: {doc.get('summary', '')}"
                    for i, doc in enumerate(top_docs)
                ])
                
                synthesis_prompt = f"""Based on these top search results for the query "{query}", provide a comprehensive 3-4 sentence synthesis that:
1. Identifies the main themes and findings
2. Notes any consensus or disagreements across sources
3. Highlights the most important insights

Results:
{combined_text}

Synthesis:"""
                
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a research analyst who synthesizes information from multiple sources into coherent insights."},
                        {"role": "user", "content": synthesis_prompt}
                    ],
                    max_tokens=200,
                    temperature=0.4
                )
                
                overall_synthesis = response.choices[0].message.content.strip()
                
            except Exception as e:
                _log.error(f"Synthesis failed: {e}")
                overall_synthesis = f"Found {len(top_docs)} relevant sources on {query}."
        else:
            overall_synthesis = f"Found {len(top_docs)} relevant sources on {query}."
        
        return {
            "query": query,
            "key_points": key_points,
            "overall_synthesis": overall_synthesis,
            "num_sources": len(enriched_docs),
            "num_academic_sources": sum(1 for d in enriched_docs if d.get("is_academic", False))
        }


def enrich_extracted_content(
    extracted_content: List[Dict[str, Any]],
    ranked_results: List[Dict[str, Any]],
    query: str,
    summarization_engine: SummarizationEngine,
    max_enrichments: int = 5
) -> List[Dict[str, Any]]:
    """
    Enrich extracted content with summaries and metadata.
    
    Args:
        extracted_content: Raw extracted content from URLs
        ranked_results: Ranked search results for matching metadata
        query: Original search query
        summarization_engine: Engine for summarization
        max_enrichments: Maximum number of documents to enrich with summaries
        
    Returns:
        List of enriched documents with summaries
    """
    from urllib.parse import urlparse
    
    enriched_docs = []
    
    for idx, item in enumerate(extracted_content[:max_enrichments]):
        if not item.get("success"):
            continue
        
        url = item.get("url", "")
        content = item.get("content", "")
        
        # Find matching ranked result for metadata
        matching_result = next(
            (r for r in ranked_results if r.get("url") == url),
            {}
        )
        
        # Generate summary if content extraction was successful
        summary = ""
        if content:
            summary = summarization_engine.summarize_document(content, query)
        
        enriched_doc = {
            "url": url,
            "title": item.get("title") or matching_result.get("title", "Untitled"),
            "summary": summary,
            "content_length": item.get("content_length", 0),
            "source_domain": matching_result.get("source_domain", urlparse(url).netloc if url else ""),
            "is_academic": matching_result.get("is_academic", False),
            "final_score": matching_result.get("final_score", 0),
            "base_score": matching_result.get("base_score", item.get("original_score", 0)),
            "content_type": item.get("content_type", "text/html"),
            "extraction_success": True
        }
        
        enriched_docs.append(enriched_doc)
    
    return enriched_docs

