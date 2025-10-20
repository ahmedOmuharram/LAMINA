Search Handler
==============

Functions for searching the web and scientific literature using SearXNG metasearch engine.

.. _search_web:

search_web
----------

Search the web and scientific literature for any topic.

**Parameters:**

- ``query`` (str): Search query string
- ``search_type`` (str, optional): Type of search:
  
  - ``'general'``: Web search (default)
  - ``'scientific'``: Academic literature search
  - ``'materials_science'``: Materials-specific search

- ``include_arxiv`` (bool, optional): Include arXiv preprints (default: True for scientific)
- ``include_pubmed`` (bool, optional): Include PubMed medical literature (default: True for scientific)
- ``include_scholar`` (bool, optional): Include Google Scholar (default: True for scientific)
- ``include_phase_diagrams`` (bool, optional): Include phase diagram results (default: True for materials)
- ``include_thermodynamics`` (bool, optional): Include thermodynamic properties (default: True for materials)
- ``language`` (str, optional): Search language ('auto', 'en', 'de', etc.) (default: 'auto')
- ``time_range`` (str, optional): Time range for results ('', 'day', 'week', 'month', 'year') (default: '')
- ``extract_content`` (bool, optional): Extract full content from high-scoring URLs (default: True)
- ``min_score`` (float, optional): Minimum score threshold for content extraction (default: 5.0)

**Returns:** Dictionary containing search results with titles, URLs, snippets, and optionally extracted content

**Search Types:**

- **General**: Standard web search across multiple engines (Google, Bing, DuckDuckGo, etc.)
- **Scientific**: Academic literature from arXiv, Google Scholar, PubMed, Semantic Scholar
- **Materials Science**: Materials-specific literature with enhanced query terms

.. _get_search_engines:

get_search_engines
------------------

Get information about available search engines and their status.

**Parameters:** None

**Returns:** Dictionary containing available search engines, categories, and system status
