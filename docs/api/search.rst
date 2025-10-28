Search Handler
==============

The Search handler provides AI functions for searching the web and scientific literature using SearXNG metasearch engine. This is the primary tool for gathering current information, finding recent research, and verifying facts from external sources.

Overview
--------

The Search handler enables:

1. **Web Search**: General internet search across multiple engines
2. **Scientific Literature Search**: Academic papers from arXiv, PubMed, Google Scholar
3. **Materials Science Search**: Specialized queries for phase diagrams, thermodynamic properties
4. **Content Extraction**: Automatic extraction and summarization of high-quality sources
5. **Ranked Results**: Intelligent scoring and ranking of search results

Core Functions
--------------

.. _search_web:

search_web
^^^^^^^^^^

Search the web and scientific literature using SearXNG metasearch engine. **This is your primary tool for gathering current information and verifying facts.**

**When to Use:**

- Finding recent research papers and articles
- Gathering current information on any topic
- Verifying facts and claims
- Accessing technical documentation
- Searching for materials science information
- Finding news and general web resources

**Parameters:**

- ``query`` (str, required): Search query string - can be about any topic
- ``search_type`` (str, optional): Type of search. Options:
  
  - ``'general'``: Web search across multiple engines (default)
  - ``'scientific'``: Academic literature (arXiv, PubMed, Google Scholar)
  - ``'materials_science'``: Materials-specific search with phase diagrams and thermodynamics

- ``include_arxiv`` (bool, optional): Include arXiv preprints for scientific search. Default: True
- ``include_pubmed`` (bool, optional): Include PubMed medical literature for scientific search. Default: True
- ``include_scholar`` (bool, optional): Include Google Scholar for scientific search. Default: True
- ``include_phase_diagrams`` (bool, optional): Include phase diagram related results for materials science. Default: True
- ``include_thermodynamics`` (bool, optional): Include thermodynamic property results for materials science. Default: True
- ``language`` (str, optional): Search language (``'auto'``, ``'en'``, ``'de'``, etc.). Default: ``'auto'``
- ``time_range`` (str, optional): Time range for results. Options: ``''`` (all time), ``'day'``, ``'week'``, ``'month'``, ``'year'``. Default: ``''``
- ``extract_content`` (bool, optional): Extract content from high-scoring URLs (score >= min_score). Default: True
- ``min_score`` (float, optional): Minimum score threshold for content extraction. Default: 5.0

**Returns:**

Dictionary containing layered information:

**Layer 1: High-Level Summary**

- ``high_level_summary``: Synthesized overview of search results with:
  
  - ``key_findings``: Main takeaways from top results
  - ``source_diversity``: Assessment of source variety
  - ``consensus_level``: Agreement across sources
  - ``academic_vs_general``: Mix of academic and general sources

**Layer 2: Top Results**

- ``top_results``: Top 10 ranked results with:
  
  - ``title``: Page title
  - ``url``: URL
  - ``snippet``: Description/snippet
  - ``score``: Relevance score (0-10)
  - ``source_domain``: Domain name
  - ``is_academic``: Boolean indicating academic source
  - ``engine``: Search engine that provided the result

**Layer 3: Enriched Content**

- ``extracted_enriched``: Top 5 results with full content extraction:
  
  - ``title``: Page title
  - ``url``: URL
  - ``score``: Relevance score
  - ``summary``: AI-generated summary of content
  - ``key_points``: Extracted key points
  - ``content_length``: Length of extracted content
  - ``extraction_successful``: Boolean

**Layer 4: Statistics**

- ``stats``: Search statistics with:
  
  - ``total_results``: Total number of results found
  - ``total_enriched``: Number of results with content extraction
  - ``num_academic``: Number of academic sources
  - ``unique_domains``: Number of unique domains
  - ``extraction_summary``: Summary of extraction performance

**Layer 5: Raw Data**

- ``raw``: Complete raw data for debugging and deep analysis

**Example - General Web Search:**

.. code-block:: python

   # Search for information about lithium-ion battery safety
   result = await handler.search_web(
       query="lithium ion battery safety hazards",
       search_type="general",
       time_range="year",
       extract_content=True
   )

**Example - Scientific Literature Search:**

.. code-block:: python

   # Search for recent papers on solid electrolytes
   result = await handler.search_web(
       query="solid electrolyte ionic conductivity",
       search_type="scientific",
       include_arxiv=True,
       include_scholar=True,
       time_range="year"
   )

**Example - Materials Science Search:**

.. code-block:: python

   # Search for Al-Mg phase diagram information
   result = await handler.search_web(
       query="Al-Mg aluminum magnesium phase diagram",
       search_type="materials_science",
       include_phase_diagrams=True,
       include_thermodynamics=True
   )

**Technical Details:**

- Uses SearXNG metasearch engine (aggregates multiple search engines)
- Intelligently ranks and scores results based on relevance, source quality, and recency
- Automatically extracts and summarizes content from high-scoring pages
- Identifies academic sources (papers, preprints, journals)
- Provides layered information from quick summary to full raw data
- Confidence level calculated based on source diversity and consensus

**Search Engine Coverage:**

- **General Search**: Google, Bing, DuckDuckGo, Qwant, Brave
- **Academic**: arXiv, PubMed, Google Scholar, Semantic Scholar
- **Technical**: GitHub, StackOverflow, technical documentation sites
- **Materials Science**: Phase diagram databases, thermodynamic databases

.. _get_search_engines:

get_search_engines
^^^^^^^^^^^^^^^^^^

Get information about available search engines and their status in the SearXNG instance.

**When to Use:**

- Checking which search engines are available
- Debugging search issues
- Understanding search engine coverage

**Parameters:** None

**Returns:**

Dictionary containing:

- ``engines``: List of available search engines with:
  
  - ``name``: Engine name
  - ``categories``: Search categories supported
  - ``shortcut``: Shortcut code
  - ``enabled``: Whether engine is enabled
  - ``timeout``: Engine timeout setting

- ``categories``: Available search categories
- ``total_engines``: Total number of engines
- ``enabled_engines``: Number of enabled engines

**Example:**

.. code-block:: python

   # Get available search engines
   result = await handler.get_search_engines()

**Technical Details:**

- Returns real-time status of SearXNG instance
- Shows which engines are operational
- Useful for troubleshooting search issues

Search Types
------------

**General Search (search_type='general')**

- Broad web search across multiple search engines
- Suitable for current events, news, general information
- Includes blogs, news sites, technical documentation
- Fast and comprehensive coverage

**Scientific Search (search_type='scientific')**

- Focused on academic literature and research papers
- Sources: arXiv, PubMed, Google Scholar, Semantic Scholar
- Filters for peer-reviewed content when possible
- Prioritizes recent publications
- Ideal for research questions and literature review

**Materials Science Search (search_type='materials_science')**

- Specialized for materials science queries
- Enhanced for phase diagrams and thermodynamic properties
- Includes materials databases and specialized resources
- Augmented queries for better results (e.g., adds "phase diagram" context)
- Prioritizes technical accuracy over general popularity

Content Extraction and Summarization
-------------------------------------

**Extraction Process:**

1. **Scoring**: All results scored 0-10 based on:
   
   - Query relevance
   - Source quality and authority
   - Content freshness
   - Academic vs general source weighting

2. **Selection**: High-scoring results (score >= min_score) selected for extraction

3. **Content Extraction**: Full page content extracted using readability algorithms

4. **Summarization**: AI generates concise summaries of extracted content

5. **Key Points**: Important facts and findings extracted

**Quality Indicators:**

- ``score``: Relevance and quality score (0-10)
- ``is_academic``: Scholarly source flag
- ``source_domain``: Domain reputation
- ``extraction_successful``: Whether content was successfully extracted

Confidence Levels
-----------------

Search results include confidence levels based on:

- **HIGH**: Multiple high-quality sources with consensus, diverse domains
- **MEDIUM**: Reasonable sources but limited diversity or some disagreement
- **LOW**: Few sources, low-quality sources, or significant inconsistencies

Factors affecting confidence:

- Number of high-quality results
- Source diversity (different domains)
- Academic vs general source mix
- Consensus across sources
- Recency of information

Best Practices
--------------

**Query Formulation:**

- Be specific with technical terms
- Include context for ambiguous terms
- Use Boolean operators when needed (AND, OR, NOT)
- Add year or time period for recent information

**Search Type Selection:**

- Use ``'scientific'`` for research questions and peer-reviewed information
- Use ``'materials_science'`` for phase diagrams, thermodynamics, material properties
- Use ``'general'`` for current events, news, and broad topics

**Content Extraction:**

- Enable ``extract_content=True`` for detailed information
- Adjust ``min_score`` to control extraction threshold (5.0 is balanced)
- Higher ``min_score`` = fewer but higher-quality extractions

**Time Filtering:**

- Use ``time_range='year'`` for recent research
- Use ``time_range='month'`` or ``'week'`` for very current information
- Leave empty (default) for comprehensive historical search

Citations
---------

**Data Source:**

- **SearXNG**: Privacy-respecting metasearch engine that aggregates results from multiple search engines
- **Search Engines**: Google, Bing, DuckDuckGo, arXiv, PubMed, Google Scholar, and others

**Content Processing:**

- Readability algorithms for content extraction
- AI summarization for key findings
- Intelligent ranking and scoring

Notes
-----

- Search results are real-time and reflect current web content
- Academic sources prioritized in scientific search mode
- Content extraction respects robots.txt and ethical scraping guidelines
- Summaries generated from extracted content, not from snippets
- Confidence levels help assess reliability of information
- Multiple search engines provide redundancy and diversity
- Results ranked by relevance, not just by original search engine ranking
