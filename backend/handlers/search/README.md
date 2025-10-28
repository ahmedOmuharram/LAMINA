# Search Handlers

This module provides AI-accessible functions for searching the web and scientific literature using the SearXNG metasearch engine.

## Overview

SearXNG is a privacy-respecting, open-source metasearch engine that aggregates results from multiple search engines. This handler integrates SearXNG into the MCP Materials Project system, providing powerful search capabilities for:

- Scientific literature and research papers
- Technical documentation
- Web resources and general information
- Materials science specific content
- Phase diagrams and thermodynamic data

## Available Functions

### 1. `search_web`

Universal search function for web, scientific literature, and materials science content.

**Purpose**: Find information on any topic using multiple search engines.

**Parameters**:
- `query` (required): Search query string
- `search_type` (optional): Type of search
  - `"general"`: Web search (default)
  - `"scientific"`: Academic literature search
  - `"materials_science"`: Materials science specific search
- `include_arxiv` (optional): Include arXiv preprints (default True for scientific)
- `include_pubmed` (optional): Include PubMed medical literature (default True for scientific)
- `include_scholar` (optional): Include Google Scholar (default True for scientific)
- `include_phase_diagrams` (optional): Include phase diagram results (default True for materials)
- `include_thermodynamics` (optional): Include thermodynamic properties (default True for materials)
- `language` (optional): Search language (default "auto")
  - `"auto"`: Automatic language detection
  - `"en"`: English
  - `"de"`: German
  - etc.
- `time_range` (optional): Time range for results
  - `""`: All time (default)
  - `"day"`: Last 24 hours
  - `"week"`: Last week
  - `"month"`: Last month
  - `"year"`: Last year
- `extract_content` (optional): Extract full content from high-scoring URLs (default True)
- `min_score` (optional): Minimum score threshold for content extraction (default 5.0)

**Returns**: Dictionary containing:
- `query`: The search query used
- `results`: List of search results, each with:
  - `url`: URL of the result
  - `title`: Title of the page
  - `content`: Description/snippet
  - `score`: Relevance score
  - `engine`: Search engine that provided the result
- `extracted_content` (if extract_content=True): Full content from high-scoring URLs
- `extraction_summary`: Statistics about content extraction

**Example Usage**:

```python
# General web search
result = await handler.search_web(
    query="aluminum zinc phase diagram"
)

# Scientific literature search
result = await handler.search_web(
    query="lithium ion battery cathode materials",
    search_type="scientific",
    include_arxiv=True,
    include_scholar=True,
    time_range="year"
)

# Materials science specific search
result = await handler.search_web(
    query="Al-Zn",
    search_type="materials_science",
    include_phase_diagrams=True,
    include_thermodynamics=True
)

# Recent news search
result = await handler.search_web(
    query="solid state battery breakthrough",
    search_type="general",
    time_range="week"
)
```

**Use Cases**:
- **Research**: Finding recent papers on a specific topic
- **Verification**: Checking current facts and information
- **Documentation**: Locating technical documentation and guides
- **Discovery**: Exploring materials science literature
- **Context**: Getting background information for analysis
- **News**: Finding recent developments in a field

---

### 2. `get_search_engines`

Get information about available search engines in the SearXNG instance.

**Purpose**: Check which search engines are available and their status.

**Parameters**: None

**Returns**: Dictionary containing:
- `engines`: List of available search engines with their status
- `statistics`: Usage statistics for each engine
- `categories`: Available search categories

**Example Usage**:
```python
engines = await handler.get_search_engines()
```

**Use Cases**:
- Checking system status
- Understanding available search capabilities
- Debugging search issues
- Verifying engine availability

---

## Search Types

### General Search
- **Purpose**: Standard web search across multiple engines
- **Best for**: Websites, documentation, news, general information
- **Engines**: Google, Bing, DuckDuckGo, Qwant, etc.
- **Language**: Supports all languages

### Scientific Search
- **Purpose**: Academic and research literature
- **Best for**: Peer-reviewed papers, preprints, academic articles
- **Engines**:
  - arXiv: Physics, mathematics, computer science preprints
  - Google Scholar: Cross-disciplinary academic search
  - PubMed: Medical and life sciences literature
  - Semantic Scholar: Computer science and biomedicine
- **Features**:
  - Optimized for scientific terminology
  - Filters for peer-reviewed content
  - Author and citation information
  - DOI links when available

### Materials Science Search
- **Purpose**: Materials-specific literature and data
- **Best for**: Phase diagrams, crystal structures, materials properties
- **Features**:
  - Query enhancement with materials science terms
  - Automatic inclusion of "phase diagram" for applicable queries
  - Thermodynamics keyword addition when relevant
  - Focus on computational materials databases

---

## Content Extraction

The `search_web` function can automatically extract full content from high-scoring search results.

**How it works**:
1. Perform search and get results with scores
2. Filter results with `score >= min_score` (default 5.0)
3. Fetch and extract content from each URL
4. Clean HTML and extract text
5. Return extracted content with metadata

**Extracted content includes**:
- Full cleaned text content
- Page title
- URL and status code
- Content length and type
- Success/failure status

**Benefits**:
- Get full context beyond snippets
- Access detailed information without manual browsing
- Aggregate content from multiple sources
- Support for follow-up analysis

**Configuration**:
```python
# Enable content extraction with custom threshold
result = await handler.search_web(
    query="lithium battery phase diagrams",
    extract_content=True,
    min_score=6.0  # Only extract from highly relevant results
)

# Disable content extraction for faster searches
result = await handler.search_web(
    query="quick search",
    extract_content=False
)
```

---

## Typical Workflows

### Workflow 1: Literature Review
```python
# Step 1: Find recent papers
papers = await handler.search_web(
    query="solid electrolyte battery",
    search_type="scientific",
    time_range="year",
    include_arxiv=True,
    include_scholar=True
)

# Step 2: Extract content from top results
for result in papers['results'][:5]:
    if result['score'] >= 7:
        print(f"Title: {result['title']}")
        print(f"URL: {result['url']}")
```

### Workflow 2: Materials Discovery
```python
# Search for phase diagram information
phase_info = await handler.search_web(
    query="Cu-Al-Li ternary",
    search_type="materials_science",
    include_phase_diagrams=True
)

# Get thermodynamic data
thermo_data = await handler.search_web(
    query="Cu-Al-Li formation energy",
    search_type="scientific"
)
```

### Workflow 3: Fact Verification
```python
# Check recent information
current_info = await handler.search_web(
    query="lithium metal anode voltage",
    search_type="scientific",
    time_range="year",
    extract_content=True
)
```

---

## Implementation Details

### SearXNG Integration
- **Backend**: Local SearXNG instance (default: http://localhost:8080)
- **Privacy**: No tracking, no profiling, no user data collection
- **Engines**: Configurable set of search engines
- **Format**: JSON API responses

### Content Extraction
- **Method**: HTTP requests with appropriate headers
- **Parsing**: HTML title and content extraction
- **Cleaning**: Removes scripts, styles, and HTML tags
- **Limits**: Content limited to 10,000 characters per page
- **Timeout**: 10 seconds per URL
- **Rate limiting**: 0.5 second delay between requests

### Error Handling
- Request timeouts return error status
- Failed extractions include error messages
- Graceful fallback when engines are unavailable
- Detailed error information in responses

---

## Configuration

### SearXNG Instance
The handler connects to a local SearXNG instance. Default URL: `http://localhost:8080`

To use a different instance:
```python
handler = SearXNGSearchHandler(searxng_url="http://your-searxng-instance:8080")
```

### Search Engines
Configure which engines are used by the SearXNG instance in its configuration files:
- `searxng-docker/searxng/settings.yml`: Engine configuration
- Enable/disable specific engines
- Set engine weights and priorities
- Configure rate limits

---

## Best Practices

1. **Choose the right search type**:
   - Use `"scientific"` for research papers and academic content
   - Use `"materials_science"` for phase diagrams and materials data
   - Use `"general"` for documentation and web resources

2. **Use time_range for recent information**:
   - `"year"` for literature reviews
   - `"month"` for recent developments
   - `"week"` or `"day"` for breaking news

3. **Content extraction**:
   - Enable for detailed analysis
   - Disable for quick searches or large result sets
   - Adjust `min_score` to control extraction threshold

4. **Query formulation**:
   - Be specific with technical terms
   - Use chemical formulas and symbols
   - Include context keywords (e.g., "phase diagram", "DFT", "experimental")

5. **Language**:
   - Use `"en"` for scientific literature (most papers are in English)
   - Use `"auto"` for general searches
   - Specify language for region-specific searches

---

## Related Modules

- **materials/**: Materials Project database search
- **electrochemistry/**: Battery and electrode calculations
- **calphad/**: CALPHAD thermodynamic calculations

**When to use search vs. Materials Project**:
- Use **search** for: Literature, recent papers, external databases, documentation
- Use **materials/** for: DFT-computed properties, crystal structures, Materials Project data

---

## Troubleshooting

### No results returned
- Check SearXNG instance is running
- Verify network connectivity
- Try a different search type
- Broaden your query

### Content extraction failures
- High-scoring results may be behind paywalls
- Some sites block automated access
- Increase timeout if needed
- Check extraction_summary for details

### Slow searches
- Disable content extraction: `extract_content=False`
- Reduce number of results
- Use more specific queries
- Check SearXNG instance performance

---

## Privacy and Ethics

- **Privacy**: SearXNG doesn't track users or store search history
- **Rate limiting**: Built-in delays respect server resources
- **User agent**: Clearly identifies as academic research bot
- **Paywalls**: Respects access restrictions
- **Terms of service**: Complies with search engine ToS

