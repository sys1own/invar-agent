"""Universal Pan-Scientific Literature Federation (arXiv, OpenAlex, Semantic Scholar,
Crossref, Wikipedia, Wikidata, OEIS, PubChem, Europe PMC REST API, and NCBI PubMed E-Utilities).
"""

from __future__ import annotations

import asyncio
from typing import List, Optional

from .search import (
    AR5IV_HTML,
    ARXIV_API,
    CROSSREF_API,
    EUROPE_PMC_API,
    NCBI_ESEARCH_API,
    NCBI_ESUMMARY_API,
    OEIS_SEARCH,
    OPENALEX_API,
    PUBCHEM_PROPERTIES,
    SEMANTIC_SCHOLAR_API,
    USER_AGENT,
    WIKIDATA_ENTITIES,
    WIKIDATA_SEARCH,
    WIKIPEDIA_OPENSEARCH,
    WIKIPEDIA_SUMMARY,
    WIKIPEDIA_WIKITEXT,
    FetchFn,
    LiteratureResult,
    LiteratureSearchClient,
    _channels_for_prompt,
    anchor_or_queries,
    bigram_relaxations,
    classify_research_domain,
    entity_concept,
    relaxed_queries,
    strip_internal_diagnostic_tokens,
)


async def async_search_europe_pmc(
    query: str,
    limit: int = 10,
    cursor_mark: str = "*",
    timeout: float = 8.0,
    fetch: Optional[FetchFn] = None,
) -> List[LiteratureResult]:
    """Resilient async Europe PMC REST API connector with strict 8-second timeout."""
    client = LiteratureSearchClient(timeout=min(float(timeout), 8.0), fetch=fetch)
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(client._search_europe_pmc, query, limit, cursor_mark),
            timeout=min(float(timeout), 8.0),
        )
    except Exception:  # noqa: BLE001
        return []


async def async_search_pubmed(
    query: str,
    limit: int = 10,
    timeout: float = 8.0,
    fetch: Optional[FetchFn] = None,
) -> List[LiteratureResult]:
    """Resilient async NCBI PubMed E-utilities connector (esearch.fcgi + esummary.fcgi) with strict 8-second timeout."""
    client = LiteratureSearchClient(timeout=min(float(timeout), 8.0), fetch=fetch)
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(client._search_pubmed, query, limit),
            timeout=min(float(timeout), 8.0),
        )
    except Exception:  # noqa: BLE001
        return []


__all__ = [
    "AR5IV_HTML",
    "ARXIV_API",
    "CROSSREF_API",
    "EUROPE_PMC_API",
    "NCBI_ESEARCH_API",
    "NCBI_ESUMMARY_API",
    "OEIS_SEARCH",
    "OPENALEX_API",
    "PUBCHEM_PROPERTIES",
    "SEMANTIC_SCHOLAR_API",
    "USER_AGENT",
    "WIKIDATA_ENTITIES",
    "WIKIDATA_SEARCH",
    "WIKIPEDIA_OPENSEARCH",
    "WIKIPEDIA_SUMMARY",
    "WIKIPEDIA_WIKITEXT",
    "FetchFn",
    "LiteratureResult",
    "LiteratureSearchClient",
    "_channels_for_prompt",
    "anchor_or_queries",
    "async_search_europe_pmc",
    "async_search_pubmed",
    "bigram_relaxations",
    "classify_research_domain",
    "entity_concept",
    "relaxed_queries",
    "strip_internal_diagnostic_tokens",
]
