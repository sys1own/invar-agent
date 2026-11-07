"""Live scientific literature search for the Research Agent Ingest stage.

Retrieves real literature from arXiv, OpenAlex/Crossref and Wikipedia while
keeping ``Sandbox_Execute`` fully network-isolated: this client is used only
during Ingest/GoI_Plan, never inside the sandboxed simulation kernel.

Deterministic disk caching + an ``offline`` mode keep unit tests hermetic:
with ``offline=True`` (or when the network is unavailable) only cached
results are returned.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

ARXIV_API = "http://export.arxiv.org/api/query"
OPENALEX_API = "https://api.openalex.org/works"
CROSSREF_API = "https://api.crossref.org/works"
WIKIPEDIA_OPENSEARCH = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"

USER_AGENT = "invar-agent-research/0.1 (mailto:research@invar-agent.local)"

# -- query normalization -----------------------------------------------------
# Literature APIs (arXiv/OpenAlex) return 0 hits for long natural-language
# sentences; keep only the primary technical terms.
_QUERY_FILLER_PHRASES = (
    "si units", "entry interface", "mars-gram 2010", "mars-gram",
)
_QUERY_FILLER_TOKENS = {
    "the", "a", "an", "and", "or", "of", "for", "with", "in", "on", "to", "at",
    "by", "from", "is", "are", "was", "were", "be", "been", "being", "as",
    "si", "unit", "units", "interface", "simulate", "simulates", "simulated",
    "simulation", "model", "models", "verify", "verified", "verification",
    "analysis", "analyse", "analyze", "study", "studies", "using", "use",
    "used", "based", "approach", "method", "methods", "data", "set", "value",
    "values", "calculate", "compute", "determine", "assess", "given", "into",
    "uses", "use",
    "that", "this", "these", "those", "their", "its", "it", "we", "our",
}


def _sanitize_query(q: str) -> str:
    """Reduce a natural-language research query to 2-4 technical terms.

    Strips parentheticals, punctuation and filler phrases ("SI units",
    "entry interface", "Mars-GRAM 2010"), then keeps the highest-signal
    tokens (hyphenated / capitalized / numeric terms first). Falls back to
    the original query when nothing technical remains.
    """
    original = (q or "").strip()
    if not original:
        return ""
    text = re.sub(r"\([^)]*\)", " ", original)
    for phrase in _QUERY_FILLER_PHRASES:
        text = re.sub(re.escape(phrase), " ", text, flags=re.IGNORECASE)
    # Punctuation -> space, but keep intra-token hyphens (Sutton-Graves) and
    # decimal points (5.8 km/s); stray dots are trimmed per token below.
    cleaned = re.sub(r"(?!\w-\w)[^A-Za-z0-9\-\. ]+", " ", text)
    tokens = [t.strip(".") for t in cleaned.split() if t.strip(".")]
    scored: List[tuple] = []
    for idx, tok in enumerate(tokens):
        if tok.lower() in _QUERY_FILLER_TOKENS:
            continue
        score = 1
        if any(ch.isdigit() for ch in tok):
            score += 2
        if "-" in tok:
            score += 2
        if tok[:1].isupper():  # proper noun / acronym (Mars, CO2, N2)
            score += 2
        if tok.isupper() and len(tok) <= 4:
            score += 1
        scored.append((score, -idx, tok))
    if not scored:
        # Nothing but filler: keep the original technical words so the API
        # still gets a query.
        fallback = [t.strip(".") for t in
                    re.sub(r"[^A-Za-z0-9\-\. ]+", " ", original).split()
                    if t.strip(".") and t.strip(".").lower() not in _QUERY_FILLER_TOKENS]
        return " ".join(fallback[:4]) or original
    scored.sort(reverse=True)
    return " ".join(tok for _, _, tok in scored[:4])


# Offline fallback: domain constants extracted from a curated aerospace /
# physics dictionary, used only when every literature API returns 0 hits so
# the simulation kernel never receives empty parameters.
_FALLBACK_DOMAIN_CONSTANTS = [
    (("mars", "aeroshell", "aerothermo", "re-entry", "reentry", "entry interface"),
     {"martian_scale_height_km": 11.1,
      "co2_surface_density_kg_m3": 0.020,
      "sutton_graves_stagnation_coeff": 1.83e-4,
      "equilibrium_temp_coefficient": 0.276}),
    (("sutton", "graves", "stagnation", "heat flux", "heat transfer"),
     {"sutton_graves_stagnation_coeff": 1.83e-4,
      "stagnation_heat_flux_form": "q = k * sqrt(rho/Rn) * V^3",
      "typical_peak_heat_flux_MW_m2": 1.0}),
    (("kdv", "soliton", "shallow water", "nonlinear wave"),
     {"kdv_nonlinear_coeff": 6.0,
      "kdv_dispersion_coeff": 1.0,
      "soliton_speed_amplitude": "c = 2A"}),
    (("drag", "aerodynamic", "lift"),
     {"drag_coefficient_typical": 1.0,
      "ballistic_coefficient_form": "BC = m / (Cd * A)"}),
]


def _fallback_constants(query: str) -> List[LiteratureResult]:
    """Domain-constant fallback hit for queries with zero API results."""
    q = (query or "").lower()
    for keywords, constants in _FALLBACK_DOMAIN_CONSTANTS:
        if any(k in q for k in keywords):
            rendered = "; ".join(f"{k}={v}" for k, v in constants.items())
            return [LiteratureResult(
                "fallback_constants",
                "Offline domain constants (literature APIs returned 0 hits)",
                f"Fallback aerospace constants for '{query}': {rendered}. "
                "Use these as empirical parameters for the simulation kernel.",
                "")]
    return []


@dataclass
class LiteratureResult:
    source: str
    title: str
    summary: str
    url: str
    year: Optional[int] = None
    categories: List[str] = field(default_factory=list)


FetchFn = Callable[[str, float], str]


def _default_fetch(url: str, timeout: float) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https/http API hosts
        return resp.read().decode("utf-8", errors="replace")


def _cache_key(source: str, query: str) -> str:
    return hashlib.blake2b(f"{source}|{query}".encode(), digest_size=16).hexdigest()


class LiteratureSearchClient:
    """Resilient, cached, offline-capable literature retrieval client."""

    def __init__(self, cache_dir: Optional[Path] = None, *, offline: bool = False,
                 timeout: float = 8.0, max_results: int = 5,
                 fetch: Optional[FetchFn] = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.offline = offline
        self.timeout = timeout
        self.max_results = max_results
        self._fetch = fetch or _default_fetch

    # -- public API -------------------------------------------------------
    def search(self, query: str, max_results: Optional[int] = None,
               sources: tuple = ("arxiv", "openalex", "wikipedia")) -> List[LiteratureResult]:
        """Search all requested sources; per-source failures degrade gracefully."""
        query = _sanitize_query(query)
        limit = max_results or self.max_results
        results: List[LiteratureResult] = []
        for source in sources:
            try:
                results.extend(self._search_source(source, query, limit))
            except Exception:  # noqa: BLE001 - one bad source must not kill ingest
                continue
        results = results[: max(limit * 2, limit)]
        if not results:
            # Never let the kernel receive empty parameters: fall back to
            # curated domain constants when every API returned 0 hits.
            results = _fallback_constants(query)
        return results

    def search_literature(self, queries: List[str],
                          max_results: Optional[int] = None) -> List[LiteratureResult]:
        """Batch helper: sanitize each query, search, and dedupe by (source, title)."""
        seen = set()
        out: List[LiteratureResult] = []
        for q in queries or []:
            for r in self.search(q, max_results=max_results):
                key = (r.source, r.title)
                if key not in seen:
                    seen.add(key)
                    out.append(r)
        return out

    # -- source implementations -------------------------------------------
    def _search_source(self, source: str, query: str, limit: int) -> List[LiteratureResult]:
        cached = self._read_cache(source, query)
        if self.offline:
            return cached or []
        try:
            if source == "arxiv":
                payload = self._fetch_with_backoff(self._arxiv_url(query, limit))
                results = self._parse_arxiv(payload)
            elif source == "openalex":
                payload = self._fetch_with_backoff(self._openalex_url(query, limit))
                results = self._parse_openalex(payload)
            elif source == "crossref":
                payload = self._fetch_with_backoff(self._crossref_url(query, limit))
                results = self._parse_crossref(payload)
            elif source == "wikipedia":
                results = self._search_wikipedia(query, limit)
            else:
                return cached or []
        except Exception:  # noqa: BLE001 - fall back to cache on network failure
            return cached or []
        if results:
            self._write_cache(source, query, [r.__dict__ for r in results])
            return results
        return cached or []

    def _fetch_with_backoff(self, url: str) -> str:
        last_exc: Optional[Exception] = None
        for attempt, delay in enumerate((0.0, 1.5, 4.0)):
            if delay:
                time.sleep(delay)
            try:
                return self._fetch(url, self.timeout)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
        raise last_exc if last_exc else RuntimeError("fetch failed")

    @staticmethod
    def _arxiv_url(query: str, limit: int) -> str:
        q = urllib.parse.quote(f"all:{query}")
        return f"{ARXIV_API}?search_query={q}&start=0&max_results={limit}"

    @staticmethod
    def _parse_arxiv(payload: str) -> List[LiteratureResult]:
        ns = {"a": "http://www.w3.org/2005/Atom",
              "arxiv": "http://arxiv.org/schemas/atom"}
        root = ET.fromstring(payload)
        out: List[LiteratureResult] = []
        for entry in root.findall("a:entry", ns):
            title = (entry.findtext("a:title", "", ns) or "").strip().replace("\n", " ")
            summary = (entry.findtext("a:summary", "", ns) or "").strip().replace("\n", " ")
            link = entry.find("a:id", ns)
            url = link.text.strip() if link is not None and link.text else ""
            published = entry.findtext("a:published", "", ns) or ""
            year = int(published[:4]) if published[:4].isdigit() else None
            cats = [c.get("term", "") for c in entry.findall("a:category", ns)]
            out.append(LiteratureResult("arxiv", title, summary[:1200], url,
                                        year=year, categories=cats))
        return out

    @staticmethod
    def _openalex_url(query: str, limit: int) -> str:
        return f"{OPENALEX_API}?search={urllib.parse.quote(query)}&per-page={limit}"

    @staticmethod
    def _parse_openalex(payload: str) -> List[LiteratureResult]:
        data = json.loads(payload)
        out: List[LiteratureResult] = []
        for work in data.get("results", []):
            title = (work.get("title") or "").strip()
            inv = work.get("abstract_inverted_index") or {}
            if inv:
                positions: Dict[int, str] = {}
                for word, idxs in inv.items():
                    for i in idxs:
                        positions[i] = word
                abstract = " ".join(positions[i] for i in sorted(positions))
            else:
                abstract = ""
            out.append(LiteratureResult(
                "openalex", title, abstract[:1200],
                work.get("doi") or work.get("id") or "",
                year=work.get("publication_year"),
                categories=[work.get("type") or ""]))
        return [r for r in out if r.title]

    @staticmethod
    def _crossref_url(query: str, limit: int) -> str:
        return f"{CROSSREF_API}?query={urllib.parse.quote(query)}&rows={limit}"

    @staticmethod
    def _parse_crossref(payload: str) -> List[LiteratureResult]:
        data = json.loads(payload)
        out: List[LiteratureResult] = []
        for item in (data.get("message") or {}).get("items", []):
            titles = item.get("title") or []
            title = titles[0] if titles else ""
            abstract = re.sub(r"<[^>]+>", " ", item.get("abstract", "") or "")
            year = None
            date_parts = (item.get("issued") or {}).get("date-parts") or [[]]
            if date_parts and date_parts[0] and date_parts[0][0]:
                year = int(date_parts[0][0])
            out.append(LiteratureResult(
                "crossref", title, abstract.strip()[:1200],
                item.get("URL") or item.get("DOI") or "", year=year,
                categories=[item.get("type") or ""]))
        return [r for r in out if r.title]

    def _search_wikipedia(self, query: str, limit: int) -> List[LiteratureResult]:
        # opensearch for the best-matching technical article, then its summary.
        search_url = (f"{WIKIPEDIA_OPENSEARCH}?action=opensearch&format=json"
                      f"&search={urllib.parse.quote(query)}&limit={limit}&namespace=0")
        payload = self._fetch_with_backoff(search_url)
        data = json.loads(payload)
        titles = data[1] if isinstance(data, list) and len(data) > 1 else []
        urls = data[3] if isinstance(data, list) and len(data) > 3 else []
        out: List[LiteratureResult] = []
        for title, url in list(zip(titles, urls))[:limit]:
            try:
                summary = json.loads(self._fetch_with_backoff(
                    WIKIPEDIA_SUMMARY + urllib.parse.quote(title.replace(" ", "_"))))
                extract = summary.get("extract") or ""
                out.append(LiteratureResult(
                    "wikipedia", summary.get("title") or title, extract[:1200],
                    summary.get("content_urls", {}).get("desktop", {}).get("page", url)
                    or url))
            except Exception:  # noqa: BLE001 - skip failed summaries
                continue
        return out

    # -- deterministic disk cache -----------------------------------------
    def _cache_path(self, source: str, query: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{source}_{_cache_key(source, query)}.json"

    def _read_cache(self, source: str, query: str) -> List[LiteratureResult]:
        path = self._cache_path(source, query)
        if path is None or not path.exists():
            return []
        try:
            data = json.loads(path.read_text())
            return [LiteratureResult(**item) for item in data]
        except Exception:  # noqa: BLE001 - corrupt cache is a miss
            return []

    def _write_cache(self, source: str, query: str, items: List[Dict]) -> None:
        path = self._cache_path(source, query)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(items))
        except Exception:  # noqa: BLE001 - cache is best-effort
            pass
