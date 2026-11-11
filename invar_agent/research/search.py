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
from typing import Any, Callable, Dict, List, Optional

ARXIV_API = "https://export.arxiv.org/api/query"
AR5IV_HTML = "https://ar5iv.labs.arxiv.org/html/"
OPENALEX_API = "https://api.openalex.org/works"
CROSSREF_API = "https://api.crossref.org/works"
WIKIPEDIA_OPENSEARCH = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"
WIKIPEDIA_WIKITEXT = "https://en.wikipedia.org/w/api.php"
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"
WIKIDATA_SEARCH = "https://www.wikidata.org/w/api.php"
WIKIDATA_ENTITIES = "https://www.wikidata.org/w/api.php"
OEIS_SEARCH = "https://oeis.org/search"
PUBCHEM_PROPERTIES = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"

USER_AGENT = ("InvarAgent/1.0 (https://github.com/sys1own/invar-agent; "
              "mailto:research@invar-agent.local")

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
    "simulation", "model", "models", "parameter", "parameters", "equation",
    "equations", "closed", "form", "forms", "value", "values", "numerical",
    "integration", "integrate", "stoichiometric", "verify", "verified",
    "verification", "analysis", "analyse", "analyze", "study", "studies",
    "using", "use", "used", "based", "approach", "method", "methods", "data",
    "set", "calculate", "compute", "determine", "assess", "given", "into",
    "uses", "use",
    "that", "this", "these", "those", "their", "its", "it", "we", "our",
    # Conversational/conjunction filler that makes strict academic APIs
    # (arXiv/OpenAlex/Wikidata) return 0 hits on 6-9 word planner queries.
    "vs", "versus", "exact", "between", "comparison", "compare", "compared",
    "generic",
}


def _token_score(tok: str) -> int:
    score = 1
    if any(ch.isdigit() for ch in tok):
        score += 2
    if "-" in tok:
        score += 2
    if tok[:1].isupper():  # proper noun / acronym (Mars, CO2, N2)
        score += 2
    if tok.isupper() and len(tok) <= 4:
        score += 1
    return score


def _kept_tokens(q: str) -> List[str]:
    """Filler-stripped technical tokens of a raw query, in original order.

    Unlike :func:`_sanitize_query` this keeps EVERY substantive token (no
    top-3 truncation) so Pass-2 relaxation can still see long-tail domain
    keywords (``metriplectic``, ``gradient``) that the compact Pass-1 query
    drops.
    """
    original = (q or "").strip()
    if not original:
        return []
    text = re.sub(r"\([^)]*\)", " ", original)
    for phrase in _QUERY_FILLER_PHRASES:
        text = re.sub(re.escape(phrase), " ", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"(?!\w-\w)[^A-Za-z0-9\-\. ]+", " ", text)
    out: List[str] = []
    for tok in cleaned.split():
        tok = tok.strip(".")
        if not tok:
            continue
        low = tok.lower()
        if low in _QUERY_FILLER_TOKENS:
            continue
        if len(tok) == 1 and tok.isalpha():
            continue
        out.append(tok)
    return out


def _sanitize_query(q: str) -> str:
    """Reduce a sentence-like research query to <=3 substantive entities.

    Aggressively strips filler (``model``, ``parameters``, ``equations``,
    ``values``, ``closed``, ``form``, ``simulation``, ``numerical``,
    ``integration``, ``units``, ...) and single-letter tokens (``s``, ``q``,
    ``w``, ``f``), merges consecutive capitalized tokens into multi-word
    entities (``Field Noyes``), then keeps the top 3 by signal score.
    """
    original = (q or "").strip()
    if not original:
        return ""
    kept_pairs = [(_token_score(tok), tok) for tok in _kept_tokens(original)]
    kept: List[Tuple[int, str]] = kept_pairs
    if not kept:
        # Nothing but filler: keep the original technical words so the API
        # still gets a query.
        fallback = [t.strip(".") for t in
                    re.sub(r"[^A-Za-z0-9\-\. ]+", " ", original).split()
                    if t.strip(".") and len(t.strip(".")) > 1
                    and t.strip(".").lower() not in _QUERY_FILLER_TOKENS]
        return " ".join(fallback[:4]) or original
    # Merge consecutive capitalized tokens into one entity ("Field Noyes",
    # "Mars CO2 N2") so proper-noun phrases survive truncation as a unit.
    entities: List[Tuple[int, int, str]] = []  # (score, -first_idx, words)
    idx = 0
    while idx < len(kept):
        score, tok = kept[idx]
        if tok[:1].isupper():
            words = [tok]
            first_idx = idx
            while idx + 1 < len(kept) and kept[idx + 1][1][:1].isupper():
                idx += 1
                words.append(kept[idx][1])
                score = max(score, kept[idx][0])
            entities.append((score, -first_idx, " ".join(words)))
        else:
            entities.append((score, -idx, tok))
        idx += 1
    entities.sort(reverse=True)
    return " ".join(words for _, _, words in entities[:3])


# -- progressive query relaxation (Pass 2 / Pass 3) --------------------------
# Domain vocabulary that marks a token as a high-IDF technical keyword even
# when it is short or lowercase (numerical analysis / theoretical mechanics).
_DOMAIN_KEYWORD_BOOST = {
    "metriplectic", "gradient", "discrete", "integrator", "integration",
    "splitting", "strang", "gonzalez", "symplectic", "verlet", "euler",
    "conservation", "entropy", "stiff", "stability", "cayley", "newton",
    "automata", "kdv", "soliton", "hamiltonian", "lagrangian", "poisson",
    "runge", "kutta", "collocation", "finite", "element", "volume",
}


# Outcome/process descriptors that must not dominate a relaxed core query:
# "conservation"/"energy" describe WHAT is measured, not the method.
_RELAX_STOP = {"conservation", "energy", "stiff", "stability", "systems",
               "system", "time", "behaviour", "behavior", "exact"}


def _relax_token_score(tok: str) -> int:
    """IDF-like keyword score: domain vocabulary > length > technical shape."""
    score = len(tok)
    low = tok.lower()
    if low in _DOMAIN_KEYWORD_BOOST:
        score += 8
    if low in _RELAX_STOP:
        score -= 12
    if tok[:1].isupper():
        score += 3
    if "-" in tok or any(ch.isdigit() for ch in tok):
        score += 3
    return score


def _relax_keywords(query: str) -> List[str]:
    """Core technical keywords of a sanitized query, best-first, deduped."""
    toks = [t for t in (query or "").split() if t]
    scored = sorted(toks, key=lambda t: (-_relax_token_score(t), -len(t), t))
    seen, out = set(), []
    for tok in scored:
        low = tok.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(tok)
    return out


def relaxed_queries(query: str, max_queries: int = 2) -> List[str]:
    """Pass-2 relaxed queries: top 2-3 highest-IDF domain keywords.

    ``"GENERIC metriplectic discrete gradient"`` -> ``["metriplectic discrete
    gradient", "discrete gradient metriplectic"]`` style cores that strict
    academic APIs can actually match.
    """
    kws = _relax_keywords(query)
    if len(kws) < 3:
        return []
    queries: List[str] = []
    # Windowed cores over the score-sorted keywords, each rendered in the
    # query's original word order: [k0 k1 k2] and [k1 k2 k3] cover both the
    # leading method terms and the trailing method+object pairs.
    order = {t.lower(): i for i, t in enumerate((query or "").split())}

    def _render(window: List[str]) -> str:
        return " ".join(sorted(window, key=lambda t: order.get(t.lower(), 99)))

    if len(kws) >= 3:
        queries.append(_render(kws[:3]))
    if len(kws) >= 4:
        queries.append(_render(kws[1:4]))
    elif len(kws) == 3:
        queries.append(_render(kws[:3]))
    # De-duplicate while preserving order.
    seen, unique = set(), []
    for q in queries:
        if q not in seen:
            seen.add(q)
            unique.append(q)
    return unique[:max_queries]


def entity_concept(query: str) -> str:
    """Compact 1-2 word canonical concept for entity APIs (Pass 3).

    Wikidata/OEIS/PubChem must never receive full 7-word sentences.
    """
    kws = _relax_keywords(query)
    return " ".join(kws[:2]) if kws else ""


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
                "", badge="ReferenceRegistry")]
    return []


def _sanitize_key(text: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"\W+", "_", (text or "").strip().lower())).strip("_")


# Common compound names accepted by PubChem PUG REST.
_COMPOUND_ALIASES = {
    "carbon dioxide": "carbon dioxide", "co2": "carbon dioxide",
    "carbon monoxide": "carbon monoxide", "co": "carbon monoxide",
    "nitrogen": "nitrogen", "n2": "nitrogen", "oxygen": "oxygen", "o2": "oxygen",
    "methane": "methane", "ch4": "methane", "ammonia": "ammonia", "nh3": "ammonia",
    "water": "water", "h2o": "water", "hydrogen": "hydrogen", "h2": "hydrogen",
    "helium": "helium", "argon": "argon", "neon": "neon",
    "ethanol": "ethanol", "methanol": "methanol", "acetone": "acetone",
    "benzene": "benzene", "titanium": "titanium", "aluminum": "aluminium",
    "iron": "iron", "copper": "copper", "silicon": "silicon", "carbon": "carbon",
}


# -- research-domain routing -------------------------------------------------
_DOMAIN_KEYWORDS = {
    "discrete_oeis": ("fibonacci", "catalan", "prime", "recurrence", "combinator",
                      "integer sequence", "number theory", "partition", "oeis",
                      "binomial", "permutation", "graph", "lattice path",
                      "discrete", "sequence", "modular", "black-scholes", "heston",
                      "fft", "cooley", "finance", "option"),
    "chemical_pubchem": ("chemical", "chemistry", "compound", "molecule", "molecular",
                         "kinetic", "reaction", "combustion", "arrhenius", "enzyme",
                         "fuel", "gas", "methane", "hydrogen", "oxygen", "nitrogen",
                         "polymer", "thermodynamic", "fuel", "propellant", "oxidizer"),
}
_CHANNEL_BY_DOMAIN = {
    "equations": ("arxiv", "openalex", "semantic_scholar", "wikipedia"),
    "peer_reviewed": ("arxiv", "openalex", "semantic_scholar", "crossref"),
    "constants": ("wikipedia", "wikidata", "openalex"),
    "discrete_oeis": ("oeis", "arxiv"),
    "chemical_pubchem": ("pubchem", "semantic_scholar"),
}


def classify_research_domain(prompt_or_query: str) -> set:
    """Route a prompt to research channels.

    Returns a subset of ``{"equations", "peer_reviewed", "constants",
    "discrete_oeis", "chemical_pubchem"}``. Every prompt gets
    ``peer_reviewed`` + ``equations`` + ``constants`` unless it is clearly
    discrete or chemical, in which case the dedicated channels are added.
    """
    t = (prompt_or_query or "").lower()
    channels = {"equations", "peer_reviewed", "constants"}
    padded = f" {t} "
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        for k in keywords:
            # Word-boundary match so "aerothermodynamics" does not hit
            # the chemical keyword "thermodynamic".
            if re.search(rf"(?<![a-z0-9]){re.escape(k)}(?![a-z0-9])", padded):
                channels.add(domain)
                break
    # Pure discrete/chemical prompts keep the core channels too (equations
    # still apply), but skip nothing: OEIS/PubChem are additive.
    return channels


def _channels_for_prompt(prompt_or_query: str) -> List[str]:
    channels = classify_research_domain(prompt_or_query)
    sources: List[str] = []
    for channel in ("peer_reviewed", "equations", "constants",
                    "discrete_oeis", "chemical_pubchem"):
        if channel in channels:
            for src in _CHANNEL_BY_DOMAIN[channel]:
                if src not in sources:
                    sources.append(src)
    return sources


@dataclass
class LiteratureResult:
    source: str
    title: str
    summary: str
    url: str
    year: Optional[int] = None
    categories: List[str] = field(default_factory=list)
    # Quantitative payload (all optional, connector-populated):
    equations: List[str] = field(default_factory=list)       # raw LaTeX strings
    constants: Dict[str, Any] = field(default_factory=dict)  # name -> value|{value,unit}
    benchmarks: List[Dict[str, Any]] = field(default_factory=list)
    badge: str = ""                                           # arXiv | OEIS | PubChem | ...
    citation_count: Optional[int] = None
    tldr: str = ""


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
        # Diagnostic telemetry of the most recent search() call (Pass 1/2/3).
        self.last_query_meta: Dict[str, Any] = {}

    # -- public API -------------------------------------------------------
    def search(self, query: str, max_results: Optional[int] = None,
               sources: Optional[tuple] = None,
               prompt_or_query: Optional[str] = None) -> List[LiteratureResult]:
        """Multi-channel quantitative literature search.

        Active connectors (routed by :func:`classify_research_domain`) run
        concurrently with strict per-connector timeouts; per-source failures
        degrade gracefully to cached results. ``sources`` may still pin an
        explicit legacy tuple (arxiv/openalex/wikipedia/...).

        Progressive 3-pass relaxation (zero extra cost when Pass 1 hits):
        Pass 1 queries the primary academic APIs with the sanitized technical
        keywords; on 0 results for a >3-word query, Pass 2 re-queries arXiv
        and OpenAlex with the top 2-3 highest-IDF domain keywords; Pass 3
        retries the entity APIs (Wikidata/OEIS/PubChem) with a compact 1-2
        word canonical concept only. Diagnostics land in
        ``self.last_query_meta`` for ``search_log`` telemetry.
        """
        raw_query = (query or "").strip()
        query = _sanitize_query(raw_query)
        limit = max_results or self.max_results
        if sources is None:
            sources = tuple(_channels_for_prompt(prompt_or_query or raw_query))
        channel_counts: Dict[str, int] = {}
        relaxed_query = ""
        error_note = ""

        def _run(pass_sources: tuple, pass_query: str) -> List[LiteratureResult]:
            nonlocal error_note
            pass_results: List[LiteratureResult] = []
            if self.offline or len(pass_sources) <= 1:
                for source in pass_sources:
                    try:
                        got = self._search_source(source, pass_query, limit)
                        channel_counts[source] = channel_counts.get(source, 0) + len(got)
                        pass_results.extend(got)
                    except Exception as exc:  # noqa: BLE001
                        error_note = f"{source}: {str(exc)[:120]}"
            else:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                with ThreadPoolExecutor(max_workers=min(4, len(pass_sources))) as pool:
                    futures = {pool.submit(self._search_source_safe, source,
                                           pass_query, limit): source
                               for source in pass_sources}
                    for fut in as_completed(futures):
                        src = futures[fut]
                        try:
                            got = fut.result()
                            channel_counts[src] = channel_counts.get(src, 0) + len(got)
                            pass_results.extend(got)
                        except Exception as exc:  # noqa: BLE001
                            error_note = f"{src}: {str(exc)[:120]}"
            return pass_results

        # Pass 1: primary academic APIs with cleaned technical keywords.
        results = _run(sources, query)

        # Pass 2: automatic relaxation on 0 results for >3-word queries. The
        # keyword pool comes from the FULL filler-stripped token list, not the
        # top-3-truncated Pass-1 query, so long-tail domain terms survive.
        keyword_pool = " ".join(_kept_tokens(raw_query)) or query
        if not results and len(keyword_pool.split()) > 3:
            for relaxed in relaxed_queries(keyword_pool):
                pass2_sources = tuple(s for s in ("arxiv", "openalex") if s in sources) \
                    or ("arxiv", "openalex")
                relaxed_results = _run(pass2_sources, relaxed)
                if relaxed_results:
                    relaxed_query = relaxed
                    results = relaxed_results
                    break

        # Pass 3: entity APIs with a compact 1-2 word canonical concept only.
        if not results:
            concept = entity_concept(keyword_pool)
            pass3_sources = tuple(s for s in sources
                                  if s in ("wikidata", "oeis", "pubchem"))
            if concept and pass3_sources:
                results = _run(pass3_sources, concept)
                if results:
                    relaxed_query = concept

        # Stable order: keep connector grouping deterministic for caching/tests.
        order = {s: i for i, s in enumerate(sources)}
        results.sort(key=lambda r: order.get(r.source, 99))
        results = results[: max(limit * 2, limit)]
        # Diagnostic telemetry for search_log query_done events.
        self.last_query_meta = {
            "query": query,
            "raw_query": raw_query,
            "status": "ok" if results else "empty",
            "channel_counts": channel_counts,
            "relaxed_query": relaxed_query,
            "error": error_note,
        }
        if not results:
            # Never let the kernel receive empty parameters: fall back to
            # curated domain constants when every API returned 0 hits.
            results = _fallback_constants(query)
        return results

    def _search_source_safe(self, source: str, query: str,
                            limit: int) -> List[LiteratureResult]:
        try:
            return self._search_source(source, query, limit)
        except Exception:  # noqa: BLE001
            return []

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
            elif source == "semantic_scholar":
                results = self._search_semantic_scholar(query, limit)
            elif source == "wikidata":
                results = self._search_wikidata_constants(query, limit)
            elif source == "oeis":
                results = self._search_oeis(query, limit)
            elif source == "pubchem":
                results = self._search_pubchem(query, limit)
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
                result = LiteratureResult(
                    "wikipedia", summary.get("title") or title, extract[:1200],
                    summary.get("content_urls", {}).get("desktop", {}).get("page", url)
                    or url, badge="Wikipedia")
                # High-mathematical-density upgrade: wikitext <math> blocks,
                # {{math|...}} templates and infobox numerical pairs (4s budget).
                # Enrichment is best-effort: a wikitext failure must NOT drop
                # the summary result itself.
                try:
                    wikitext = self._fetch_wikitext(title)
                except Exception:  # noqa: BLE001
                    wikitext = ""
                if wikitext:
                    from .extractor import (extract_infobox_parameters,
                                            extract_wikitext_math)
                    result.equations = extract_wikitext_math(wikitext)[:8]
                    for p in extract_infobox_parameters(wikitext,
                                                         source="Wikipedia-Math")[:12]:
                        result.constants[p.name] = {
                            "value": p.si_value, "unit": p.si_unit,
                            "raw_value": p.raw_value, "raw_unit": p.raw_unit}
                out.append(result)
            except Exception:  # noqa: BLE001 - skip failed summaries
                continue
        return out

    def _fetch_wikitext(self, title: str) -> str:
        """MediaWiki revisions API: full wikitext for <math>/infobox mining."""
        url = (f"{WIKIPEDIA_WIKITEXT}?action=query&prop=revisions&rvprop=content"
               f"&rvslots=main&titles={urllib.parse.quote(title)}"
               f"&format=json&origin=*")
        payload = self._fetch(url, 4.0)
        data = json.loads(payload)
        pages = (data.get("query") or {}).get("pages") or {}
        for page in pages.values():
            revisions = page.get("revisions") or []
            if revisions:
                slots = revisions[0].get("slots") or {}
                return (slots.get("main") or {}).get("content") or ""
        return ""

    def _search_semantic_scholar(self, query: str, limit: int) -> List[LiteratureResult]:
        q = urllib.parse.quote(query)
        url = (f"{SEMANTIC_SCHOLAR_API}?query={q}&limit={min(limit, 5)}"
               f"&fields=title,abstract,tldr,year,citationCount,externalIds")
        data = json.loads(self._fetch_with_backoff(url))
        out: List[LiteratureResult] = []
        for paper in data.get("data") or []:
            tldr = ((paper.get("tldr") or {}).get("text") or "").strip()
            abstract = (paper.get("abstract") or "").strip()
            summary = (tldr + "\n\n" + abstract).strip()[:1200]
            ext = paper.get("externalIds") or {}
            arxiv_id = ext.get("ArXiv") or ""
            out.append(LiteratureResult(
                "semanticscholar",
                (paper.get("title") or "").strip(), summary,
                (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id
                 else ext.get("DOI") or paper.get("paperId") or ""),
                year=paper.get("year"), badge="SemanticScholar",
                citation_count=paper.get("citationCount"), tldr=tldr))
            # ar5iv equation upgrade for arXiv-identified papers (live only).
            if arxiv_id and not self.offline:
                eqs = self._fetch_ar5iv_equations(arxiv_id)
                if eqs:
                    out[-1].equations = eqs
        return [r for r in out if r.title]

    def _fetch_ar5iv_equations(self, arxiv_id: str, max_equations: int = 8) -> List[str]:
        """Display equations from the ar5iv HTML rendering (no PDF download)."""
        try:
            payload = self._fetch(f"{AR5IV_HTML}{urllib.parse.quote(arxiv_id)}", 3.0)
        except Exception:  # noqa: BLE001 - 404/timeout is graceful
            return []
        from .extractor import extract_ar5iv_equations
        return extract_ar5iv_equations(payload, max_equations=max_equations)

    def _search_wikidata_constants(self, query: str, limit: int) -> List[LiteratureResult]:
        search_url = (f"{WIKIDATA_SEARCH}?action=wbsearchentities&search="
                      f"{urllib.parse.quote(query)}&language=en&limit=3&format=json")
        data = json.loads(self._fetch_with_backoff(search_url))
        entity_ids = [e.get("id") for e in (data.get("search") or [])
                      if e.get("id")]
        if not entity_ids:
            return []
        ent_url = (f"{WIKIDATA_ENTITIES}?action=wbgetentities&ids="
                   f"{urllib.parse.quote('|'.join(entity_ids))}&props=claims|labels"
                   f"&languages=en&format=json&origin=*")
        ents = json.loads(self._fetch_with_backoff(ent_url)).get("entities") or {}
        out: List[LiteratureResult] = []
        for ent_id, entity in ents.items():
            label = ((entity.get("labels") or {}).get("en") or {}).get("value") or ent_id
            constants: Dict[str, Any] = {}
            snippets: List[str] = []
            for prop_id, claims in (entity.get("claims") or {}).items():
                for claim in claims or []:
                    dv = ((claim.get("mainsnak") or {}).get("datavalue") or {})
                    if dv.get("type") != "quantity":
                        continue
                    amount = dv.get("value") or {}
                    value = amount.get("amount")
                    try:
                        num = float(str(value).replace("+", ""))
                    except (TypeError, ValueError):
                        continue
                    unit_qid = (amount.get("unit") or "").rsplit("/", 1)[-1]
                    unit_label = self._wikidata_unit_label(unit_qid) if unit_qid else ""
                    pname = self._wikidata_property_label(prop_id)
                    key = _sanitize_key(pname or prop_id)
                    constants[key] = {"value": num, "unit": unit_label or ""}
                    snippets.append(f"{pname or prop_id}: {num} {unit_label}")
                    break
                if len(snippets) >= 8:
                    break
            if constants:
                out.append(LiteratureResult(
                    "wikidata", label, "; ".join(snippets)[:1200],
                    f"https://www.wikidata.org/wiki/{ent_id}",
                    badge="Wikidata", constants=constants))
        return out

    def _wikidata_unit_label(self, unit_qid: str) -> str:
        try:
            url = (f"{WIKIDATA_ENTITIES}?action=wbgetentities&ids={unit_qid}"
                   f"&props=labels&languages=en&format=json&origin=*")
            data = json.loads(self._fetch(url, 4.0))
            entity = (data.get("entities") or {}).get(unit_qid) or {}
            return (((entity.get("labels") or {}).get("en") or {}).get("value") or "")
        except Exception:  # noqa: BLE001
            return ""

    def _wikidata_property_label(self, prop_id: str) -> str:
        try:
            url = (f"{WIKIDATA_ENTITIES}?action=wbgetentities&ids={prop_id}"
                   f"&props=labels&languages=en&format=json&origin=*")
            data = json.loads(self._fetch(url, 4.0))
            entity = (data.get("entities") or {}).get(prop_id) or {}
            return (((entity.get("labels") or {}).get("en") or {}).get("value") or "")
        except Exception:  # noqa: BLE001
            return ""

    def _search_oeis(self, query: str, limit: int) -> List[LiteratureResult]:
        url = (f"{OEIS_SEARCH}?q={urllib.parse.quote(query)}&fmt=json")
        data = json.loads(self._fetch_with_backoff(url))
        out: List[LiteratureResult] = []
        for seq in (data.get("results") or [])[:limit]:
            number = str(seq.get("number") or "")
            name = seq.get("name") or ""
            data_str = seq.get("data") or ""
            terms = [int(t) for t in re.findall(r"-?\d+", data_str)][:32]
            formula = seq.get("formula") or ""
            constants: Dict[str, Any] = {}
            benchmarks: List[Dict[str, Any]] = []
            if terms:
                benchmarks.append({
                    "name": f"OEIS {number} initial terms",
                    "input_regime": name[:200], "expected_value": terms,
                    "unit": "", "tolerance": 0.0,
                    "source": f"https://oeis.org/{number}"})
                constants["initial_terms"] = terms
            if formula:
                constants["formula"] = formula
            summary = f"{name}\nData: {data_str}\nFormula: {formula}"
            out.append(LiteratureResult(
                "oeis", f"OEIS A{number}: {name}", summary[:1200],
                f"https://oeis.org/{number}", badge="OEIS",
                constants=constants, benchmarks=benchmarks))
        return out

    def _search_pubchem(self, query: str, limit: int) -> List[LiteratureResult]:
        # PubChem PUG REST needs a compound identifier; extract the most
        # chemical-looking token (gas/formula names like "CO2", "methane").
        candidates = [t for t in re.split(r"[^A-Za-z0-9]+", query or "")
                      if len(t) >= 2 and (t[:1].isupper() or t.isupper()
                                          or t in _COMPOUND_ALIASES)]
        if not candidates:
            return []
        compound = candidates[0]
        compound = _COMPOUND_ALIASES.get(compound.lower(), compound)
        props = "MolecularWeight,ExactMass,XLogP,TPSA,Complexity,IUPACName"
        url = f"{PUBCHEM_PROPERTIES}{urllib.parse.quote(compound)}/property/{props}/JSON"
        try:
            data = json.loads(self._fetch_with_backoff(url))
        except Exception:  # noqa: BLE001 - unknown compound name -> 404
            return []
        table = ((data.get("PropertyTable") or {}).get("Properties") or [])
        if not table:
            return []
        row = table[0]
        constants: Dict[str, Any] = {}
        snippets: List[str] = []
        for prop, unit, si_unit in (
                ("MolecularWeight", "g/mol", "kg/mol"),
                ("ExactMass", "g/mol", "kg/mol"),
                ("XLogP", "", ""), ("TPSA", "A^2", ""),
                ("Complexity", "", "")):
            if row.get(prop) is not None:
                constants[prop] = {"value": float(row[prop]), "unit": unit}
                snippets.append(f"{prop}={row[prop]} {unit}".strip())
        cid = row.get("CID")
        title = row.get("IUPACName") or f"CID {cid} ({compound})"
        return [LiteratureResult(
            "pubchem", f"PubChem: {title}", "; ".join(snippets)[:1200],
            f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}", badge="PubChem",
            constants=constants)]

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
