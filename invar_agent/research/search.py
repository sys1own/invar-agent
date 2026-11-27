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
import ssl
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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
EUROPE_PMC_API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
NCBI_ESEARCH_API = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_ESUMMARY_API = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

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

_UNIT_NOISE_TOKENS = {
    "m2", "m3", "cm2", "cm3", "mm2", "mm3", "km2", "km3",
    "cm", "mm", "um", "nm", "pm", "km", "kg", "mg", "ug", "ng",
    "mol", "mmol", "umol", "nmol", "pmol", "kcal", "cal", "kj", "mj", "gj",
    "kw", "mw", "gw", "hz", "khz", "mhz", "ghz", "kpa", "mpa", "gpa", "pa",
    "atm", "bar", "sec", "msec", "usec", "nsec", "min", "mins", "hr", "hrs",
    "yr", "yrs", "gtc", "pgc", "ppm", "ppb", "ppt", "ev", "kev", "mev", "gev",
    "bits", "nats", "deg", "rad", "bps", "dt", "dx", "dy", "dz",
    "s-1", "m-3", "cm-3",
}

_COMPOUND_UNIT_STR_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:"
    r"(?:mol|mmol|umol|nmol|pmol|kg|mg|ug|ng|g|k?[wW]|M?[jJ]|GtC|PgC|gtco2|m|km|cm|mm|um|M|mM|uM|nM|1)"
    r"\s*/\s*(?:\([^)]+\)|[A-Za-zμµ]+(?:\^?-?[0-9²³]+)?(?:\s*/\s*[A-Za-z]+)?)"
    r"|[A-Za-z]+\^-[123]"
    r")(?![A-Za-z0-9_])",
    re.I,
)
_SCI_LITERAL_RE = re.compile(r"(?<![A-Za-z0-9_])[+-]?\d+(?:\.\d+)?[eE][+-]?\d+(?![A-Za-z0-9_])")


def _strip_unit_and_numeric_noise(text: str) -> str:
    """Strip compound unit expressions (``mol/m3``, ``W/m^2``, ``M/min``) and sci-notation literals."""
    s = _COMPOUND_UNIT_STR_RE.sub(" ", text or "")
    s = _SCI_LITERAL_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


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
    original = _strip_unit_and_numeric_noise((q or "").strip())
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
        if low in _QUERY_FILLER_TOKENS or low in _UNIT_NOISE_TOKENS:
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


def _stem_token(tok: str) -> str:
    """Lightweight scientific de-pluralization for API query terms.

    ``integrators`` -> ``integrator``, ``gradients`` -> ``gradient``,
    ``equations`` -> ``equation``, ``systems`` -> ``system``,
    ``discretizations`` -> ``discretization``. Conservative: only strips a
    single trailing plural suffix and never touches short tokens.
    """
    t = (tok or "").strip().lower()
    if not t:
        return t
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    if len(t) > 4 and t.endswith("es") and not t.endswith(("ses", "xes", "zes")):
        return t[:-2]
    return t


def _normalize_terms(q: str) -> List[str]:
    """API-query terms: filler-stripped, hyphen-split, de-pluralized.

    ``structure-preserving`` becomes ``structure preserving``; every token is
    stemmed so ``integrators discrete gradients`` queries match singular
    index entries. Order-preserving and de-duplicated (case-insensitive).
    """
    out: List[str] = []
    seen = set()
    for tok in _kept_tokens(q):
        for part in tok.replace("-", " ").split():
            stem = _stem_token(part)
            if not stem or stem in seen:
                continue
            seen.add(stem)
            out.append(stem)
    return out


def _relax_token_score(tok: str) -> int:
    """IDF-like keyword score: domain vocabulary > length > technical shape."""
    score = len(_stem_token(tok))
    low = _stem_token(tok)
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
    """Core technical keywords of a query, best-first, de-pluralized.

    Operates on :func:`_normalize_terms` output, so hyphenated compounds are
    split and plurals stemmed before scoring.
    """
    toks = _normalize_terms(query) or [t for t in (query or "").split() if t]
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


def bigram_relaxations(query: str, max_queries: int = 4) -> List[str]:
    """Pass-2 2-word core decompositions of ANY query (2+ terms).

    Generates adjacent pairs plus the first+last domain anchor pair from the
    normalized, de-pluralized term list, e.g.::

        metriplectic integrators discrete gradients
          -> ["metriplectic integrator", "integrator discrete",
              "discrete gradient", "metriplectic gradient"]

    Adjacent pairs come first (in query order), then the first+last anchor
    pair, de-duplicated. Never returns the full query itself.
    """
    terms = _normalize_terms(query)
    if len(terms) < 2:
        return []
    full = " ".join(terms)
    pairs: List[str] = []

    def _add(a: str, b: str) -> None:
        cand = f"{a} {b}"
        if cand != full and cand not in pairs:
            pairs.append(cand)

    for i in range(len(terms) - 1):
        _add(terms[i], terms[i + 1])
    if len(terms) > 2:
        _add(terms[0], terms[-1])
    return pairs[:max_queries]


def anchor_or_queries(query: str, max_queries: int = 2) -> List[str]:
    """Pass-3 broad fallback: single highest-specificity anchor, then an
    explicit 2-anchor OR pair for arXiv/OpenAlex broad recall."""
    kws = _relax_keywords(query)
    if not kws:
        return []
    out: List[str] = [kws[0]]
    if len(kws) > 1:
        out.append(f"{kws[0]} OR {kws[1]}")
    return out[:max_queries]


_ARXIV_DOMAIN_CATS: Dict[str, Tuple[str, ...]] = {
    "LIFE_SCIENCES_AND_BIOMEDICINE": ("q-bio.*", "physics.bio-ph"),
    "EPIDEMIOLOGY_AND_ECOLOGY": ("q-bio.PE", "q-bio.QM", "physics.soc-ph"),
    "EARTH_CLIMATE_AND_ENVIRONMENT": ("physics.ao-ph", "physics.geo-ph", "physics.flu-dyn"),
    "STATISTICS_AND_CAUSAL_INFERENCE": ("stat.ME", "stat.TH", "stat.AP", "stat.ML", "econ.EM", "math.ST"),
    "ECONOMICS_AND_GAME_THEORY": ("econ.TH", "cs.GT", "q-fin.EC", "physics.soc-ph"),
    "MATHEMATICS_CS_AND_INFO_THEORY": ("cs.IT", "math.IT", "cs.GT", "math.DS", "math.PR", "nlin.CD"),
    "CHEMICAL_AND_MATERIALS": ("physics.chem-ph", "cond-mat.mtrl-sci", "q-bio.BM"),
}


_NUMERICAL_METHOD_QUERY_RE = re.compile(
    r"\b(?:patankar|runge[- ]kutta|rk\d+|rk45|dopri\d*|semi[- ]implicit|"
    r"implicit[- ]euler|explicit[- ]euler|positivity[- ]preserving|"
    r"stiff\s+(?:ode|system|solver|integrat)|jacobian|newton[- ]raphson|"
    r"bdf\d*|rosenbrock|crank[- ]nicolson|finite[- ](?:difference|volume|element)|"
    r"cfl|time[- ]stepping|symplectic|splitting\s+scheme|numerical\s+integrat)\b",
    re.I,
)
_CLINICAL_SUBJECT_RE = re.compile(
    r"\b(?:patient|patients|clinical\s+trial|cohort|hospital|covid|sars-cov-2|"
    r"burn|contracture|wound|surgery|surgical|chemotherapy|diagnosis|prognosis)\b",
    re.I,
)


def is_numerical_method_query(query: str) -> bool:
    """Return True when ``query`` specifically targets numerical ODE/PDE solver methods."""
    q = (query or "").strip()
    return bool(_NUMERICAL_METHOD_QUERY_RE.search(q) and not _CLINICAL_SUBJECT_RE.search(q))


def _arxiv_categories_for_prompt(prompt_or_query: str) -> Tuple[str, ...]:
    """Derive macro-domain arXiv category filters from ``classify_macro_scientific_domains``."""
    if is_numerical_method_query(prompt_or_query or ""):
        return ("math.NA", "cs.NA", "physics.comp-ph")
    try:
        from .extractor import classify_macro_scientific_domains
        domains = classify_macro_scientific_domains(prompt_or_query or "")
    except Exception:  # noqa: BLE001
        return ()
    cats: List[str] = []
    for dom in domains[:2]:
        for c in _ARXIV_DOMAIN_CATS.get(dom, ()):
            if c not in cats:
                cats.append(c)
    return tuple(cats)


def _arxiv_query_string(query: str, *, categories: Optional[Sequence[str]] = None) -> str:
    """Build an arXiv ``search_query`` value that the API parser accepts.

    Hyphens are replaced with spaces (unquoted hyphens inside ``all:`` break
    arXiv's parser). Queries with >= 3 terms never require a 4-way AND:
    terms are paired into 2-term ``(all:a AND all:b)`` bigrams joined by OR,
    giving broad recall with precise pairs. 1-2 term queries stay a plain
    AND. An explicit ``OR``/``all:`` query is passed through untouched.
    """
    raw = (query or "").strip()
    if not raw:
        return "all:"
    if " OR " in raw or raw.startswith(("all:", "ti:", "abs:")):
        base = raw
    else:
        terms = [t for t in raw.replace("-", " ").split() if t]
        if not terms:
            return "all:"
        if len(terms) <= 2:
            base = " AND ".join(f"all:{t}" for t in terms)
        else:
            bigrams = [f"(all:{terms[i]} AND all:{terms[i + 1]})"
                       for i in range(len(terms) - 1)]
            base = " OR ".join(bigrams[:3])
    cats = [c.strip() for c in (categories or ()) if c and c.strip()]
    if cats:
        cat_clause = " OR ".join(f"cat:{c}" for c in cats)
        return f"({base}) AND ({cat_clause})"
    return base


def _count_json_list(payload: str, key: str) -> int:
    """Number of raw HTTP records in a JSON API payload (before filtering)."""
    try:
        data = json.loads(payload)
    except Exception:  # noqa: BLE001
        return 0
    items = data.get(key)
    return len(items) if isinstance(items, list) else 0


def _count_crossref_items(payload: str) -> int:
    try:
        data = json.loads(payload)
    except Exception:  # noqa: BLE001
        return 0
    items = ((data.get("message") or {}).get("items")) or []
    return len(items)


_STRIP_HTML_TAGS = re.compile(r"<[^>]+>")


def _strip_markup(text: str) -> str:
    """Strip JATS XML / HTML tags (incl. Wikipedia ``searchmatch`` spans)."""
    return _STRIP_HTML_TAGS.sub(" ", text or "")


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
    "biomed_epmc_pubmed": ("europe_pmc", "pubmed", "biorxiv", "medrxiv"),
}


def classify_research_domain(prompt_or_query: str) -> set:
    """Route a prompt to research channels.

    Returns a subset of ``{"equations", "peer_reviewed", "constants",
    "discrete_oeis", "chemical_pubchem", "biomed_epmc_pubmed"}``. Every prompt
    gets ``peer_reviewed`` + ``equations`` + ``constants`` unless it is clearly
    discrete, chemical, or biomedical/epidemiological, in which case the
    dedicated channels are added.
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
    try:
        from .extractor import classify_macro_scientific_domains
        macro_domains = set(classify_macro_scientific_domains(prompt_or_query or ""))
        if macro_domains & {"LIFE_SCIENCES_AND_BIOMEDICINE", "EPIDEMIOLOGY_AND_ECOLOGY"}:
            channels.add("biomed_epmc_pubmed")
        if macro_domains & {"LIFE_SCIENCES_AND_BIOMEDICINE", "CHEMICAL_AND_MATERIALS"}:
            channels.add("chemical_pubchem")
            channels.add("biomed_epmc_pubmed")
    except Exception:  # noqa: BLE001
        pass
    # Pure discrete/chemical/biomedical prompts keep the core channels too
    # (equations still apply), but skip nothing: OEIS/PubChem/EuropePMC/PubMed are additive.
    return channels


def _channels_for_prompt(prompt_or_query: str) -> List[str]:
    if is_numerical_method_query(prompt_or_query or ""):
        return ["arxiv", "openalex", "crossref", "semantic_scholar", "wikipedia"]
    channels = classify_research_domain(prompt_or_query)
    sources: List[str] = []
    for channel in ("biomed_epmc_pubmed", "peer_reviewed", "equations", "constants",
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


def _default_ssl_context() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001
        return ssl.create_default_context()


def _default_fetch(url: str, timeout: float) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    ctx = _default_ssl_context() if url.lower().startswith("https://") else None
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:  # noqa: S310 - fixed https/http API hosts
        return resp.read().decode("utf-8", errors="replace")


def _cache_key(source: str, query: str) -> str:
    return hashlib.blake2b(f"{source}|{query}".encode(), digest_size=16).hexdigest()


_INTERNAL_DIAG_RE = re.compile(
    r"\b(?:cabi|fallback|zero[- ]step|prewrite_guard|python_smoke|clang\+\+|clang|"
    r"sandbox|tier[- ]?[012]|accretion|overlay|_invar_\w+|buf\.data)\b"
    r"|state\[\d+\]",
    re.I,
)
_OFF_DOMAIN_CV_RE = re.compile(
    r"\b(?:ego4d|vision-and-language\s+navigation|object\s+detection|image\s+segmentation|"
    r"video\s+grounding|embodied\s+navigation|3d\s+pose\s+estimation)\b",
    re.I,
)
_CV_PROMPT_RE = re.compile(
    r"\b(?:computer\s+vision|object\s+detection|image\s+segmentation|video|robotics|vln|ego4d)\b",
    re.I,
)


_OFF_DOMAIN_CLINICAL_RE = re.compile(
    r"\b(?:covid[- ]?19|sars[- ]?cov[- ]?2|burn\s+contracture|post[- ]?burn|"
    r"wound\s+healing|randomized\s+controlled\s+trial|surgical\s+reconstruction|"
    r"case\s+report|chemotherapy\s+regimen|outpatient\s+clinic)\b",
    re.I,
)


def strip_internal_diagnostic_tokens(text: str) -> str:
    """Remove internal invar-agent diagnostic tokens and raw unit/sci-notation noise from search queries."""
    cleaned = _INTERNAL_DIAG_RE.sub(" ", text or "")
    cleaned = _strip_unit_and_numeric_noise(cleaned)
    toks = [t for t in cleaned.split() if t.lower() not in _UNIT_NOISE_TOKENS]
    return " ".join(toks).strip()


def _is_off_domain_hit(hit: LiteratureResult, context: str) -> bool:
    ctx = context or ""
    blob = f"{hit.title} {hit.summary}"
    if not _CV_PROMPT_RE.search(ctx) and _OFF_DOMAIN_CV_RE.search(blob):
        return True
    if not _CLINICAL_SUBJECT_RE.search(ctx) and _OFF_DOMAIN_CLINICAL_RE.search(blob):
        return True
    return False


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
        raw_query = strip_internal_diagnostic_tokens((query or "").strip())
        started = time.perf_counter()
        # Pass 1: cleaned, de-pluralized, hyphen-normalized FULL query.
        # Short planner queries (<=5 terms) keep every technical token;
        # sentence-like input still goes through the <=3-entity sanitizer.
        kept = _kept_tokens(raw_query)
        if 0 < len(kept) <= 5:
            query = " ".join(_normalize_terms(raw_query)) or _sanitize_query(raw_query)
        else:
            query = _sanitize_query(raw_query)
        limit = max_results or self.max_results
        routing_ctx = raw_query if is_numerical_method_query(raw_query) else (prompt_or_query or raw_query)
        if sources is None:
            sources = tuple(_channels_for_prompt(routing_ctx))
        arxiv_cats = _arxiv_categories_for_prompt(routing_ctx)
        channel_counts: Dict[str, int] = {}
        raw_counts: Dict[str, int] = {}
        channel_errors: Dict[str, str] = {}
        relaxed_trail: List[str] = []
        error_note = ""

        def _run(pass_sources: tuple, pass_query: str, *, use_arxiv_cats: bool = True) -> List[LiteratureResult]:
            nonlocal error_note
            pass_results: List[LiteratureResult] = []
            cats_arg = arxiv_cats if use_arxiv_cats else ()
            if self.offline or len(pass_sources) <= 1:
                for source in pass_sources:
                    try:
                        got = self._search_source(source, pass_query, limit, arxiv_categories=cats_arg)
                        channel_counts[source] = channel_counts.get(source, 0) + len(got)
                        raw_counts[source] = raw_counts.get(source, 0) + \
                            getattr(self, "_last_raw_count", 0)
                        pass_results.extend(got)
                    except Exception as exc:  # noqa: BLE001
                        channel_errors[source] = f"{type(exc).__name__}: {str(exc)[:120]}"
                        error_note = f"{source}: {str(exc)[:120]}"
            else:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                with ThreadPoolExecutor(max_workers=min(4, len(pass_sources))) as pool:
                    futures = {pool.submit(self._search_source_safe, source,
                                           pass_query, limit, cats_arg): source
                               for source in pass_sources}
                    for fut in as_completed(futures):
                        src = futures[fut]
                        try:
                            got = fut.result()
                            channel_counts[src] = channel_counts.get(src, 0) + len(got)
                            pass_results.extend(got)
                        except Exception as exc:  # noqa: BLE001
                            channel_errors[src] = f"{type(exc).__name__}: {str(exc)[:120]}"
                            error_note = f"{src}: {str(exc)[:120]}"
            return pass_results

        # Merge per-channel raw counts / errors recorded by _search_source.
        def _merge_diag() -> None:
            for src, n in (getattr(self, "_channel_raw", {}) or {}).items():
                raw_counts[src] = raw_counts.get(src, 0) + int(n or 0)
            for src, msg in (getattr(self, "_channel_errors", {}) or {}).items():
                channel_errors.setdefault(src, str(msg)[:160])

        # Pass 1: primary academic APIs with the normalized full query.
        self._channel_raw = {}
        self._channel_errors = {}
        results = _run(sources, query, use_arxiv_cats=True)
        _merge_diag()

        # Pass 2: 2-word bigram decomposition — runs whenever Pass 1 accepted
        # 0 sources, regardless of the initial word count (2-, 3- or 4-word
        # queries included). Prioritizes adjacent pairs and first+last domain
        # anchors from the de-pluralized term list.
        if not results:
            pass2_sources = tuple(s for s in ("arxiv", "openalex") if s in sources) \
                or ("arxiv", "openalex")
            for relaxed in bigram_relaxations(raw_query):
                relaxed_trail.append(relaxed)
                relaxed_results = _run(pass2_sources, relaxed, use_arxiv_cats=True)
                _merge_diag()
                if relaxed_results:
                    results = relaxed_results
                    break

        # Pass 3: broad single-anchor / explicit OR query on arXiv & OpenAlex.
        if not results:
            pass2_sources = tuple(s for s in ("arxiv", "openalex") if s in sources) \
                or ("arxiv", "openalex")
            for anchor_q in anchor_or_queries(raw_query):
                relaxed_trail.append(anchor_q)
                relaxed_results = _run(pass2_sources, anchor_q, use_arxiv_cats=True)
                _merge_diag()
                if relaxed_results:
                    results = relaxed_results
                    break

        # Pass 4: entity APIs (Wikidata/OEIS/PubChem) with a compact 1-2 word
        # canonical concept only — never full sentences.
        if not results:
            concept = entity_concept(" ".join(_kept_tokens(raw_query)) or raw_query)
            pass3_sources = tuple(s for s in sources
                                  if s in ("wikidata", "oeis", "pubchem"))
            if concept and pass3_sources:
                relaxed_trail.append(concept)
                results = _run(pass3_sources, concept, use_arxiv_cats=False)
                _merge_diag()

        relaxed_query = " | ".join(relaxed_trail) if relaxed_trail else ""

        # Stable order: keep connector grouping deterministic for caching/tests.
        results = [r for r in results if not _is_off_domain_hit(r, prompt_or_query or raw_query)]
        order = {s: i for i, s in enumerate(sources)}
        results.sort(key=lambda r: order.get(r.source, 99))

        # Non-intrusive SHBT Federation: inject only when SHBTSolvabilityResolver activates.
        from .shbt_federation import resolve_shbt_solvability, shbt_literature_dicts
        shbt_res = resolve_shbt_solvability(
            prompt_or_query or raw_query, allow_remote_fetch=not self.offline
        )
        if shbt_res.should_activate:
            shbt_items = [
                LiteratureResult(
                    source=d["source"],
                    title=d["title"],
                    url=d["url"],
                    year=d["year"],
                    summary=d["summary"],
                    badge=d["badge"],
                    tldr=d["tldr"],
                    equations=d["equations"],
                    constants=d["constants"],
                    benchmarks=d["benchmarks"],
                )
                for d in shbt_literature_dicts(shbt_res)
            ]
            channel_counts["shbt_federation"] = len(shbt_items)
            results = shbt_items + results
        results = results[: max(limit * len(sources), limit * 3, 12)]
        # Diagnostic telemetry for search_log query_done events.
        self.last_query_meta = {
            "query": query,
            "raw_query": raw_query,
            "status": "ok" if results else "empty",
            "arxiv_categories": list(arxiv_cats),
            "channel_counts": channel_counts,
            "by_source": {src: {"count": channel_counts.get(src, 0), "error": channel_errors.get(src)} for src in sources},
            "raw_counts": raw_counts,
            "channel_errors": channel_errors,
            "relaxed_query": relaxed_query,
            "error": error_note,
            "offline": self.offline,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
        }
        if not results:
            # Never let the kernel receive empty parameters: fall back to
            # curated domain constants when every API returned 0 hits.
            results = _fallback_constants(query)
        return results

    def _search_source_safe(self, source: str, query: str,
                            limit: int,
                            arxiv_categories: Sequence[str] = ()) -> List[LiteratureResult]:
        try:
            return self._search_source(source, query, limit, arxiv_categories=arxiv_categories)
        except Exception as exc:  # noqa: BLE001
            errs = getattr(self, "_channel_errors", None)
            if errs is not None:
                errs[source] = f"{type(exc).__name__}: {str(exc)[:140]}"
            raise

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
    def _search_source(self, source: str, query: str, limit: int,
                       arxiv_categories: Sequence[str] = ()) -> List[LiteratureResult]:
        cache_q = f"{query}|cat={','.join(arxiv_categories)}" if (source == "arxiv" and arxiv_categories) else query
        cached = self._read_cache(source, cache_q)
        raw = getattr(self, "_channel_raw", None)
        if self.offline:
            if raw is not None:
                raw[source] = raw.get(source, 0) + len(cached)
            return cached or []
        raw_count = 0
        try:
            if source == "arxiv":
                payload = self._fetch_with_backoff(self._arxiv_url(query, limit, categories=arxiv_categories))
                raw_count = payload.count("<entry>")
                results = self._parse_arxiv(payload)
                upgraded = 0
                for item in results:
                    if upgraded >= 2:
                        break
                    m_arx = re.search(r"arxiv\.org/abs/([0-9]+\.[0-9]+|[a-z\-]+/[0-9]+)", item.url or "", re.I)
                    if m_arx:
                        eqs = self._fetch_ar5iv_equations(m_arx.group(1))
                        if eqs:
                            item.equations = eqs
                            upgraded += 1
            elif source == "openalex":
                payload = self._fetch_with_backoff(self._openalex_url(query, limit))
                raw_count = _count_json_list(payload, "results")
                results = self._parse_openalex(payload)
            elif source == "crossref":
                payload = self._fetch_with_backoff(self._crossref_url(query, limit))
                raw_count = _count_crossref_items(payload)
                results = self._parse_crossref(payload)
            elif source == "wikipedia":
                results = self._search_wikipedia(query, limit)
                raw_count = len(results)
            elif source == "semantic_scholar":
                results = self._search_semantic_scholar(query, limit)
                raw_count = len(results)
            elif source == "wikidata":
                results = self._search_wikidata_constants(query, limit)
                raw_count = len(results)
            elif source == "oeis":
                results = self._search_oeis(query, limit)
                raw_count = len(results)
            elif source == "pubchem":
                results = self._search_pubchem(query, limit)
                raw_count = len(results)
            elif source in ("europe_pmc", "europepmc"):
                results = self._search_europe_pmc(query, limit)
                raw_count = len(results)
            elif source in ("pubmed", "ncbi_pubmed"):
                results = self._search_pubmed(query, limit)
                raw_count = len(results)
            elif source in ("biorxiv", "medrxiv"):
                results = self._search_europe_pmc(f'({query}) AND PUBLISHER:"{source}" AND SRC:PPR', limit, source=source)
                raw_count = len(results)
            else:
                return cached or []
        except Exception as exc:  # noqa: BLE001 - record error, fall back to cache
            errs = getattr(self, "_channel_errors", None)
            if errs is not None:
                errs[source] = f"{type(exc).__name__}: {str(exc)[:140]}"
            return cached or []
        if raw is not None:
            raw[source] = raw.get(source, 0) + (raw_count or len(results))
        # No relevance gate here: academic papers are ingested even with short
        # or missing abstracts — extractor.py handles symbolic content.
        results = [r for r in results if r.title]
        if results:
            self._write_cache(source, cache_q, [r.__dict__ for r in results])
            return results
        # NEVER cache a 0-result response: an empty answer must be retried
        # live on the next call instead of poisoning the disk cache.
        return []

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
    def _arxiv_url(query: str, limit: int, *, categories: Sequence[str] = ()) -> str:
        q = urllib.parse.quote(_arxiv_query_string(query, categories=categories))
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

    HUMANITIES_EXCLUDE = {
        "gedichte", "shakespeare", "poetry", "poesie", "theater", "theatre",
        "literatur", "drama", "journal bearing", "journal bearings", "literary",
    }
    SCIENTIFIC_TOKENS = {
        "quantum", "operator", "entropy", "dissipat", "unitary", "cayley",
        "lindblad", "matrix", "equation", "numerical", "integrat", "hamilton",
        "mechanic", "algebra", "markov", "stochastic", "pade", "algorithm",
        "convergence", "discrete", "gradient", "manifold", "symplectic",
        "energy", "drift", "invariant", "physics", "system",
    }

    @staticmethod
    def _crossref_url(query: str, limit: int) -> str:
        words = query.strip().split()
        if len(words) == 1:
            low = words[0].lower()
            if low in {"kraus", "pade", "lindblad", "cayley", "runge", "kutta", "strang", "liouville"}:
                query = f"{query} quantum operator"
        return f"{CROSSREF_API}?query={urllib.parse.quote(query)}&rows={limit}"

    @staticmethod
    def _parse_crossref(payload: str) -> List[LiteratureResult]:
        data = json.loads(payload)
        out: List[LiteratureResult] = []
        for item in (data.get("message") or {}).get("items", []):
            titles = item.get("title") or []
            title = titles[0] if titles else ""
            low_title = title.lower()
            container = " ".join(item.get("container-title") or []).lower()
            if any(tok in low_title or tok in container for tok in LiteratureSearchClient.HUMANITIES_EXCLUDE):
                continue
            abstract = _strip_markup(item.get("abstract", "") or "")
            if not abstract.strip():
                if not any(tok in low_title or tok in container for tok in LiteratureSearchClient.SCIENTIFIC_TOKENS):
                    continue
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
                extract = _strip_markup(summary.get("extract") or "")
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

    def _search_europe_pmc(self, query: str, limit: int = 10,
                           cursor_mark: str = "*", *, source: str = "europe_pmc") -> List[LiteratureResult]:
        """Zero-API-key Europe PMC REST connector with strict 8s timeout and cursorMark support."""
        page_size = min(max(int(limit or 10), 1), 10)
        url = (
            f"{EUROPE_PMC_API}?query={urllib.parse.quote(query)}"
            f"&format=json&resultType=core&pageSize={page_size}"
            f"&cursorMark={urllib.parse.quote(cursor_mark)}"
        )
        timeout = min(float(self.timeout or 8.0), 8.0)
        try:
            payload = self._fetch(url, timeout)
            data = json.loads(payload)
        except Exception as exc:  # noqa: BLE001
            errs = getattr(self, "_channel_errors", None)
            if errs is not None:
                errs[source] = f"{type(exc).__name__}: {str(exc)[:140]}"
            return []
        items = ((data.get("resultList") or {}).get("result")) or []
        out: List[LiteratureResult] = []
        for item in items[:page_size]:
            if not isinstance(item, dict):
                continue
            title = _strip_markup(str(item.get("title") or "")).strip()
            if not title:
                continue
            abstract = _strip_markup(str(item.get("abstractText") or "")).strip()
            pmid = str(item.get("pmid") or "").strip()
            pmcid = str(item.get("pmcid") or "").strip()
            doi = str(item.get("doi") or "").strip()
            pub_year = item.get("pubYear")
            year = int(str(pub_year)[:4]) if pub_year and str(pub_year)[:4].isdigit() else None
            cited = item.get("citedByCount")
            citation_count = int(cited) if isinstance(cited, (int, float)) else None
            if pmcid:
                article_url = f"https://europepmc.org/article/PMC/{pmcid}"
            elif pmid:
                article_url = f"https://europepmc.org/article/MED/{pmid}"
            elif doi:
                article_url = f"https://doi.org/{doi}"
            else:
                article_url = ""
            constants: Dict[str, Any] = {}
            if abstract:
                try:
                    from .extractor import extract_parameters_from_text
                    for p in extract_parameters_from_text(abstract, source="EuropePMC")[:10]:
                        constants[p.name] = {
                            "value": p.si_value,
                            "unit": p.si_unit,
                            "raw_value": p.raw_value,
                            "raw_unit": p.raw_unit,
                        }
                except Exception:  # noqa: BLE001
                    pass
            out.append(
                LiteratureResult(
                    source=source,
                    title=title,
                    summary=abstract[:1200],
                    url=article_url,
                    year=year,
                    badge="EuropePMC" if source == "europe_pmc" else source,
                    citation_count=citation_count,
                    constants=constants,
                )
            )
        return out

    def _search_pubmed(self, query: str, limit: int = 10) -> List[LiteratureResult]:
        """Zero-API-key NCBI PubMed E-utilities connector (esearch.fcgi + esummary.fcgi, retmode=json)."""
        retmax = min(max(int(limit or 10), 1), 10)
        esearch_url = (
            f"{NCBI_ESEARCH_API}?db=pubmed&term={urllib.parse.quote(query)}"
            f"&usehistory=y&retmode=json&retmax={retmax}"
        )
        timeout = min(float(self.timeout or 8.0), 8.0)
        try:
            esearch_payload = self._fetch(esearch_url, timeout)
            esearch_data = json.loads(esearch_payload)
        except Exception as exc:  # noqa: BLE001
            errs = getattr(self, "_channel_errors", None)
            if errs is not None:
                errs["pubmed"] = f"{type(exc).__name__}: {str(exc)[:140]}"
            return []
        es_res = esearch_data.get("esearchresult") or {}
        idlist = [str(x).strip() for x in (es_res.get("idlist") or []) if str(x).strip()]
        webenv = str(es_res.get("webenv") or "").strip()
        querykey = str(es_res.get("querykey") or "").strip()
        if not idlist and not (webenv and querykey):
            return []
        if idlist:
            esummary_url = (
                f"{NCBI_ESUMMARY_API}?db=pubmed&id={urllib.parse.quote(','.join(idlist[:retmax]))}"
                f"&retmode=json"
            )
        else:
            esummary_url = (
                f"{NCBI_ESUMMARY_API}?db=pubmed&query_key={urllib.parse.quote(querykey)}"
                f"&WebEnv={urllib.parse.quote(webenv)}&retmax={retmax}&retmode=json"
            )
        try:
            esummary_payload = self._fetch(esummary_url, timeout)
            esummary_data = json.loads(esummary_payload)
        except Exception as exc:  # noqa: BLE001
            errs = getattr(self, "_channel_errors", None)
            if errs is not None:
                errs["pubmed"] = f"{type(exc).__name__}: {str(exc)[:140]}"
            return []
        res_map = esummary_data.get("result") or {}
        uids = res_map.get("uids") or idlist
        out: List[LiteratureResult] = []
        for uid in uids[:retmax]:
            doc = res_map.get(str(uid))
            if not isinstance(doc, dict):
                continue
            title = _strip_markup(str(doc.get("title") or "")).strip()
            if not title:
                continue
            source_j = str(doc.get("fulljournalname") or doc.get("source") or "").strip()
            pubdate = str(doc.get("pubdate") or "")
            m_yr = re.search(r"\b(19\d\d|20\d\d)\b", pubdate)
            year = int(m_yr.group(1)) if m_yr else None
            authors = ", ".join(
                str(a.get("name") or "") for a in (doc.get("authors") or [])[:4] if isinstance(a, dict)
            )
            summary_parts = [p for p in (source_j, pubdate, authors) if p]
            summary = " — ".join(summary_parts)
            out.append(
                LiteratureResult(
                    source="pubmed",
                    title=title,
                    summary=summary[:1200],
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{uid}/",
                    year=year,
                    badge="PubMed",
                )
            )
        return out

    # -- deterministic disk cache -----------------------------------------
    def clear_cache(self) -> int:
        """Delete every cached literature response; returns files removed."""
        removed = 0
        if self.cache_dir is None or not self.cache_dir.exists():
            return 0
        for path in self.cache_dir.glob("*.json"):
            try:
                path.unlink()
                removed += 1
            except Exception:  # noqa: BLE001 - best-effort cleanup
                continue
        return removed

    def _cache_path(self, source: str, query: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{source}_{_cache_key(source, query)}.json"

    def _read_cache(self, source: str, query: str) -> List[LiteratureResult]:
        path = self._cache_path(source, query)
        if path is None or not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return [LiteratureResult(**item) for item in data]
        except Exception:  # noqa: BLE001 - corrupt cache is a miss
            return []

    def _write_cache(self, source: str, query: str, items: List[Dict]) -> None:
        path = self._cache_path(source, query)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(items), encoding="utf-8")
        except Exception:  # noqa: BLE001 - cache is best-effort
            pass
