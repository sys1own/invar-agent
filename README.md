# invar-agent

**Mathematically Verified Autonomous Agent Kernel and Web Console.**

`invar-agent` is a formally grounded autonomous software-engineering kernel. Every mutation an agent proposes passes through a nine-stage verified state machine before it can touch disk:

```
Ingest -> GoI_Plan -> Wavefront_Schedule -> Overlay_Mutate -> PreWrite_Guard
       -> SMT_Concolic_Verify -> Sandbox_Execute -> Reflex_Heal
       -> Adjoint_Commit_or_Rollback
```

## Architecture

- **Rust PyO3 native core** (`invar_agent/_native`): Girard Geometry-of-Interaction nilpotency solver, arena-allocated Lafont Interaction Net engine, zero-token deterministic healer transducers, and the 256-bit BLAKE3 Holographic Boundary Register — all exposed as `invar_agent._native` with a pure-Python fallback.
- **Python kernel** (`invar_agent/`): pydantic v2 contracts, transactional Copy-on-Write overlay with algebraic adjoint rollback, Precision Shield (FOGE + Z3 SMT + concolic execution + Chiasmus symmetry + holographic parity + Rust shield), hermetic sandbox with command inspection, three-tier cognitive router (Tier-0 Reflex / Tier-1 Flash / Tier-2 Pro), and curated exports.
- **Web console** (`invar_agent/server.py` + `static/index.html`): FastAPI + SSE-style single-page console with live stage visualization, overlay/shield inspector, and export menu.

## Quickstart

```bash
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -e .
maturin develop --release   # builds invar_agent._native
invar-agent serve ./my-workspace
pytest -v                   # full verification suite
cargo test --manifest-path invar_agent/_native/Cargo.toml
```

## LLM providers

The copilot chat and the builder (Tier-1 repairs, Tier-2 code generation) call an
OpenAI-compatible chat API. Supported providers: **Gemini**, **OpenAI**, **DeepSeek**,
**OpenRouter**.

### Setup

1. Start the console: `invar-agent serve ./my-workspace` (add `--provider gemini|openai|deepseek|openrouter`
   to force a provider).
2. Open the web UI, click the **gear** icon and paste the API key. On save the server calls
   `GET {base}/models` for every provider, checks the resolved fast/reasoning models against the
   live list (auto-picking a fallback when the default is unavailable), and shows the resolved
   models, the model count and any warnings. Use **Test connection** for a one-token FAST-tier check.
3. Keys and the chosen provider are saved in memory for the server session. To persist them across
   restarts the server writes `~/.invar-agent/config.json` (mode `0600`, path overridable with
   `INVAR_CONFIG_PATH`) — the API key itself is only written when `INVAR_PERSIST_KEY=1`.

### Environment variables

| Variable | Purpose |
| --- | --- |
| `GEMINI_API_KEY`, `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY` | Per-provider keys |
| `INVAR_API_KEY` | Generic key when `INVAR_LLM_PROVIDER` is set |
| `INVAR_LLM_PROVIDER` | Provider to use when several keys are present (also `--provider`) |
| `INVAR_CONFIG_PATH` | Location of the persisted config file |
| `INVAR_PERSIST_KEY` | `1` to persist the API key in the config file |
| `GEMINI_MODEL_FAST` / `GEMINI_MODEL_REASONING` (etc. per provider) | Tier model override |
| `INVAR_LLM_TIER_FAST` / `INVAR_LLM_TIER_REASONING` | Generic tier overrides (JSON mapping or model name) |

At startup the server auto-detects the key from the environment (`INVAR_LLM_PROVIDER` picks the
provider; otherwise exactly one provider key present selects it) and falls back to the persisted file.
Precedence for model selection: explicit override > provider env > generic env > config file > defaults
(`DEFAULT_TIER_MODELS` in `invar_agent/config.py`, verified against the live `/models` listings).

### Google Colab

Add the key to **Colab Secrets** (key icon, left sidebar) with the exact variable name
(e.g. `GEMINI_API_KEY` or `DEEPSEEK_API_KEY`), then, before starting the server, export it into the
environment from the secrets:

```python
from google.colab import userdata
import os
os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")
# or: os.environ["DEEPSEEK_API_KEY"] = userdata.get("DEEPSEEK_API_KEY")
```

Run the server afterwards (`!invar-agent serve ./workspace`). Colab secrets are injected into
`os.environ` only through this call, so the export must happen in the same notebook cell or earlier.

### Default models

| Provider | Fast tier | Reasoning tier |
| --- | --- | --- |
| Gemini | `gemini-3.8-flash` | `gemini-3.1-pro-preview` |
| OpenAI | `gpt-5.4-mini` | `gpt-5.5-pro` |
| DeepSeek | `deepseek-flash` | `deepseek-v4-pro` |
| OpenRouter | `openrouter/auto` | `openrouter/auto` |

Gemini `gemini-2.5-*` IDs are legacy (access-limited for new projects per
https://ai.google.dev/gemini-api/docs/deprecations) and are kept only as last-resort preference
fallbacks, never as defaults. DeepSeek retired `deepseek-chat` / `deepseek-reasoner` on 2026-07-24.

### Troubleshooting

- **400 Bad Request** — usually a model that does not exist for your key/project. Open the gear
  modal: validation lists the provider's live models and auto-selects an available fallback for
  defaults; explicit overrides are kept but flagged with suggestions. Some models reject sampling
  parameters (`temperature`/`top_p`); the client retries once without them and remembers the
  provider/model pair.
- **401 Unauthorized / API key not valid** — wrong key for the provider selected (each provider has
  its own key format: Gemini `AIza…`, OpenAI/DeepSeek `sk-…`, OpenRouter `sk-or-…`). The settings
  modal shows a soft warning on prefix mismatch.
- **403 Forbidden / permission denied** — the key is valid but lacks access to the model (e.g. a
  preview model not enabled for the project, or a region-restricted provider). Pick another model via
  the tier env vars or the provider's model list shown in the modal.
- **404 Not Found** — the endpoint or model path is wrong (e.g. an outdated base URL); not retried.
- **429 Too Many Requests / 5xx** — retried automatically up to 2 times with exponential backoff and
  jitter; other 4xx are never retried. Use a provider with a paid quota or lower request rate.
- Every failed provider call is recorded in `/api/status` under `llm.last_error` with a timestamp,
  and the chat shows a banner with the provider, model, HTTP status and the provider's own message
  (API keys are redacted before anything is logged, returned, or shown).

### Live smoke tests

`tests/test_llm_live.py` is opt-in and skipped by default: run `pytest -m live` with
`GEMINI_API_KEY` or `DEEPSEEK_API_KEY` set to validate the key, list models and complete one token
per fast tier.

## Research Agent

invar-agent ships a second, selectable **Research Agent** alongside the Builder Agent. It runs the
same nine-stage verified kernel as a *cyclic* research loop: Ingest (question + falsifiable
hypothesis + search queries) → GoI_Plan → Wavefront_Schedule (nilpotency ρ(UM) < 1, with
Sherman-Morrison-Woodbury rank-k incremental updates) → Overlay_Mutate (CoW scratch) →
PreWrite_Guard (SHACL/Chiasmus firewall + v9 transpiler compliance) →
SMT_Concolic_Verify (real Z3 consistency check via `research/smt.py`) → Sandbox_Execute
(isolated subprocess C-ABI with rlimits; `rust_hin`, `cpp`, or `multi_variant` Pareto
benchmarks) → Reflex_Heal (deterministic Tier-0 source repair + egg simplification) →
Adjoint_Commit_or_Rollback.

### Cumulative Blueprint Evolution

A `ResearchBlueprint` can target **multiple research stages** (`target_research_stages`, default 1).
Stage 1 synthesises and verifies a foundation module, then pins it as a checkpoint layer
(`ModuleLayer` with BLAKE3 bundle hash, contracts, and baseline invariants). Stage k > 1 re-enters
Ingest → GoI_Plan in **accretive mode**: telemetry from stage k-1 (residuals, energy drift,
finite/divergence flags) steers targeted literature queries; the Tier-2 LLM receives the exact
verified code and contracts of the previous stages and must *extend and couple* into the same
unified module; the GoI wavefront scheduler is updated incrementally via Sherman-Morrison-Woodbury
ΔM = P·Qᵀ (`PyGoIWavefrontScheduler.apply_incremental_update`); and Sandbox_Execute re-verifies
every pinned layer inside the same compiled artefact as the new logic (regression contracts are
prefixed `regression:<layer>:<contract>`). Rollback is **delta-scoped**: a failed stage k rolls back
only its delta to the stage k-1 checkpoint, feeds the counterexample back into GoI_Plan, and when
retries are exhausted the last verified unified checkpoint is preserved and exported under
`<workspace>/checkpoints/<blueprint_id>/stage_k_unified_module.py` + `blueprint.json` rather than
losing verified work. The accretion validator (`research/accretion.py`) makes pinned functions
immutable (redefinitions are reverted, not rejected), restores deleted pinned functions, requires
the new driver to call at least one pinned function, and extracts the stage delta for rollback.

### Live Literature Search

`research/search.py` provides `LiteratureSearchClient` with three sources: arXiv
(`export.arxiv.org/api/query`), OpenAlex (`api.openalex.org/works`, with Crossref fallback) and a
Wikipedia opensearch+summary fallback. It has clean timeout handling, rate-limit backoff
(0s/1.5s/4s), and deterministic BLAKE2b-keyed caching so offline unit tests stay hermetic
(`offline=True` serves only the cache; network failure falls back to cache). Search runs strictly
during Ingest/GoI_Plan — Sandbox_Execute stays network-isolated (subprocess with rlimits). Every
query, retrieved paper and extracted mathematical insight (equations/coefficients/scaling laws) is
emitted as structured `search_log` SSE events (`query` / `paper` / `insight` / `query_done`).
Live fetching is opt-in via `INVAR_RESEARCH_SEARCH_LIVE=1`; otherwise the offline cache is used.

### Epistemic Feasibility Guardrails

Before any blueprint is synthesised or simulated, `research/feasibility.py` classifies the request
into one of four verdicts (`FeasibilityAssessment` on the blueprint):

- `OPTIMAL_DIRECT` — quantitative research suited for literature + deterministic native verify.
- `REFORMULATED_SURROGATE` — too large for local execution (3D DNS turbulence, ab-initio,
  macro-scale); the engine must define a reduced-order surrogate and justify the reduction.
- `REFUSED_IMPOSSIBLE` — axiomatic impossibility (net positive energy with conservation
  invariants, perpetual motion, undecidable/contradictory constraints). Also enforced *during*
  execution: if Z3 proves a contract's invariants mutually unsatisfiable, the run halts as
  REFUSED before any simulation.
- `REFUSED_UNSUITABLE` — subjective/historical/literary/clinical tasks without falsifiable
  governing equations ("math-washing").

A deterministic heuristic gate runs first (and is authoritative for refusals); the Tier-2 REASONING
model may escalate or refine the verdict. The gate runs in `POST /api/v1/research/plan`, in Copilot
`SUGGEST_RESEARCH_PLAN` (which returns a `RESEARCH_REFUSED` card), and defensively inside the state
machine. Refusals never reach compilation or simulation; the API returns
`{"status": "refused", "feasibility": {...}}` and the Web Console renders a red refusal card
explaining **why**, **which epistemic boundary** was violated, and **how to reformulate**.

### Prompt Optimisation

The planner prompts (`research/planner.py`) encode tool mastery heuristics: `cpp` C-ABI for
RK4 ODE/PDE, matrix ops and heavy loops; `rust_hin` for scalar/vector recurrences, discrete math
and algebraic invariants (HIN structural normalisation telemetry `hin_steps`/`hin_stalled_pairs`);
`multi_variant` for Pareto comparison of candidate algorithms (residual vs wall-time frontier).
All synthesis prompts enforce the strict v9_transpiler_friendly subset: primitive
int/float/bool/`list[float]`/`list[list[float]]`, for/while with guards, `range`, `math.*`,
deterministic intrinsics (`prng_uniform`, `clamp`, `sign`) and top-level pure functions — with
classes, try/except, with, yield, async, `import os/sys/socket/random/re/json`, nested closures
and unbounded loops forbidden (enforced both in the prompt and by the PreWrite_Guard AST check).
Accretive prompts receive the exact verified code + contracts of stage k-1 plus sandbox observables
(`state`, `initial`, `trace_min`, `trace_max`, `telemetry`, `variants`) and instruct the model to
extend and couple rather than overwrite.

- **Cognitive routing**: Tier-2 REASONING synthesises blueprints and accretive stages; Tier-1 FAST
  formulates telemetry-guided literature queries; Tier-0 Reflex handles SMT/Z3/concolic checks,
  the security firewall, compilation and egg-based repair.
- **Web console**: a `[ Builder Agent | Research Agent ]` toggle in the navbar switches the main
  pane. Research mode shows a plan card (editable blueprint JSON with **Approve & Run Deep
  Research** / **Edit Plan** / **Abort**), three live log panes (`SEARCH LOG`, `GoI/HIN STREAM`,
  `ACCELERATOR LOG`), and a telemetry strip (contract, status, execution time, residuals).
  Copilot answers to research questions render a `SUGGEST_RESEARCH_PLAN` action card.
- **API**: `POST /api/v1/research/plan` (synthesise a `ResearchBlueprint`),
  `POST /api/v1/research/action` (`approve_plan` / `edit_plan` / `abort`),
  `GET /api/v1/research/status`, and `GET /api/v1/research/stream` (SSE across the three channels).
- **Native acceleration**: `invar_agent._native` exposes `PyUASTGraph`, `PyHINArenaEngine`,
  `PyGoIWavefrontScheduler` (with Sherman-Morrison incremental updates), `GoIProofNet`,
  `GoIWavefrontSolverNative`, `PyGraphPolyglotMaterializer` and the egg-based
  `PyProofTheoreticHealer` / `repair_uast_expression`.

## Research-to-Software Builder Bridge

`invar_agent/builder/research_bridge.py` turns Research Vault runs into build specifications for
the Builder Agent in both Direct and Plan Mode.

- **Resolution** (`resolve_relevant_research_runs`): explicit `@research:<id>` tags and
  `attached_research_ids` first, then keyword/symbol overlap with run titles, constants and
  equations, then implicit phrases ("from our research", "turn this into", "uploaded research")
  bound to the most recent run. Unrelated greenfield prompts bind no research.
- **Archetypes** (`classify_software_archetype`): `SCIENTIFIC_SDK_LIBRARY` (default),
  `CLI_SIMULATOR_SWEEP`, `INTERACTIVE_WEB_SIMULATOR`, `INVERSE_DESIGN_OPTIMIZER`,
  `MULTI_SOLVER_BENCHMARK`, `TELEMETRY_DIGITAL_TWIN`, `STANDALONE_NATIVE_CRATE`, each with a
  file layout, architecture guidelines and verification requirements.
- **Build packet** (`compile_research_build_packet`): SI constants, governing equations, the
  verified `simulation_step(buf, dt)` kernel, contracts and key metrics, rendered as a
  `<research_to_software_specification>` block injected into Builder prompts.
- **Research-Parity tests**: Direct-Mode builds write `tests/test_research_parity.py` and
  `.invar_research_provenance.json`, emit a `research_context_bound` activity event and execute
  the parity tests after the build. The test checks that every research constant is preserved
  (relative tolerance $10^{-12}$), that the built solver stays finite under the baseline initial
  conditions, satisfies every `VerificationContract` invariant, and reproduces each key metric
  $m$ within $|m_{\text{built}} - m_{\text{research}}| \le 10^{-6}\,|m_{\text{research}}| + 10^{-9}$.

### User research uploads

`POST /api/v1/research/vault/upload` (alias `/import`) accepts multipart files or JSON
`{"files": [{"filename", "content", "encoding"}], "title"}` with `.md .tex .txt .py .json .csv
.zip` sources. A `.invar-research.zip` bundle is restored as-is; anything else is stored under
`uploaded_sources/` and mined by the zero-token `QuantitativeInsightExtractor` (LaTeX and plain
equations, SI-normalised parameters such as `rho = 15.6 g/cm^3` to `15600 kg/m^3`, benchmark tables
from Markdown/JSON/CSV). An uploaded `.py` with `simulation_step` becomes `unified_module.py`;
otherwise a baseline module is synthesised. Uploads are tagged `origin="user_upload"`,
`status="UPLOADED_READY"` and are usable by both agents; a Research Agent run with the upload
attached is pre-seeded with its constants and evidence and flips it to `VERIFIED` on success.

### Zip export / import

| Endpoint | Result |
|---|---|
| `GET /api/v1/workspace/export-zip` | `invar_software_workspace.zip` with `INVAR_BUILD_MANIFEST.json`; excludes `.invar/`, `.git/`, `.venv/`, caches and build scratch |
| `POST /api/v1/workspace/import-zip` | extracts a codebase zip; absolute or `..` members are rejected with HTTP 400, `.git/` and `.invar/` are ignored |
| `GET /api/v1/research/vault/{run_id}/export-zip` | `<run_id>.invar-research.zip` |
| `GET /api/v1/research/vault/export-all-zip` | `invar_research_vault_bundle.zip` |

The web console exposes these as Export/Import buttons on the Codebase pane, Upload/Export on the
Research Vault pane, per-run export, origin badges, and a Research Viewer archetype selector with
**Build Software from this Research** and **Verify / Extend** actions.

## Verification

- `cargo test` — Rust core unit tests (GoI nilpotency, HIN reduction, reflex healer, HIS register,
  GoI wavefront Sherman-Morrison updates).
- `pytest` — kernel, overlay/adjoint, precision shield, reflex/router, end-to-end agent, web server/export,
  and research suites (schemas, transpiler/emitters, C-ABI bridge, algorithms, state machine, API).
