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
PreWrite_Guard (SHACL/Chiasmus firewall rejecting `os`/`sys`/`socket`/unbounded-loop code) →
SMT_Concolic_Verify → Sandbox_Execute (`rust_hin`, `cpp` C-ABI, or `multi_variant` Pareto
benchmarks) → Reflex_Heal (egg equality-saturation, Tier-0) → Adjoint_Commit_or_Rollback
(commits telemetry + a BLAKE3 bundle hash on VERIFIED; rolls back and feeds the counterexample
back into the next round on FALSIFIED/DIVERGED/ERROR).

- **Cognitive routing**: Tier-2 REASONING synthesises the research blueprint (search queries,
  invariants, simulation code template); Tier-1 FAST handles search-query expansion; Tier-0 Reflex
  handles SMT/Z3/concolic checks, the security firewall, compilation and egg-based repair.
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

## Verification

- `cargo test` — Rust core unit tests (GoI nilpotency, HIN reduction, reflex healer, HIS register,
  GoI wavefront Sherman-Morrison updates).
- `pytest` — kernel, overlay/adjoint, precision shield, reflex/router, end-to-end agent, web server/export,
  and research suites (schemas, transpiler/emitters, C-ABI bridge, algorithms, state machine, API).
