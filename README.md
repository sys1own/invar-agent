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

## Verification

- `cargo test` — Rust core unit tests (GoI nilpotency, HIN reduction, reflex healer, HIS register).
- `pytest` — kernel, overlay/adjoint, precision shield, reflex/router, end-to-end agent, and web server/export suites.
