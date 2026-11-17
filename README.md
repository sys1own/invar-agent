# invar-agent

Formally Grounded Autonomous Software Engineering Kernel and Computational Research Platform.

`invar-agent` combines a Rust/PyO3 native verification and algebraic-rewriting core (`invar_agent._native`), a deterministic Python/C++20/Z3 verification substrate (`invar_agent/`), and an interactive FastAPI + SSE web console (`invar_agent/server.py`, `invar_agent/static/index.html`) to synthesize, formally verify, and execute numerical simulation kernels and software modules with zero unverified claims.

---

## 1. Architectural Overview

`invar-agent` is structured into three tightly coupled layers:

1. **Native Acceleration Core (`invar_agent/_native`)**:
   Rust crate (`invar_agent_native`) compiled via `maturin` and exposed to Python through PyO3:
   - **Girard GoI Wavefront Solver (`goi_wavefront.rs`, `goi_solver.rs`, `goi_matrix.rs`)**: Computes execution matrices `EX(M, U) = (I - U * M)^(-1) * U`, checks nilpotency (`rho(U * M) < 1`), partitions acyclic DAGs into parallelizable wavefronts, and performs Sherman-Morrison-Woodbury rank-k incremental updates.
   - **Holographic Interaction Net (HIN) Arena (`hin_arena.rs`, `hin_engine.rs`, `holographic_hin.rs`)**: Flat arena graph reducer supporting MELL linear typing (`tau_wire = (MELLType, SIDimensionVector, IntervalBound)`), automatic promotion/dereliction, and causal fault slicing.
   - **Reverse-Mode Adjoint HIN (`adjoint.rs`)**: Symbolic Jacobian differentiation `J_ij = df_i / dx_j` and spectrum/stiffness analysis.
   - **`egg` E-Graph Reflex Healer (`csel_egraph.rs`, `deterministic_healer.rs`)**: Sub-2 ms Tier-0 algebraic canonicalization, CSEL desugaring, and syntax/import reflex repair.
   - **Holographic Interface Signatures (`his.rs`)**: BLAKE3 structural digests that track public signature boundaries independently of private body edits.
   - **Declarative UAST (`uast.rs`)**: Typed representation for `ResearchComputeUAST` and Closed Symbolic Expression Language (CSEL).

2. **Kernel and Verification Substrate (`invar_agent/`)**:
   - **9-Stage Verification State Machine (`invar_agent/research/state_machine.py`)**: Orchestrates literature ingestion, GoI scheduling, copy-on-write overlay mutation, PreWrite guards, Z3 SMT/concolic verification, C++20 C-ABI sandbox execution, Tier-0/1/2 healing, and adjoint commit/rollback.
   - **Tier-0 Deterministic AST Sanitizer (`invar_agent/healing/deterministic_ast.py`)**: Rewrites domains, signatures, comprehensions, tuple unpacking, relational contracts, and manifold projections before compilation or execution.
   - **Polyglot Transpiler & C-ABI Bridge (`invar_agent/transpiler/`, `invar_agent/runtime/cabi_bridge.py`)**: Lowers restricted Python (`v9_transpiler_friendly`) to UAST and C++20 shared libraries (`-O3 -std=c++20`), with isolated subprocess execution and deterministic Python fallback.
   - **Deterministic LaTeX Canonicalizer (`invar_agent/latex_normalizer.py`)**: Repairs JSON control-character escapes, wraps bare equations, canonicalizes ASCII-math hybrids, and balances delimiters/braces.

3. **Interactive Web Platform (`invar_agent/server.py`, `invar_agent/static/index.html`)**:
   - FastAPI backend with 3-channel SSE streaming (`search_log`, `goi_stream`, `accelerator_log`), Research Vault persistence, zip export/import, Markdown-protected KaTeX rendering, and an integrated `xterm.js` workspace terminal.

---

## 2. The 9-Stage Verified State Machine

Every research stage and software mutation traverses the same deterministic 9-stage pipeline:

```text
  [1. Ingest]
       |
       v
  [2. GoI_Plan] ---> [3. Wavefront_Schedule] ---> [4. Overlay_Mutate]
                                                         |
                                                         v
  [8. Reflex_Heal] <--- [7. Sandbox_Execute] <--- [6. SMT_Concolic_Verify] <--- [5. PreWrite_Guard]
       |                                                 |
       +----------------------> [9. Adjoint_Commit_or_Rollback]
```

1. **Ingest**: Canonicalizes blueprint slots, auto-populates `variable_map`, and indexes workspace/literature evidence.
2. **GoI_Plan**: Evaluates epistemic feasibility, queries federated literature channels, and plans stage contracts.
3. **Wavefront_Schedule**: Verifies nilpotency (`rho(U * M) < 1`) over regression and new-stage contracts via the GoI solver, computing Sherman-Morrison-Woodbury rank-k delta updates.
4. **Overlay_Mutate**: Materializes the candidate `unified_module.py` inside an isolated copy-on-write scratch directory (`round_<k>`).
5. **PreWrite_Guard**: Enforces the SHACL/Chiasmus firewall (`v9_transpiler_friendly` subset), runs a 1-step Python smoke execution (`simulation_step(buf, 0.01)`), and lowers Python -> UAST -> C++20.
6. **SMT_Concolic_Verify**: Checks contract satisfiability and invariant expressions using Z3 and safe merged-globals concolic evaluation (`all(math.isfinite(state[i]) for i in range(N))`).
7. **Sandbox_Execute**: Compiles and executes the C++20 C-ABI kernel (or isolated Python fallback) across all stage and regression contracts (`dt`, `steps`, `initial_state` guarded by `auto-init`).
8. **Reflex_Heal**: On `FALSIFIED` / `DIVERGED` / `ERROR`, applies Tier-0 deterministic AST repairs (`< 2 ms`, 0 tokens), Tier-1 structured directives, and Tier-2 counterexample-guided refinement.
9. **Adjoint_Commit_or_Rollback**: Pins verified layers into `unified_module.py` and persists checkpoints to the Research Vault, or rolls back failed stage deltas without discarding earlier verified stages.

---

## 3. Declarative UAST, Totalized Semantics, and Tier-0 Stabilization

To prevent runtime domain errors (`ZeroDivisionError`, `ValueError`, `OverflowError`) and numerical blowups (`NaN`, `Inf`), `invar-agent` enforces totalized semantics and Tier-0 stabilization:

- **Declarative UAST & CSEL**: Compute kernels operate over flat scalar buffers (`buf.data[i]`) and pure helper functions (`v9_transpiler_friendly`), excluding classes, dynamic imports, comprehensions, tuple unpacking, `global`/`nonlocal`, and exceptions.
- **Totalized Algebraic Semantics**:
  - `_invar_sdiv(a, b)` / `div_eps(a, b) = a / (sgn(b) * max(|b|, eps))` (with `eps = 1e-12` or `1e-30`) replaces unsafe divisions across all helper and step functions.
  - Domain-guarded `safe_sqrt`, `safe_log`, `safe_exp`, `safe_pow`, and `smooth_clamp` prevent negative-argument `ValueError` and exponential overflow.
- **Tier-0 Dynamical State Divergence Shield (`_invar_stabilize_state` / `_invar_stabilize_state_6`)**:
  - Snapshots pre-step primary state slots (`0..3` or `0..5` for 6-mode triads) and applies a Rational Pade step limiter `p + _invar_sdiv(d, 1.0 + |d|)` whenever `|d| > 0.5`, plus full-buffer non-finite clamping (`[-1e12, 1e12]`) and local-variable re-sync for downstream diagnostics.
- **Physical Manifold Projectors**:
  - **Density-Matrix Bloch/Simplex Projector (`_invar_project_density`)**: Enforces unit trace (`p0 + p1 = 1.0`, `p0, p1 >= 0.0`) and coherence positivity `c0^2 + c1^2 <= p0 * p1`.
  - **Stoichiometric Mass-Simplex Projector (`_invar_project_mass_simplex`)**: Clamps species concentrations `buf.data[0..3] >= 0.0` and rescales by `_invar_sdiv(M_prev, M_cur)` to preserve exact total mass `M_0`.
  - **Unit-Sphere `S^2` Projector (`_invar_project_sphere_s2`)**: Normalizes 3-component spin/magnetization vectors `(m_x, m_y, m_z)` onto `||m||_2 = 1.0`.
- **Universal Contract-Slot Conditioning & Signature Normalization**:
  - Normalizes 1-arg `simulation_step(buf)` to `(buf, dt=0.01)` (rewriting `buf.dt -> dt`) and defaults trailing args `> 2` to `0.0`.
  - Evaluates constant-expression bounds (`3.0 - 1e-12`), unifies lower bounds across stages, preserves running-minimum sentinels, and conditions single-slot (`<=`, `<`, `>=`, `>`) and relational (`state[i] <= coeff * state[j]`) invariants across all blueprint contracts.
- **Automatic Stiffness Detection & Counterexample-Guided Re-Prompting**:
  - Detects stiff/dissipative regimes (`kappa >> 1`, `alpha * dt >> 1`) and routes exact falsified counterexamples (`observed {...}`) into targeted literature re-search and Tier-1/Tier-2 healing.

---

## 4. Dual Agent Modalities and Context Architecture

### Research Agent
- **Epistemic Feasibility Gate (`invar_agent/research/feasibility.py`)**: Classifies requests into `OPTIMAL_DIRECT`, `REFORMULATED_SURROGATE`, `REFUSED_IMPOSSIBLE`, or `REFUSED_UNSUITABLE`. Automatically converts over-constrained computational physics/chemistry/engineering benchmarks into `REFORMULATED_SURROGATE` with compatible surrogate coupling parameters.
- **Multi-Channel Literature Federation**: Queries arXiv, bioRxiv/medRxiv, Crossref, OpenAlex, Semantic Scholar, and Wikipedia with automatic query relaxation.
- **Zero-Token SI Normalization & Reference Registry (`invar_agent/research/extractor.py`)**: Extracts empirical constants and equations with word-boundary domain gating (preventing cross-domain pollution).
- **Multi-Variant Pareto Benchmarking**: Evaluates competing numerical schemes (e.g., Cayley/Patankar vs. RK4/Euler) under identical contract grids.
- **Cumulative Blueprint Evolution & Non-Aborting Rollback**: Stages `1..N` accrete new capabilities onto pinned verified layers; exhausted retries on stage `k` roll back only stage `k`'s delta without aborting subsequent stages.
- **Deterministic LaTeX Canonicalization (`invar_agent/latex_normalizer.py`)**: Repairs JSON control-character escapes (`\x0crac -> \frac`, `\x08eta -> \beta`, `\x09heta -> \theta`, `\x0dho -> \rho`), auto-wraps bare math, canonicalizes ASCII-math hybrids, and balances `{}` and `\left`/`\right`.
- **Stateless Direct Research vs. Context-Aware Planner Mode**:
  - **Direct Research Mode (`/api/v1/research/run`)**: Strictly task-scoped and stateless; receives only the active research prompt (and explicitly attached `@research:<id>` items), never prior chat history or unrelated vault runs.
  - **Planner Mode (`/api/v1/research/plan`, `/api/v1/research/plan-prompt`)**: Context-aware; receives chat context plus a structured Research Vault digest (`@research:<id>`, hypothesis, verified status, metrics, invariants) for follow-up planning.

### Builder Agent
- **Polyglot Compilation**: Synthesizes and verifies Python, C++20 C-ABI, and Rust targets.
- **Research-to-Software Bridge (`invar_agent/builder/research_bridge.py`)**: Binds verified Research Vault artifacts across all 7 scientific/engineering archetypes using Three-Tier Research Resolution (explicit `@research:<id>`, keyword match, or implicit recent verified run).
- **Automated Research-Parity Verification (`tests/test_research_parity.py`)**: Generates and executes `test_research_parity.py` asserting that software constants and invariants match the verified research bundle within tolerance.
- **Precision Shield**: Enforces `math.isfinite`, bounded residuals, and zero unverified numerical drift during software builds.

---

## 5. Workspace Platform and Web Console

- **Partitioned File Explorers**: Separate `[Codebase]` workspace tree and `[Research Vault]` archive browser.
- **Center-Stage Research Viewer**: Displays verified numerical metrics tables, extracted constants, governing equations, unified kernel code, and full reports with Markdown-protected KaTeX rendering (`katex.renderToString` after masking code spans and math delimiters).
- **One-Click Software & Vault Archive Management**: Zip export/import for both software workspaces and `.invar/research_vault/` bundles.
- **Zero-Leak Multi-Run Lifecycle**: Every run instantiates an isolated `ResearchStateMachine`, clears SSE replay history (`clear_history()`) on start so prior commit banners never bleed across runs, unloads compiled C-ABI shared library handles (`FreeLibrary`/`dlclose`), removes temporary `.cpp`/`.so`/`.dll` build artifacts in `finally:`, and runs `gc.collect()`.
- **Integrated Workspace Terminal**: Embedded `xterm.js` terminal with fit addon for direct shell inspection.

---

## 6. Native Core Implementation Details

- **Girard GoI Wavefront Solver**: Constructs dependency adjacency matrix `M` and feedback/control matrix `U`, verifies `rho(U * M) < 1`, and executes rank-k Sherman-Morrison-Woodbury updates `(A + P * Q^T)^(-1) = A^(-1) - A^(-1) * P * (I + Q^T * A^(-1) * P)^(-1) * Q^T * A^(-1)` when adding stage contracts.
- **Holographic Interaction Net (HIN) Arena**: Fixed-index arena nodes with active-pair interaction rules (annihilation, commutation), MELL exponential promotion/dereliction, and 3-hop causal fault slicing.
- **Reverse-Mode Adjoint HIN**: Computes exact symbolic partial derivatives `J_ij = df_i / dx_j` and eigenvalue/stiffness bounds for ODE/PDE systems.
- **MELL Linear Typing & `egg` E-Graph Reflex Healer**: Enforces linear resource usage and dimensional consistency (`SIDimensionVector`) alongside `egg` equality saturation for algebraic simplification (`log1p`, stiff IMEX forms) and sub-2 ms syntax/import healing.

---

## 7. Cognitive Router

| Tier | Role | Default Model Routing | Typical Latency / Cost |
|---|---|---|---|
| **Tier-2 Pro** | Blueprint synthesis, Stage `2..N` accretion, final healing | Reasoning model (e.g., `deepseek-v4-pro`, `gemini-2.5-pro`) | Complex synthesis |
| **Tier-1 Flash** | Planning, feasibility gate, literature queries, fast healing directives | Fast model (e.g., `deepseek-flash`, `gemini-2.5-flash`) | Low latency |
| **Tier-0 Reflex** | `egg` e-graph, AST sanitizer, manifold projectors, contract conditioning | Deterministic Rust + Python AST | `< 2 ms` \| `0 Tokens` |

Supported LLM providers (`invar_agent/llm/`): **DeepSeek**, **Gemini**, **OpenAI**, and **OpenRouter**.

---

## 8. Quickstart and Verification Suites

### Prerequisites
- Python 3.12+
- Rust toolchain (`cargo`, `rustc`) and `maturin`
- C++20 compiler (`clang++` or `g++`) on `PATH` for native C-ABI kernel execution

### Installation
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate  |  Linux/macOS: source .venv/bin/activate
pip install maturin
maturin develop --release
pip install -e .
```

### Environment Variables
Set one or more provider keys (or configure them in the Web Console settings modal):
- `DEEPSEEK_API_KEY`
- `GEMINI_API_KEY`
- `OPENAI_API_KEY`
- `OPENROUTER_API_KEY`

### Launch the Web Console
```bash
uvicorn invar_agent.server:app --host 127.0.0.1 --port 8000
```

### Run Verification Suites
```bash
# Rust native core tests
cargo test --lib

# Python unit and integration tests (offline/deterministic suite)
pytest -q
```

### Repository Directory Tree
```text
invar-agent/
├── Cargo.toml
├── pyproject.toml
├── README.md
├── invar_agent/
│   ├── _native/                  # Rust PyO3 crate (GoI, HIN, Adjoint, CSEL egg, HIS, Healer)
│   ├── builder/                  # Research-to-Software Bridge & parity test generator
│   ├── guards/                   # PreWrite_Guard, SHACL/Chiasmus, import pruner
│   ├── healing/                  # Tier-0 deterministic AST sanitizer & reflex healer
│   ├── llm/                      # Provider clients (DeepSeek, Gemini, OpenAI, OpenRouter)
│   ├── research/                 # State machine, feasibility, planner, extractor, report, vault
│   ├── runtime/                  # C-ABI bridge & isolated sandbox runner
│   ├── schemas/                  # Pydantic v2 models for blueprints, contracts, and telemetry
│   ├── static/                   # Web UI (index.html, marked, KaTeX, xterm.js)
│   ├── transpiler/               # Python -> UAST -> C++20 / HIN emitters
│   ├── verification/             # Z3 SMT & concolic verification
│   ├── latex_normalizer.py       # Deterministic JSON escape repair & LaTeX canonicalizer
│   └── server.py                 # FastAPI + SSE server
└── tests/                        # Unit, regression, parity, and live integration tests
```
