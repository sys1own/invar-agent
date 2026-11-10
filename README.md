# invar-agent

Formally Grounded Autonomous Software Engineering Kernel and Computational Research Platform.

invar-agent is an autonomous engineering and scientific discovery system. Every mutation proposed by an agent passes through a 9-stage formal verification state machine before touching disk or entering source control. The platform combines a high-performance native core written in Rust (PyO3) with an extensible Python kernel, providing dual operational modes: a polyglot software engineering Builder Agent and an accretive computational Research Agent.

---

## Architectural Overview

invar-agent enforces transactional correctness and mathematical consistency across three coordinated subsystems:

* Native Acceleration Core (`invar_agent/_native`): High-throughput Rust engine implementing Girard Geometry of Interaction (GoI) nilpotency solving, an arena-allocated Lafont-style Hierarchical Interaction Net (HIN) engine, zero-token deterministic rewrite heuristics, and content-addressed BLAKE3 state registers.
* Kernel and Verification Substrate (`invar_agent/`): Python runtime managing Pydantic v2 schemas, transactional Copy-on-Write (CoW) overlays, algebraic adjoint rollbacks, a three-tier cognitive router, and the Precision Shield.
* Interactive Web Platform (`invar_agent/server.py` + `static/index.html`): FastAPI and Server-Sent Events (SSE) single-page application featuring partitioned codebase and research explorers, center-stage research inspection, plan-approval action cards, and an integrated xterm.js workspace terminal drawer.

---

## The 9-Stage Verified State Machine

Every task progresses through an invariant-preserving lifecycle:

```
[ Ingest ]
  |
  v
[ GoI_Plan ]
  |
  v
[ Wavefront_Schedule ] ----- (Nilpotency Check: rho(U * M) < 1)
  |
  v
[ Overlay_Mutate ] ------- (Transactional Copy-on-Write Scratch)
  |
  v
[ PreWrite_Guard ] ------- (AST Policy Firewall & Import Checks)
  |
  v
[ SMT_Concolic_Verify ] -- (Z3 Constraint & Boundary Validation)
  |
  v
[ Sandbox_Execute ] ------ (Resource-Limited Isolated Subprocess: RLIMIT_CPU / RLIMIT_AS)
  |
  v
[ Reflex_Heal ] ---------- (Tier-0 In-Memory Structural Repair)
  |
  v
[ Adjoint_Commit_or_Rollback ]
```

1. Ingest: Validates user specifications, empirical data, or literature findings into strictly typed Pydantic schemas.
2. GoI_Plan: Formulates task dependencies, causal contracts, and execution assertions as a directed proof net.
3. Wavefront_Schedule: Solves the linear execution system `EX(M, U) = (I - U * M)^(-1) * U`. Enforces the spectral radius nilpotency invariant `rho(U * M) < 1` to guarantee acyclic execution and absence of deadlocks.
4. Overlay_Mutate: Mounts an isolated tmpfs Copy-on-Write directory, ensuring baseline project files remain untouched during compilation and execution.
5. PreWrite_Guard: Performs static AST analysis enforcing `v9_transpiler_friendly` constraints, rejecting unauthorized system imports, network socket allocations, and unbounded iteration.
6. SMT_Concolic_Verify: Z3 SMT solver validates concrete types, structural invariant contracts, and numerical boundary conditions.
7. Sandbox_Execute: Executes compiled artifacts (`rust_hin`, C-ABI C++, or Python) within short-lived isolated subprocesses constrained by CPU and memory rlimits (`RLIMIT_CPU`, `RLIMIT_AS`).
8. Reflex_Heal: Applies deterministic E-graph equality saturation and syntactic rewrites without consuming LLM tokens when recoverable errors occur.
9. Adjoint_Commit_or_Rollback: If all verification contracts pass, changes commit transactionally to the active workspace. If verification fails, an algebraic adjoint rollback cleanly unlinks scratch directories and restores the last verified state.

---

## Dual Agent Modalities

### 1. Research Agent

The Research Agent conducts computational scientific research where findings are validated through native simulation:

* Epistemic Feasibility Gate: Evaluates research requests prior to execution, classifying tasks into four operational regimes:
  * `OPTIMAL_DIRECT`: Quantitative problems with explicit governing equations and testable invariants.
  * `REFORMULATED_SURROGATE`: High-dimensional or macro-scale systems downscaled into rigorous 0D/1D asymptotic models.
  * `REFUSED_IMPOSSIBLE`: Problems violating axiomatic physical laws (such as net-positive energy with closed-loop Hamiltonian invariants). Execution halts immediately without generating ungrounded simulations.
  * `REFUSED_UNSUITABLE`: Subjective or non-computational questions lacking formal state variables, prevented to avoid synthetic math-washing.
* Multi-Channel Literature Federation: Retrieves domain equations, parameters, and benchmark values across arXiv, ar5iv (HTML5 equation extraction), OpenAlex, Semantic Scholar (TLDR extraction), Wikipedia wikitext (`<math>` and infobox extraction), Wikidata quantities, OEIS (integer sequences and recurrence relations), and PubChem (molecular properties).
* Zero-Token SI Normalization: Deterministically converts mixed physical units to standard SI base units (m, kg, s, K, Pa, W/m^2, J) prior to code generation, preventing parameter scaling errors.
* Multi-Variant Pareto Benchmarking: Compiles and evaluates competing numerical integrators and closures concurrently (e.g., standard explicit RK4 versus semi-implicit Euler on stiff kinetics), selecting optimal formulations along Pareto frontiers of execution latency versus invariant residual error.
* Cumulative Blueprint Evolution: Verified computational kernels are checkpointed as pinned module layers. Follow-up research rounds integrate higher-order terms into the same unified computation module (`unified_module.py`).
* Adjoint Delta Rollback: If an added physical term causes divergence or invariant falsification, an adjoint rollback unlinks the active delta, preserves the verified baseline checkpoint, and feeds the numerical counterexample to the planner.
* User-Uploaded Research Ingestion: Ingests external research documents (`.md`, `.tex`, `.txt`, `.py`, `.json`, `.csv`, `.zip`) into `.invar/research_vault/<run_id>/`, extracting governing equations and SI constants via the `QuantitativeInsightExtractor`.

### 2. Builder Agent

The Builder Agent generates, verifies, and packages multi-file software projects:

* Polyglot Compilation: Synthesizes and tests code across Rust (Cargo crates with PyO3 bindings), C++ (C-ABI zero-copy shared libraries compiled via Clang++ with `-O3 -std=c++20 -fPIC -shared`), and Python.
* Research-to-Software Bridge: Ingests completed or uploaded research runs from the Research Vault and classifies build intent into one of seven software archetypes:
  1. `SCIENTIFIC_SDK_LIBRARY`: Reusable Python package with optional native C++ or Rust accelerators, typed interfaces, and test suites.
  2. `CLI_SIMULATOR_SWEEP`: Standalone command-line application supporting parameter sweeps, invariant guards, and structured export.
  3. `INTERACTIVE_WEB_SIMULATOR`: FastAPI and HTML5/JS dashboard exposing simulation steppers with real-time controls.
  4. `INVERSE_DESIGN_OPTIMIZER`: Constrained parameter optimization harness wrapping the verified forward research kernel.
  5. `MULTI_SOLVER_BENCHMARK`: Benchmarking harness comparing accuracy, convergence, and execution latency across parameter regimes.
  6. `TELEMETRY_DIGITAL_TWIN`: Streaming data processor and real-time invariant anomaly detector grounded in research equations.
  7. `STANDALONE_NATIVE_CRATE`: Zero-Python C++20 or Rust standalone library and binary.
* Three-Tier Research Resolution: Resolves research context in both Direct Build and Plan modes via explicit `@research:<id>` tags, semantic keyword matching, or implicit reference to the most recently completed research run.
* Automated Research-Parity Verification: Compiles and executes `tests/test_research_parity.py` during builds, asserting that the synthesized software preserves all SI constants and reproduces verified research metrics within tolerance.
* Precision Shield: Enforces boundary verification via Fock-Space Graph Encoding (FoGE), concolic Z3 checks, and the Chiasmus firewall.

---

## Workspace Platform and Web Console

The web interface (`http://localhost:8000`) provides an integrated development and research environment:

* Partitioned File Explorers: Strict separation between the Software Codebase (`[Codebase]`) and the Research Vault (`[Research Vault]`). Research runs are isolated under `.invar/research_vault/<run_id>/`, keeping the codebase root clean.
* Center-Stage Research Viewer: Dedicated inspector interface providing full-text search across research runs, executive metric summaries, governing equation and constant cards, and verified `unified_module.py` kernels. Includes one-click context forwarding (`Build Software from this Research`) to the Builder Agent.
* One-Click Software and Vault Archive Management:
  * Export Software Codebase: Generates a sanitized `.zip` containing project source files, test suites, and build manifests while excluding scratch directories and virtual environments.
  * Import Software Codebase: Extracts uploaded `.zip` archives with integrated Zip-Slip path-traversal validation.
  * Export/Import Research Vault: Exports individual research runs or entire vault bundles as portable `.zip` archives, and ingests user research archives directly into the vault.
* Integrated Workspace Terminal: Full-featured interactive terminal drawer powered by xterm.js, connected via WebSocket (`/api/v1/terminal/ws`) to a non-blocking asynchronous PTY process anchored in the active workspace directory.

---

## Native Core Implementation Details

The native core (`invar_agent._native`) executes performance-critical verification algorithms:

* Girard GoI Wavefront Solver: Solves `EX(M, U) = (I - U * M)^(-1) * U`. When task graphs update incrementally, rank-k updates are evaluated via the Sherman-Morrison-Woodbury formula:
  `K' = K + K * U * P * (I - Q^T * K * U * P)^(-1) * Q^T * K`
  where `K = (I - U * M)^(-1)` and `Delta M = P * Q^T`. Includes condition number validation (kappa > 1e8 triggers a full recomputation) to protect against numerical instability.
* Hierarchical Interaction Net (HIN) Arena: Slab-allocated, zero-heap interaction-net arena supporting active-pair reduction with the Python GIL released and no dynamic heap allocations.
* MELL Linear Typing: Wires carry Multiplicative-Exponential Linear Logic types (`I`, `Tensor`, `Implication`, `Bang`), tracking resource consumption and preserving linearity.
* Reflex Healer: Employs equality saturation (E-graphs) via the `egg` framework to simplify arithmetic expressions and repair ASTs deterministically without invoking LLMs.

---

## Cognitive Router

Operations are routed across a three-tier cognitive hierarchy:

| Tier | Engine | Responsibilities |
|---|---|---|
| Tier-2 Pro | LLM (Reasoning) | Upfront planning, research hypothesis synthesis, final report formulation, and complex software scaffolding. |
| Tier-1 Flash | LLM (Fast) | Search query formulation, parameter extraction, and bounded AST hole completion. |
| Tier-0 Reflex | Rust Native & Z3 | 0-token operations: AST parsing, SMT checks, C-ABI/HIN compilation, execution, and E-graph repair. |

Supported LLM providers include Gemini, OpenAI, DeepSeek, and OpenRouter.

---

## Quickstart

### Prerequisites

* Python 3.10+
* Rust toolchain (Cargo)
* Clang++ with C++20 support
* `uv` (recommended) or `pip`

### Installation

```bash
# Clone the repository
git clone https://github.com/sys1own/invar-agent.git
cd invar-agent

# Set up virtual environment
uv venv .venv --python 3.10
source .venv/bin/activate

# Install dependencies and build native extension
uv pip install -e .
maturin develop --release
```

### Launching the Workspace

```bash
# Start the web console serving a workspace directory
invar-agent serve ./my-workspace

# Start with a specific provider preselected
invar-agent serve ./my-workspace --provider deepseek
```

Open `http://localhost:8000` in a browser. Configure API keys in the settings modal or export them in your shell environment:

```bash
export DEEPSEEK_API_KEY="your-api-key"
export INVAR_LLM_PROVIDER="deepseek"
```

---

## Verification Suite

Run the formal verification and integration test suite:

```bash
# Run Rust native unit tests
cargo test --manifest-path invar_agent/_native/Cargo.toml

# Run Python kernel and verification tests
pytest -v -m "not live"

# Run live provider end-to-end smoke test (requires API key)
pytest -v -m live tests/test_llm_live.py
```

---

## License

This project is licensed under the MIT License. See LICENSE for details.
