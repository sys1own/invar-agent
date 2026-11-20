# invar-agent

Formally Grounded Autonomous Software Engineering Kernel and Universal Pan-Scientific Hypothesis Verification Engine.

`invar-agent` combines a Rust/PyO3 native verification and algebraic-rewriting core (`invar_agent._native`), a deterministic Python/C++20/Z3 verification substrate (`invar_agent/`), and an interactive FastAPI + SSE web console (`invar_agent/server.py`, `invar_agent/static/index.html`) to synthesize, formally verify, and execute numerical simulation kernels, multi-domain scientific hypotheses across all 8 macro-scientific domains, and software modules with zero unverified claims.

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
- **Epistemic Feasibility Gate (`invar_agent/research/feasibility.py`)**: Classifies requests into `OPTIMAL_DIRECT`, `REFORMULATED_SURROGATE`, `REFUSED_IMPOSSIBLE`, or `REFUSED_UNSUITABLE`. Automatically converts over-constrained computational physics/chemistry/engineering benchmarks into `REFORMULATED_SURROGATE` with compatible surrogate coupling parameters, and rescues frontier open-boundary tasks covered by the SHBT Frontier Solvability Federation to `OPTIMAL_DIRECT`.
- **Literature and Frontier Solvability Federation (`invar_agent/research/search.py`, `invar_agent/research/shbt_federation.py`)**:
  - **Standard Multi-Channel Literature Federation**: Default retrieval across arXiv, ar5iv, bioRxiv/medRxiv, Crossref, OpenAlex, Semantic Scholar, Wikipedia, Wikidata, OEIS, PubChem, and CODATA/CRC `ReferenceRegistry` (with automatic query relaxation) for all conventional physics, chemistry, mathematics, and engineering tasks.
  - **Bottleneck-Gated Frontier Solvability Federation (10-Repository SHBT Ecosystem)**: For research tasks involving fundamental physics or advanced technological bottlenecks that are unsolvable within conventional closed-system models (such as sub-Landauer cryogenic reset dissipation, solid-state Floquet-screened Coulomb barrier penetration, macroscopic Stinespring state translocation, reactionless geodesic traction, holographic metric stabilization, or zero-parameter gauge coupling closure), the engine selectively queries the 10-repository SHBT ecosystem (`sys1own/shbt-precision`, `sys1own/shbt-qc`, `sys1own/shbt-power`, `sys1own/shbt-cf`, `sys1own/shbt-ghost`, `sys1own/shbt-recon`, `sys1own/shbt-sglt`, `sys1own/shbt-exotic`, `sys1own/shbt-warp`, `sys1own/shbt-swarm`) to supply verified open-boundary algebraic closures and SI constants with relevance-filtered constant/equation injection and zero intrusion on standard research tasks.
- **Zero-Token SI Normalization & Reference Registry (`invar_agent/research/extractor.py`)**: Extracts empirical constants and equations with word-boundary domain gating (preventing cross-domain pollution).
- **Multi-Variant Pareto Benchmarking**: Evaluates competing numerical schemes (e.g., Cayley/Patankar vs. RK4/Euler) under identical contract grids.
- **Cumulative Blueprint Evolution & Non-Aborting Rollback**: Stages `1..N` accrete new capabilities onto pinned verified layers; exhausted retries on stage `k` roll back only stage `k`'s delta without aborting subsequent stages.
- **Deterministic LaTeX Canonicalization (`invar_agent/latex_normalizer.py`)**: Repairs JSON control-character escapes (`\x0crac -> \frac`, `\x08eta -> \beta`, `\x09heta -> \theta`, `\x0dho -> \rho`), auto-wraps bare math, canonicalizes ASCII-math hybrids, and balances `{}` and `\left`/`\right`.
- **Stateless Direct Research vs. Context-Aware Planner Mode**:
  - **Direct Research Mode (`/api/v1/research/run`)**: Strictly task-scoped and stateless; receives only the active research prompt (and explicitly attached `@research:<id>` items), never prior chat history or unrelated vault runs.
  - **Planner Mode (`/api/v1/research/plan`, `/api/v1/research/plan-prompt`)**: Context-aware; receives chat context plus a structured Research Vault digest (`@research:<id>`, hypothesis, verified status, metrics, invariants) for follow-up planning.

### Builder Agent
- **Builder Studio**: Run `invar-agent serve --workspace-dir /path/to/workspace` and open the console's **Builder Studio**. Install `pip install -e ".[ingest,scientific]"` for document ingestion and scientific profiles.
- **Verified instruments**: Vault references (`@research:<id>`), uploaded LaTeX/PDF/CSV, or hybrid inputs become a typed, SI-normalized `CanonicalResearchIR`, checked with HIN/Z3 before deterministic compilation. Composable profiles are `hybrid_compute_library`, `interactive_workbench`, `executable_paper_audit`, and `calibration_inverse_design`.
- **Builder API**: Under `/api/v1/builder`, use `POST /ingest`, `PATCH /ir/{ir_id}/override`, `POST /plan`, `POST /execute`, `GET /status/{build_id}`, `GET /stream`, `POST /rebuild/{build_id}`, and `GET /export/{build_id}.zip`. Workbenches use `POST`/`DELETE /instrument/{build_id}/launch`. SSE events use `event_version: "2.0"`, `mode: "builder"`, and the `goi_stream`, `accelerator_log`, and `log` channels.
- **Scoped extensions and provenance**: Frozen interfaces constrain LLM leaf slots; bounded healing and verified deterministic fallbacks never rewrite scientific kernels. HIS-gated incremental rebuilds reuse unchanged artifacts, and BLAKE3 receipts detect tampering. Redacted leaf responses are retained privately under the external workspace's `.invar/builds/<build_id>/diagnostics/leaf_responses/` for diagnosis; they are not exported with the instrument.
- **Polyglot Compilation**: Synthesizes and verifies Python, C++20 C-ABI, and Rust targets.
- **Research-to-Software Bridge (`invar_agent/builder/research_bridge.py`)**: Binds verified Research Vault artifacts across all 7 scientific/engineering archetypes using Three-Tier Research Resolution (explicit `@research:<id>`, keyword match, or implicit recent verified run).
- **Automated Research-Parity Verification (`tests/test_research_parity.py`)**: Generates and executes `test_research_parity.py` asserting that software constants and invariants match the verified research bundle within tolerance.
- **Precision Shield**: Enforces `math.isfinite`, bounded residuals, and zero unverified numerical drift during software builds.

---

## Universal Pan-Scientific Hypothesis Verification Engine

`invar-agent` extends its deterministic verification substrate across all **8 Macro-Scientific Domains**, unifying continuous ODE/PDE dynamical steppers, deterministic Monte Carlo ensembles, fixed-point game/equilibrium solvers, and parameter/bifurcation sweeps under a single zero-token unit normalizer, semantic-tagged Holographic Interaction Net (HIN), and C++20 C-ABI verification pipeline.

### 8 Macro-Scientific Domains & Federated Literature Routing

`classify_macro_scientific_domains` and `classify_primary_macro_domain` (`invar_agent/research/extractor.py`) classify every hypothesis across 8 macro-domains and route literature queries to specialized zero-key scientific connectors (`invar_agent/research/search.py`, `invar_agent/research/literature.py`):

| Macro-Domain ID | Representative Sub-Disciplines | Primary Literature & Data Channels |
|---|---|---|
| `PHYSICAL_SCIENCES_AND_ENG` | Quantum mechanics, relativity, plasma/fluid dynamics, condensed matter, astrodynamics, thermodynamics, electromagnetism | arXiv, ar5iv, Crossref, OpenAlex, Wikipedia, CODATA `ReferenceRegistry` |
| `CHEMICAL_AND_MATERIALS` | Reaction kinetics, catalysis, electrochemistry, battery degradation, polymers, thermochemistry, materials thermophysics | PubChem, Crossref, OpenAlex, Semantic Scholar, Wikipedia, `ReferenceRegistry` |
| `LIFE_SCIENCES_AND_BIOMEDICINE` | Systems biology, enzyme kinetics, genomics, population genetics, pharmacology (PK/PD), oncology, neuroscience, biophysics | Europe PMC REST API, NCBI PubMed E-Utilities, bioRxiv/medRxiv, PubChem, OpenAlex |
| `EPIDEMIOLOGY_AND_ECOLOGY` | Compartmental infectious disease (`SIR`, `SEIR`, `SEIR-V`, `SEIRHD`), vector-borne transmission, predator-prey, food webs, biodiversity | Europe PMC REST API, NCBI PubMed E-Utilities, bioRxiv/medRxiv, Crossref, OpenAlex |
| `EARTH_CLIMATE_AND_ENVIRONMENT` | Radiative-convective energy balance, global carbon cycle reservoirs, ocean carbonate chemistry (Revelle factor), hydrology, geophysics | arXiv (`physics.ao-ph`), Crossref, OpenAlex, Wikipedia, `ReferenceRegistry` |
| `STATISTICS_AND_CAUSAL_INFERENCE` | Frequentist/Bayesian estimation, Structural Causal Models (SCM), `do`-calculus, Instrumental Variables (`2SLS` / Wald), Cramér-Rao bounds | arXiv (`stat.ME`, `stat.ML`, `econ.EM`), Crossref, OpenAlex, Semantic Scholar |
| `ECONOMICS_AND_GAME_THEORY` | Evolutionary game theory (replicator-mutator dynamics), Nash equilibrium, mechanism design, Bellman dynamic programming, Walrasian equilibria | arXiv (`econ.TH`, `cs.GT`, `q-fin`), Crossref, OpenAlex, Semantic Scholar |
| `MATHEMATICS_CS_AND_INFO_THEORY` | Information theory (Shannon entropy, KL divergence, channel capacity), Markov chains, spectral graph theory, combinatorics | arXiv (`cs.IT`, `math.*`), OEIS, Crossref, OpenAlex, Wikipedia |

When `LIFE_SCIENCES_AND_BIOMEDICINE` or `EPIDEMIOLOGY_AND_ECOLOGY` is active, `LiteratureSearchOrchestrator` and `FederatedLiteratureEngine` automatically activate resilient zero-key async connectors for **Europe PMC** (`ebi.ac.uk/europepmc/webservices/rest/search`) and **NCBI PubMed E-Utilities** (`esearch.fcgi` + `esummary.fcgi`) with strict 8-second HTTP timeouts and graceful fallback.

### Pan-Scientific Unit Normalizer & Gated ReferenceRegistry

The zero-token unit normalizer (`normalize_to_si` / `normalize_unit` in `invar_agent/research/extractor.py`) converts clinical, pharmacological, biological, epidemiological, climate, and information-theoretic units into canonical base units while preserving case-sensitive disambiguation (such as `cm = 0.01 m` versus `cM = 0.01 Morgan` and `M = 1000 mol/m^3`):

- **Clinical, Pharmacological & Biochemical**: `M` (`1e3 mol/m^3`), `mM` (`1.0 mol/m^3`), `uM` (`1e-3 mol/m^3`), `nM` (`1e-6 mol/m^3`), `pM` (`1e-9 mol/m^3`), `mg/dL` (`1e-2 kg/m^3`), `ug/mL` (`1e-3 kg/m^3`), `ng/mL` (`1e-6 kg/m^3`), `mg/kg` (`1e-6 kg/kg`), `IU/L` (`1e3 IU/m^3`), `Da` (`1.66053906660e-27 kg`), `kDa` (`1.66053906660e-24 kg`), `kcal/mol` (`4184.0 J/mol`).
- **Biological, Epidemiological & Genomic**: `bp` (`1.0 bp`), `kb` (`1e3 bp`), `Mb` (`1e6 bp`), `cM` (`0.01 Morgan`), `CFU/mL` (`1e6 m^-3`), `PFU/mL` (`1e6 m^-3`), `1/day` (`1 / 86400 1/s`), `1/year` (`1 / 31557600 1/s`).
- **Climate, Earth-System & Astronomical**: `ppm` (`1e-6 fraction`), `ppb` (`1e-9 fraction`), `GtC` (`1e12 kg`), `PgC` (`1e12 kg`), `GtC/yr` (`1e12 / 31557600 kg/s`), `Sv` (`1e6 m^3/s`), `bar` (`1e5 Pa`), `atm` (`101325.0 Pa`), `au` (`1.495978707e11 m`), `pc` (`3.085677581491367e16 m`).
- **Statistical, Economic & Information-Theoretic**: `bit` (`1.0 bit`), `nat` (`1 / ln(2) bit ≈ 1.4426950408889634 bit`), `dB` (`1.0 dB`), `%` (`0.01 fraction`), `bps` (`1e-4 fraction`).

The curated `ReferenceRegistry` includes 7 word-boundary-gated pan-scientific domain packs (`epidemiology_population`, `pharmacology_biochemistry`, `genetics_evolution`, `climate_earth_systems`, `statistics_causal_inference`, `economics_game_theory`, `math_info_theory`) supplying canonical equations, benchmark values, and invariant templates only when explicit domain anchors appear in the research prompt.

### Dimensionless Semantic Tags in Holographic Interaction Nets (HIN)

In non-physical sciences, many state variables are dimensionless (`[0, 0, 0, 0, 0, 0, 0]`) yet semantically incompatible under addition or direct comparison (for example, adding a probability `p ∈ [0, 1]` to an economic utility `U` or Shannon entropy `H` in nats). `SIDimensionVector` (`invar_agent/_native/src/holographic_hin.rs`, `invar_agent/uast/schemas.py`) carries an orthogonal `semantic_tag: u8` field (default `0` for 100% backward compatibility):

| `semantic_tag` | Semantic Domain | Domain & Algebraic Constraints |
|---|---|---|
| `0` (`SEMANTIC_PHYSICAL`) | Standard physical SI quantity or pure scalar | Checked by the 7-component SI exponent vector `[m, kg, s, A, K, mol, cd]` |
| `1` (`SEMANTIC_PROBABILITY`) | Probability, allele/compartment fraction, receptor occupancy, prevalence | Bounded to `[0, 1]`; valid input to `logit(p)`, `norm_ppf(p)`, `kl_term(p, q)`, `shannon_term(p)` |
| `2` (`SEMANTIC_INFORMATION`) | Shannon entropy, KL divergence, mutual information, log-likelihood (`nat` / `bit`) | Non-negative (`≥ 0` for entropy and KL divergence); incompatible with raw probabilities or utilities |
| `3` (`SEMANTIC_POPULATION`) | Population count, individual/cell/virion count, base-pair length | Non-negative (`≥ 0`); ratios of identical-tag counts normalize to `SEMANTIC_PROBABILITY` (`tag = 1`) |
| `4` (`SEMANTIC_STAT_METRIC`) | Correlation `r ∈ [-1, 1]`, `R^2 ∈ [0, 1]`, z-score, t-statistic, treatment effect `β` | Checked by statistical domain conditioners (Cauchy-Schwarz, Cramér-Rao, correlation bounds) |
| `5` (`SEMANTIC_UTILITY`) | Economic payoff, game utility, welfare, log-return, fitness | Compatible with Bellman discounting and Nikaido-Isoda Nash exploitability regret |

In the Rust HIN verifier (`unify_additive_dimensions`), adding two dimensionless wires with distinct non-zero semantic tags (`tag_a != 0 && tag_b != 0 && tag_a != tag_b`) immediately emits a `Semantic Tag Mismatch` diagnostic, while the `egg` equality-saturation e-graph (`csel_egraph.rs`) recognizes pan-scientific operators (`norm_cdf`, `norm_ppf`, `logistic_sigmoid`, `logit`, `hill_eq`, `michaelis_menten`, `kl_term`, `shannon_term`, `gammainc_reg`, `betainc_reg`) and algebraic identities (`kl_term(p, p) -> 0`, `logit(logistic_sigmoid(x)) -> x`, `logistic_sigmoid(logit(p)) -> p`).

### Deterministic Stateless SplitMix64 PRNG, Statistical Intrinsics & 4 Execution Archetypes

To make Monte Carlo simulations, stochastic biostatistics, causal inference, and quantitative biology 100% reproducible across Python smoke checks, Z3 concolic traces, and `-O3 -std=c++20` C-ABI execution without mutable global PRNG state:

1. **Collision-Free Stateless SplitMix64 PRNG (`invar_splitmix64_stateless`)**:
   Mixes `(seed, step, channel)` using distinct 64-bit Weyl constants (`0x9e3779b97f4a7c15` for `step` and `0xbf58476d1ce4e5b9` for `channel`) so diagonal calls `step == channel` never collide:
   - `invar_rand_u01(seed, step, channel)`: Uniform variate strictly clamped to `[1e-15, 1.0 - 1e-15]`.
   - `invar_rand_normal(seed, step, channel)`: Exact Box-Muller standard normal variate `N(0, 1)` guaranteed free of `log(0)` or `NaN`/`Inf`.
   - `invar_rand_poisson(seed, step, lam)` and `invar_rand_binomial(seed, step, n, p)`: Deterministic Knuth/normal-approximated discrete variates.
2. **Automatic Tier-0 `random.*` AST Lowering (`_lower_random_calls_tree`)**:
   Strips `import random` before `PreWrite_Guard`, maintains an automatic monotonic step counter in `buf.data[62]`, and lowers `random.random()`, `random.uniform(a, b)`, `random.gauss(mu, sigma)`, `random.normalvariate(mu, sigma)`, `random.expovariate(lam)`, and `random.randint(a, b)` into stateless PRNG calls with unique per-callsite channel IDs.
3. **Closed-Form Statistical, Pharmacological & Information-Theoretic Intrinsics**:
   Exposed identically in Python globals, on the `math` module, and in the C++20 transpiler prelude:
   - Online Welford accumulators: `welford_mean_update(mean, x, k)`, `welford_m2_update(m2, mean_old, mean_new, x)`, `welford_cov_update(c_xy, mean_x_old, x, mean_y_new, y)`
   - Gaussian & link functions: `norm_cdf(x)`, `norm_ppf(p)` (Acklam rational approximation on `clamp(p, 1e-15, 1 - 1e-15)`), `logistic_sigmoid(x)`, `logit(p)`
   - Dose-response & enzyme kinetics: `hill_eq(c, ec50, n) = c^n / (ec50^n + c^n)`, `michaelis_menten(s, vmax, km) = (vmax * s) / (km + s)`
   - Information theory & special functions: `kl_term(p, q) = p * ln(p / q) - p + q`, `shannon_term(p) = -p * ln(p)`, `gammainc_reg(a, x)`, `betainc_reg(a, b, x)`
4. **4 Execution Archetypes over `SimulationBuffer`**:
   - `ARCHETYPE_DYNAMICAL_STEPPER`: Continuous/discrete time-stepping `z[k+1] = Phi_dt(z[k])` for ODEs, PDEs, compartmental epidemiology, and carbon-cycle box models.
   - `ARCHETYPE_MONTE_CARLO_ENSEMBLE`: Each step `k ∈ {0..N-1}` draws deterministic SplitMix64 variates and updates online Welford moments, 2SLS/OLS estimators, and confidence intervals.
   - `ARCHETYPE_EQUILIBRIUM_FIXED_POINT`: Iterates contractive/damped best-response, replicator-mutator, or Bellman operators tracking Lyapunov decay (`D_KL(x* || x(t))`) and Nash regret `max_i(u_i - u_bar) -> 0`.
   - `ARCHETYPE_PARAMETER_BIFURCATION_SWEEP`: Sweeps a control parameter (dose, reproduction number, coupling) across `steps` grid points to verify monotonicity, EC50 thresholds, and bifurcation invariants.

### Generalized Arbitrary-K Simplex Projection & Statistical Domain Conditioners

The Tier-0 deterministic AST normalizer (`invar_agent/healing/deterministic_ast.py`) inspects all blueprint contract invariants and automatically synthesizes domain-specific manifold projectors and statistical conditioners:

- **Generalized Arbitrary-`K` Simplex Projector (`_invar_project_k_simplex`)**:
  Automatically detects any `K`-slot sum conservation law (`K ≥ 2`), including `K = 2` biallelic frequencies (`p + q = 1`), `K = 3` `SIR` or 3-strategy Rock-Paper-Scissors replicator simplices (`x_1 + x_2 + x_3 = 1`), `K = 4` `SEIR` or 4-reservoir global carbon cycle inventories (`M_atm + M_surf + M_deep + M_bio = 41000 GtC`), `K = 5` `SEIR-V` vaccine models, and `K = 6` `SEIRHD` clinical cohorts. At the end of every step function, it clamps non-negative slots `buf.data[i_m] = max(0.0, buf.data[i_m])`, computes the current sum `_s_cur`, rescales each participating slot by `_invar_sdiv(_target_sum, _s_cur)` (or distributes `_target_sum / K` uniformly on underflow), and re-synchronizes local variable aliases.
- **Statistical & Causal Inference Domain Conditioners**:
  Enforces Cauchy-Schwarz covariance bounds (`|Cov(X, Y)| ≤ √(Var(X) * Var(Y))`), Cramér-Rao / Fisher information variance lower bounds (`Var(θ_hat) ≥ 1 / I_F(θ)`), Pearson correlation bounds (`-1.0 ≤ r ≤ 1.0`), probability/AUC/`R^2`/p-value unit-interval bounds (`0.0 ≤ p ≤ 1.0`), and Instrumental Variable (`2SLS` / Wald) versus confounded OLS estimation error ordering.

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
