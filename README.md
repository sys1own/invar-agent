# invar-agent

An **8-Gate Formal Epistemic Meaningfulness, Uniqueness & Justified Falsification Engine**, paired with a software Builder and an interactive research workspace.

The Rust/PyO3 core, Python/Z3 substrate, restricted-Python-to-C++20 compiler and isolated execution bridge connect declared scientific hypotheses to executable evidence. Results carry BLAKE3 commitments and auditable certificates. **Verification is conditional on the sealed model, domain, numerical reliability and trust boundary—not a guarantee of physical truth or universal implementation correctness.** Missing evidence and unsupported obligations fail closed.

## Scientific contract and eight gates

`invar-agent` implements the **Invar-Agent vNext Correct-by-Construction Scientific Compiler Architecture**, enforcing a strict **Two-Boundary Architecture** separating natural-language interpretation from deterministic compilation and verification:

1. **Staged 3-Pass Scientific IR Synthesizer (`invar_agent/ir/`)**: Only the LLM Interpreter layer (assisted by Literature Federation, `ReferenceRegistry`, and bottleneck-gated SHBT Federation) reads raw user prose. Using append-only `Block A || Block B` KV-cache-friendly prompt templates and rationale-before-declaration JSON ordering:
   - **Pass 1 (`FrozenSymbolTable`)** partitions every symbol into strongly typed roles (`STATE`, `CONTROL_PARAM`, `CONSTANT`, `DERIVED_OBSERVABLE`, `INDEPENDENT_VAR`), declares conserved stoichiometric pools (`conservation_pools`), and logs every inferred physical constant/domain/initial condition transparently in `assumptions_ledger`.
   - **Pass 2 (`ModelBodyDraft`)** synthesizes acyclic `let_bindings`, `dynamics_or_residuals`, `observables`, and faithful `obligations` against the immutable `FrozenSymbolTable`.
   - **Pass 3 (Deterministic Rust Scope/Type/DAG Checker & Bounded Micro-Repair)** parses expressions via a zero-eval Pratt parser (`expr_parser.rs`), enforces the 4-scope lexical graph, Kahn `let`-DAG topological ordering + Tarjan SCC cycle extraction, Theorem A.1 4-tier injective alias resolution (`pi_1`/`pi_2`/`pi_3` with `⊥_AMBIG` collision poisoning), `Chk-ODE-RHS` (banning `DERIVED_OBSERVABLE` from dynamics/let-bindings), and bidirectional `Q^7` SI dimensional checking (`scope_check.rs`), with bounded single-symbol micro-repair (`<= 2` repairs, `<= 256` tokens, strict epistemic lock).
2. **Single-Authority Rust RFC 8785 JCS Sealing & 1-Byte Domain BLAKE3 Commitments (`his.rs`, `ihcg_bridge.py`)**: Sealed `SciIR` (`invar.sciir.v2`, aliased as `CanonicalResearchIR` for Builder Studio) is canonicalized via self-contained Rust RFC 8785 UTF-16 JCS (`float_roundtrip` exact IEEE-754 round-trip, duplicate-key & NaN/Inf rejection) and bound with 1-byte domain-separated BLAKE3 commitments: `0x01 || JCS(SciIR)`, `0x02 || JCS(IHCG_v1)`, `0x03 || raw_kernel_source`, `0x04 || JCS(EvidenceRecord)`, and `0x05 || JCS(VerdictReceipt)`.
3. **Deterministic Archetype Kernel Compiler & Native Rust Solver Suite (`kernel_compiler.rs`, `solvers/`)**: Lowers sealed `SciIR` through diagonal affine Buckingham–Π scaling (`buckingham_pi.rs`) to flat `SciBufferSpec` C-ABI C++20 (`-std=c++20 -O3 -fno-fast-math -ffp-contract=off`) and native Rust solvers:
   - **ODE (`solvers/ode.rs`)**: `RK4`, `DOPRI5`, `MPRK22(1.0)` (strictly rejecting `alpha != 1.0`), `ROS2(γ₊)` (`< 1e-13` linear invariant drift on overlapping stoichiometry), `Rosenbrock23` (`+ h d f_t`), `SDIRK2`, and symplectic `ImplicitMidpoint` (`< 1e-3` non-secular Hamiltonian energy error over 10,000 steps).
   - **Monte Carlo (`solvers/mc.rs`)**: Stateless `SplitMix64` CRN, Welford–CGL parallel merge, Maurer–Pontil Empirical Bernstein CI, zero-dependency pure-math Lentz CF Clopper–Pearson (`< 1e-12` vs SciPy), and Vehtari rank-normalized split-`R_hat` & bulk `ESS`.
   - **Fixed-Point & Sweep (`solvers/fixed_point.rs`, `solvers/sweep.rs`)**: Deuflhard affine-invariant damped Newton, central-FD Jacobian cross-check (`JACOBIAN_MISMATCH`), 3-tier stability certification (Routh–Hurwitz, Gershgorin, and continuous Lyapunov `J^T P + P J = -I` Cholesky SPD for non-normal/near-defective Jacobians), parallel Rayon parameter sweeps with adaptive bisection, and algebraic-closure Newton solving.
4. **Provably Total 6-Absorbing-State FSM, SSE v2.0 & Bespoke Hole Verifier (`total_fsm.py`, `hole_verifier.py`)**: Lexicographic Lyapunov budgets `(b_pass, b_repair, b_exec) <= (2, 2, 2)` and `TotalBoundaryAdapter` lift every boundary fault into a cryptographically committed `VerdictReceipt` (0% HTTP 500/502s). SSE v2.0 streams emit `X-Accel-Buffering: no`, `Cache-Control: no-cache, no-transform`, monotonic `id:` frames with `Last-Event-ID` replay, and 15 s keepalive heartbeats. Any `<5%` bespoke Python hole is statically verified by `hole_verifier.py` (`<= 50 KB`) for AST safety, static loop bounds, and MELL observational purity.
5. **Deterministic 9-Stage / 8-Gate Engine & Builder Studio Unification**: Once sealed, **the deterministic engine never parses raw natural-language user prompts**. All 8 epistemic gates, Duhem–Quine falsification certifiers, and Builder Studio `CanonicalResearchIR.from_sci_ir` admission checks share the exact same `0x01 || JCS(SciIR)` BLAKE3 commitment and lossless 8-macro-domain bijection.

| Gate | Requirement |
|---|---|
| Φ<sub>lock</sub> | Compiled contracts entail the original hypothesis; bounds, strictness and quantifiers remain sealed. |
| Φ<sub>nonvac</sub> | Reject tautologies, unreachable antecedents and redundant/vacuous constraints; distinguish a consistent model/environment from a model-incompatible hypothesis. |
| Φ<sub>pure</sub> | Observational purity and immutable controls; projection/stabilization is enforcement, not preservation. |
| Φ<sub>fid</sub> | Formal/executable equivalence, SI/semantic consistency, convergence, physical checks and applicable statistical reliability. |
| Φ<sub>active</sub> | Mechanism-to-observable coupling via derivative/FD/native-SVD and archetype-specific sensitivity/restoring-force checks. |
| Φ<sub>epistemic</sub> | Certified falsification, boundary evidence or significance against a baseline—not a convenient sample. |
| Φ<sub>novel</sub> | Canonical/reference/vault comparison rejects duplicates and validates scoped extensions or sharpenings. |
| Φ<sub>nontrivial</sub> | The environment and domain alone must not already imply the claim without the proposed model. |

Deductive refutation has an explicit activity shortcut after certified model-relative infeasibility; fidelity, purity and reliability still apply. Certified zero-dissipation cancellations can be excluded from activity checks, but hardcoded zero/no-op implementations are not discoveries.

### Four terminal epistemic states

| Verdict | Meaning |
|---|---|
| `VERIFIED_DISCOVERY` | A supported, significant, nontrivial model-dependent result passes the applicable gates and uniqueness policy in its recorded domain. |
| `VERIFIED_FALSIFICATION` | A sealed hypothesis is refuted within its recorded model/domain by certified deductive or numerical evidence with passed fidelity/reliability guards. |
| `VERIFIED_BOUNDARY_DISCOVERY` | Uncertainty-separated witnesses, a bounded bracket and a certified valid-domain argument establish a supported boundary. |
| `INCONCLUSIVE_UNVERIFIED` | Missing, unsupported, unreliable, trivial, duplicate or unresolved evidence; no discovery/refutation is asserted. |

Operational statuses such as `PARTIAL`, `ERROR` and legacy `VERIFIED` describe workflow progress, not additional scientific verdicts. Feasibility assessments (`OPTIMAL_DIRECT`, `REFORMULATED_SURROGATE`, `REFUSED_IMPOSSIBLE`, `REFUSED_UNSUITABLE`, `NEEDS_CLARIFICATION`) likewise do not certify a hypothesis.

### Security, Trust-Boundary & Runtime Architecture (v2 Hardening)

- **Real Execution Isolation (`invar_agent/runtime/isolation.py`)**: Zero in-process `exec` of generated or client-supplied code. Candidate kernels are validated by a strict **AST allow-list** (names resolve strictly to parameters, locals, `math`, or safe scalar builtins; all dunder/private attribute chains are rejected) and executed inside an OS-isolated child process (Win32 Job Objects enforcing `ActiveProcessLimit=1`, memory, and CPU limits on Windows; `rlimits` on Linux; strict `sys.addaudithook` denying file, process, and network access; sanitized environment with secrets stripped). State-changing HTTP endpoints enforce optional bearer authentication when `INVAR_API_TOKEN` is set.
- **Vault Manifest v2 & Strict Cryptographic Admission (`research/vault.py`, `builder/ingestion/vault_adapter.py`)**: Every research run (all verdicts) is archived atomically under lock with an RFC 8785 canonical `manifest.json` (`invar.vault.manifest.v2`) binding domain-separated BLAKE3 digests over every artifact (`blueprint.json`, `unified_module.py`, `meaningfulness_certificate.json`, `report.md`, `metadata.json`, and optional `sci_ir.json`) plus per-run `llm_telemetry`. Builder verifies the manifest and every artifact digest before parsing any file, rejecting missing certificates, inconclusive runs, or any 1-byte tamper with HTTP `422`. `VERIFIED_FALSIFICATION` entries are admitted strictly as `NEGATIVE_CONSTRAINT_ONLY` (rejected at ingest for positive executable profiles). Export (`GET /api/v1/builder/export/{build_id}.zip`) requires `COMPLETED*` status and verified HIS receipts (`409` otherwise), and strips absolute host paths from provenance.
- **Run Isolation, Z3 Worker Serialization & Total SSE v2.0 (`research/broadcaster.py`, `ir/total_fsm.py`)**: `run_uid` is allocated before planning begins; a bounded worker pool gives each run its own scratch directory; all Z3 calls across Research and Builder serialize through a dedicated single-thread OS worker (`run_on_z3_thread`) eliminating cross-thread Z3 crashes; `GET /api/v1/research/stream?run_uid=` delivers per-run history with zero stale cross-run replay and guarantees exactly one `terminal_verdict` SSE v2.0 event per `run_uid`.
- **Claim Fidelity Ledger & Verdict Algebra (`research/claim_ledger.py`, `research/uniqueness.py`, `research/shbt_federation.py`)**: User-fixed prompt literals are pinned to immutable point domains; tautological obligations entailed by guards/environment alone are rejected (`TAUTOLOGICAL_TARGET`); SI horizons match explicit durations or cover `>= 10x` the slowest rate timescale; unrelated model families in a shared Vault classify as `NOVEL_INDEPENDENT` and pass; non-verified certificates always carry a non-null `failure_reason`; capability-based SHBT routing requires explicit opt-in or multi-signal registry concept evidence, labels registry constants as `REGISTRY_ASSUMPTION`, and never overrides feasibility refusals.
- **DeepSeek Provider Layer (`llm/deepseek.py`)**: Planning and formalization use strict `response_format={"type": "json_object"}` via `generate_structured` with typed errors (`EMPTY`, `TRUNCATED`, `INVALID_JSON`, `SCHEMA`) and a 2nd-fingerprint retry circuit breaker.
- **Builder Studio Unification & Platform Hygiene (`builder/ingestion/`, `builder/emitters/`, `_native/build.rs`)**: Strict symbol identity preserves user identifiers (`e`, `E`, `pi`, `I`, `S`, `N`, `lambda`, `gamma`); ODE state SI dimensions are solved from declared parameter units (including compound units like `W/(m²·K)`); algebraic closures and temporal contract predicates (`FINAL`/`MAX`/`MIN`) are preserved cleanly; generated workbenches use UTF-8 and strict `SweepRequest(extra="forbid")`; explicit `encoding="utf-8"` is enforced repo-wide (`PLW1514`); and `invar_agent/_native/build.rs` automatically resolves Python DLLs on Windows so plain `cargo test` runs out of the box.

### Reliability-guarded Duhem–Quine routing

A failed numerical contract does not by itself falsify a scientific model.

- Implementation/formal mismatch, prohibited writes, nonfinite behavior or unsupported bindings are implementation failures or inconclusive evidence.
- Primary/secondary schemes evaluate refined grids and the **same oriented hypothesis functional**. Inter-scheme discrepancy divided by combined GCI uncertainty, with a rounding floor, defines ρ<sub>conv</sub> (`rho_conv`, stored as `convergence_reliability_ratio`).
- If ρ<sub>conv</sub> > 1, `CONVERGENCE_ESTIMATOR_UNRELIABLE` blocks verified falsification and must not halt healing as though the model were refuted. Bounded precision refinement can be requested; persistent uncertainty terminates inconclusively.
- MC variance/drift/stream checks can produce `STATISTICAL_ESTIMATOR_UNRELIABLE`. An unreliable estimator cannot certify a scientific conclusion.
- Only passed fidelity, purity and reliability permit the applicable negative-margin certifier. The hypothesis context and uncertainty are not retuned to pass.

**GCI, variance consistency and FD agreement are reliability diagnostics, not analytic whole-domain floating-point error proofs.**

### Falsification and boundary certifiers

| Path | Evidence and limits |
|---|---|
| Mode 1: deductive | SMT model/environment consistency, QuickXplain minimal unsatisfiable core, serialized constraints and receipt-bound proof payload. Early deductive termination requires zero scientific heals; Z3 remains trusted. |
| Mode 2A: universal | Executable counterexample with a negative uncertainty-aware margin and certified witness enclosure. Sampled failure alone is insufficient. |
| Mode 2B: existential | Exhaustive interval branch-and-bound or validated whole-cell Lipschitz coverage must close the entire domain with a strictly negative upper margin. Failed search is not exhaustive refutation. |
| Boundary | Opposite-side uncertainty-separated witnesses, bounded bracket and separate valid-domain proof. A bracket alone does not certify an entire domain. |

Mode 2B defaults to `B(n) = min(32768, 2048 × 2^max(0,n−3))`, where *n* counts IHCG domain coordinates. The runtime budget function returns 8,192 boxes for five coordinates and caps at 32,768. Caller budgets can lower, not raise, that cap.

Telemetry includes effective budget, boxes evaluated, certified subboxes, refuted-volume fraction, unresolved upper bound/unboundedness and `budget_exhausted`. Incomplete coverage or unresolved/exhausted work yields `INCONCLUSIVE_UNVERIFIED`. Deadlines are **cooperative at checked boundaries**, not hard preemption of arbitrary library/native calls.

## Architecture and execution

### Native core and nine-stage Research loop

- **GoI:** execution/dependency/feedback matrices, spectral checks, acyclic wavefront scheduling and Sherman–Morrison–Woodbury incremental updates.
- **Holographic Interaction Nets:** flat arena reduction, MELL linear typing, promotion/dereliction, SI dimensions, semantic tags, interval bounds and causal fault slicing.
- **Adjoint/numerical analysis:** symbolic derivatives and interval extensions, native SVD, IFT sensitivity, Krawczyk certification, Bauer–Fike/Gershgorin stability bounds.
- **Canonicalization:** Buckingham–Π reduction, audited proof-set `egg` rewrites, exact algebraic DAG normalization and 1-WL structural refinement.
- **Commitments:** Holographic Interface Signatures (HIS), IHCG/stage-chain digests and canonical discovery signatures (CDS) binding model, claim and full domain/verdict commitments.

```text
Ingest → GoI_Plan → Wavefront_Schedule → Overlay_Mutate → PreWrite_Guard
       → SMT_Concolic_Verify → Sandbox_Execute → Reflex_Heal
       → Adjoint_Commit_or_Rollback
```

Copy-on-write overlays isolate candidates. Stages pin immutable layers and commitments; later stages extend rather than overwrite the foundation. Intermediate pinning also requires every declared legacy contract to verify. Rollback preserves earlier verified work without promoting an incomplete extension.

The transpiler lowers restricted Python through UAST to C++20 shared libraries. The C-ABI bridge executes isolated subprocesses, with an isolated Python fallback where supported. Selected-slot trajectories and diagnostics live separately from physical state. Networking and arbitrary filesystem access are not sandbox kernel operations.

### Default verification and four archetypes

An explicit verification runner takes precedence; otherwise Research installs the default runner.

| Canonical archetype | Supported execution/evidence |
|---|---|
| `ODE_PDE` | Autonomous RK4 systems and algebraic readouts; exact stage/assignment-level RHS and RK4 equivalence, primary/secondary refinement, trajectory extrema, FD/SVD activity and executed null/prior baselines. |
| `MONTE_CARLO` | Role-bound stateless SplitMix64 replicates, common random inputs, statistical reliability, supported moments/structural estimators, empirical-Bernstein/CI comparisons and Clopper–Pearson where Bernoulli win semantics are explicitly bound. |
| `FIXED_POINT` | Bound residuals/maps, executable Newton centers, interval Jacobians, Krawczyk, FD enclosure cross-check, IFT sensitivity and interval stability. |
| `PARAM_SWEEP` | Supported scalar sweeps, oriented margins, uncertainty-aware boundary searches and separately certified valid domains. |

These are formal archetypes, **not universal support for arbitrary programs**. General distributions, rational estimator moments, root families, piecewise protocols and unsupported library operations may remain unbindable. Unproved assignments/enclosures fail closed; approximate equivalence and larger solver caps are not substitutes for proof.

### Z3 decidability ladder

`research/meaningfulness.py::solve_query` shares a bounded query budget across:

| Tier | Method |
|---|---|
| 1 | Exact linear real arithmetic, `QF_LRA`. |
| 2 | Exact supported nonlinear real arithmetic, `QF_NRA`. |
| 3 | Sound outer relaxations, `MCCORMICK_QF_LRA`, and interval-verified `WITNESS_PROBE`. |
| 4 | `UNKNOWN` when no supported proof/witness closes within budget. |

Outer-relaxation UNSAT can discharge a query; a relaxation's SAT point is not automatically a real witness. Exact acyclic definition elimination retains domain constraints. The scientific benchmark includes nonlinear/transcendental queries with analytically justified ground truth and checks every non-UNKNOWN answer.

Recorded output from a fresh hermetic run of `tests/test_z3_ladder_benchmark.py`:

```text
Z3 LADDER BENCHMARK: 60 problems / 8 macro-domains / 4 archetypes
Tier 1: 9/60 (15.00%)
Tier 2: 8/60 (13.33%)
Tier 3: 38/60 (63.33%)
Tier 4: 5/60 (8.33%)
Tier 3 detail: McCormick=38, interval-verified witness probe=0
Wrong answers: 0; UNKNOWN problems: oscillatory_bound, exponential_crossing, log_calibration, harmonic_event, poisson_extinction
```

These are corpus-specific empirical results, not a universal solver soundness or completeness proof. UNKNOWN remains unresolved rather than receiving a forced answer.

### Preserved versus enforced telemetry

Certificates, reports and SSE purity events expose `total_steps`, `preserved_steps`, `projected_steps`, `stabilized_steps`, `enforcement_ratio` and stage/domain labels. The ratio is `(projected_steps + stabilized_steps) / max(1, total_steps)`, including empty audits.

Legacy numerical tools include guarded division/log/sqrt/exp/power, Pade-style step limiting, energy backtracking, arbitrary-size simplex projection and covariance/probability/domain conditioners. These may support operational stabilization, but **do not establish observational preservation**. Protected-state/control/hypothesis mutations are rejected or recorded as enforcement. A clamped invariant is not a discovery.

Proof rewrites and healing-only rewrites are separate inventories. Totalizing a singular expression or projecting onto a manifold is not unconditional equivalence of the original model.

### UAST differential translation-validation harness

`research/differential_check.py` compares Python execution, emitted/compiled C++ and independently computed native interval enclosures on identical inputs. `tests/test_uast_differential_lowering.py` covers time stepping, fixed-sample MC reductions, equilibrium iteration, transcendental kernels, simplex normalization and equivalent cubic lowerings.

Numeric agreement and containment are checked without fitting intervals to observations. Arithmetic and libm paths have distinct tolerances. Native exp/log/sqrt widening is an **empirical platform-libm assumption**; sin/cos can use their full codomain. Unsupported interval operations and interval-dependent control flow fail closed.

### Trust Boundary & Empirical Validation Matrix

**Formally-proved-modulo-TCB** means a supported formal query, exact equivalence, interval inclusion or exhaustive enclosure closes relative to trusted components and the declared model. It does **not** mean the entire implementation, compiler, libraries or prose interpretation has a machine-checked correctness proof.

**Empirically-verified-by-differential-testing** means tests compare implementations/oracles on finite corpora and adversarial inputs. Passing increases confidence, not universal certainty.

| TCB assumption | Conditional formal role | Empirical validation and limitation |
|---|---|---|
| 1. Z3 per-logic soundness | Entailment, consistency and MUS queries trust the solver and encoding. | `test_z3_ladder_benchmark.py` compares known answers and reports tier/UNKNOWN telemetry; finite benchmarks do not prove Z3/encoder soundness. |
| 2. `egg` rewrite-axiom soundness | Canonical equivalence trusts admitted rules and valid-domain preconditions. | `test_tcb_soundness.py` inventories proof rules, high-precision samples/interval checks and healing exclusions. **Rewrite sampling is empirical**, not a universal axiom proof. |
| 3. `IntervalF64` IEEE-754 outward rounding | Interval-based conclusions require containment of real results. | IEEE edge tests cover subnormals, signed zero, infinity and near-overflow; differential checks cover containment. **Libm widening is assumed**, not a correctly-rounded-library proof. |
| 4. HIN interval typing soundness | Typed wiring must preserve dimensions, semantic/role constraints and bounds. | Native typing/purity tests and differential lowering exercise accepted/rejected wires; finite examples and type acceptance do not prove arbitrary compiler correctness. |
| 5. `adjoint.rs` symbolic derivative and interval-extension soundness | Jacobian, Krawczyk/IFT, sensitivity and stability arguments rely on these implementations. | Symbolic results are compared to complex-step/FD, including FD interval-Jacobian containment. FD agreement is not a proof for all inputs; unsupported extensions fail closed. |
| 6. [1. Ingest] prose → `FormalResearchSpec` AST binding | The LLM Interpreter translates user prose into a typed Blueprint AST (`FormalResearchSpec`), logging every added equation, SI constant, operating domain, initial condition, or null baseline in `assumptions_ledger`. Once sealed into the IHCG, the deterministic engine never reads raw prose and all 8 formal gates hold strictly modulo the sealed Blueprint AST. | `test_blueprint_compiler.py` and replay tests verify four-namespace resolution, non-blocking provenance annotation, bounded Tier-1 AST repair, and post-seal anti-gaming. Automated translation does not prove semantic equivalence of arbitrary prose and a formula; human audit of `assumptions_ledger` and sealed obligations remains part of the TCB. |

Additional operational assumptions include correct sandbox/compiler execution and scientific routines such as beta quantiles. Cooperative deadlines can overshoot inside library/native calls. Receipts prove artifact integrity/identity, not scientific truth. Unsupported operators, incomplete bindings and solver UNKNOWN must not become positive evidence or falsification certificates.

### Reference coverage and novelty limits

A runtime inventory of the default registry reports **14 packs / 55 legacy equations**, **16 typed annotation records**, **15 indexed native model hashes** and **40 unresolved references**; `coverage_complete` is **False**. These are distinct inventories, not interchangeable coverage percentages.

Unannotated laws do not receive invented SI/role metadata. Unresolved references remain visible; supported independent nontriviality paths do not prove novelty against all scientific literature.

CDS normalization combines typed equations, exact rational unit scales, Π-domain mapping and structural refinement. Vault comparisons distinguish duplicates, valid domain extensions, bound sharpenings, coupled extensions and refutations, with lineage edges. Hash difference alone is not substantive novelty, and 1-WL is not a complete general graph-isomorphism procedure.

## Pan-scientific research and federation

### Eight macro-scientific domains

Canonical IDs in `research/extractor.py`:

| Domain | Representative scope / literature-data routing |
|---|---|
| `PHYSICAL_SCIENCES_AND_ENG` | Mechanics, relativity, plasma, aerospace, thermodynamics; arXiv/ar5iv, Crossref, OpenAlex, Wikipedia, reference constants. |
| `CHEMICAL_AND_MATERIALS` | Kinetics, electrochemistry, materials; PubChem, Crossref, OpenAlex, Semantic Scholar. |
| `LIFE_SCIENCES_AND_BIOMEDICINE` | Systems biology, genomics, PK/PD; Europe PMC, PubMed, bioRxiv/medRxiv, PubChem. |
| `EPIDEMIOLOGY_AND_ECOLOGY` | Compartmental epidemics, populations, ecosystems; Europe PMC, PubMed, bioRxiv/medRxiv, Crossref, OpenAlex. |
| `EARTH_CLIMATE_AND_ENVIRONMENT` | Climate forcing, carbon reservoirs, oceans; arXiv, Crossref, OpenAlex, reference packs. |
| `STATISTICS_AND_CAUSAL_INFERENCE` | Estimation, IV/2SLS, confounding, uncertainty; arXiv, Crossref, OpenAlex, reference packs. |
| `ECONOMICS_AND_GAME_THEORY` | Payoffs, equilibria, welfare; arXiv, Crossref, OpenAlex, reference packs. |
| `MATHEMATICS_CS_AND_INFO_THEORY` | Algorithms, entropy, information; arXiv, OEIS, Wikidata, OpenAlex, reference packs. |

Connectors are relevance/availability-dependent, not a guarantee of live service access. Search, extraction and feasibility are distinct from certification.

SI normalization covers concentrations, clearance, populations, carbon inventories/fluxes, pressure and information units. Domain-gated packs include `epidemiology_population`, `pharmacology_biochemistry`, `genetics_evolution`, `climate_earth_systems`, `statistics_causal_inference`, `economics_game_theory` and `math_info_theory`.

Dimensionless probability, information, population, statistical-metric and utility tags are not interchangeable scalars. Intrinsics include stateless SplitMix64 uniform/normal/discrete samplers, Welford accumulators, Gaussian/link functions, Hill/Michaelis–Menten laws and information/special functions. Discrete approximations and guarded functions are not claims of exact real-distribution semantics.

### SHBT frontier solvability federation

`research/shbt_federation.py` gates equations/constants by bottleneck and relevance across the ten-repository ecosystem:

- `sys1own/shbt-precision` — precision constants and foundational closures.
- `sys1own/shbt-qc` — quantum/photonic and modular-flow/reset models.
- `sys1own/shbt-cf` — cold-fusion frontier and thermal/power bookkeeping.
- `sys1own/shbt-power` — power and open-boundary energy models.
- `sys1own/shbt-ghost` — state-translocation models.
- `sys1own/shbt-sglt` — geodesic-traction models.
- `sys1own/shbt-exotic` — exotic/metric-related models.
- `sys1own/shbt-recon` — reconstruction and gauge-related closures.
- `sys1own/shbt-warp` — metric/warp models.
- `sys1own/shbt-swarm` — collective frontier architectures.

Unrelated conventional tasks are not blanket-routed into SHBT. The closure control uses registry frontier equations, typed SI, explicit external Landauer power debt and internal zero-dissipation cancellations. **Success verifies conditional consistency/execution of that declared model—not experimental validation of SHBT, free energy or a thermodynamic violation.**

### Research context and cognitive routing

Direct research is scoped to the active prompt and explicit `@research:<id>` references, not unrelated chat/vault history. Planner mode can use chat and a structured vault digest. LaTeX/JSON normalization repairs transport/display syntax, not scientific obligations.

Tier-0 uses deterministic native/AST checks; FAST steers queries/directives; REASONING handles synthesis/refinement. DeepSeek defaults resolve via `config.py` to `deepseek-flash` and `deepseek-v4-pro`; Gemini, OpenAI and OpenRouter are also supported.

DeepSeek uses streaming idle/read deadlines, not a fixed total streaming timeout. Research retries provider errors with bounded backoff; adapter/callback budgets compose. Exhaustion yields `INCONCLUSIVE_UNVERIFIED` / `LLM_PROVIDER_RETRIES_EXHAUSTED`. `ResearchStateMachine.plan_and_run()` also persists pre-blueprint validation failures as `PLANNING_VALIDATION_FAILED`, with diagnostics but no fabricated hypothesis graph. The standalone planner retains its exception contract.

## Builder Studio v2.0 and Research-to-Software bridge

Vault references, LaTeX/PDF/CSV or hybrid inputs become typed `CanonicalResearchIR` and composable instruments. Profiles are `hybrid_compute_library`, `interactive_workbench`, `executable_paper_audit` and `calibration_inverse_design`.

The bridge checks IR/certificate BLAKE3 receipts, certificate copies, run identities and evidence roles. Falsification imports are **`NEGATIVE_CONSTRAINT_ONLY`**: never positive executable laws, reference kernels or positive metrics. Fully audited inconclusive certificates cannot be scientific evidence; legacy artifacts remain explicitly unverified.

Frozen interfaces constrain LLM leaf slots. HIS-gated rebuilds reuse unchanged artifacts; bounded healing, deterministic fallbacks, polyglot emitters and research-parity tests protect scientific contracts. Explicit-reference, keyword and scoped recent-run resolution do not waive admission rules. Private redacted leaf-response diagnostics under `.invar/builds/<build_id>/diagnostics/leaf_responses/` are excluded from instrument exports.

## Workspace, Vault and APIs

The console separates workspace and Research Vault, with report/code/metrics viewing, Markdown-protected KaTeX, archive import/export, Builder Studio and xterm terminal. Run-local SSE history and compiled-library/scratch cleanup limit accidental cross-run state reuse; this is not a blanket security proof.

Vault records under `<workspace>/.invar/research_vault/<run_id>/` can include `blueprint.json`, `unified_module.py`, `metadata.json`, `report.md`, `meaningfulness_certificate.json` and canonical-signature/lineage artifacts. Receipts and executable/IHCG/stage commitments bind admitted evidence to those artifacts.

### Research and Vault endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/v1/research/plan` | Blueprint/feasibility planning. |
| POST | `/api/v1/research/plan-prompt` | Context-aware expert research-prompt planning. |
| POST | `/api/v1/research/run` | Start scoped research. |
| POST / GET | `/api/v1/research/action` / `/api/v1/research/status` | Control / status. |
| GET | `/api/v1/research/stream` | Research SSE. |
| GET | `/api/research/literature/health` | Connector health. |
| GET | `/api/v1/research/vault` | List runs. |
| GET | `/api/v1/research/vault/search` | Search records. |
| GET / DELETE | `/api/v1/research/vault/{run_id}` | Inspect / delete. |
| GET | `/api/v1/research/vault/{run_id}/export-zip` | Export run. |
| GET | `/api/v1/research/vault/export-all-zip` | Export vault. |
| POST | `/api/v1/research/vault/import` or `/api/v1/research/vault/upload` | Import vault archive. |
| GET / POST | `/api/v1/workspace/export-zip` / `/api/v1/workspace/import-zip` | Workspace archive transfer. |
| POST | `/api/v1/workspace/clear` | Clear selected workspace. |

### Builder endpoints

All relative paths below are under `/api/v1/builder`:

| Method | Path | Purpose |
|---|---|---|
| POST | `/ingest` | Upload/reference/hybrid ingestion. |
| GET / PATCH | `/ir/{ir_id}` / `/ir/{ir_id}/override` | Inspect / explicitly override IR. |
| POST | `/plan`, `/execute` | Plan / execute build. |
| GET | `/status/{build_id}`, `/stream` | Status / SSE. |
| POST | `/rebuild/{build_id}` | Incremental rebuild. |
| GET | `/export/{build_id}.zip` | Export build. |
| POST / DELETE | `/instrument/{build_id}/launch` | Launch / stop. |
| GET / DELETE | `/instrument/{build_id}` | Inspect / stop. |

**SSE v2.0:** `event_version: "2.0"`. Research channels are `search_log`, `goi_stream` and `accelerator_log`. Builder events include `mode: "builder"` and `goi_stream`, `accelerator_log` and `log`. Consumers should retain purity, uncertainty, receipt and reason-code telemetry and distinguish workflow status from epistemic verdicts.

## Build, run and test

Use a Python supported by `pyproject.toml`, Rust and a C++20 compiler on `PATH`:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install maturin
python -m pip install -e ".[dev,ingest,scientific]"
maturin develop --release

invar-agent serve --workspace-dir /path/to/workspace
# Alternative:
uvicorn invar_agent.server:app --host 127.0.0.1 --port 8000
```

Configure an existing `DEEPSEEK_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY` or `OPENROUTER_API_KEY` in the environment/console. Never commit credentials. Offline verification does not require provider access.

```bash
# Native workspace, from repository root
cargo test

# Offline Python
pytest -v -m "not live"

# Hermetic environment
env -u GEMINI_API_KEY -u OPENROUTER_API_KEY -u OPENAI_API_KEY \
    -u DEEPSEEK_API_KEY -u DEEPSEEK_API \
    INVAR_OFFLINE=1 INVAR_RESEARCH_SEARCH_LIVE=0 \
    pytest -v -m "not live"

# TCB checks and current Z3 tier breakdown
pytest -v -s -m "not live" tests/test_tcb_soundness.py tests/test_z3_ladder_benchmark.py

# Translation validation
pytest -v -m "not live" tests/test_uast_differential_lowering.py

# Opt-in live acceptance, with an existing provider key
pytest -v -m live
```

Offline traps cover weakening/vacuity, frozen/uncoupled dynamics, deductive Carnot refutation, universal/existential numerical refutation and algebraic/unit-rescaled duplicates. Positive controls cover SEIR-V boundaries, structural IV/2SLS versus OLS, SIRS equilibrium and conditional SHBT closure. Pipeline tests exercise persisted receipts, purity, lineage and Builder admission.

These commands are reproducible checks, not claims that a particular machine ran them. Do not copy old test totals, benchmark percentages or timings into a new report without fresh applicable measurements. Offline controls do not establish live success.

### Live E2E results (`deepseek-v4-pro` / `deepseek-flash`)

Verified across all live computational archetypes, terminal epistemic verdicts, SHBT/non-SHBT routing, and Builder Studio v2.0 admission controls (`R7b` and `B1`–`B4` executed end-to-end fresh; live DeepSeek-synthesized artifacts for `R1`–`R7a` were finalized and certified unchanged through `ResearchStateMachine` + Vault persistence after locking universal compiler/formalizer fixes surfaced during live verification with offline regression tests in `tests/test_release_hardening_regressions.py`):

| Scenario | Expected verdict | Observed verdict / mode | ρ<sub>conv</sub> | Enforcement ratio | BLAKE3 receipt | Key metrics & tokens |
|---|---|---|---|---|---|---|
| **R1** SEIR-V boundary (`seirv_vaccine_threshold`, Non-SHBT) | `VERIFIED_BOUNDARY_DISCOVERY` | `VERIFIED_BOUNDARY_DISCOVERY` (`ODE_GCI_SEPARATION`) | `0.0` | `0.0` | `c5ab7ae0cc6b5969345aab0fc4267a2518ea14c5904cd66e3258ea6c229e3781` | `domain_compression_bits=3.0875`, bracket `0.78431320..0.78431416`, 0 SHBT, 4 calls (`54,732` in / `151,517` out) |
| **R2** IV/2SLS vs OLS MC (`iv_mse_separation_mc`, Non-SHBT) | `VERIFIED_DISCOVERY` | `VERIFIED_DISCOVERY` (`MC_CLOPPER_PEARSON`) | — | `0.0` | `57cb725443da27d0eda1e21b7a87859d15d798b2ecb6e94e707d970c239498b6` | `conf=0.9967`, `margin=+0.9013`, `256` CRN replicates, 0 SHBT, 2 calls (`17,506` in / `84,977` out) |
| **R3** RPS replicator-mutator (`rps_replicator_mutator`, Non-SHBT) | `VERIFIED_DISCOVERY` | `VERIFIED_DISCOVERY` (`FIXED_POINT_KRAWCZYK_HURWITZ`) | — | `0.0` | `0f2d18404956def383a6add3b9b3a1278c87d4a0b071ca23aad040584d3718b1` | `margin=+0.0505`, `dead_parameters=[]`, `epsilon_total=2.23e-5`, 0 SHBT |
| **R4** SHBT open-boundary closure (`open_landauer_lanr_ledger`, SHBT) | `VERIFIED_DISCOVERY` | `VERIFIED_DISCOVERY` (`PARETO_EPS_DOMINANCE`) | `0.0` | `0.0` | `0da82a63b6bd6a5631d4687ae86b4cf745b8d2acba5e5e997e3cd1ff8a23dbf6` | `margin=9.99e-9`, `effect=1,047,054`, 565 SHBT refs, 2 calls (`22,082` in / `76,851` out) |
| **R5** Carnot ceiling (`thermo_cycle_eta_feasibility`) | `VERIFIED_FALSIFICATION` | `VERIFIED_FALSIFICATION` (`MODE1_DEDUCTIVE_MUS`) | `0.0` | `0.0` | `74da474644c89034182b749a81b854d9616a7821b109b376714e8f64e680d205` | `heal_count=0`, deductive MUS refutation, 3 calls (`25,447` in / `70,081` out) |
| **R6** SIR `R_max` universal bound (`sir_peak_rmax_bound`) | `VERIFIED_FALSIFICATION` | `VERIFIED_FALSIFICATION` (`MODE2A_COUNTEREXAMPLE`) | `0.7957` | `0.0` | `a750e16098ad71f7f4fddd505bc6b18bc4e0fcd8763f5c33505f71143fc3eef8` | Counterexample witness certified, 2 calls (`15,019` in / `50,363` out) |
| **R7a** Duplicate vs R6 (`sir_peak_universal_bound_test`) | `INCONCLUSIVE_UNVERIFIED` | `INCONCLUSIVE_UNVERIFIED` (`uniqueness_audit.passed=False`) | `0.7956` | `0.0` | `fd4de850101d4381a2ca70cc606f62d9c70b09ab206eccf5567aa579a0388196` | Caught Vault duplicate against R6, 4 calls (`43,894` in / `118,973` out) |
| **R7b** Contradictory impossibility (`T>=500 and T<=300`) | `REFUSED_IMPOSSIBLE` | `REFUSED_IMPOSSIBLE` | — | — | `480b130ba07303ed6593225cca98574e9c34269763b9a88bc9366517e56d9c7f` | Fast precheck refusal in `11.38s`, 1 call (`7,119` in / `607` out) |

**Invar-Agent vNext Live E2E Matrix (`L1`–`L5`, DeepSeek `deepseek-flash` / `deepseek-v4-pro`):**
- **L1** (Casual natural-language OXPHOS): `VERIFIED` (`14.04s`, 1st-try Pass 1 + Pass 2, 0 micro-repairs, pool conservation drift `1.15e-14`).
- **L2** (Detailed multi-pool conserved OXPHOS): `VERIFIED` (`17.20s`, both adenine & redox pools conserved to `3.73e-14`).
- **L3** (Carnot efficiency falsification trap, `η >= 0.75` for `T_H = 600 K`, `T_C ∈ [250, 350] K`): `FALSIFIED` (`8.16s`, `heal_count == 0`, 25 sweep points).
- **L4** (Open-boundary radiative-convective energy balance): `VERIFIED` (`12.55s`, `ALGEBRAIC_CLOSURE` Newton closure solved and certified).
- **L5** (Builder Studio `CanonicalResearchIR` unification from sealed `SciIR`): `PASS` (`commitment_match=True`, identical `0x01 || JCS(SciIR)` BLAKE3 hash across Python/C++20/Rust targets + `hole_verifier.py` audit).
- **Telemetry**: 8 LLM calls total (`51.89s`, `8,576 / 18,231` KV-cache hit tokens = `47.0%`, 0% HTTP 500/502s).

**Builder Studio v2.0 E2E (`B1`–`B4`):**
- **B1** (`seirv_boundary_workbench` from **R1**): `POSITIVE_EVIDENCE`, `ir_blake3_digest=0120b2e5...`, `parity_rc=0`, live launch health `200`, clean stop `200`, export `81,908 bytes` (30 files) (`41.66s`).
- **B2** (`shbt_frontier_calibration_lib` from **R4**): `COMPLETED` build + `PATCH /override` (`ir_digest` `1e162f80...` → `68054338...`) + `COMPLETED` incremental rebuild, export `41,662 bytes` (19 files) (`17.68s`).
- **B3** (**R5** & **R6** falsifications): both admitted as `NEGATIVE_CONSTRAINT_ONLY` with `equations_count=0` and empty pinned kernel (`0.41s`).
- **B4** (Anti-laundering & tamper rejection): HTTP `422` rejecting **R7a** (`INCONCLUSIVE_UNVERIFIED`) and HTTP `422` rejecting 1-byte tampered **R1** certificate (`meaningfulness certificate receipt mismatch`) (`0.51s`).

## Repository map

```text
invar-agent/
├── Cargo.toml                       # Rust workspace
├── pyproject.toml                   # Dependencies, CLI, maturin configuration
├── README.md
├── invar_agent/
│   ├── _native/
│   │   ├── Cargo.toml
│   │   └── src/                     # SciIR, Pratt parser, scope/SI checker, RFC 8785 JCS, kernel compiler, solvers (ODE/MC/FP/Sweep), GoI, HIN, adjoint, egg, HIS, Π
│   ├── ir/                          # SciIR v2 schema, 3-pass staged synthesizer, prefix templates, JSON recovery, IHCG/JCS bridge, TotalBoundaryAdapter FSM, UAST lowering
│   ├── compiler/                    # hole_verifier.py (<= 50 KB AST safety, static loop bounds, MELL purity)
│   ├── research/
│   │   ├── state_machine.py         # Research loop and provider handling
│   │   ├── planner.py               # LLM Interpreter synthesis and bounded repair
│   │   ├── blueprint_compiler.py    # Tier-0 four-namespace Blueprint AST compiler & preflight
│   │   ├── formal_evidence.py       # Optional non-blocking provenance annotation
│   │   ├── meaningfulness.py        # IHCG, gates, SMT, reliability, certificates
│   │   ├── falsification.py         # Deductive, numerical, boundary certifiers
│   │   ├── default_verification.py  # Executable binding and baseline evidence
│   │   ├── stochastic_equilibrium.py
│   │   ├── differential_check.py
│   │   ├── uniqueness.py            # CDS, references, vault comparison, lineage
│   │   ├── reference_annotations.py
│   │   ├── epistemic.py             # Receipts and execution/certificate bridge
│   │   ├── shbt_federation.py
│   │   └── ...                      # Feasibility, search/extraction, reports, vault, SSE v2.0
│   ├── builder/                     # Studio API, ingestion/IR (CanonicalResearchIR.from_sci_ir), guards, emitters
│   ├── uast/                        # Schemas, compiler, totalized math, native bridge
│   ├── transpiler/                  # Restricted Python frontend and C++/HIN
│   ├── runtime/                     # C-ABI and isolated Python execution
│   ├── verification/                # SMT, concolic checks, precision shield
│   ├── guards/                      # Firewall and prewrite checks
│   ├── healing/                     # AST repair, purity instrumentation
│   ├── llm/                         # Providers, errors, retries, telemetry
│   ├── schemas/                     # Blueprint/contract/shared models
│   ├── static/                      # Workspace, Research, Builder UI
│   ├── latex_normalizer.py
│   └── server.py                    # FastAPI and SSE v2.0
├── scripts/
└── tests/                           # Unit, CI gates, 540-system Hypothesis fuzz suite, differential, traps/controls, opt-in live
```

## Roadmap

**Full Bayesian/KL prior-relative informativeness is a next-phase capability.** Current model dependence, significance and canonical/reference-relative nontriviality are not a full Bayesian update; having a KL intrinsic does not implement that gate. Broader certified sampler/residual bindings, analytic floating-point error arguments and expanded typed reference coverage remain distinct work.
