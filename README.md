# invar-agent

An **8-Gate Formal Epistemic Meaningfulness, Uniqueness & Justified Falsification Engine**, paired with a software Builder and an interactive research workspace.

The Rust/PyO3 core, Python/Z3 substrate, restricted-Python-to-C++20 compiler and isolated execution bridge connect declared scientific hypotheses to executable evidence. Results carry BLAKE3 commitments and auditable certificates. **Verification is conditional on the sealed model, domain, numerical reliability and trust boundary—not a guarantee of physical truth or universal implementation correctness.** Missing evidence and unsupported obligations fail closed.

## Scientific contract and eight gates

Ingest seals the IHCG: original prompt, quoted obligations and character spans, quantifier, finite domain Ω, typed symbols, governing equations, immutable controls and stage commitments. Formal evidence must bind to the executable slots and contracts, not to an input containing the desired answer.

Literal extraction requires unambiguous formula/domain/unit declarations; it does not invent units or ambiguous prose aliases. Quote re-anchoring tolerates whitespace and equivalent Unicode comparison/minus tokens, then stores the exact original substring. Paraphrases, changed numbers/strictness and ambiguous matches are rejected. Strict quantitative planning requests bounded repair for absent or invalid evidence. Healing cannot recalibrate sealed observables or silently weaken a hypothesis.

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

Operational statuses such as `PARTIAL`, `ERROR` and legacy `VERIFIED` describe workflow progress, not additional scientific verdicts. Feasibility assessments (`OPTIMAL_DIRECT`, `REFORMULATED_SURROGATE`, `REFUSED_IMPOSSIBLE`, `REFUSED_UNSUITABLE`) likewise do not certify a hypothesis.

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
| 6. [1. Ingest] prose → H<sub>user</sub> binding | The formal graph must faithfully encode original claim, assumptions, units, domain and quantifier. | Exact-span/replay tests catch evidence loss, normalization drift and weakened contracts. A literal quote proves provenance, **not semantic equivalence of arbitrary prose and a formula**; human/model-binding audit remains necessary. |

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

### Live E2E results — user-maintained placeholder

**PENDING LIVE VALIDATION. Fill only from retained real-run artifacts; no live success is asserted here.**

| Scenario | Expected verdict | Observed verdict / reason | ρ<sub>conv</sub> | Enforcement ratio | BLAKE3 receipt | Directory / tokens |
|---|---|---|---|---|---|---|
| SEIR-V boundary | `VERIFIED_BOUNDARY_DISCOVERY` | **PENDING USER RESULT** | — | — | — | — |
| Carnot ceiling | `VERIFIED_FALSIFICATION` | **PENDING USER RESULT** | — | — | — | — |
| SHBT open-boundary closure | `VERIFIED_DISCOVERY` | **PENDING USER RESULT** | — | — | — | — |

Also retain applicable fidelity/statistical guards, zero-heal deductive behavior, purity events, certificate/IR checks and model IDs. Live artifacts stay outside the repo under `~/.invar-live-tests/meaningfulness/`, including plans/response content, SSE events and summaries where available.

## Repository map

```text
invar-agent/
├── Cargo.toml                       # Rust workspace
├── pyproject.toml                   # Dependencies, CLI, maturin configuration
├── README.md
├── invar_agent/
│   ├── _native/
│   │   ├── Cargo.toml
│   │   └── src/                     # GoI, HIN, adjoint, egg, HIS, Π, intervals
│   ├── research/
│   │   ├── state_machine.py         # Research loop and provider handling
│   │   ├── planner.py               # Synthesis and formal validation
│   │   ├── formal_evidence.py       # Literal extraction and quote anchoring
│   │   ├── meaningfulness.py        # IHCG, gates, SMT, reliability, certificates
│   │   ├── falsification.py         # Deductive, numerical, boundary certifiers
│   │   ├── default_verification.py  # Executable binding and baseline evidence
│   │   ├── stochastic_equilibrium.py
│   │   ├── differential_check.py
│   │   ├── uniqueness.py            # CDS, references, vault comparison, lineage
│   │   ├── reference_annotations.py
│   │   ├── epistemic.py             # Receipts and execution/certificate bridge
│   │   ├── shbt_federation.py
│   │   └── ...                      # Feasibility, search/extraction, reports, vault, SSE
│   ├── builder/                     # Studio API, ingestion/IR, guards, emitters
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
│   └── server.py                    # FastAPI and SSE
├── scripts/
└── tests/                           # Unit, differential, traps/controls, opt-in live
```

## Roadmap

**Full Bayesian/KL prior-relative informativeness is a next-phase capability.** Current model dependence, significance and canonical/reference-relative nontriviality are not a full Bayesian update; having a KL intrinsic does not implement that gate. Broader certified sampler/residual bindings, analytic floating-point error arguments and expanded typed reference coverage remain distinct work.
