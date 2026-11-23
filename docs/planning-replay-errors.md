# Planning replay findings

These are offline replays of the exact saved provider content, not new provider calls.
The historical messages below were recorded before normalization and the evidence-only pipeline.

| Run | Scenario | Attempt | Exact validation error |
|---|---|---|---|
| 3 | SEIR-V | 1 | `slots.12.role`, `slots.13.role`: `Input should be 'CONTROL_PARAM', 'DYNAMIC_STATE', 'DERIVED_OBSERVABLE' or 'CONSERVATION_AUX'`; supplied `DIAGNOSTIC` |
| 3 | SEIR-V | 2 | `IHCG_INGEST_REJECTED: CONSTRAINING` |
| 3 | Carnot | 1 | `compiled contract has no sealed symbolic binding` |
| 3 | Carnot | 2 | `compiled contract has no sealed symbolic binding` |
| 3 | SHBT | 1 | `IHCG_INGEST_REJECTED: CONSTRAINING` |
| 3 | SHBT | 2 | `IHCG_INGEST_REJECTED: CONSTRAINING` |

`CONSTRAINING` alone concealed a failing redundancy check, not a failed classification.
The SEIR-V second response has redundant conjunct indices `4`, `5`; SHBT has index `1`
in its first response and indices `1`, `2` in its second. These remain rejected. New
diagnostics expose redundancy indices and solver-query statuses without changing the gates.

Both Carnot responses explicitly declare every buffer index in their typed slots but use
different descriptive names in `variable_map`. Binding now uses the sealed slot indices,
rejecting conflicting same-name indices. This exposes a subsequent genuine `Phi_lock`
failure: their contracts assert the ceiling's refutation rather than retain the sealed
hypothesis. Normalization must not invert these inequalities to manufacture acceptance.

## Normalization boundary

- Exact lower/upper and min/max objects become arrays; a dimensionless flag is accepted
  only when consistent with explicit slot dimensions.
- Equations can use a uniquely bound scalar assignment or derivative. Missing dimensions
  come only from explicit output/state SI declarations; derivative dimensions subtract time.
- Only fixed, context-unique aliases are supported. `ALGEBRAIC` is not an archetype alias;
  `ALGEBRAIC_INEQUALITY` and `DIAGNOSTIC` remain unsupported, ambiguous encodings.
- Presentation fields and original transformed values survive under
  `ihcg_evidence.execution.normalization_source`, outside the scientific graph.
- Unknown semantic fields, incomplete declarations, conflicting assignments, and duplicate
  normalized outputs fail closed. No number, bound, inequality or quantifier is rewritten.

Strict quantitative planning now requests only the blueprint, followed by an evidence-only
call using the compiler models' JSON schema and the compiler-checked worked example.
The existing configured reasoning callback is the default; an explicit `llm_formalizer`
callback can select the configured fast tier. Evidence repair is limited to three calls,
each receiving the exact previous error and response. The blueprint is immutable during
this loop. An incorrect executable or contradictory contracts therefore still require a
new blueprint, not a fabricated evidence repair. Exhaustion persists
`PLANNING_VALIDATION_FAILED` with all attempt diagnostics and a receipt-bearing incomplete
certificate. Nonquantitative legacy and explicit offline/fallback planning retain their
existing behavior.

## Evidence-only replay findings

The next saved runs made one blueprint call and three evidence-only calls each. The
first-error messages below are historical; the complete original messages, including
every missing symbol, are stored in `tests/fixtures/live_run4_raw_plans.json`.

| Scenario | Evidence attempt | Historical validation error | Decision |
|---|---|---|---|
| SEIR-V | 1 | `obligation quote has no unique immutable prompt span` | Gate bug after off-by-one offsets: the quote exists exactly, but the lexer consumed the following sentence period as part of the number. Fix numeric-token boundaries without accepting changed numbers. The replay then reaches the missing-domain error. |
| SEIR-V | 2 | `IHCG slots and Omega must be one-to-one; missing_domain=...` | Mis-encoding: 22 slots, only 3 domains. All slots, including derived quantities, require justified finite bounds. |
| SEIR-V | 3 | `ihcg_evidence.evidence.environmental_assumptions[0]: invalid syntax (<unknown>, line 1)` | Mis-encoding: English prose in an executable expression field. Never discard or guess its meaning. |
| SHBT | 1 | `IHCG slots and Omega must be one-to-one; missing_domain=...` | Mis-encoding: 33 slots, only 13 domains. Report the domain omissions, variable cap and invalid prose together in repair feedback. |
| SHBT | 2 | `at most 32 bounded variables are supported` | Unsupported encoding: 33 symbolic variables exceed the existing deterministic cap. Do not raise the cap or silently drop scientific variables. |
| SHBT | 3 | `ihcg_evidence.evidence.environmental_assumptions[0]: invalid syntax (<unknown>, line 1)` | Mis-encoding: prose remains after the variable/domain repair. |

None of these six attempts reaches the redundancy gate. The non-vacuity requirement
demands zero redundant primary conjuncts in an admitted sealed graph. Q4 checks
entailment by the environment **and the other primary conjuncts**, which is stronger
than environment-only entailment; a Q4 UNSAT result alone does not authorize pruning.
An environment-proved equivalence is mathematically non-weakening, but would require
audited canonicalization before sealing (or an explicitly checked resealing rule),
not deletion of clauses from an already sealed hypothesis. The current gate is retained;
no redundancy-policy change is needed to address these failures.

Incomplete certificates now bind structured per-attempt diagnostics into their receipt;
metadata and terminal SSE carry the same data, and the opt-in live summary prints it.
Legacy incomplete receipts without diagnostics remain readable. Repair feedback reports
independent encoding errors together, without changing the scientific gates or inventing
bounds. The worked example includes an executable environmental predicate.

The previous planning identifier was a hash of the question. The two historical live
runs used separate scratch workspace vault roots, so their repeated identifier did not
address the same files. Within a shared workspace, repeated failures could overwrite
one another. Planning failures now allocate a fresh UUID identifier for every invocation.
