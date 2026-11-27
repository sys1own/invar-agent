//! Pass 3 Deterministic Scope Graph, Kahn let-DAG Sort, Tarjan SCC Cycle Extractor,
//! Provably Injective 4-Tier Alias Resolver (Theorem A.1), and Bidirectional Q^7 SI Checker.

use crate::expr_parser::{parse_expr, RawExpr};
use crate::sci_ir::{Archetype, BinaryOp, ExprNode, SiDimension, SlotRole, SymbolDecl, TrajectoryReduction, UnaryOp};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AliasResolution {
    Exact { canonical: String },
    Resolved { canonical: String, tier: u8 },
    AmbiguousCollision { reference: String, tier: u8, candidates: Vec<String> },
    Unresolved { reference: String },
}

/// Tier 1 projection `pi_1`: NFKC-like subscript digits -> ASCII, Greek transliteration
/// inserting `_` at Greek-to-letter boundaries, preserving case and underscores.
pub fn pi_1(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    let chars: Vec<char> = s.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        let greek = match c {
            'Δ' => Some("Delta"),
            'δ' => Some("delta"),
            'Ψ' => Some("Psi"),
            'ψ' => Some("psi"),
            'Φ' => Some("Phi"),
            'φ' | 'ϕ' => Some("phi"),
            'α' => Some("alpha"),
            'β' => Some("beta"),
            'γ' => Some("gamma"),
            'Γ' => Some("Gamma"),
            'ε' | 'ϵ' => Some("eps"),
            'η' => Some("eta"),
            'θ' | 'ϑ' => Some("theta"),
            'Θ' => Some("Theta"),
            'κ' => Some("kappa"),
            'λ' => Some("lambda"),
            'Λ' => Some("Lambda"),
            'μ' | 'µ' => Some("mu"),
            'ν' => Some("nu"),
            'π' => Some("pi"),
            'Π' => Some("Pi"),
            'ρ' => Some("rho"),
            'σ' => Some("sigma"),
            'Σ' => Some("Sigma"),
            'τ' => Some("tau"),
            'ω' => Some("omega"),
            'Ω' => Some("Omega"),
            _ => None,
        };
        if let Some(word) = greek {
            if !out.is_empty() && !out.ends_with('_') {
                out.push('_');
            }
            out.push_str(word);
            if i + 1 < chars.len() {
                let next = chars[i + 1];
                if next.is_alphabetic() || next.is_ascii_digit() {
                    out.push('_');
                }
            }
            continue;
        }
        if ('\u{2080}'..='\u{2089}').contains(&c) {
            let digit = (c as u32 - 0x2080) as u8;
            if !out.is_empty() && !out.ends_with('_') && out.chars().last().map(|x| x.is_alphabetic()).unwrap_or(false) {
                out.push('_');
            }
            out.push((b'0' + digit) as char);
            continue;
        }
        out.push(c);
    }
    // Collapse accidental double underscores.
    while out.contains("__") {
        out = out.replace("__", "_");
    }
    out.trim_matches('_').to_string()
}

/// Tier 2 projection `pi_2`: underscore stripping on `pi_1(s)`, strictly case-sensitive.
pub fn pi_2(s: &str) -> String {
    pi_1(s).replace('_', "")
}

/// Tier 3 projection `pi_3`: rewrite leading `^(?:Delta|delta)_?([A-Za-z]) -> d\1` on `pi_1(s)`,
/// strip underscores, and lowercase.
pub fn pi_3(s: &str) -> String {
    let p1 = pi_1(s);
    let rewritten = if let Some(rest) = p1.strip_prefix("Delta_") {
        format!("d{}", rest)
    } else if let Some(rest) = p1.strip_prefix("delta_") {
        format!("d{}", rest)
    } else if p1.starts_with("Delta") && p1.len() > 5 && p1.as_bytes()[5].is_ascii_alphabetic() {
        format!("d{}", &p1[5..])
    } else if p1.starts_with("delta") && p1.len() > 5 && p1.as_bytes()[5].is_ascii_alphabetic() {
        format!("d{}", &p1[5..])
    } else {
        p1
    };
    rewritten.replace('_', "").to_ascii_lowercase()
}

/// Theorem A.1: Non-Merging Injective 4-Tier Alias Resolver with `⊥_AMBIG` poisoning.
pub struct InjectiveAliasResolver {
    exact: HashSet<String>,
    tier1: HashMap<String, Result<String, Vec<String>>>,
    tier2: HashMap<String, Result<String, Vec<String>>>,
    tier3: HashMap<String, Result<String, Vec<String>>>,
}

impl InjectiveAliasResolver {
    pub fn new(declared: &[String]) -> Self {
        let mut exact = HashSet::new();
        let mut f1: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let mut f2: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let mut f3: BTreeMap<String, Vec<String>> = BTreeMap::new();

        for sym in declared {
            exact.insert(sym.clone());
            f1.entry(pi_1(sym)).or_default().push(sym.clone());
            f2.entry(pi_2(sym)).or_default().push(sym.clone());
            f3.entry(pi_3(sym)).or_default().push(sym.clone());
        }

        let build = |fibers: BTreeMap<String, Vec<String>>| {
            let mut map = HashMap::new();
            for (k, mut v) in fibers {
                v.sort();
                v.dedup();
                if v.len() == 1 {
                    map.insert(k, Ok(v.pop().unwrap()));
                } else {
                    map.insert(k, Err(v));
                }
            }
            map
        };

        Self {
            exact,
            tier1: build(f1),
            tier2: build(f2),
            tier3: build(f3),
        }
    }

    pub fn resolve(&self, reference: &str) -> AliasResolution {
        if self.exact.contains(reference) {
            return AliasResolution::Exact {
                canonical: reference.to_string(),
            };
        }
        let k1 = pi_1(reference);
        if let Some(entry) = self.tier1.get(&k1) {
            return match entry {
                Ok(c) => AliasResolution::Resolved { canonical: c.clone(), tier: 1 },
                Err(cands) => AliasResolution::AmbiguousCollision {
                    reference: reference.to_string(),
                    tier: 1,
                    candidates: cands.clone(),
                },
            };
        }
        let k2 = pi_2(reference);
        if let Some(entry) = self.tier2.get(&k2) {
            return match entry {
                Ok(c) => AliasResolution::Resolved { canonical: c.clone(), tier: 2 },
                Err(cands) => AliasResolution::AmbiguousCollision {
                    reference: reference.to_string(),
                    tier: 2,
                    candidates: cands.clone(),
                },
            };
        }
        let k3 = pi_3(reference);
        if let Some(entry) = self.tier3.get(&k3) {
            return match entry {
                Ok(c) => AliasResolution::Resolved { canonical: c.clone(), tier: 3 },
                Err(cands) => AliasResolution::AmbiguousCollision {
                    reference: reference.to_string(),
                    tier: 3,
                    candidates: cands.clone(),
                },
            };
        }
        AliasResolution::Unresolved {
            reference: reference.to_string(),
        }
    }
}

/// Rational arithmetic for Q^7 SI dimension vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Rat(i32, i32);

fn gcd_i32(mut a: i32, mut b: i32) -> i32 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    if a == 0 { 1 } else { a }
}

impl Rat {
    fn new(n: i32, d: i32) -> Self {
        if d == 0 {
            return Self(0, 1);
        }
        let sign = if d < 0 { -1 } else { 1 };
        let g = gcd_i32(n, d);
        Self((n * sign) / g, (d * sign) / g)
    }
    fn add(self, other: Self) -> Self {
        Self::new(self.0 * other.1 + other.0 * self.1, self.1 * other.1)
    }
    fn sub(self, other: Self) -> Self {
        Self::new(self.0 * other.1 - other.0 * self.1, self.1 * other.1)
    }
    fn mul(self, other: Self) -> Self {
        Self::new(self.0 * other.0, self.1 * other.1)
    }
}

fn dim_add(a: &[(i32, i32); 7], b: &[(i32, i32); 7]) -> [(i32, i32); 7] {
    let mut out = [(0, 1); 7];
    for i in 0..7 {
        let r = Rat::new(a[i].0, a[i].1).add(Rat::new(b[i].0, b[i].1));
        out[i] = (r.0, r.1);
    }
    out
}

fn dim_sub(a: &[(i32, i32); 7], b: &[(i32, i32); 7]) -> [(i32, i32); 7] {
    let mut out = [(0, 1); 7];
    for i in 0..7 {
        let r = Rat::new(a[i].0, a[i].1).sub(Rat::new(b[i].0, b[i].1));
        out[i] = (r.0, r.1);
    }
    out
}

fn dim_scale(a: &[(i32, i32); 7], factor: Rat) -> [(i32, i32); 7] {
    let mut out = [(0, 1); 7];
    for i in 0..7 {
        let r = Rat::new(a[i].0, a[i].1).mul(factor);
        out[i] = (r.0, r.1);
    }
    out
}

fn dim_eq(a: &[(i32, i32); 7], b: &[(i32, i32); 7]) -> bool {
    for i in 0..7 {
        let ra = Rat::new(a[i].0, a[i].1);
        let rb = Rat::new(b[i].0, b[i].1);
        if ra != rb {
            return false;
        }
    }
    true
}

fn dim_is_zero(a: &[(i32, i32); 7]) -> bool {
    a.iter().all(|(n, _)| *n == 0)
}

/// Kahn's topological sort + Tarjan's SCC cycle extractor for `let_bindings`.
pub fn sort_let_dag(deps: &BTreeMap<String, Vec<String>>) -> Result<Vec<String>, Vec<Vec<String>>> {
    let mut indeg: BTreeMap<String, usize> = BTreeMap::new();
    let mut adj: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for name in deps.keys() {
        indeg.insert(name.clone(), 0);
        adj.entry(name.clone()).or_default();
    }
    for (node, reqs) in deps {
        for req in reqs {
            if deps.contains_key(req) {
                adj.entry(req.clone()).or_default().push(node.clone());
                *indeg.entry(node.clone()).or_insert(0) += 1;
            }
        }
    }
    let mut q: VecDeque<String> = indeg
        .iter()
        .filter(|(_, &d)| d == 0)
        .map(|(k, _)| k.clone())
        .collect();
    let mut order = Vec::with_capacity(deps.len());
    while let Some(u) = q.pop_front() {
        order.push(u.clone());
        if let Some(neighbors) = adj.get(&u) {
            for v in neighbors {
                let d = indeg.get_mut(v).unwrap();
                *d -= 1;
                if *d == 0 {
                    q.push_back(v.clone());
                }
            }
        }
    }
    if order.len() == deps.len() {
        return Ok(order);
    }
    // Extract minimal cycles via Tarjan's SCC algorithm.
    let mut index = 0usize;
    let mut indices: HashMap<String, usize> = HashMap::new();
    let mut lowlink: HashMap<String, usize> = HashMap::new();
    let mut stack: Vec<String> = Vec::new();
    let mut on_stack: HashSet<String> = HashSet::new();
    let mut cycles: Vec<Vec<String>> = Vec::new();

    #[allow(clippy::too_many_arguments)]
    fn strongconnect(
        v: &str,
        deps: &BTreeMap<String, Vec<String>>,
        index: &mut usize,
        indices: &mut HashMap<String, usize>,
        lowlink: &mut HashMap<String, usize>,
        stack: &mut Vec<String>,
        on_stack: &mut HashSet<String>,
        cycles: &mut Vec<Vec<String>>,
    ) {
        indices.insert(v.to_string(), *index);
        lowlink.insert(v.to_string(), *index);
        *index += 1;
        stack.push(v.to_string());
        on_stack.insert(v.to_string());

        if let Some(reqs) = deps.get(v) {
            for w in reqs {
                if !deps.contains_key(w) {
                    continue;
                }
                if !indices.contains_key(w) {
                    strongconnect(w, deps, index, indices, lowlink, stack, on_stack, cycles);
                    let lw = lowlink[w];
                    let lv = lowlink[v];
                    lowlink.insert(v.to_string(), lv.min(lw));
                } else if on_stack.contains(w) {
                    let iw = indices[w];
                    let lv = lowlink[v];
                    lowlink.insert(v.to_string(), lv.min(iw));
                }
            }
        }

        if lowlink[v] == indices[v] {
            let mut comp = Vec::new();
            while let Some(w) = stack.pop() {
                on_stack.remove(&w);
                comp.push(w.clone());
                if w == v {
                    break;
                }
            }
            let self_loop = comp.len() == 1
                && deps.get(&comp[0]).map(|r| r.contains(&comp[0])).unwrap_or(false);
            if comp.len() > 1 || self_loop {
                comp.reverse();
                cycles.push(comp);
            }
        }
    }

    for v in deps.keys() {
        if !indices.contains_key(v) {
            strongconnect(
                v,
                deps,
                &mut index,
                &mut indices,
                &mut lowlink,
                &mut stack,
                &mut on_stack,
                &mut cycles,
            );
        }
    }
    Err(cycles)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScopeDiagnostic {
    pub code: String,
    pub field: String,
    pub symbol: Option<String>,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawLetDraft {
    pub name: String,
    pub expr_text: String,
    #[serde(default)]
    pub dimension: Option<SiDimension>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawEquationDraft {
    pub target: String,
    pub expr_text: String,
    #[serde(default)]
    pub reduction: Option<TrajectoryReduction>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawObligationDraft {
    pub id: String,
    pub quantifier: String,
    pub lhs_text: String,
    pub comparator: String,
    pub rhs_text: String,
    #[serde(default = "default_tol")]
    pub tolerance: f64,
    pub epistemic_origin: String,
    #[serde(default)]
    pub source_text: String,
}

fn default_tol() -> f64 {
    1e-6
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScopeCheckRequest {
    pub archetype: Archetype,
    pub symbols: Vec<SymbolDecl>,
    #[serde(default)]
    pub let_bindings: Vec<RawLetDraft>,
    #[serde(default)]
    pub dynamics_or_residuals: Vec<RawEquationDraft>,
    #[serde(default)]
    pub observables: Vec<RawEquationDraft>,
    #[serde(default)]
    pub obligations: Vec<RawObligationDraft>,
    #[serde(default)]
    pub check_dimensions: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedLet {
    pub name: String,
    pub dimension: SiDimension,
    pub expr: ExprNode,
    pub source_text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedEquation {
    pub target: String,
    pub target_slot: usize,
    pub expr: ExprNode,
    pub source_text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reduction: Option<TrajectoryReduction>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedObligation {
    pub id: String,
    pub quantifier: String,
    pub lhs_expr: ExprNode,
    pub comparator: String,
    pub rhs_expr: ExprNode,
    pub tolerance: f64,
    pub epistemic_origin: String,
    pub source_text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScopeCheckResponse {
    pub ok: bool,
    pub diagnostics: Vec<ScopeDiagnostic>,
    pub resolved_aliases: BTreeMap<String, String>,
    pub let_bindings: Vec<ResolvedLet>,
    pub dynamics_or_residuals: Vec<ResolvedEquation>,
    pub observables: Vec<ResolvedEquation>,
    pub obligations: Vec<ResolvedObligation>,
}

fn collect_raw_idents(expr: &RawExpr, out: &mut BTreeSet<String>) {
    match expr {
        RawExpr::Const { .. } | RawExpr::Rand { .. } => {}
        RawExpr::Ident { name } => {
            out.insert(name.clone());
        }
        RawExpr::Unary { arg, .. } => collect_raw_idents(arg, out),
        RawExpr::Binary { lhs, rhs, .. } => {
            collect_raw_idents(lhs, out);
            collect_raw_idents(rhs, out);
        }
        RawExpr::IfThenElse {
            lhs,
            rhs,
            then_branch,
            else_branch,
            ..
        } => {
            collect_raw_idents(lhs, out);
            collect_raw_idents(rhs, out);
            collect_raw_idents(then_branch, out);
            collect_raw_idents(else_branch, out);
        }
    }
}

/// Resolve a `RawExpr` in a lexical scope, applying Theorem A.1 alias resolution and
/// returning `(ExprNode, inferred_si_dim)`.
type OptDim = Option<[(i32, i32); 7]>;

fn merge_additive_dim(
    l_dim: OptDim,
    r_dim: OptDim,
    check_dimensions: bool,
    field: &str,
    msg: &str,
    diags: &mut Vec<ScopeDiagnostic>,
) -> OptDim {
    match (l_dim, r_dim) {
        (None, None) => None,
        (Some(d), None) | (None, Some(d)) => Some(d),
        (Some(ld), Some(rd)) => {
            if check_dimensions && !dim_eq(&ld, &rd) {
                diags.push(ScopeDiagnostic {
                    code: "DIMENSIONAL_INCONSISTENCY".into(),
                    field: field.into(),
                    symbol: None,
                    message: format!("{}: {:?} vs {:?}", msg, ld, rd),
                });
            }
            Some(ld)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn lower_and_check_expr(
    raw: &RawExpr,
    field: &str,
    archetype: Archetype,
    allow_observables: bool,
    sym_map: &HashMap<String, &SymbolDecl>,
    let_map: &HashMap<String, (ExprNode, OptDim)>,
    resolver: &InjectiveAliasResolver,
    check_dimensions: bool,
    resolved_aliases: &mut BTreeMap<String, String>,
    diags: &mut Vec<ScopeDiagnostic>,
) -> Option<(ExprNode, OptDim)> {
    match raw {
        RawExpr::Const { value } => Some((ExprNode::Const { value: *value }, None)),
        RawExpr::Rand {
            dist,
            draw_stream_id,
        } => {
            if archetype != Archetype::MonteCarlo {
                diags.push(ScopeDiagnostic {
                    code: "RAND_OUTSIDE_MONTE_CARLO".into(),
                    field: field.into(),
                    symbol: None,
                    message: "Random draws are permitted exclusively in MONTE_CARLO archetype".into(),
                });
                return None;
            }
            Some((
                ExprNode::Rand {
                    dist: *dist,
                    draw_stream_id: *draw_stream_id,
                },
                Some([(0, 1); 7]),
            ))
        }
        RawExpr::Ident { name } => {
            // Built-in constants `pi` and `e_euler` (when not shadowed by a declared symbol).
            if !sym_map.contains_key(name) && !let_map.contains_key(name) {
                if name == "pi" {
                    return Some((
                        ExprNode::Const {
                            value: std::f64::consts::PI,
                        },
                        None,
                    ));
                }
                if name == "e_euler" {
                    return Some((
                        ExprNode::Const {
                            value: std::f64::consts::E,
                        },
                        None,
                    ));
                }
            }
            let canonical = match resolver.resolve(name) {
                AliasResolution::Exact { canonical } => canonical,
                AliasResolution::Resolved { canonical, .. } => {
                    resolved_aliases.insert(name.clone(), canonical.clone());
                    canonical
                }
                AliasResolution::AmbiguousCollision {
                    reference,
                    tier,
                    candidates,
                } => {
                    diags.push(ScopeDiagnostic {
                        code: "AMBIGUOUS_COLLISION".into(),
                        field: field.into(),
                        symbol: Some(reference.clone()),
                        message: format!(
                            "Symbol {:?} collides ambiguously at alias tier {} with {:?}",
                            reference, tier, candidates
                        ),
                    });
                    return None;
                }
                AliasResolution::Unresolved { reference } => {
                    diags.push(ScopeDiagnostic {
                        code: "UNRESOLVED_SYMBOL".into(),
                        field: field.into(),
                        symbol: Some(reference.clone()),
                        message: format!(
                            "Symbol {:?} is not declared in FrozenSymbolTable or let_bindings",
                            reference
                        ),
                    });
                    return None;
                }
            };
            if let Some((let_expr, let_dim)) = let_map.get(&canonical) {
                return Some((let_expr.clone(), *let_dim));
            }
            if let Some(decl) = sym_map.get(&canonical) {
                if decl.role == SlotRole::DerivedObservable && !allow_observables {
                    diags.push(ScopeDiagnostic {
                        code: "DERIVED_OBSERVABLE_IN_DYNAMICS".into(),
                        field: field.into(),
                        symbol: Some(canonical.clone()),
                        message: format!(
                            "Chk-ODE-RHS violation: DERIVED_OBSERVABLE {:?} cannot be read inside dynamics_or_residuals or let_bindings",
                            canonical
                        ),
                    });
                    return None;
                }
                return Some((
                    ExprNode::Sym {
                        name: decl.name.clone(),
                        role: decl.role,
                        slot_index: decl.slot_index,
                    },
                    Some(decl.dimension.exponents),
                ));
            }
            diags.push(ScopeDiagnostic {
                code: "UNRESOLVED_SYMBOL".into(),
                field: field.into(),
                symbol: Some(canonical),
                message: "Resolved symbol missing from scope map".into(),
            });
            None
        }
        RawExpr::Unary { op, arg } => {
            let (a_node, a_dim) = lower_and_check_expr(
                arg,
                field,
                archetype,
                allow_observables,
                sym_map,
                let_map,
                resolver,
                check_dimensions,
                resolved_aliases,
                diags,
            )?;
            let out_dim: OptDim = match op {
                UnaryOp::Neg | UnaryOp::Abs => a_dim,
                UnaryOp::Sqrt => a_dim.map(|d| dim_scale(&d, Rat::new(1, 2))),
                UnaryOp::Exp | UnaryOp::Log | UnaryOp::Sin | UnaryOp::Cos | UnaryOp::Tanh => {
                    if check_dimensions {
                        if let Some(d) = a_dim {
                            if !dim_is_zero(&d) {
                                diags.push(ScopeDiagnostic {
                                    code: "DIMENSIONAL_INCONSISTENCY".into(),
                                    field: field.into(),
                                    symbol: None,
                                    message: format!("Transcendental {:?} requires dimensionless argument", op),
                                });
                            }
                        }
                    }
                    Some([(0, 1); 7])
                }
                UnaryOp::Step => Some([(0, 1); 7]),
            };
            Some((
                ExprNode::Unary {
                    op: *op,
                    arg: Box::new(a_node),
                },
                out_dim,
            ))
        }
        RawExpr::Binary { op, lhs, rhs } => {
            let (l_node, l_dim) = lower_and_check_expr(
                lhs,
                field,
                archetype,
                allow_observables,
                sym_map,
                let_map,
                resolver,
                check_dimensions,
                resolved_aliases,
                diags,
            )?;
            let (r_node, r_dim) = lower_and_check_expr(
                rhs,
                field,
                archetype,
                allow_observables,
                sym_map,
                let_map,
                resolver,
                check_dimensions,
                resolved_aliases,
                diags,
            )?;
            let out_dim: OptDim = match op {
                BinaryOp::Add | BinaryOp::Sub | BinaryOp::Min | BinaryOp::Max => {
                    merge_additive_dim(
                        l_dim,
                        r_dim,
                        check_dimensions,
                        field,
                        &format!("Binary {:?} operand SI dimensions disagree", op),
                        diags,
                    )
                }
                BinaryOp::Mul => match (l_dim, r_dim) {
                    (None, None) => None,
                    (Some(ld), None) => Some(ld),
                    (None, Some(rd)) => Some(rd),
                    (Some(ld), Some(rd)) => Some(dim_add(&ld, &rd)),
                },
                BinaryOp::Div => match (l_dim, r_dim) {
                    (None, None) => None,
                    (Some(ld), None) => Some(ld),
                    (None, Some(rd)) => Some(dim_sub(&[(0, 1); 7], &rd)),
                    (Some(ld), Some(rd)) => Some(dim_sub(&ld, &rd)),
                },
                BinaryOp::Pow => {
                    if check_dimensions {
                        if let Some(rd) = r_dim {
                            if !dim_is_zero(&rd) {
                                diags.push(ScopeDiagnostic {
                                    code: "DIMENSIONAL_INCONSISTENCY".into(),
                                    field: field.into(),
                                    symbol: None,
                                    message: "Power exponent must be dimensionless".into(),
                                });
                            }
                        }
                    }
                    if let ExprNode::Const { value } = r_node {
                        if value.fract() == 0.0 && value.abs() <= 16.0 {
                            l_dim.map(|ld| dim_scale(&ld, Rat::new(value as i32, 1)))
                        } else if (value * 2.0).fract() == 0.0 && value.abs() <= 16.0 {
                            l_dim.map(|ld| dim_scale(&ld, Rat::new((value * 2.0) as i32, 2)))
                        } else {
                            Some([(0, 1); 7])
                        }
                    } else {
                        Some([(0, 1); 7])
                    }
                }
            };
            Some((
                ExprNode::Binary {
                    op: *op,
                    lhs: Box::new(l_node),
                    rhs: Box::new(r_node),
                },
                out_dim,
            ))
        }
        RawExpr::IfThenElse {
            cond_op,
            lhs,
            rhs,
            then_branch,
            else_branch,
        } => {
            let (cl, cl_dim) = lower_and_check_expr(
                lhs, field, archetype, allow_observables, sym_map, let_map, resolver,
                check_dimensions, resolved_aliases, diags,
            )?;
            let (cr, cr_dim) = lower_and_check_expr(
                rhs, field, archetype, allow_observables, sym_map, let_map, resolver,
                check_dimensions, resolved_aliases, diags,
            )?;
            let (tb, t_dim) = lower_and_check_expr(
                then_branch, field, archetype, allow_observables, sym_map, let_map, resolver,
                check_dimensions, resolved_aliases, diags,
            )?;
            let (eb, e_dim) = lower_and_check_expr(
                else_branch, field, archetype, allow_observables, sym_map, let_map, resolver,
                check_dimensions, resolved_aliases, diags,
            )?;
            let _ = merge_additive_dim(
                cl_dim,
                cr_dim,
                check_dimensions,
                field,
                "if() condition operands have mismatched SI dimensions",
                diags,
            );
            let branch_dim = merge_additive_dim(
                t_dim,
                e_dim,
                check_dimensions,
                field,
                "if() then/else branches have mismatched SI dimensions",
                diags,
            );
            Some((
                ExprNode::IfThenElse {
                    cond_op: *cond_op,
                    lhs: Box::new(cl),
                    rhs: Box::new(cr),
                    then_branch: Box::new(tb),
                    else_branch: Box::new(eb),
                },
                branch_dim,
            ))
        }
    }
}

/// Run full Pass 3 scope-graph checking, Kahn let-DAG ordering, 4-tier injective alias resolution,
/// `Chk-ODE-RHS` enforcement, and bidirectional Q^7 SI dimensional checking.
pub fn check_scope_and_lower(req: &ScopeCheckRequest) -> ScopeCheckResponse {
    let mut diags: Vec<ScopeDiagnostic> = Vec::new();
    let mut resolved_aliases: BTreeMap<String, String> = BTreeMap::new();

    let sym_map: HashMap<String, &SymbolDecl> = req
        .symbols
        .iter()
        .map(|s| (s.name.clone(), s))
        .collect();

    // Check duplicate let names or shadowing of S_frozen.
    let mut let_raw: BTreeMap<String, (&RawLetDraft, RawExpr)> = BTreeMap::new();
    let mut let_deps: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let let_names_set: HashSet<String> = req.let_bindings.iter().map(|l| l.name.clone()).collect();

    for lb in &req.let_bindings {
        if sym_map.contains_key(&lb.name) {
            diags.push(ScopeDiagnostic {
                code: "LET_SHADOWS_FROZEN".into(),
                field: format!("let_bindings.{}", lb.name),
                symbol: Some(lb.name.clone()),
                message: format!("let_binding {:?} shadows a frozen Pass 1 symbol", lb.name),
            });
            continue;
        }
        if let_raw.contains_key(&lb.name) {
            diags.push(ScopeDiagnostic {
                code: "DUPLICATE_LET_BINDING".into(),
                field: format!("let_bindings.{}", lb.name),
                symbol: Some(lb.name.clone()),
                message: format!("Duplicate let_binding {:?}", lb.name),
            });
            continue;
        }
        match parse_expr(&lb.expr_text) {
            Ok(ast) => {
                let mut idents = BTreeSet::new();
                collect_raw_idents(&ast, &mut idents);
                let deps: Vec<String> = idents
                    .into_iter()
                    .filter(|id| let_names_set.contains(id))
                    .collect();
                let_deps.insert(lb.name.clone(), deps);
                let_raw.insert(lb.name.clone(), (lb, ast));
            }
            Err(err) => {
                diags.push(ScopeDiagnostic {
                    code: "PARSE_ERROR".into(),
                    field: format!("let_bindings.{}", lb.name),
                    symbol: Some(lb.name.clone()),
                    message: err,
                });
            }
        }
    }

    // Build resolver for S_frozen + S_let (for dynamics & let_bindings) and S_obs (for observables & obligations).
    let mut all_declared: Vec<String> = req.symbols.iter().map(|s| s.name.clone()).collect();
    all_declared.extend(let_raw.keys().cloned());
    let resolver = InjectiveAliasResolver::new(&all_declared);

    let mut let_map: HashMap<String, (ExprNode, OptDim)> = HashMap::new();
    let mut resolved_lets: Vec<ResolvedLet> = Vec::new();
    let all_dimensionless = req.symbols.iter().all(|s| dim_is_zero(&s.dimension.exponents));
    let time_dim: [(i32, i32); 7] = if all_dimensionless {
        [(0, 1); 7]
    } else {
        [(0, 1), (0, 1), (1, 1), (0, 1), (0, 1), (0, 1), (0, 1)]
    };

    match sort_let_dag(&let_deps) {
        Ok(order) => {
            for name in order {
                let (draft, ast) = &let_raw[&name];
                if let Some((node, dim)) = lower_and_check_expr(
                    ast,
                    &format!("let_bindings.{}", name),
                    req.archetype,
                    false, // Chk-ODE-RHS: DERIVED_OBSERVABLE banned in let_bindings
                    &sym_map,
                    &let_map,
                    &resolver,
                    req.check_dimensions,
                    &mut resolved_aliases,
                    &mut diags,
                ) {
                    let concrete_dim = dim.unwrap_or([(0, 1); 7]);
                    let si_dim = draft
                        .dimension
                        .clone()
                        .unwrap_or_else(|| SiDimension {
                            exponents: concrete_dim,
                            unit_label: "1".into(),
                        });
                    let_map.insert(name.clone(), (node.clone(), dim));
                    resolved_lets.push(ResolvedLet {
                        name,
                        dimension: si_dim,
                        expr: node,
                        source_text: draft.expr_text.clone(),
                    });
                }
            }
        }
        Err(cycles) => {
            for cyc in cycles {
                diags.push(ScopeDiagnostic {
                    code: "CYCLIC_LET_DAG".into(),
                    field: "let_bindings".into(),
                    symbol: cyc.first().cloned(),
                    message: format!("Cyclic dependency in let_bindings: {}", cyc.join(" -> ")),
                });
            }
        }
    }

    // Check dynamics_or_residuals (DERIVED_OBSERVABLE banned!).
    let mut resolved_dyn: Vec<ResolvedEquation> = Vec::new();
    for eq in &req.dynamics_or_residuals {
        let Some(target_decl) = sym_map.get(&eq.target) else {
            diags.push(ScopeDiagnostic {
                code: "UNRESOLVED_TARGET".into(),
                field: format!("dynamics_or_residuals.{}", eq.target),
                symbol: Some(eq.target.clone()),
                message: format!("Equation target {:?} is not declared in symbols", eq.target),
            });
            continue;
        };
        if eq.reduction.is_some() {
            diags.push(ScopeDiagnostic {
                code: "REDUCTION_NOT_ALLOWED_IN_DYNAMICS".into(),
                field: format!("dynamics_or_residuals.{}", eq.target),
                symbol: Some(eq.target.clone()),
                message: format!("Dynamics equation target {:?} may not specify a trajectory reduction", eq.target),
            });
            continue;
        }
        if req.archetype == Archetype::OdePde && target_decl.role != SlotRole::State {
            diags.push(ScopeDiagnostic {
                code: "INVALID_DYNAMICS_TARGET".into(),
                field: format!("dynamics_or_residuals.{}", eq.target),
                symbol: Some(eq.target.clone()),
                message: format!("ODE_PDE dynamics target {:?} must have role STATE", eq.target),
            });
            continue;
        }
        match parse_expr(&eq.expr_text) {
            Ok(ast) => {
                if let Some((node, rhs_dim)) = lower_and_check_expr(
                    &ast,
                    &format!("dynamics_or_residuals.{}", eq.target),
                    req.archetype,
                    false, // Chk-ODE-RHS strictly bans DERIVED_OBSERVABLE in dynamics!
                    &sym_map,
                    &let_map,
                    &resolver,
                    req.check_dimensions,
                    &mut resolved_aliases,
                    &mut diags,
                ) {
                    if req.check_dimensions {
                        let expected_dim = if req.archetype == Archetype::OdePde && target_decl.role == SlotRole::State {
                            dim_sub(&target_decl.dimension.exponents, &time_dim)
                        } else {
                            target_decl.dimension.exponents
                        };
                        let is_literal_zero = matches!(node, ExprNode::Const { value } if value == 0.0);
                        if !is_literal_zero {
                            if let Some(rd) = rhs_dim {
                                if !dim_eq(&rd, &expected_dim) {
                                    diags.push(ScopeDiagnostic {
                                        code: "DIMENSIONAL_INCONSISTENCY".into(),
                                        field: format!("dynamics_or_residuals.{}", eq.target),
                                        symbol: Some(eq.target.clone()),
                                        message: format!(
                                            "RHS dimension {:?} does not match target {:?} expected dimension {:?}",
                                            rd, eq.target, expected_dim
                                        ),
                                    });
                                }
                            }
                        }
                    }
                    resolved_dyn.push(ResolvedEquation {
                        target: target_decl.name.clone(),
                        target_slot: target_decl.slot_index,
                        expr: node,
                        source_text: eq.expr_text.clone(),
                        reduction: None,
                    });
                }
            }
            Err(err) => {
                diags.push(ScopeDiagnostic {
                    code: "PARSE_ERROR".into(),
                    field: format!("dynamics_or_residuals.{}", eq.target),
                    symbol: Some(eq.target.clone()),
                    message: err,
                });
            }
        }
    }

    // Check observables (DERIVED_OBSERVABLE allowed in S_obs).
    let mut resolved_obs: Vec<ResolvedEquation> = Vec::new();
    for eq in &req.observables {
        let Some(target_decl) = sym_map.get(&eq.target) else {
            diags.push(ScopeDiagnostic {
                code: "UNRESOLVED_TARGET".into(),
                field: format!("observables.{}", eq.target),
                symbol: Some(eq.target.clone()),
                message: format!("Observable target {:?} is not declared in symbols", eq.target),
            });
            continue;
        };
        match parse_expr(&eq.expr_text) {
            Ok(ast) => {
                if let Some((node, rhs_dim)) = lower_and_check_expr(
                    &ast,
                    &format!("observables.{}", eq.target),
                    req.archetype,
                    true,
                    &sym_map,
                    &let_map,
                    &resolver,
                    req.check_dimensions,
                    &mut resolved_aliases,
                    &mut diags,
                ) {
                    if req.check_dimensions {
                        let is_literal_zero = matches!(node, ExprNode::Const { value } if value == 0.0);
                        if !is_literal_zero {
                            if let Some(rd) = rhs_dim {
                                if !dim_eq(&rd, &target_decl.dimension.exponents) {
                                    diags.push(ScopeDiagnostic {
                                        code: "DIMENSIONAL_INCONSISTENCY".into(),
                                        field: format!("observables.{}", eq.target),
                                        symbol: Some(eq.target.clone()),
                                        message: format!(
                                            "Observable RHS dimension {:?} does not match target {:?} dimension {:?}",
                                            rd, eq.target, target_decl.dimension.exponents
                                        ),
                                    });
                                }
                            }
                        }
                    }
                    resolved_obs.push(ResolvedEquation {
                        target: target_decl.name.clone(),
                        target_slot: target_decl.slot_index,
                        expr: node,
                        source_text: eq.expr_text.clone(),
                        reduction: eq.reduction,
                    });
                }
            }
            Err(err) => {
                diags.push(ScopeDiagnostic {
                    code: "PARSE_ERROR".into(),
                    field: format!("observables.{}", eq.target),
                    symbol: Some(eq.target.clone()),
                    message: err,
                });
            }
        }
    }

    // Check obligations (DERIVED_OBSERVABLE allowed).
    let mut resolved_obs_list: Vec<ResolvedObligation> = Vec::new();
    for ob in &req.obligations {
        let lhs_res = parse_expr(&ob.lhs_text).and_then(|ast| {
            lower_and_check_expr(
                &ast,
                &format!("obligations.{}.lhs", ob.id),
                req.archetype,
                true,
                &sym_map,
                &let_map,
                &resolver,
                req.check_dimensions,
                &mut resolved_aliases,
                &mut diags,
            )
            .ok_or_else(|| "lhs resolution failed".into())
        });
        let rhs_res = parse_expr(&ob.rhs_text).and_then(|ast| {
            lower_and_check_expr(
                &ast,
                &format!("obligations.{}.rhs", ob.id),
                req.archetype,
                true,
                &sym_map,
                &let_map,
                &resolver,
                req.check_dimensions,
                &mut resolved_aliases,
                &mut diags,
            )
            .ok_or_else(|| "rhs resolution failed".into())
        });
        if let (Ok((lhs_node, lhs_dim)), Ok((rhs_node, rhs_dim))) = (lhs_res, rhs_res) {
            if req.check_dimensions {
                let _ = merge_additive_dim(
                    lhs_dim,
                    rhs_dim,
                    true,
                    &format!("obligations.{}", ob.id),
                    "Obligation LHS and RHS SI dimensions disagree",
                    &mut diags,
                );
            }
            resolved_obs_list.push(ResolvedObligation {
                id: ob.id.clone(),
                quantifier: ob.quantifier.clone(),
                lhs_expr: lhs_node,
                comparator: ob.comparator.clone(),
                rhs_expr: rhs_node,
                tolerance: ob.tolerance,
                epistemic_origin: ob.epistemic_origin.clone(),
                source_text: ob.source_text.clone(),
            });
        }
    }

    ScopeCheckResponse {
        ok: diags.is_empty(),
        diagnostics: diags,
        resolved_aliases,
        let_bindings: resolved_lets,
        dynamics_or_residuals: resolved_dyn,
        observables: resolved_obs,
        obligations: resolved_obs_list,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn twenty_case_injective_alias_collision_suite() {
        // Base symbols: simultaneous T/t, C/c, C_m, Delta_Psi, k_1, g_leak, g_unc, ATP, ADP, NADH
        let declared: Vec<String> = vec![
            "T", "t", "C", "c", "C_m", "Delta_Psi", "k_1", "g_leak", "g_unc", "ATP", "ADP", "NADH",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        let r = InjectiveAliasResolver::new(&declared);

        // 1-4: Exact matches (Tier 0)
        assert_eq!(r.resolve("T"), AliasResolution::Exact { canonical: "T".into() });
        assert_eq!(r.resolve("t"), AliasResolution::Exact { canonical: "t".into() });
        assert_eq!(r.resolve("C_m"), AliasResolution::Exact { canonical: "C_m".into() });
        assert_eq!(r.resolve("Delta_Psi"), AliasResolution::Exact { canonical: "Delta_Psi".into() });

        // 5-8: Tier 1 Unicode Greek & subscript transliteration
        assert_eq!(r.resolve("ΔΨ"), AliasResolution::Resolved { canonical: "Delta_Psi".into(), tier: 1 });
        assert_eq!(r.resolve("k₁"), AliasResolution::Resolved { canonical: "k_1".into(), tier: 1 });
        assert_eq!(r.resolve("Δ_Ψ"), AliasResolution::Resolved { canonical: "Delta_Psi".into(), tier: 1 });
        assert_eq!(r.resolve("Delta_Ψ"), AliasResolution::Resolved { canonical: "Delta_Psi".into(), tier: 1 });

        // 9-13: Tier 2 case-sensitive underscore stripping (never conflating T/t or C/c)
        assert_eq!(r.resolve("Cm"), AliasResolution::Resolved { canonical: "C_m".into(), tier: 2 });
        assert_eq!(r.resolve("gleak"), AliasResolution::Resolved { canonical: "g_leak".into(), tier: 2 });
        assert_eq!(r.resolve("gunc"), AliasResolution::Resolved { canonical: "g_unc".into(), tier: 2 });
        assert_eq!(r.resolve("k1"), AliasResolution::Resolved { canonical: "k_1".into(), tier: 2 });
        assert_eq!(r.resolve("t_"), AliasResolution::Resolved { canonical: "t".into(), tier: 1 });

        // 14-16: Tier 3 Delta/d prefix & case-insensitive resolution
        assert_eq!(r.resolve("dPsi"), AliasResolution::Resolved { canonical: "Delta_Psi".into(), tier: 3 });
        assert_eq!(r.resolve("delta_psi"), AliasResolution::Resolved { canonical: "Delta_Psi".into(), tier: 3 });
        assert_eq!(r.resolve("atp"), AliasResolution::Resolved { canonical: "ATP".into(), tier: 3 });

        // 17-18: Unresolved symbols stay UNRESOLVED
        assert!(matches!(r.resolve("R_gas"), AliasResolution::Unresolved { .. }));
        assert!(matches!(r.resolve("F_far"), AliasResolution::Unresolved { .. }));

        // 19-20: Adversarial collision pair (C_m and Cm both declared) -> AMBIGUOUS_COLLISION on c_m / CM
        let adv: Vec<String> = vec!["C_m", "Cm", "T", "t"].into_iter().map(String::from).collect();
        let r_adv = InjectiveAliasResolver::new(&adv);
        assert!(matches!(r_adv.resolve("c_m"), AliasResolution::AmbiguousCollision { .. }));
        assert!(matches!(r_adv.resolve("CM"), AliasResolution::AmbiguousCollision { .. }));
    }

    #[test]
    fn kahn_and_tarjan_detect_order_and_cycles() {
        let mut dag = BTreeMap::new();
        dag.insert("b".into(), vec!["a".into()]);
        dag.insert("a".into(), vec![]);
        assert_eq!(sort_let_dag(&dag).unwrap(), vec!["a".to_string(), "b".to_string()]);

        let mut cyc = BTreeMap::new();
        cyc.insert("x".into(), vec!["y".into()]);
        cyc.insert("y".into(), vec!["x".into()]);
        let err = sort_let_dag(&cyc).unwrap_err();
        assert_eq!(err.len(), 1);
        assert_eq!(err[0].len(), 2);
    }

    #[test]
    fn chk_ode_rhs_bans_derived_observable_in_dynamics() {
        let req = ScopeCheckRequest {
            archetype: Archetype::OdePde,
            symbols: vec![
                SymbolDecl {
                    name: "x".into(), display_symbol: "x".into(), role: SlotRole::State,
                    slot_index: 0, dimension: SiDimension::dimensionless(), initial_or_default: 1.0,
                    domain: None, discrete_values: None, strictly_positive: false, pool_id: None, description: "".into(),
                },
                SymbolDecl {
                    name: "obs_y".into(), display_symbol: "y".into(), role: SlotRole::DerivedObservable,
                    slot_index: 0, dimension: SiDimension::dimensionless(), initial_or_default: 0.0,
                    domain: None, discrete_values: None, strictly_positive: false, pool_id: None, description: "".into(),
                },
            ],
            let_bindings: vec![],
            dynamics_or_residuals: vec![RawEquationDraft { target: "x".into(), expr_text: "-obs_y".into(), reduction: None }],
            observables: vec![RawEquationDraft { target: "obs_y".into(), expr_text: "2*x".into(), reduction: None }],
            obligations: vec![],
            check_dimensions: false,
        };
        let res = check_scope_and_lower(&req);
        assert!(!res.ok);
        assert!(res.diagnostics.iter().any(|d| d.code == "DERIVED_OBSERVABLE_IN_DYNAMICS"));
    }
}
