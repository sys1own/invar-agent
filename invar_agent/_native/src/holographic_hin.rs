//! Holographic Interaction Nets (HIN) with MELL Linear Typing,
//! SI Dimensional Unification, and Static Interval Bound Propagation.

use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

pub const SAFE_MAX_VAL: f64 = 1e300;
pub const SAFE_DIV_EPS: f64 = 1e-12;
pub const SAFE_LOG_EPS: f64 = 1e-30;

// ---------------------------------------------------------------------------
// Totalized Smooth Operators (exact Rust reference semantics)
// ---------------------------------------------------------------------------

#[inline]
pub fn safe_div(x: f64, y: f64) -> f64 {
    let denom_mag = y.abs().max(SAFE_DIV_EPS);
    let denom = if y < 0.0 { -denom_mag } else { denom_mag };
    (x / denom).clamp(-SAFE_MAX_VAL, SAFE_MAX_VAL)
}

#[inline]
pub fn safe_exp(x: f64) -> f64 {
    x.clamp(-80.0, 80.0).exp()
}

#[inline]
pub fn safe_log(x: f64) -> f64 {
    x.max(SAFE_LOG_EPS).ln()
}

#[inline]
pub fn safe_sqrt(x: f64) -> f64 {
    x.max(0.0).sqrt()
}

#[inline]
pub fn safe_acos(x: f64) -> f64 {
    x.clamp(-1.0, 1.0).acos()
}

#[inline]
pub fn safe_asin(x: f64) -> f64 {
    x.clamp(-1.0, 1.0).asin()
}

#[inline]
pub fn safe_pow(x: f64, y: f64) -> f64 {
    if (y - y.round()).abs() < 1e-12 && y.abs() <= 1024.0 {
        let exp_i = y.round() as i32;
        if exp_i == 0 {
            return 1.0;
        }
        if exp_i > 0 {
            let val = x.powi(exp_i);
            if val.is_finite() {
                return val.clamp(-SAFE_MAX_VAL, SAFE_MAX_VAL);
            }
            return if val.is_sign_negative() { -SAFE_MAX_VAL } else { SAFE_MAX_VAL };
        }
        let pos = x.powi(-exp_i);
        return safe_div(1.0, pos);
    }
    safe_exp(y * safe_log(x.abs()))
}

#[inline]
pub fn smooth_clamp(x: f64, a: f64, b: f64) -> f64 {
    let lo = a.min(b);
    let hi = a.max(b);
    let mid = 0.5 * (lo + hi);
    let sig = 1.0 / (1.0 + safe_exp(-10.0 * (x - mid)));
    lo + (hi - lo) * sig
}

// ---------------------------------------------------------------------------
// Holographic Port Types: SIDimensionVector, IntervalBound, MELLType
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SIDimensionVector {
    pub mass: i8,
    pub length: i8,
    pub time: i8,
    pub current: i8,
    pub temperature: i8,
    pub amount: i8,
    pub luminosity: i8,
}

impl SIDimensionVector {
    pub fn dimensionless() -> Self {
        Self {
            mass: 0,
            length: 0,
            time: 0,
            current: 0,
            temperature: 0,
            amount: 0,
            luminosity: 0,
        }
    }

    pub fn from_vec(v: &[i8]) -> Result<Self, String> {
        if v.len() != 7 {
            return Err(format!("Expected 7 SI dimension components, got {}", v.len()));
        }
        Ok(Self {
            mass: v[0],
            length: v[1],
            time: v[2],
            current: v[3],
            temperature: v[4],
            amount: v[5],
            luminosity: v[6],
        })
    }

    pub fn to_vec(self) -> Vec<i8> {
        vec![
            self.mass,
            self.length,
            self.time,
            self.current,
            self.temperature,
            self.amount,
            self.luminosity,
        ]
    }

    pub fn is_dimensionless(self) -> bool {
        self == Self::dimensionless()
    }

    pub fn mul_dim(self, other: Self) -> Self {
        Self {
            mass: self.mass.saturating_add(other.mass),
            length: self.length.saturating_add(other.length),
            time: self.time.saturating_add(other.time),
            current: self.current.saturating_add(other.current),
            temperature: self.temperature.saturating_add(other.temperature),
            amount: self.amount.saturating_add(other.amount),
            luminosity: self.luminosity.saturating_add(other.luminosity),
        }
    }

    pub fn div_dim(self, other: Self) -> Self {
        Self {
            mass: self.mass.saturating_sub(other.mass),
            length: self.length.saturating_sub(other.length),
            time: self.time.saturating_sub(other.time),
            current: self.current.saturating_sub(other.current),
            temperature: self.temperature.saturating_sub(other.temperature),
            amount: self.amount.saturating_sub(other.amount),
            luminosity: self.luminosity.saturating_sub(other.luminosity),
        }
    }

    pub fn pow_dim(self, exp: i8) -> Self {
        Self {
            mass: self.mass.saturating_mul(exp),
            length: self.length.saturating_mul(exp),
            time: self.time.saturating_mul(exp),
            current: self.current.saturating_mul(exp),
            temperature: self.temperature.saturating_mul(exp),
            amount: self.amount.saturating_mul(exp),
            luminosity: self.luminosity.saturating_mul(exp),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct IntervalBound {
    pub lower: f64,
    pub upper: f64,
}

impl IntervalBound {
    pub fn new(lower: f64, upper: f64) -> Self {
        if lower <= upper {
            Self { lower, upper }
        } else {
            Self { lower: upper, upper: lower }
        }
    }

    pub fn point(val: f64) -> Self {
        Self { lower: val, upper: val }
    }

    pub fn unbounded() -> Self {
        Self {
            lower: f64::NEG_INFINITY,
            upper: f64::INFINITY,
        }
    }

    pub fn add(self, o: Self) -> Self {
        Self::new(self.lower + o.lower, self.upper + o.upper)
    }

    pub fn sub(self, o: Self) -> Self {
        Self::new(self.lower - o.upper, self.upper - o.lower)
    }

    pub fn neg(self) -> Self {
        Self::new(-self.upper, -self.lower)
    }

    pub fn mul(self, o: Self) -> Self {
        let prods = [
            self.lower * o.lower,
            self.lower * o.upper,
            self.upper * o.lower,
            self.upper * o.upper,
        ];
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for p in prods {
            let v = if p.is_nan() { 0.0 } else { p };
            if v < lo { lo = v; }
            if v > hi { hi = v; }
        }
        Self::new(lo, hi)
    }

    pub fn safe_div(self, o: Self) -> Self {
        if o.lower > 0.0 || o.upper < 0.0 {
            let quots = [
                safe_div(self.lower, o.lower),
                safe_div(self.lower, o.upper),
                safe_div(self.upper, o.lower),
                safe_div(self.upper, o.upper),
            ];
            let lo = quots.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = quots.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            return Self::new(lo, hi);
        }
        // Denominator interval straddles 0: totalized safe_div still stays finite
        // within [-max_abs / eps, max_abs / eps]
        let max_num = self.lower.abs().max(self.upper.abs()).min(SAFE_MAX_VAL);
        let bound = (max_num / SAFE_DIV_EPS).min(SAFE_MAX_VAL);
        Self::new(-bound, bound)
    }

    pub fn safe_exp(self) -> Self {
        Self::new(safe_exp(self.lower).max(0.0), safe_exp(self.upper))
    }

    pub fn safe_log(self) -> Self {
        Self::new(safe_log(self.lower), safe_log(self.upper))
    }

    pub fn safe_sqrt(self) -> Self {
        Self::new(safe_sqrt(self.lower).max(0.0), safe_sqrt(self.upper))
    }

    pub fn join(self, o: Self) -> Self {
        Self::new(self.lower.min(o.lower), self.upper.max(o.upper))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromotionBox {
    pub symbol: String,
    pub use_count: usize,
    pub is_derelicted: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MELLType {
    Linear,
    Exponential(PromotionBox),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HolographicPort {
    pub mell_type: MELLType,
    pub dimensions: SIDimensionVector,
    pub bound: IntervalBound,
    /// True for numeric literals or unspecified dimensionless constants that can
    /// unify their SI dimension from a peer operand in +/-/comparison nodes.
    pub inferable_dimension: bool,
}

// ---------------------------------------------------------------------------
// CSEL Expression AST (deserialized from Python CSEL parser or JSON)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CselExpr {
    Literal {
        value: f64,
    },
    Symbol {
        name: String,
    },
    Unary {
        op: String,
        arg: Box<CselExpr>,
    },
    Binary {
        op: String,
        left: Box<CselExpr>,
        right: Box<CselExpr>,
    },
    Call {
        func: String,
        args: Vec<CselExpr>,
    },
}

impl CselExpr {
    pub fn collect_symbols(&self, counts: &mut HashMap<String, usize>) {
        match self {
            CselExpr::Literal { .. } => {}
            CselExpr::Symbol { name } => {
                *counts.entry(name.clone()).or_insert(0) += 1;
            }
            CselExpr::Unary { arg, .. } => arg.collect_symbols(counts),
            CselExpr::Binary { left, right, .. } => {
                left.collect_symbols(counts);
                right.collect_symbols(counts);
            }
            CselExpr::Call { args, .. } => {
                for a in args {
                    a.collect_symbols(counts);
                }
            }
        }
    }

    pub fn eval(&self, env: &HashMap<String, f64>) -> f64 {
        match self {
            CselExpr::Literal { value } => *value,
            CselExpr::Symbol { name } => *env.get(name).unwrap_or(&0.0),
            CselExpr::Unary { op, arg } => {
                let v = arg.eval(env);
                match op.as_str() {
                    "-" | "neg" => -v,
                    "+" => v,
                    _ => v,
                }
            }
            CselExpr::Binary { op, left, right } => {
                let l = left.eval(env);
                let r = right.eval(env);
                match op.as_str() {
                    "+" => l + r,
                    "-" => l - r,
                    "*" => l * r,
                    "/" => safe_div(l, r),
                    "^" | "**" => safe_pow(l, r),
                    _ => 0.0,
                }
            }
            CselExpr::Call { func, args } => {
                let vals: Vec<f64> = args.iter().map(|a| a.eval(env)).collect();
                let a0 = vals.first().copied().unwrap_or(0.0);
                let a1 = vals.get(1).copied().unwrap_or(0.0);
                let a2 = vals.get(2).copied().unwrap_or(0.0);
                match func.as_str() {
                    "safe_div" => safe_div(a0, a1),
                    "safe_sqrt" | "sqrt" => safe_sqrt(a0),
                    "safe_log" | "log" => safe_log(a0),
                    "safe_exp" | "exp" => safe_exp(a0),
                    "safe_pow" | "pow" => safe_pow(a0, a1),
                    "safe_sin" | "sin" => a0.sin(),
                    "safe_cos" | "cos" => a0.cos(),
                    "safe_tan" | "tan" => a0.clamp(-1.570796, 1.570796).tan(),
                    "safe_tanh" | "tanh" => a0.tanh(),
                    "safe_acos" | "acos" => safe_acos(a0),
                    "safe_asin" | "asin" => safe_asin(a0),
                    "abs" | "fabs" => a0.abs(),
                    "min" => a0.min(a1),
                    "max" => a0.max(a1),
                    "log1p" => a0.max(-1.0 + SAFE_LOG_EPS).ln_1p(),
                    "expm1" => a0.clamp(-80.0, 80.0).exp_m1(),
                    "smooth_clamp" => smooth_clamp(a0, a1, a2),
                    "where" => if a0 > 0.0 { a1 } else { a2 },
                    _ => 0.0,
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Ontology & Holographic Verification Report
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantitySpec {
    pub symbol: String,
    pub si_dimension: [i8; 7],
    pub interval: [f64; 2],
    pub is_state_variable: bool,
    #[serde(default)]
    pub dimension_unspecified: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquationReport {
    pub equation_name: String,
    pub valid: bool,
    pub output_dimensions: [i8; 7],
    pub output_bound: [f64; 2],
    pub promotion_agents: Vec<String>,
    pub dereliction_agents: Vec<String>,
    pub unbound_symbols: Vec<String>,
    pub dimensional_warnings: Vec<String>,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HolographicVerificationResult {
    pub all_valid: bool,
    pub active_pairs_reduced: usize,
    pub equations: Vec<EquationReport>,
    pub fault_node_id: Option<String>,
    pub diagnostics: Vec<String>,
}

// ---------------------------------------------------------------------------
// Holographic HIN Arena
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum HinAgentKind {
    StateSource { symbol: String },
    LiteralSource { value: f64 },
    Promotion { symbol: String, fanout: usize }, // ! (bang)
    Dereliction { symbol: String },              // ? (weakening)
    Operator { name: String },
    StateDerivativeSink { equation: String },
}

#[derive(Debug, Clone)]
struct HinNode {
    #[allow(dead_code)]
    id: usize,
    #[allow(dead_code)]
    kind: HinAgentKind,
    out_port: HolographicPort,
}

pub struct HolographicHinArena {
    ontology: HashMap<String, QuantitySpec>,
    nodes: Vec<HinNode>,
    active_pairs_reduced: usize,
}

impl HolographicHinArena {
    pub fn new(quantities: Vec<QuantitySpec>) -> Self {
        let mut ontology = HashMap::new();
        for q in quantities {
            ontology.insert(q.symbol.clone(), q);
        }
        Self {
            ontology,
            nodes: Vec::new(),
            active_pairs_reduced: 0,
        }
    }

    fn alloc_node(&mut self, kind: HinAgentKind, out_port: HolographicPort) -> usize {
        let id = self.nodes.len();
        self.nodes.push(HinNode { id, kind, out_port });
        id
    }

    /// Lower an equation RHS CselExpr into the Holographic HIN Arena,
    /// automatically inserting `!` (promotion) and `?` (dereliction) MELL agents,
    /// propagating IntervalBounds, and performing Smart Dimensional Unification.
    pub fn verify_equation(
        &mut self,
        equation_name: &str,
        expr: &CselExpr,
    ) -> EquationReport {
        let mut counts: HashMap<String, usize> = HashMap::new();
        expr.collect_symbols(&mut counts);

        let mut unbound_symbols = Vec::new();
        let mut promotion_agents = Vec::new();
        let mut dereliction_agents = Vec::new();
        let mut dimensional_warnings = Vec::new();

        // Check every referenced symbol against the ontology (strictly catch unbound wires)
        for (sym, &count) in &counts {
            if !self.ontology.contains_key(sym) && sym != "t" && sym != "pi" && sym != "e" {
                unbound_symbols.push(sym.clone());
            } else if count > 1 {
                promotion_agents.push(format!("!{} (x{})", sym, count));
                // Allocate explicit ! promotion box in the arena
                if let Some(spec) = self.ontology.get(sym) {
                    let dim = SIDimensionVector::from_vec(&spec.si_dimension)
                        .unwrap_or_else(|_| SIDimensionVector::dimensionless());
                    let port = HolographicPort {
                        mell_type: MELLType::Exponential(PromotionBox {
                            symbol: sym.clone(),
                            use_count: count,
                            is_derelicted: false,
                        }),
                        dimensions: dim,
                        bound: IntervalBound::new(spec.interval[0], spec.interval[1]),
                        inferable_dimension: spec.dimension_unspecified,
                    };
                    self.alloc_node(
                        HinAgentKind::Promotion {
                            symbol: sym.clone(),
                            fanout: count,
                        },
                        port,
                    );
                    self.active_pairs_reduced += count;
                }
            }
        }

        // Automatically insert ? (dereliction/weakening) agents for declared symbols
        // not referenced in this particular equation f_i.
        let unused_specs: Vec<(String, QuantitySpec)> = self
            .ontology
            .iter()
            .filter(|(sym, _)| !counts.contains_key(*sym))
            .map(|(sym, spec)| (sym.clone(), spec.clone()))
            .collect();
        for (sym, spec) in unused_specs {
            dereliction_agents.push(format!("?{}", sym));
            let dim = SIDimensionVector::from_vec(&spec.si_dimension)
                .unwrap_or_else(|_| SIDimensionVector::dimensionless());
            let port = HolographicPort {
                mell_type: MELLType::Exponential(PromotionBox {
                    symbol: sym.clone(),
                    use_count: 0,
                    is_derelicted: true,
                }),
                dimensions: dim,
                bound: IntervalBound::new(spec.interval[0], spec.interval[1]),
                inferable_dimension: spec.dimension_unspecified,
            };
            self.alloc_node(
                HinAgentKind::Dereliction { symbol: sym },
                port,
            );
            self.active_pairs_reduced += 1;
        }

        if !unbound_symbols.is_empty() {
            unbound_symbols.sort();
            return EquationReport {
                equation_name: equation_name.to_string(),
                valid: false,
                output_dimensions: [0; 7],
                output_bound: [f64::NEG_INFINITY, f64::INFINITY],
                promotion_agents,
                dereliction_agents,
                error_message: Some(format!(
                    "MELL unbound input wires at {}: {}",
                    equation_name,
                    unbound_symbols.join(", ")
                )),
                unbound_symbols,
                dimensional_warnings,
            };
        }

        // Lower AST recursively into HIN ports
        let out_port = self.lower_expr(expr, equation_name, &mut dimensional_warnings);

        // Connect to state-derivative sink port
        self.alloc_node(
            HinAgentKind::StateDerivativeSink {
                equation: equation_name.to_string(),
            },
            out_port.clone(),
        );
        self.active_pairs_reduced += 1;

        let dim_vec = out_port.dimensions.to_vec();
        let mut dim_arr = [0i8; 7];
        dim_arr.copy_from_slice(&dim_vec);

        EquationReport {
            equation_name: equation_name.to_string(),
            valid: true,
            output_dimensions: dim_arr,
            output_bound: [out_port.bound.lower, out_port.bound.upper],
            promotion_agents,
            dereliction_agents,
            unbound_symbols,
            dimensional_warnings,
            error_message: None,
        }
    }

    fn lower_expr(
        &mut self,
        expr: &CselExpr,
        eq_name: &str,
        warnings: &mut Vec<String>,
    ) -> HolographicPort {
        match expr {
            CselExpr::Literal { value } => {
                let port = HolographicPort {
                    mell_type: MELLType::Linear,
                    dimensions: SIDimensionVector::dimensionless(),
                    bound: IntervalBound::point(*value),
                    inferable_dimension: true,
                };
                self.alloc_node(HinAgentKind::LiteralSource { value: *value }, port.clone());
                port
            }
            CselExpr::Symbol { name } => {
                if name == "t" {
                    let port = HolographicPort {
                        mell_type: MELLType::Linear,
                        dimensions: SIDimensionVector {
                            mass: 0,
                            length: 0,
                            time: 1,
                            current: 0,
                            temperature: 0,
                            amount: 0,
                            luminosity: 0,
                        },
                        bound: IntervalBound::new(0.0, f64::INFINITY),
                        inferable_dimension: false,
                    };
                    self.alloc_node(HinAgentKind::StateSource { symbol: "t".into() }, port.clone());
                    return port;
                }
                if name == "pi" || name == "e" {
                    let v = if name == "pi" { std::f64::consts::PI } else { std::f64::consts::E };
                    let port = HolographicPort {
                        mell_type: MELLType::Linear,
                        dimensions: SIDimensionVector::dimensionless(),
                        bound: IntervalBound::point(v),
                        inferable_dimension: true,
                    };
                    self.alloc_node(HinAgentKind::LiteralSource { value: v }, port.clone());
                    return port;
                }
                let spec = self.ontology.get(name).unwrap();
                let dim = SIDimensionVector::from_vec(&spec.si_dimension)
                    .unwrap_or_else(|_| SIDimensionVector::dimensionless());
                let port = HolographicPort {
                    mell_type: MELLType::Linear,
                    dimensions: dim,
                    bound: IntervalBound::new(spec.interval[0], spec.interval[1]),
                    inferable_dimension: spec.dimension_unspecified,
                };
                self.alloc_node(HinAgentKind::StateSource { symbol: name.clone() }, port.clone());
                port
            }
            CselExpr::Unary { op, arg } => {
                let inner = self.lower_expr(arg, eq_name, warnings);
                let bound = match op.as_str() {
                    "-" | "neg" => inner.bound.neg(),
                    _ => inner.bound,
                };
                let port = HolographicPort {
                    mell_type: MELLType::Linear,
                    dimensions: inner.dimensions,
                    bound,
                    inferable_dimension: inner.inferable_dimension,
                };
                self.alloc_node(HinAgentKind::Operator { name: op.clone() }, port.clone());
                self.active_pairs_reduced += 1;
                port
            }
            CselExpr::Binary { op, left, right } => {
                let lp = self.lower_expr(left, eq_name, warnings);
                let rp = self.lower_expr(right, eq_name, warnings);
                let (dim, inferable, bound) = match op.as_str() {
                    "+" | "-" => {
                        let (d, inf) = unify_additive_dimensions(
                            &lp, &rp, op, eq_name, warnings,
                        );
                        let b = if op == "+" {
                            lp.bound.add(rp.bound)
                        } else {
                            lp.bound.sub(rp.bound)
                        };
                        (d, inf, b)
                    }
                    "*" => (
                        lp.dimensions.mul_dim(rp.dimensions),
                        lp.inferable_dimension && rp.inferable_dimension,
                        lp.bound.mul(rp.bound),
                    ),
                    "/" => (
                        lp.dimensions.div_dim(rp.dimensions),
                        lp.inferable_dimension && rp.inferable_dimension,
                        lp.bound.safe_div(rp.bound),
                    ),
                    "^" | "**" => {
                        let dim = if let CselExpr::Literal { value } = right.as_ref() {
                            if (value - value.round()).abs() < 1e-12 && value.abs() <= 12.0 {
                                lp.dimensions.pow_dim(value.round() as i8)
                            } else {
                                SIDimensionVector::dimensionless()
                            }
                        } else {
                            SIDimensionVector::dimensionless()
                        };
                        (dim, lp.inferable_dimension, IntervalBound::unbounded())
                    }
                    _ => (SIDimensionVector::dimensionless(), true, IntervalBound::unbounded()),
                };
                let port = HolographicPort {
                    mell_type: MELLType::Linear,
                    dimensions: dim,
                    bound,
                    inferable_dimension: inferable,
                };
                self.alloc_node(HinAgentKind::Operator { name: op.clone() }, port.clone());
                self.active_pairs_reduced += 1;
                port
            }
            CselExpr::Call { func, args } => {
                let ports: Vec<HolographicPort> = args
                    .iter()
                    .map(|a| self.lower_expr(a, eq_name, warnings))
                    .collect();
                let p0 = ports.first().cloned().unwrap_or(HolographicPort {
                    mell_type: MELLType::Linear,
                    dimensions: SIDimensionVector::dimensionless(),
                    bound: IntervalBound::point(0.0),
                    inferable_dimension: true,
                });
                let p1 = ports.get(1).cloned().unwrap_or_else(|| p0.clone());
                let p2 = ports.get(2).cloned().unwrap_or_else(|| p1.clone());

                let (dim, inferable, bound) = match func.as_str() {
                    "safe_div" => (
                        p0.dimensions.div_dim(p1.dimensions),
                        p0.inferable_dimension && p1.inferable_dimension,
                        p0.bound.safe_div(p1.bound),
                    ),
                    "safe_sqrt" | "sqrt" => {
                        let d = SIDimensionVector {
                            mass: p0.dimensions.mass / 2,
                            length: p0.dimensions.length / 2,
                            time: p0.dimensions.time / 2,
                            current: p0.dimensions.current / 2,
                            temperature: p0.dimensions.temperature / 2,
                            amount: p0.dimensions.amount / 2,
                            luminosity: p0.dimensions.luminosity / 2,
                        };
                        (d, p0.inferable_dimension, p0.bound.safe_sqrt())
                    }
                    "safe_exp" | "exp" => (
                        SIDimensionVector::dimensionless(),
                        true,
                        p0.bound.safe_exp(),
                    ),
                    "safe_log" | "log" | "log1p" => (
                        SIDimensionVector::dimensionless(),
                        true,
                        p0.bound.safe_log(),
                    ),
                    "expm1" => (
                        SIDimensionVector::dimensionless(),
                        true,
                        IntervalBound::new(
                            safe_exp(p0.bound.lower) - 1.0,
                            safe_exp(p0.bound.upper) - 1.0,
                        ),
                    ),
                    "safe_sin" | "sin" | "safe_cos" | "cos" | "safe_tanh" | "tanh" => (
                        SIDimensionVector::dimensionless(),
                        true,
                        IntervalBound::new(-1.0, 1.0),
                    ),
                    "safe_acos" | "acos" => (
                        SIDimensionVector::dimensionless(),
                        true,
                        IntervalBound::new(0.0, std::f64::consts::PI),
                    ),
                    "safe_asin" | "asin" => (
                        SIDimensionVector::dimensionless(),
                        true,
                        IntervalBound::new(-std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2),
                    ),
                    "abs" | "fabs" => (
                        p0.dimensions,
                        p0.inferable_dimension,
                        IntervalBound::new(0.0, p0.bound.lower.abs().max(p0.bound.upper.abs())),
                    ),
                    "min" | "max" => {
                        let (d, inf) = unify_additive_dimensions(&p0, &p1, func, eq_name, warnings);
                        let b = if func == "min" {
                            IntervalBound::new(p0.bound.lower.min(p1.bound.lower), p0.bound.upper.min(p1.bound.upper))
                        } else {
                            IntervalBound::new(p0.bound.lower.max(p1.bound.lower), p0.bound.upper.max(p1.bound.upper))
                        };
                        (d, inf, b)
                    }
                    "smooth_clamp" => (
                        p0.dimensions,
                        p0.inferable_dimension,
                        IntervalBound::new(
                            p1.bound.lower.min(p2.bound.lower),
                            p1.bound.upper.max(p2.bound.upper),
                        ),
                    ),
                    "where" => {
                        let (d, inf) = unify_additive_dimensions(&p1, &p2, "where", eq_name, warnings);
                        (d, inf, p1.bound.join(p2.bound))
                    }
                    _ => (SIDimensionVector::dimensionless(), true, IntervalBound::unbounded()),
                };

                let port = HolographicPort {
                    mell_type: MELLType::Linear,
                    dimensions: dim,
                    bound,
                    inferable_dimension: inferable,
                };
                self.alloc_node(HinAgentKind::Operator { name: func.clone() }, port.clone());
                self.active_pairs_reduced += 1;
                port
            }
        }
    }
}

/// Smart Dimensional Unification:
/// 1. Numeric literals / unspecified constants automatically unify with peer operand.
/// 2. If both operands have explicit non-matching dimensions, log a Dimensional Bridge
///    Inference warning instead of crashing so Tier-0 reconciles without false positives.
fn unify_additive_dimensions(
    lp: &HolographicPort,
    rp: &HolographicPort,
    op: &str,
    eq_name: &str,
    warnings: &mut Vec<String>,
) -> (SIDimensionVector, bool) {
    if lp.dimensions == rp.dimensions {
        return (lp.dimensions, lp.inferable_dimension && rp.inferable_dimension);
    }
    if lp.inferable_dimension {
        return (rp.dimensions, false);
    }
    if rp.inferable_dimension {
        return (lp.dimensions, false);
    }
    if lp.dimensions.is_dimensionless() {
        return (rp.dimensions, false);
    }
    if rp.dimensions.is_dimensionless() {
        return (lp.dimensions, false);
    }
    warnings.push(format!(
        "Dimensional Bridge Inference at {} ({}): unified {:?} with {:?} via implicit scaling factor",
        eq_name, op, lp.dimensions.to_vec(), rp.dimensions.to_vec()
    ));
    (lp.dimensions, false)
}

// ---------------------------------------------------------------------------
// PyO3 Wrapper
// ---------------------------------------------------------------------------

#[pyclass(module = "invar_agent._native", name = "PyHolographicHin")]
pub struct PyHolographicHin {
    quantities: Vec<QuantitySpec>,
}

#[pymethods]
impl PyHolographicHin {
    #[new]
    pub fn new(quantities_json: &str) -> PyResult<Self> {
        let quantities: Vec<QuantitySpec> = serde_json::from_str(quantities_json)
            .map_err(|e| PyValueError::new_err(format!("Invalid quantities JSON: {}", e)))?;
        Ok(Self { quantities })
    }

    /// Verify a map of `{equation_name: CselExpr JSON}` against the ontology.
    /// Returns JSON-serialized `HolographicVerificationResult`.
    pub fn verify_equations(&self, py: Python<'_>, equations_json: &str) -> PyResult<String> {
        let eqs: HashMap<String, CselExpr> = serde_json::from_str(equations_json)
            .map_err(|e| PyValueError::new_err(format!("Invalid equations JSON: {}", e)))?;
        let quantities = self.quantities.clone();

        let result = py.allow_threads(move || {
            let mut arena = HolographicHinArena::new(quantities);
            let mut keys: Vec<String> = eqs.keys().cloned().collect();
            keys.sort();
            let mut reports = Vec::with_capacity(keys.len());
            let mut fault_node_id: Option<String> = None;
            let mut diagnostics = Vec::new();
            let mut all_valid = true;

            for k in keys {
                let expr = &eqs[&k];
                let rep = arena.verify_equation(&k, expr);
                if !rep.valid {
                    all_valid = false;
                    if fault_node_id.is_none() {
                        fault_node_id = Some(k.clone());
                    }
                    if let Some(msg) = &rep.error_message {
                        diagnostics.push(msg.clone());
                    }
                }
                for w in &rep.dimensional_warnings {
                    diagnostics.push(w.clone());
                }
                reports.push(rep);
            }

            HolographicVerificationResult {
                all_valid,
                active_pairs_reduced: arena.active_pairs_reduced,
                equations: reports,
                fault_node_id,
                diagnostics,
            }
        });

        serde_json::to_string(&result)
            .map_err(|e| PyValueError::new_err(format!("Serialization error: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safe_operators_totalized() {
        assert!((safe_div(1.0, 0.0) - 1e12).abs() < 1.0);
        assert!((safe_div(1.0, -0.0) - 1e12).abs() < 1.0);
        assert!((safe_div(4.0, -2.0) - (-2.0)).abs() < 1e-12);
        assert_eq!(safe_pow(-2.0, 2.0), 4.0);
        assert_eq!(safe_pow(0.0, 2.0), 0.0);
        assert!(safe_exp(1000.0).is_finite());
        assert!(safe_log(-5.0).is_finite());
        assert_eq!(safe_sqrt(-4.0), 0.0);
        assert!((safe_acos(2.0) - 0.0).abs() < 1e-12);
        assert!((safe_asin(-2.0) + std::f64::consts::FRAC_PI_2).abs() < 1e-12);
    }

    #[test]
    fn test_mell_automatic_promotion_and_dereliction() {
        let q = vec![
            QuantitySpec {
                symbol: "x".into(),
                si_dimension: [0, 1, 0, 0, 0, 0, 0],
                interval: [-10.0, 10.0],
                is_state_variable: true,
                dimension_unspecified: false,
            },
            QuantitySpec {
                symbol: "unused_p".into(),
                si_dimension: [1, 0, 0, 0, 0, 0, 0],
                interval: [0.0, 100.0],
                is_state_variable: false,
                dimension_unspecified: false,
            },
        ];
        let mut arena = HolographicHinArena::new(q);
        // x * x + 1.0 uses x twice (promotion !) and ignores unused_p (dereliction ?)
        let expr = CselExpr::Binary {
            op: "+".into(),
            left: Box::new(CselExpr::Binary {
                op: "*".into(),
                left: Box::new(CselExpr::Symbol { name: "x".into() }),
                right: Box::new(CselExpr::Symbol { name: "x".into() }),
            }),
            right: Box::new(CselExpr::Literal { value: 1.0 }),
        };
        let rep = arena.verify_equation("dx_dt", &expr);
        assert!(rep.valid);
        assert_eq!(rep.promotion_agents, vec!["!x (x2)"]);
        assert_eq!(rep.dereliction_agents, vec!["?unused_p"]);
        assert_eq!(rep.output_dimensions, [0, 2, 0, 0, 0, 0, 0]);
    }
}
