//! Unified `SciIR` (`invar.sciir.v2`) exact Rust `serde` mirror of `invar_agent/ir/schema.py`.
//!
//! All discriminated unions use internal `"kind"` tags (`#[serde(tag = "kind", rename_all = "snake_case")]`),
//! reject unknown fields (`#[serde(deny_unknown_fields)]`), and reject NaN/Inf at validation time.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Archetype {
    OdePde,
    MonteCarlo,
    FixedPoint,
    ParamSweep,
    AlgebraicClosure,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ScientificDomain {
    PhysicsThermo,
    BiochemSystems,
    ClimateEarth,
    EcologyEpidem,
    FluidContinuum,
    QuantumStatmech,
    EngineeringCtrl,
    EconComplex,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SlotRole {
    State,
    ControlParam,
    Constant,
    DerivedObservable,
    IndependentVar,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SiDimension {
    /// Rational exponents (num, den) over [M, L, T, I, Theta, N, J].
    pub exponents: [(i32, i32); 7],
    pub unit_label: String,
}

impl Default for SiDimension {
    fn default() -> Self {
        Self {
            exponents: [(0, 1); 7],
            unit_label: "1".to_string(),
        }
    }
}

impl SiDimension {
    pub fn dimensionless() -> Self {
        Self::default()
    }

    pub fn is_dimensionless(&self) -> bool {
        self.exponents.iter().all(|(n, _)| *n == 0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IntervalBound {
    pub lo: f64,
    pub hi: f64,
}

impl IntervalBound {
    pub fn validate(&self) -> Result<(), String> {
        if !self.lo.is_finite() || !self.hi.is_finite() || self.lo > self.hi {
            return Err(format!("Invalid finite interval [{}, {}]", self.lo, self.hi));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SymbolDecl {
    pub name: String,
    pub display_symbol: String,
    pub role: SlotRole,
    pub slot_index: usize,
    pub dimension: SiDimension,
    pub initial_or_default: f64,
    #[serde(default)]
    pub domain: Option<IntervalBound>,
    #[serde(default)]
    pub discrete_values: Option<Vec<f64>>,
    #[serde(default)]
    pub strictly_positive: bool,
    #[serde(default)]
    pub pool_id: Option<String>,
    #[serde(default)]
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConservationPoolSpec {
    pub pool_id: String,
    pub member_states: Vec<String>,
    pub stoichiometry: Vec<f64>,
    pub expected_total: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssumptionEntry {
    pub id: String,
    pub statement: String,
    pub source: String,
    pub falsifiable_impact: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnaryOp {
    Neg,
    Abs,
    Exp,
    Log,
    Sqrt,
    Sin,
    Cos,
    Tanh,
    Step,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Min,
    Max,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CondOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RandDist {
    Uniform01,
    Normal01,
    Exponential1,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExprNode {
    Const {
        value: f64,
    },
    Sym {
        name: String,
        role: SlotRole,
        slot_index: usize,
    },
    Unary {
        op: UnaryOp,
        arg: Box<ExprNode>,
    },
    Binary {
        op: BinaryOp,
        lhs: Box<ExprNode>,
        rhs: Box<ExprNode>,
    },
    IfThenElse {
        cond_op: CondOp,
        lhs: Box<ExprNode>,
        rhs: Box<ExprNode>,
        then_branch: Box<ExprNode>,
        else_branch: Box<ExprNode>,
    },
    Rand {
        dist: RandDist,
        draw_stream_id: u32,
    },
}

impl ExprNode {
    pub fn validate_finite(&self) -> Result<(), String> {
        match self {
            ExprNode::Const { value } => {
                if !value.is_finite() {
                    return Err("Non-finite constant in ExprNode".into());
                }
                Ok(())
            }
            ExprNode::Sym { .. } | ExprNode::Rand { .. } => Ok(()),
            ExprNode::Unary { arg, .. } => arg.validate_finite(),
            ExprNode::Binary { lhs, rhs, .. } => {
                lhs.validate_finite()?;
                rhs.validate_finite()
            }
            ExprNode::IfThenElse {
                lhs,
                rhs,
                then_branch,
                else_branch,
                ..
            } => {
                lhs.validate_finite()?;
                rhs.validate_finite()?;
                then_branch.validate_finite()?;
                else_branch.validate_finite()
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LetBinding {
    pub name: String,
    pub dimension: SiDimension,
    pub expr: ExprNode,
    pub source_text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EquationSpec {
    pub target: String,
    pub target_slot: usize,
    pub expr: ExprNode,
    pub source_text: String,
    #[serde(default)]
    pub reduction: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObligationSpec {
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
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ArchetypeConfig {
    OdePde {
        t_span: (f64, f64),
        sample_points: usize,
        rtol: f64,
        atol: f64,
        preferred_solver: String,
        enable_buckingham_scaling: bool,
    },
    MonteCarlo {
        base_seed: u64,
        replicates: usize,
        confidence_alpha: f64,
        use_crn: bool,
        #[serde(default)]
        observable_range_bounds: BTreeMap<String, IntervalBound>,
    },
    FixedPoint {
        max_newton_iters: usize,
        residual_tol: f64,
        require_krawczyk_proof: bool,
        require_stability_cert: bool,
    },
    ParamSweep {
        grid_strategy: String,
        points_per_axis: usize,
        bisection_tol: f64,
    },
    AlgebraicClosure {
        shbt_federation_enabled: bool,
        #[serde(default)]
        open_boundary_vars: Vec<String>,
    },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BuilderStudioSpec {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_target_languages")]
    pub target_languages: Vec<String>,
    #[serde(default)]
    pub export_workbench_ui: bool,
    #[serde(default)]
    pub artifact_manifest_hash: Option<String>,
}

fn default_target_languages() -> Vec<String> {
    vec!["python".to_string()]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SciIR {
    pub schema_version: String,
    pub ir_id: String,
    pub archetype: Archetype,
    pub domain: ScientificDomain,
    pub user_prompt_digest: String,
    pub assumptions_ledger: Vec<AssumptionEntry>,
    pub symbols: Vec<SymbolDecl>,
    #[serde(default)]
    pub conservation_pools: Vec<ConservationPoolSpec>,
    #[serde(default)]
    pub let_bindings: Vec<LetBinding>,
    #[serde(default)]
    pub dynamics_or_residuals: Vec<EquationSpec>,
    #[serde(default)]
    pub observables: Vec<EquationSpec>,
    pub obligations: Vec<ObligationSpec>,
    pub config: ArchetypeConfig,
    #[serde(default)]
    pub builder_studio: BuilderStudioSpec,
    #[serde(default)]
    pub ihcg_commitment_blake3: Option<String>,
}

impl SciIR {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != "invar.sciir.v2" {
            return Err(format!("Unsupported schema_version: {}", self.schema_version));
        }
        for sym in &self.symbols {
            if !sym.initial_or_default.is_finite() {
                return Err(format!("Non-finite initial_or_default for {}", sym.name));
            }
            if let Some(ref b) = sym.domain {
                b.validate()?;
            }
            if let Some(ref vals) = sym.discrete_values {
                if vals.is_empty() || vals.iter().any(|v| !v.is_finite()) {
                    return Err(format!("Invalid discrete_values for {}", sym.name));
                }
            }
        }
        for lb in &self.let_bindings {
            lb.expr.validate_finite()?;
        }
        for eq in &self.dynamics_or_residuals {
            eq.expr.validate_finite()?;
        }
        for obs in &self.observables {
            obs.expr.validate_finite()?;
        }
        for ob in &self.obligations {
            if !ob.tolerance.is_finite() || ob.tolerance <= 0.0 {
                return Err(format!("Invalid obligation tolerance for {}", ob.id));
            }
            ob.lhs_expr.validate_finite()?;
            ob.rhs_expr.validate_finite()?;
        }
        if let ArchetypeConfig::MonteCarlo { base_seed, .. } = &self.config {
            if *base_seed > 9_007_199_254_740_991 {
                return Err("MonteCarlo base_seed exceeds IEEE-754 safe integer 2^53 - 1".into());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sci_ir_roundtrip_serde() {
        let json = r#"{
            "schema_version": "invar.sciir.v2",
            "ir_id": "test_ir",
            "archetype": "ODE_PDE",
            "domain": "PHYSICS_THERMO",
            "user_prompt_digest": "abc",
            "assumptions_ledger": [],
            "symbols": [
                {"name": "x", "display_symbol": "x", "role": "STATE", "slot_index": 0, "dimension": {"exponents": [[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]], "unit_label": "1"}, "initial_or_default": 1.0}
            ],
            "obligations": [],
            "config": {"kind": "ode_pde", "t_span": [0.0, 1.0], "sample_points": 100, "rtol": 1e-7, "atol": 1e-10, "preferred_solver": "AUTO", "enable_buckingham_scaling": true}
        }"#;
        let ir: SciIR = serde_json::from_str(json).expect("valid SciIR");
        ir.validate().expect("passes validation");
        assert_eq!(ir.archetype, Archetype::OdePde);
    }
}
