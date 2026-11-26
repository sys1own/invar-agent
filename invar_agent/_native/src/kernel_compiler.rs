//! Deterministic Archetype Kernel Compiler (`kernel_compiler.rs`).
//!
//! Lowers a sealed `SciIR` (`invar.sciir.v2`) into:
//! 1. Flat `SciBufferSpec` C-ABI layout (`states`, `params`, `constants`, `observables`).
//! 2. Zero-allocation Rust interpreter + exact forward-mode symbolic Jacobian
//!    (`eval_rhs`, `eval_observables`, `eval_jacobian`, `eval_time_derivative`).
//! 3. Deterministic C++20 (`-O3 -fno-fast-math -ffp-contract=off`), Rust, and
//!    Python/NumPy kernels with strictly post-step `sci_eval_observables` and
//!    `0x03 || raw_kernel_source` BLAKE3 commitment.

use crate::buckingham_pi::{compute_diagonal_affine_scales, DiagonalAffineScaling};
use crate::his::{domain_byte_blake3_hex, DOMAIN_KERNEL_SOURCE};
use crate::sci_ir::{BinaryOp, CondOp, ExprNode, RandDist, SciIR, SlotRole, UnaryOp};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SciBufferSpec {
    pub n_states: usize,
    pub n_params: usize,
    pub n_constants: usize,
    pub n_observables: usize,
    pub n_pools: usize,
    pub state_names: Vec<String>,
    pub param_names: Vec<String>,
    pub constant_names: Vec<String>,
    pub observable_names: Vec<String>,
    pub initial_states: Vec<f64>,
    pub default_params: Vec<f64>,
    pub constants: Vec<f64>,
    pub scaling: DiagonalAffineScaling,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledKernelArtifact {
    pub language: String,
    pub compiler_flags: Vec<String>,
    pub source_code: String,
    pub kernel_commitment_blake3: String,
    pub buffer_spec: SciBufferSpec,
}

/// Deterministic stateless SplitMix64 PRNG for Monte Carlo CRN streams.
#[inline]
pub fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

#[inline]
pub fn sample_stateless_rand(base_seed: u64, replicate: u64, stream_id: u32, dist: RandDist) -> f64 {
    let key = base_seed
        .wrapping_mul(0x9e3779b97f4a7c15)
        .wrapping_add(replicate.wrapping_mul(0xbf58476d1ce4e5b9))
        .wrapping_add((stream_id as u64).wrapping_mul(0x94d049bb133111eb));
    let u1_bits = splitmix64(key);
    let u1 = ((u1_bits >> 11) as f64) * (1.0 / ((1u64 << 53) as f64));
    let u1_pos = u1.max(1e-16).min(1.0 - 1e-16);
    match dist {
        RandDist::Uniform01 => u1,
        RandDist::Exponential1 => -u1_pos.ln(),
        RandDist::Normal01 => {
            let u2_bits = splitmix64(key.wrapping_add(0x517cc1b727220a95));
            let u2 = ((u2_bits >> 11) as f64) * (1.0 / ((1u64 << 53) as f64));
            (-2.0 * u1_pos.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }
    }
}

/// Evaluation context referencing contiguous flat buffers (`SciBufferSpec` layout).
pub struct EvalContext<'a> {
    pub t: f64,
    pub states: &'a [f64],
    pub params: &'a [f64],
    pub constants: &'a [f64],
    pub observables: &'a [f64],
    pub base_seed: u64,
    pub replicate: u64,
}

/// Dual number `(val, deriv)` for exact forward-mode symbolic AD.
#[derive(Debug, Clone, Copy)]
struct Dual {
    v: f64,
    d: f64,
}

impl Dual {
    #[inline]
    fn c(v: f64) -> Self {
        Self { v, d: 0.0 }
    }
}

pub fn eval_expr(node: &ExprNode, ctx: &EvalContext<'_>) -> f64 {
    match node {
        ExprNode::Const { value } => *value,
        ExprNode::Sym {
            role, slot_index, ..
        } => match role {
            SlotRole::State => ctx.states.get(*slot_index).copied().unwrap_or(0.0),
            SlotRole::ControlParam => ctx.params.get(*slot_index).copied().unwrap_or(0.0),
            SlotRole::Constant => ctx.constants.get(*slot_index).copied().unwrap_or(0.0),
            SlotRole::DerivedObservable => ctx.observables.get(*slot_index).copied().unwrap_or(0.0),
            SlotRole::IndependentVar => ctx.t,
        },
        ExprNode::Unary { op, arg } => {
            let a = eval_expr(arg, ctx);
            match op {
                UnaryOp::Neg => -a,
                UnaryOp::Abs => a.abs(),
                UnaryOp::Exp => a.clamp(-700.0, 700.0).exp(),
                UnaryOp::Log => a.max(1e-300).ln(),
                UnaryOp::Sqrt => a.max(0.0).sqrt(),
                UnaryOp::Sin => a.sin(),
                UnaryOp::Cos => a.cos(),
                UnaryOp::Tanh => a.tanh(),
                UnaryOp::Step => {
                    if a >= 0.0 {
                        1.0
                    } else {
                        0.0
                    }
                }
            }
        }
        ExprNode::Binary { op, lhs, rhs } => {
            let l = eval_expr(lhs, ctx);
            let r = eval_expr(rhs, ctx);
            match op {
                BinaryOp::Add => l + r,
                BinaryOp::Sub => l - r,
                BinaryOp::Mul => l * r,
                BinaryOp::Div => l / r,
                BinaryOp::Pow => l.powf(r),
                BinaryOp::Min => l.min(r),
                BinaryOp::Max => l.max(r),
            }
        }
        ExprNode::IfThenElse {
            cond_op,
            lhs,
            rhs,
            then_branch,
            else_branch,
        } => {
            let l = eval_expr(lhs, ctx);
            let r = eval_expr(rhs, ctx);
            let cond = match cond_op {
                CondOp::Lt => l < r,
                CondOp::Le => l <= r,
                CondOp::Gt => l > r,
                CondOp::Ge => l >= r,
                CondOp::Eq => (l - r).abs() <= 1e-12,
            };
            if cond {
                eval_expr(then_branch, ctx)
            } else {
                eval_expr(else_branch, ctx)
            }
        }
        ExprNode::Rand {
            dist,
            draw_stream_id,
        } => sample_stateless_rand(ctx.base_seed, ctx.replicate, *draw_stream_id, *dist),
    }
}

fn eval_dual(
    node: &ExprNode,
    ctx: &EvalContext<'_>,
    wrt_role: SlotRole,
    wrt_slot: usize,
) -> Dual {
    match node {
        ExprNode::Const { value } => Dual::c(*value),
        ExprNode::Sym {
            role, slot_index, ..
        } => {
            let v = match role {
                SlotRole::State => ctx.states.get(*slot_index).copied().unwrap_or(0.0),
                SlotRole::ControlParam => ctx.params.get(*slot_index).copied().unwrap_or(0.0),
                SlotRole::Constant => ctx.constants.get(*slot_index).copied().unwrap_or(0.0),
                SlotRole::DerivedObservable => {
                    ctx.observables.get(*slot_index).copied().unwrap_or(0.0)
                }
                SlotRole::IndependentVar => ctx.t,
            };
            let d = if *role == wrt_role
                && (*role == SlotRole::IndependentVar || *slot_index == wrt_slot)
            {
                1.0
            } else {
                0.0
            };
            Dual { v, d }
        }
        ExprNode::Unary { op, arg } => {
            let a = eval_dual(arg, ctx, wrt_role, wrt_slot);
            match op {
                UnaryOp::Neg => Dual { v: -a.v, d: -a.d },
                UnaryOp::Abs => Dual {
                    v: a.v.abs(),
                    d: if a.v >= 0.0 { a.d } else { -a.d },
                },
                UnaryOp::Exp => {
                    let ev = a.v.clamp(-700.0, 700.0).exp();
                    Dual { v: ev, d: ev * a.d }
                }
                UnaryOp::Log => {
                    let av = a.v.max(1e-300);
                    Dual {
                        v: av.ln(),
                        d: a.d / av,
                    }
                }
                UnaryOp::Sqrt => {
                    let sv = a.v.max(0.0).sqrt();
                    Dual {
                        v: sv,
                        d: if sv > 1e-150 { a.d / (2.0 * sv) } else { 0.0 },
                    }
                }
                UnaryOp::Sin => Dual {
                    v: a.v.sin(),
                    d: a.v.cos() * a.d,
                },
                UnaryOp::Cos => Dual {
                    v: a.v.cos(),
                    d: -a.v.sin() * a.d,
                },
                UnaryOp::Tanh => {
                    let tv = a.v.tanh();
                    Dual {
                        v: tv,
                        d: (1.0 - tv * tv) * a.d,
                    }
                }
                UnaryOp::Step => Dual {
                    v: if a.v >= 0.0 { 1.0 } else { 0.0 },
                    d: 0.0,
                },
            }
        }
        ExprNode::Binary { op, lhs, rhs } => {
            let l = eval_dual(lhs, ctx, wrt_role, wrt_slot);
            let r = eval_dual(rhs, ctx, wrt_role, wrt_slot);
            match op {
                BinaryOp::Add => Dual {
                    v: l.v + r.v,
                    d: l.d + r.d,
                },
                BinaryOp::Sub => Dual {
                    v: l.v - r.v,
                    d: l.d - r.d,
                },
                BinaryOp::Mul => Dual {
                    v: l.v * r.v,
                    d: l.d * r.v + l.v * r.d,
                },
                BinaryOp::Div => Dual {
                    v: l.v / r.v,
                    d: (l.d * r.v - l.v * r.d) / (r.v * r.v),
                },
                BinaryOp::Pow => {
                    let pv = l.v.powf(r.v);
                    let d = if r.d == 0.0 {
                        r.v * l.v.powf(r.v - 1.0) * l.d
                    } else if l.v > 0.0 {
                        pv * (r.d * l.v.ln() + r.v * l.d / l.v)
                    } else {
                        0.0
                    };
                    Dual { v: pv, d }
                }
                BinaryOp::Min => {
                    if l.v <= r.v {
                        l
                    } else {
                        r
                    }
                }
                BinaryOp::Max => {
                    if l.v >= r.v {
                        l
                    } else {
                        r
                    }
                }
            }
        }
        ExprNode::IfThenElse {
            cond_op,
            lhs,
            rhs,
            then_branch,
            else_branch,
        } => {
            let l = eval_expr(lhs, ctx);
            let r = eval_expr(rhs, ctx);
            let cond = match cond_op {
                CondOp::Lt => l < r,
                CondOp::Le => l <= r,
                CondOp::Gt => l > r,
                CondOp::Ge => l >= r,
                CondOp::Eq => (l - r).abs() <= 1e-12,
            };
            if cond {
                eval_dual(then_branch, ctx, wrt_role, wrt_slot)
            } else {
                eval_dual(else_branch, ctx, wrt_role, wrt_slot)
            }
        }
        ExprNode::Rand {
            dist,
            draw_stream_id,
        } => Dual::c(sample_stateless_rand(
            ctx.base_seed,
            ctx.replicate,
            *draw_stream_id,
            *dist,
        )),
    }
}

pub fn build_buffer_spec(ir: &SciIR) -> SciBufferSpec {
    let mut states = Vec::new();
    let mut params = Vec::new();
    let mut constants = Vec::new();
    let mut observables = Vec::new();

    for sym in &ir.symbols {
        match sym.role {
            SlotRole::State => states.push(sym),
            SlotRole::ControlParam => params.push(sym),
            SlotRole::Constant => constants.push(sym),
            SlotRole::DerivedObservable => observables.push(sym),
            SlotRole::IndependentVar => {}
        }
    }
    states.sort_by_key(|s| s.slot_index);
    params.sort_by_key(|s| s.slot_index);
    constants.sort_by_key(|s| s.slot_index);
    observables.sort_by_key(|s| s.slot_index);

    SciBufferSpec {
        n_states: states.len(),
        n_params: params.len(),
        n_constants: constants.len(),
        n_observables: observables.len(),
        n_pools: ir.conservation_pools.len(),
        state_names: states.iter().map(|s| s.name.clone()).collect(),
        param_names: params.iter().map(|s| s.name.clone()).collect(),
        constant_names: constants.iter().map(|s| s.name.clone()).collect(),
        observable_names: observables.iter().map(|s| s.name.clone()).collect(),
        initial_states: states.iter().map(|s| s.initial_or_default).collect(),
        default_params: params.iter().map(|s| s.initial_or_default).collect(),
        constants: constants.iter().map(|s| s.initial_or_default).collect(),
        scaling: compute_diagonal_affine_scales(ir),
    }
}

/// Evaluate RHS vector `f(t, y, p, c)` into `out_rhs` (length `n_states`).
pub fn eval_ir_rhs(
    ir: &SciIR,
    t: f64,
    states: &[f64],
    params: &[f64],
    constants: &[f64],
    out_rhs: &mut [f64],
) {
    let ctx = EvalContext {
        t,
        states,
        params,
        constants,
        observables: &[],
        base_seed: 0,
        replicate: 0,
    };
    for v in out_rhs.iter_mut() {
        *v = 0.0;
    }
    for eq in &ir.dynamics_or_residuals {
        if eq.target_slot < out_rhs.len() {
            out_rhs[eq.target_slot] = eval_expr(&eq.expr, &ctx);
        }
    }
}

/// Evaluate post-step observables `g(t, y, p, c)` into `out_obs` (strictly post-step).
pub fn eval_ir_observables(
    ir: &SciIR,
    t: f64,
    states: &[f64],
    params: &[f64],
    constants: &[f64],
    out_obs: &mut [f64],
    base_seed: u64,
    replicate: u64,
) {
    for v in out_obs.iter_mut() {
        *v = 0.0;
    }
    for eq in &ir.observables {
        let ctx = EvalContext {
            t,
            states,
            params,
            constants,
            observables: out_obs,
            base_seed,
            replicate,
        };
        if eq.target_slot < out_obs.len() {
            out_obs[eq.target_slot] = eval_expr(&eq.expr, &ctx);
        }
    }
}

/// Compute exact symbolic Jacobian `J[i][j] = df_i / dy_j` via forward-mode AD.
pub fn eval_ir_jacobian(
    ir: &SciIR,
    t: f64,
    states: &[f64],
    params: &[f64],
    constants: &[f64],
) -> Vec<Vec<f64>> {
    let n = states.len();
    let mut jac = vec![vec![0.0; n]; n];
    let ctx = EvalContext {
        t,
        states,
        params,
        constants,
        observables: &[],
        base_seed: 0,
        replicate: 0,
    };
    for eq in &ir.dynamics_or_residuals {
        if eq.target_slot < n {
            for j in 0..n {
                jac[eq.target_slot][j] = eval_dual(&eq.expr, &ctx, SlotRole::State, j).d;
            }
        }
    }
    jac
}

/// Compute exact symbolic time partial derivative `df / dt` via forward-mode AD.
pub fn eval_ir_df_dt(
    ir: &SciIR,
    t: f64,
    states: &[f64],
    params: &[f64],
    constants: &[f64],
) -> Vec<f64> {
    let n = states.len();
    let mut ft = vec![0.0; n];
    let ctx = EvalContext {
        t,
        states,
        params,
        constants,
        observables: &[],
        base_seed: 0,
        replicate: 0,
    };
    for eq in &ir.dynamics_or_residuals {
        if eq.target_slot < n {
            ft[eq.target_slot] = eval_dual(&eq.expr, &ctx, SlotRole::IndependentVar, 0).d;
        }
    }
    ft
}

fn emit_cpp_expr(node: &ExprNode) -> String {
    match node {
        ExprNode::Const { value } => format!("{:.17e}", value),
        ExprNode::Sym {
            role, slot_index, ..
        } => match role {
            SlotRole::State => format!("y[{}]", slot_index),
            SlotRole::ControlParam => format!("p[{}]", slot_index),
            SlotRole::Constant => format!("c[{}]", slot_index),
            SlotRole::DerivedObservable => format!("obs[{}]", slot_index),
            SlotRole::IndependentVar => "t".to_string(),
        },
        ExprNode::Unary { op, arg } => {
            let a = emit_cpp_expr(arg);
            match op {
                UnaryOp::Neg => format!("(-({}))", a),
                UnaryOp::Abs => format!("std::fabs({})", a),
                UnaryOp::Exp => format!("std::exp(std::clamp({}, -700.0, 700.0))", a),
                UnaryOp::Log => format!("std::log(std::max({}, 1e-300))", a),
                UnaryOp::Sqrt => format!("std::sqrt(std::max({}, 0.0))", a),
                UnaryOp::Sin => format!("std::sin({})", a),
                UnaryOp::Cos => format!("std::cos({})", a),
                UnaryOp::Tanh => format!("std::tanh({})", a),
                UnaryOp::Step => format!("(({}) >= 0.0 ? 1.0 : 0.0)", a),
            }
        }
        ExprNode::Binary { op, lhs, rhs } => {
            let l = emit_cpp_expr(lhs);
            let r = emit_cpp_expr(rhs);
            match op {
                BinaryOp::Add => format!("({} + {})", l, r),
                BinaryOp::Sub => format!("({} - {})", l, r),
                BinaryOp::Mul => format!("({} * {})", l, r),
                BinaryOp::Div => format!("({} / {})", l, r),
                BinaryOp::Pow => format!("std::pow({}, {})", l, r),
                BinaryOp::Min => format!("std::min({}, {})", l, r),
                BinaryOp::Max => format!("std::max({}, {})", l, r),
            }
        }
        ExprNode::IfThenElse {
            cond_op,
            lhs,
            rhs,
            then_branch,
            else_branch,
        } => {
            let op_s = match cond_op {
                CondOp::Lt => "<",
                CondOp::Le => "<=",
                CondOp::Gt => ">",
                CondOp::Ge => ">=",
                CondOp::Eq => "==",
            };
            format!(
                "(({} {} {}) ? ({}) : ({}))",
                emit_cpp_expr(lhs),
                op_s,
                emit_cpp_expr(rhs),
                emit_cpp_expr(then_branch),
                emit_cpp_expr(else_branch)
            )
        }
        ExprNode::Rand { draw_stream_id, .. } => {
            format!("sci_rand_draw(seed, rep, {})", draw_stream_id)
        }
    }
}

/// Compile a sealed `SciIR` into a deterministic C++20 `SciBufferSpec` C-ABI kernel
/// (`-O3 -fno-fast-math -ffp-contract=off`) with `0x03 || raw_kernel_source` commitment.
pub fn compile_cpp20_kernel(ir: &SciIR) -> CompiledKernelArtifact {
    let spec = build_buffer_spec(ir);
    let mut src = String::with_capacity(2048);
    src.push_str("// Auto-generated by Invar-Agent vNext kernel_compiler.rs\n");
    src.push_str("// Strict IEEE-754 flags: -std=c++20 -O3 -fno-fast-math -ffp-contract=off\n");
    src.push_str("#include <algorithm>\n#include <cmath>\n#include <cstdint>\n\n");
    src.push_str("extern \"C\" {\n\n");
    src.push_str(
        "void sci_eval_rhs(double t, const double* y, const double* p, const double* c, double* dy) {\n",
    );
    for eq in &ir.dynamics_or_residuals {
        src.push_str(&format!(
            "    dy[{}] = {}; // {}\n",
            eq.target_slot,
            emit_cpp_expr(&eq.expr),
            eq.target
        ));
    }
    src.push_str("}\n\n");
    src.push_str(
        "void sci_eval_observables(double t, const double* y, const double* p, const double* c, double* obs) {\n",
    );
    for eq in &ir.observables {
        src.push_str(&format!(
            "    obs[{}] = {}; // {}\n",
            eq.target_slot,
            emit_cpp_expr(&eq.expr),
            eq.target
        ));
    }
    src.push_str("}\n\n} // extern \"C\"\n");

    let commitment = domain_byte_blake3_hex(DOMAIN_KERNEL_SOURCE, src.as_bytes());
    CompiledKernelArtifact {
        language: "cpp20".to_string(),
        compiler_flags: vec![
            "-std=c++20".into(),
            "-O3".into(),
            "-fno-fast-math".into(),
            "-ffp-contract=off".into(),
        ],
        source_code: src,
        kernel_commitment_blake3: commitment,
        buffer_spec: spec,
    }
}
