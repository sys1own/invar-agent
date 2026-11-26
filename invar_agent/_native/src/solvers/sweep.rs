//! Parallel Parameter Sweep, Signed Obligation Slack, Adaptive Bisection & SHBT Closure (`solvers/sweep.rs`).

use crate::kernel_compiler::{build_buffer_spec, eval_expr, eval_ir_observables, EvalContext};
use crate::sci_ir::{Archetype, ArchetypeConfig, ObligationSpec, SciIR, SlotRole};
use crate::solvers::fixed_point::solve_sci_ir_fixed_point;
use crate::solvers::mc::run_sci_ir_monte_carlo;
use crate::solvers::ode::integrate_sci_ir_ode;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepPointEvaluation {
    pub params: Vec<f64>,
    pub final_states: Vec<f64>,
    pub observables: Vec<f64>,
    pub obligation_slacks: BTreeMap<String, f64>,
    pub all_satisfied: bool,
    pub max_conservation_drift: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepVerdictReport {
    pub archetype: String,
    pub points_evaluated: usize,
    pub all_obligations_satisfied: bool,
    pub min_obligation_slacks: BTreeMap<String, f64>,
    pub worst_counterexample: Option<SweepPointEvaluation>,
    pub boundary_bisection_point: Option<Vec<f64>>,
    pub max_conservation_drift: f64,
}

/// Evaluate signed slack for a single obligation: `>= 0` means satisfied, `< 0` means violated.
pub fn evaluate_obligation_slack(ob: &ObligationSpec, ctx: &EvalContext<'_>) -> f64 {
    let lhs = eval_expr(&ob.lhs_expr, ctx);
    let rhs = eval_expr(&ob.rhs_expr, ctx);
    if !lhs.is_finite() || !rhs.is_finite() {
        return f64::NEG_INFINITY;
    }
    match ob.comparator.as_str() {
        "<=" | "<" => (rhs - lhs) + ob.tolerance,
        ">=" | ">" => (lhs - rhs) + ob.tolerance,
        "==" => ob.tolerance - (lhs - rhs).abs(),
        _ => (rhs - lhs) + ob.tolerance,
    }
}

/// Evaluate a sealed `SciIR` at a single control-parameter vector `params`.
pub fn evaluate_point(ir: &SciIR, params: &[f64]) -> Result<SweepPointEvaluation, String> {
    let spec = build_buffer_spec(ir);
    let constants = &spec.constants;

    let (t_final, final_states, obs, max_drift) = match ir.archetype {
        Archetype::OdePde => {
            let traj = integrate_sci_ir_ode(ir, Some(params))?;
            let t_end = *traj.times.last().unwrap_or(&0.0);
            let y_end = traj.states.last().cloned().unwrap_or_default();
            let o_end = traj.observables.last().cloned().unwrap_or_default();
            (t_end, y_end, o_end, traj.max_conservation_drift)
        }
        Archetype::FixedPoint => {
            let fp = solve_sci_ir_fixed_point(ir, Some(params))?;
            let mut o_end = vec![0.0; spec.n_observables];
            eval_ir_observables(ir, 0.0, &fp.fixed_point, params, constants, &mut o_end, 0, 0);
            (0.0, fp.fixed_point, o_end, 0.0)
        }
        Archetype::MonteCarlo => {
            let mc = run_sci_ir_monte_carlo(ir, Some(params))?;
            (0.0, spec.initial_states.clone(), mc.observable_means, 0.0)
        }
        Archetype::AlgebraicClosure => {
            let solved_states = if !ir.dynamics_or_residuals.is_empty() && spec.n_states > 0 {
                solve_sci_ir_fixed_point(ir, Some(params))?.fixed_point
            } else {
                spec.initial_states.clone()
            };
            let mut o_end = vec![0.0; spec.n_observables];
            eval_ir_observables(
                ir,
                0.0,
                &solved_states,
                params,
                constants,
                &mut o_end,
                0,
                0,
            );
            (0.0, solved_states, o_end, 0.0)
        }
        Archetype::ParamSweep => {
            let mut o_end = vec![0.0; spec.n_observables];
            eval_ir_observables(
                ir,
                0.0,
                &spec.initial_states,
                params,
                constants,
                &mut o_end,
                0,
                0,
            );
            (0.0, spec.initial_states.clone(), o_end, 0.0)
        }
    };

    let ctx = EvalContext {
        t: t_final,
        states: &final_states,
        params,
        constants,
        observables: &obs,
        base_seed: 0,
        replicate: 0,
    };

    let mut slacks = BTreeMap::new();
    let mut all_sat = true;
    for ob in &ir.obligations {
        let s = evaluate_obligation_slack(ob, &ctx);
        if s < 0.0 || !s.is_finite() {
            all_sat = false;
        }
        slacks.insert(ob.id.clone(), s);
    }

    Ok(SweepPointEvaluation {
        params: params.to_vec(),
        final_states,
        observables: obs,
        obligation_slacks: slacks,
        all_satisfied: all_sat,
        max_conservation_drift: max_drift,
    })
}

/// Run a parallel Rayon parameter sweep over `CONTROL_PARAM` domains with adaptive bisection.
pub fn run_sci_ir_sweep(ir: &SciIR) -> Result<SweepVerdictReport, String> {
    let spec = build_buffer_spec(ir);
    let (points_per_axis, bisection_tol) = match &ir.config {
        ArchetypeConfig::ParamSweep {
            points_per_axis,
            bisection_tol,
            ..
        } => ((*points_per_axis).clamp(2, 64), *bisection_tol),
        _ => (5, 1e-4),
    };

    let mut param_syms: Vec<&crate::sci_ir::SymbolDecl> = ir
        .symbols
        .iter()
        .filter(|s| s.role == SlotRole::ControlParam)
        .collect();
    param_syms.sort_by_key(|s| s.slot_index);

    // Build grid points (bounded to <= 256 total points for fast deterministic sweeps)
    let mut grid: Vec<Vec<f64>> = vec![spec.default_params.clone()];
    if !param_syms.is_empty() {
        let per_axis = if param_syms.len() == 1 {
            points_per_axis.min(33)
        } else if param_syms.len() == 2 {
            points_per_axis.min(9)
        } else {
            3
        };
        let mut axes: Vec<Vec<f64>> = Vec::new();
        for p in &param_syms {
            if let Some(ref disc) = p.discrete_values {
                axes.push(disc.clone());
            } else if let Some(ref dom) = p.domain {
                if (dom.hi - dom.lo).abs() <= 1e-15 {
                    axes.push(vec![dom.lo]);
                } else {
                    let vals: Vec<f64> = (0..per_axis)
                        .map(|i| dom.lo + (dom.hi - dom.lo) * (i as f64) / ((per_axis - 1) as f64))
                        .collect();
                    axes.push(vals);
                }
            } else {
                axes.push(vec![p.initial_or_default]);
            }
        }
        grid.clear();
        let mut combos = vec![Vec::new()];
        for axis in &axes {
            let mut next = Vec::new();
            for prefix in &combos {
                for &v in axis {
                    let mut row = prefix.clone();
                    row.push(v);
                    next.push(row);
                    if next.len() >= 256 {
                        break;
                    }
                }
                if next.len() >= 256 {
                    break;
                }
            }
            combos = next;
        }
        grid = combos;
    }

    let results: Result<Vec<SweepPointEvaluation>, String> = grid
        .par_iter()
        .map(|p| evaluate_point(ir, p))
        .collect();
    let evals = results?;

    let mut min_slacks: BTreeMap<String, f64> = BTreeMap::new();
    let mut max_drift = 0.0_f64;
    let mut worst_ce: Option<SweepPointEvaluation> = None;
    let mut worst_slack_val = f64::INFINITY;
    let mut first_sat: Option<Vec<f64>> = None;
    let mut first_unsat: Option<Vec<f64>> = None;

    for ev in &evals {
        max_drift = max_drift.max(ev.max_conservation_drift);
        if ev.all_satisfied && first_sat.is_none() {
            first_sat = Some(ev.params.clone());
        }
        if !ev.all_satisfied && first_unsat.is_none() {
            first_unsat = Some(ev.params.clone());
        }
        let mut pt_min = f64::INFINITY;
        for (oid, &sl) in &ev.obligation_slacks {
            pt_min = pt_min.min(sl);
            min_slacks
                .entry(oid.clone())
                .and_modify(|cur| *cur = cur.min(sl))
                .or_insert(sl);
        }
        if !ev.all_satisfied && pt_min < worst_slack_val {
            worst_slack_val = pt_min;
            worst_ce = Some(ev.clone());
        }
    }

    // Adaptive bisection between a satisfied and unsatisfied point when both exist
    let mut boundary_pt = None;
    if let (Some(mut lo), Some(mut hi)) = (first_sat, first_unsat) {
        for _ in 0..24 {
            let dist: f64 = lo
                .iter()
                .zip(hi.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();
            if dist <= bisection_tol {
                break;
            }
            let mid: Vec<f64> = lo.iter().zip(hi.iter()).map(|(a, b)| 0.5 * (a + b)).collect();
            if let Ok(ev) = evaluate_point(ir, &mid) {
                if ev.all_satisfied {
                    lo = mid;
                } else {
                    hi = mid;
                }
            } else {
                break;
            }
        }
        boundary_pt = Some(lo);
    }

    Ok(SweepVerdictReport {
        archetype: format!("{:?}", ir.archetype),
        points_evaluated: evals.len(),
        all_obligations_satisfied: worst_ce.is_none(),
        min_obligation_slacks: min_slacks,
        worst_counterexample: worst_ce,
        boundary_bisection_point: boundary_pt,
        max_conservation_drift: max_drift,
    })
}
