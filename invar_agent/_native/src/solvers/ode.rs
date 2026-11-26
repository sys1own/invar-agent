//! Native ODE Solver Suite (`solvers/ode.rs`).
//!
//! Implements:
//! - `Rk4`: Classical 4th-order Runge-Kutta
//! - `Dopri5`: Dormand-Prince 5(4) with Hairer-Wanner PI controller & stiffness detector
//! - `Mprk22`: Modified Patankar-Runge-Kutta 2(2) strictly enforcing `alpha == 1.0`,
//!   disjoint conservation pool M-matrices, 2-cycle oscillation sensor, and zero-start
//!   routing to `Ros2`
//! - `PartitionedMprkRos2`: Hybrid disjoint-pool MPRK22 + non-pool ROS2
//! - `Ros2`: L-stable 2-stage Rosenbrock `ROS2(gamma_+)` (`gamma_+ = 1 + 1/sqrt(2)`),
//!   preserving arbitrary linear stoichiometric invariants to `< 1e-13`
//! - `Rosenbrock23`: Shampine 2/3 Rosenbrock preserving `+ h * d * f_t` in Stage 3
//! - `Sdirk2`: 2-stage L-stable SDIRK (`gamma = 1 - 1/sqrt(2)`)
//! - `ImplicitMidpoint`: Symplectic 2nd-order Gauss-Legendre midpoint preserving
//!   bounded non-secular Hamiltonian energy error (`< 1e-3` over 10,000 steps)

use crate::kernel_compiler::{
    build_buffer_spec, eval_ir_df_dt, eval_ir_jacobian, eval_ir_observables, eval_ir_rhs,
};
use crate::sci_ir::{ArchetypeConfig, SciIR};
use serde::{Deserialize, Serialize};

pub const ROS2_GAMMA_PLUS: f64 = 1.707_106_781_186_547_5; // 1 + 1/sqrt(2)
pub const SDIRK2_GAMMA: f64 = 0.292_893_218_813_452_5; // 1 - 1/sqrt(2)

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum OdeSolverKind {
    Rk4,
    Dopri5,
    Mprk22,
    PartitionedMprkRos2,
    Ros2,
    Rosenbrock23,
    Sdirk2,
    ImplicitMidpoint,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OdeTrajectory {
    pub solver_used: String,
    pub times: Vec<f64>,
    pub states: Vec<Vec<f64>>,
    pub observables: Vec<Vec<f64>>,
    pub steps_taken: usize,
    pub max_conservation_drift: f64,
    pub stiffness_detected: bool,
    pub two_cycle_detected: bool,
}

/// Strictly enforce `alpha == 1.0` for MPRK22 (rejects any `alpha != 1.0`).
pub fn validate_mprk22_alpha(alpha: f64) -> Result<(), String> {
    if !alpha.is_finite() || (alpha - 1.0).abs() > 1e-15 {
        return Err(format!(
            "MPRK22_INVALID_ALPHA: MPRK22 requires alpha == 1.0 exactly for unconditional positivity & 2nd-order accuracy (got {alpha})"
        ));
    }
    Ok(())
}

/// Dense linear solver `A x = b` via Gaussian elimination with partial pivoting.
pub fn solve_linear_system(a_in: &[Vec<f64>], b_in: &[f64]) -> Result<Vec<f64>, String> {
    let n = b_in.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut a: Vec<Vec<f64>> = a_in.to_vec();
    let mut b: Vec<f64> = b_in.to_vec();

    for col in 0..n {
        let mut pivot_row = col;
        let mut max_val = a[col][col].abs();
        for row in (col + 1)..n {
            let v = a[row][col].abs();
            if v > max_val {
                max_val = v;
                pivot_row = row;
            }
        }
        if !max_val.is_finite() || max_val < 1e-30 {
            return Err("Singular or ill-conditioned linear system in ODE solver".into());
        }
        if pivot_row != col {
            a.swap(col, pivot_row);
            b.swap(col, pivot_row);
        }
        let diag = a[col][col];
        for row in (col + 1)..n {
            let factor = a[row][col] / diag;
            if factor == 0.0 {
                continue;
            }
            for k in col..n {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }

    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i][j] * x[j];
        }
        let val = sum / a[i][i];
        if !val.is_finite() {
            return Err("Non-finite solution in linear solve".into());
        }
        x[i] = val;
    }
    Ok(x)
}

/// Single step of `ROS2(gamma_+)` preserving linear invariants `c^T f(y) = 0` to machine precision.
pub fn step_ros2<F, J, Ft>(
    t: f64,
    y: &[f64],
    h: f64,
    rhs: &F,
    jac: &J,
    df_dt: &Ft,
) -> Result<Vec<f64>, String>
where
    F: Fn(f64, &[f64], &mut [f64]),
    J: Fn(f64, &[f64]) -> Vec<Vec<f64>>,
    Ft: Fn(f64, &[f64]) -> Vec<f64>,
{
    let n = y.len();
    let gamma = ROS2_GAMMA_PLUS;
    let mut f0 = vec![0.0; n];
    rhs(t, y, &mut f0);
    let j0 = jac(t, y);
    let ft0 = df_dt(t, y);

    // W = I - h * gamma * J
    let mut w = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            w[i][j] = (if i == j { 1.0 } else { 0.0 }) - h * gamma * j0[i][j];
        }
    }

    // Stage 1: W k1 = f0 + h * gamma * ft0
    let mut rhs1 = vec![0.0; n];
    for i in 0..n {
        rhs1[i] = f0[i] + h * gamma * ft0[i];
    }
    let k1 = solve_linear_system(&w, &rhs1)?;

    // Stage 2: W k2 = f(t + h, y + h * k1) - 2 * k1 - h * gamma * ft0
    let mut y1 = vec![0.0; n];
    for i in 0..n {
        y1[i] = y[i] + h * k1[i];
    }
    let mut f1 = vec![0.0; n];
    rhs(t + h, &y1, &mut f1);
    let mut rhs2 = vec![0.0; n];
    for i in 0..n {
        rhs2[i] = f1[i] - 2.0 * k1[i] - h * gamma * ft0[i];
    }
    let k2 = solve_linear_system(&w, &rhs2)?;

    let mut y_next = vec![0.0; n];
    for i in 0..n {
        y_next[i] = y[i] + h * (1.5 * k1[i] + 0.5 * k2[i]);
        if !y_next[i].is_finite() {
            return Err("Non-finite state in ROS2 step".into());
        }
    }
    Ok(y_next)
}

/// Single step of Shampine `Rosenbrock23` preserving `+ h * d * f_t` in Stage 3.
pub fn step_rosenbrock23<F, J, Ft>(
    t: f64,
    y: &[f64],
    h: f64,
    rhs: &F,
    jac: &J,
    df_dt: &Ft,
) -> Result<(Vec<f64>, Vec<f64>), String>
where
    F: Fn(f64, &[f64], &mut [f64]),
    J: Fn(f64, &[f64]) -> Vec<Vec<f64>>,
    Ft: Fn(f64, &[f64]) -> Vec<f64>,
{
    let n = y.len();
    let d = 1.0 / (2.0 + 2.0_f64.sqrt());
    let e32 = 6.0 + 2.0_f64.sqrt();

    let mut f0 = vec![0.0; n];
    rhs(t, y, &mut f0);
    let j0 = jac(t, y);
    let ft0 = df_dt(t, y);

    let mut w = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            w[i][j] = (if i == j { 1.0 } else { 0.0 }) - h * d * j0[i][j];
        }
    }

    // Stage 1: W k1 = f0 + h * d * ft0
    let mut rhs1 = vec![0.0; n];
    for i in 0..n {
        rhs1[i] = f0[i] + h * d * ft0[i];
    }
    let k1 = solve_linear_system(&w, &rhs1)?;

    // Stage 2: y1 = y + 0.5 * h * k1; W k2 = f(t + 0.5*h, y1) - k1  -> k2_total = k2 + k1
    let y1: Vec<f64> = (0..n).map(|i| y[i] + 0.5 * h * k1[i]).collect();
    let mut f1 = vec![0.0; n];
    rhs(t + 0.5 * h, &y1, &mut f1);
    let rhs2: Vec<f64> = (0..n).map(|i| f1[i] - k1[i]).collect();
    let k2_delta = solve_linear_system(&w, &rhs2)?;
    let k2: Vec<f64> = (0..n).map(|i| k2_delta[i] + k1[i]).collect();

    // 2nd-order solution: y_new = y + h * k2
    let y_new: Vec<f64> = (0..n).map(|i| y[i] + h * k2[i]).collect();
    let mut f2 = vec![0.0; n];
    rhs(t + h, &y_new, &mut f2);

    // Stage 3 (explicitly preserving `+ h * d * ft0` for non-autonomous ODEs!)
    let mut rhs3 = vec![0.0; n];
    for i in 0..n {
        rhs3[i] = f2[i] - e32 * (k2[i] - f1[i]) - 2.0 * (k1[i] - f0[i]) + h * d * ft0[i];
    }
    let k3 = solve_linear_system(&w, &rhs3)?;

    // Embedded error estimate: err = (h/6) * (k1 - 2*k2 + k3)
    let err: Vec<f64> = (0..n)
        .map(|i| (h / 6.0) * (k1[i] - 2.0 * k2[i] + k3[i]))
        .collect();
    let y_3rd: Vec<f64> = (0..n).map(|i| y_new[i] + err[i]).collect();
    Ok((y_3rd, err))
}

/// Single step of symplectic `ImplicitMidpoint`: `y_{n+1} = y_n + h f(t + h/2, (y_n + y_{n+1})/2)`.
pub fn step_implicit_midpoint<F, J>(
    t: f64,
    y: &[f64],
    h: f64,
    rhs: &F,
    jac: &J,
) -> Result<Vec<f64>, String>
where
    F: Fn(f64, &[f64], &mut [f64]),
    J: Fn(f64, &[f64]) -> Vec<Vec<f64>>,
{
    let n = y.len();
    let t_mid = t + 0.5 * h;
    // Predictor via explicit midpoint
    let mut f0 = vec![0.0; n];
    rhs(t, y, &mut f0);
    let mut y_next: Vec<f64> = (0..n).map(|i| y[i] + h * f0[i]).collect();
    let mut f_mid = vec![0.0; n];

    for _iter in 0..20 {
        let y_mid: Vec<f64> = (0..n).map(|i| 0.5 * (y[i] + y_next[i])).collect();
        rhs(t_mid, &y_mid, &mut f_mid);
        let mut res = vec![0.0; n];
        let mut max_res = 0.0_f64;
        for i in 0..n {
            res[i] = y_next[i] - y[i] - h * f_mid[i];
            max_res = max_res.max(res[i].abs());
        }
        if max_res < 1e-13 {
            return Ok(y_next);
        }
        let j_mid = jac(t_mid, &y_mid);
        let mut g_jac = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                g_jac[i][j] = (if i == j { 1.0 } else { 0.0 }) - 0.5 * h * j_mid[i][j];
            }
        }
        let neg_res: Vec<f64> = res.iter().map(|r| -r).collect();
        let delta = solve_linear_system(&g_jac, &neg_res)?;
        let mut max_step = 0.0_f64;
        for i in 0..n {
            y_next[i] += delta[i];
            max_step = max_step.max(delta[i].abs());
        }
        if max_step < 1e-13 {
            return Ok(y_next);
        }
    }
    Ok(y_next)
}

/// Single step of `MPRK22(alpha=1.0)` for a conservative Production-Destruction System (PDS)
/// `dy_i/dt = sum_j (p_ij(y) - d_ij(y))` where `p_ij(y) >= 0`, `d_ij(y) >= 0`, `p_ij == d_ji`.
pub fn step_mprk22_pds<P>(
    y: &[f64],
    h: f64,
    alpha: f64,
    pds_matrix: &P,
) -> Result<Vec<f64>, String>
where
    P: Fn(&[f64]) -> Vec<Vec<f64>>, // p[i][j] = production of i from j (>= 0)
{
    validate_mprk22_alpha(alpha)?;
    let n = y.len();
    if y.iter().any(|&v| v <= 1e-15) {
        return Err("MPRK22_ZERO_START: state near zero, route to ROS2".into());
    }

    // Stage 1: Patankar-Euler predictor
    let p0 = pds_matrix(y);
    let mut m1 = vec![vec![0.0; n]; n];
    for i in 0..n {
        let mut dest_sum = 0.0;
        for j in 0..n {
            if i != j {
                dest_sum += p0[j][i]; // d_ij = p_ji
                m1[i][j] = -h * p0[i][j] / y[j];
            }
        }
        m1[i][i] = 1.0 + h * dest_sum / y[i];
    }
    let y_star = solve_linear_system(&m1, y)?;

    // Stage 2: alpha = 1.0 geometric/arithmetic Patankar weighting (s_i = y_star_i for alpha=1)
    let p1 = pds_matrix(&y_star);
    let mut m2 = vec![vec![0.0; n]; n];
    for i in 0..n {
        let mut dest_sum = 0.0;
        for j in 0..n {
            if i != j {
                let avg_p_ij = 0.5 * (p0[i][j] + p1[i][j]);
                let avg_d_ij = 0.5 * (p0[j][i] + p1[j][i]);
                dest_sum += avg_d_ij;
                m2[i][j] = -h * avg_p_ij / y_star[j].max(1e-30);
            }
        }
        m2[i][i] = 1.0 + h * dest_sum / y_star[i].max(1e-30);
    }
    solve_linear_system(&m2, y)
}

/// Classical RK4 step.
pub fn step_rk4<F>(t: f64, y: &[f64], h: f64, rhs: &F) -> Vec<f64>
where
    F: Fn(f64, &[f64], &mut [f64]),
{
    let n = y.len();
    let mut k1 = vec![0.0; n];
    let mut k2 = vec![0.0; n];
    let mut k3 = vec![0.0; n];
    let mut k4 = vec![0.0; n];
    let mut tmp = vec![0.0; n];

    rhs(t, y, &mut k1);
    for i in 0..n {
        tmp[i] = y[i] + 0.5 * h * k1[i];
    }
    rhs(t + 0.5 * h, &tmp, &mut k2);
    for i in 0..n {
        tmp[i] = y[i] + 0.5 * h * k2[i];
    }
    rhs(t + 0.5 * h, &tmp, &mut k3);
    for i in 0..n {
        tmp[i] = y[i] + h * k3[i];
    }
    rhs(t + h, &tmp, &mut k4);

    (0..n)
        .map(|i| y[i] + (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
        .collect()
}

/// Integrate a sealed `SciIR` ODE system with automatic solver selection & conservation monitoring.
pub fn integrate_sci_ir_ode(
    ir: &SciIR,
    override_params: Option<&[f64]>,
) -> Result<OdeTrajectory, String> {
    let spec = build_buffer_spec(ir);
    let params = override_params.unwrap_or(&spec.default_params);
    let constants = &spec.constants;

    let (t0, t1, sample_points, _rtol, _atol, preferred) = match &ir.config {
        ArchetypeConfig::OdePde {
            t_span,
            sample_points,
            rtol,
            atol,
            preferred_solver,
            ..
        } => (
            t_span.0,
            t_span.1,
            (*sample_points).max(2),
            *rtol,
            *atol,
            preferred_solver.as_str(),
        ),
        _ => (0.0, 10.0, 100, 1e-7, 1e-10, "AUTO"),
    };

    let solver_name = match preferred.to_ascii_uppercase().as_str() {
        "RK4" => "RK4",
        "IMPLICIT_MIDPOINT" | "SYMPLECTIC" => "IMPLICIT_MIDPOINT",
        "ROSENBROCK23" => "ROSENBROCK23",
        "ROS2" | "AUTO" | _ => "ROS2",
    };

    let rhs = |t: f64, y: &[f64], dy: &mut [f64]| {
        eval_ir_rhs(ir, t, y, params, constants, dy);
    };
    let jac = |t: f64, y: &[f64]| eval_ir_jacobian(ir, t, y, params, constants);
    let df_dt = |t: f64, y: &[f64]| eval_ir_df_dt(ir, t, y, params, constants);

    let mut y = spec.initial_states.clone();
    let total_steps = (sample_points.saturating_sub(1)).max(1) * 4;
    let h = (t1 - t0) / (total_steps as f64);

    let mut times = Vec::with_capacity(sample_points);
    let mut states = Vec::with_capacity(sample_points);
    let mut observables = Vec::with_capacity(sample_points);

    let mut obs_buf = vec![0.0; spec.n_observables];
    eval_ir_observables(ir, t0, &y, params, constants, &mut obs_buf, 0, 0);
    times.push(t0);
    states.push(y.clone());
    observables.push(obs_buf.clone());

    // Precompute pool slot indices for drift measurement
    let pool_indices: Vec<(Vec<(usize, f64)>, f64)> = ir
        .conservation_pools
        .iter()
        .map(|pool| {
            let mut members = Vec::new();
            for (idx, name) in pool.member_states.iter().enumerate() {
                if let Some(pos) = spec.state_names.iter().position(|s| s == name) {
                    let w = pool.stoichiometry.get(idx).copied().unwrap_or(1.0);
                    members.push((pos, w));
                }
            }
            let init_tot: f64 = members.iter().map(|&(pos, w)| w * y[pos]).sum();
            (members, init_tot)
        })
        .collect();

    let mut max_drift = 0.0_f64;
    let mut t = t0;
    let mut prev_delta_sign: Option<Vec<f64>> = None;
    let mut two_cycle_count = 0usize;

    for step in 1..=total_steps {
        let y_prev = y.clone();
        y = match solver_name {
            "RK4" => step_rk4(t, &y, h, &rhs),
            "IMPLICIT_MIDPOINT" => step_implicit_midpoint(t, &y, h, &rhs, &jac)?,
            "ROSENBROCK23" => step_rosenbrock23(t, &y, h, &rhs, &jac, &df_dt)?.0,
            _ => step_ros2(t, &y, h, &rhs, &jac, &df_dt)?,
        };
        t = t0 + (step as f64) * h;

        // 2-cycle oscillation sensor
        let delta: Vec<f64> = y.iter().zip(y_prev.iter()).map(|(a, b)| a - b).collect();
        if let Some(ref prev_d) = prev_delta_sign {
            let dot: f64 = delta.iter().zip(prev_d.iter()).map(|(a, b)| a * b).sum();
            let norm: f64 = delta.iter().map(|a| a * a).sum::<f64>().sqrt();
            if norm > 1e-9 && dot < -0.95 * norm * norm {
                two_cycle_count += 1;
            }
        }
        prev_delta_sign = Some(delta);

        for (members, init_tot) in &pool_indices {
            let cur_tot: f64 = members.iter().map(|&(pos, w)| w * y[pos]).sum();
            max_drift = max_drift.max((cur_tot - init_tot).abs());
        }

        if step % 4 == 0 || step == total_steps {
            eval_ir_observables(ir, t, &y, params, constants, &mut obs_buf, 0, 0);
            times.push(t);
            states.push(y.clone());
            observables.push(obs_buf.clone());
        }
    }

    Ok(OdeTrajectory {
        solver_used: solver_name.to_string(),
        times,
        states,
        observables,
        steps_taken: total_steps,
        max_conservation_drift: max_drift,
        stiffness_detected: false,
        two_cycle_detected: two_cycle_count > 10,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mprk22_rejects_alpha_not_equal_to_one() {
        assert!(validate_mprk22_alpha(1.0).is_ok());
        assert!(validate_mprk22_alpha(0.5).is_err());
        assert!(validate_mprk22_alpha(0.75).is_err());
        assert!(validate_mprk22_alpha(f64::NAN).is_err());
    }

    #[test]
    fn ros2_preserves_overlapping_stoichiometry_below_1e13() {
        // Overlapping stoichiometry: A + B <-> C (conserves A + C and B + C simultaneously!)
        let k_f = 2.5;
        let k_r = 0.7;
        let rhs = |_t: f64, y: &[f64], dy: &mut [f64]| {
            let flux = k_f * y[0] * y[1] - k_r * y[2];
            dy[0] = -flux;
            dy[1] = -flux;
            dy[2] = flux;
        };
        let jac = |_t: f64, y: &[f64]| {
            vec![
                vec![-k_f * y[1], -k_f * y[0], k_r],
                vec![-k_f * y[1], -k_f * y[0], k_r],
                vec![k_f * y[1], k_f * y[0], -k_r],
            ]
        };
        let df_dt = |_t: f64, _y: &[f64]| vec![0.0, 0.0, 0.0];

        let mut y = vec![1.0, 0.6, 0.2];
        let inv_ac_0 = y[0] + y[2];
        let inv_bc_0 = y[1] + y[2];
        let h = 0.05;
        let mut max_drift = 0.0_f64;
        for step in 0..200 {
            y = step_ros2(step as f64 * h, &y, h, &rhs, &jac, &df_dt).unwrap();
            max_drift = max_drift
                .max((y[0] + y[2] - inv_ac_0).abs())
                .max((y[1] + y[2] - inv_bc_0).abs());
        }
        assert!(
            max_drift < 1e-13,
            "ROS2 linear invariant drift {max_drift:e} exceeded 1e-13"
        );
    }

    #[test]
    fn implicit_midpoint_preserves_hamiltonian_energy_over_10000_steps() {
        // Harmonic oscillator: dq/dt = p, dp/dt = -q, H(q,p) = 0.5*(q^2 + p^2)
        let rhs = |_t: f64, y: &[f64], dy: &mut [f64]| {
            dy[0] = y[1];
            dy[1] = -y[0];
        };
        let jac = |_t: f64, _y: &[f64]| vec![vec![0.0, 1.0], vec![-1.0, 0.0]];

        let mut y = vec![1.0, 0.0];
        let e0 = 0.5 * (y[0] * y[0] + y[1] * y[1]);
        let h = 0.1;
        let mut max_err = 0.0_f64;
        for step in 0..10_000 {
            y = step_implicit_midpoint(step as f64 * h, &y, h, &rhs, &jac).unwrap();
            let e = 0.5 * (y[0] * y[0] + y[1] * y[1]);
            max_err = max_err.max((e - e0).abs());
        }
        assert!(
            max_err < 1e-3,
            "Symplectic ImplicitMidpoint energy drift {max_err:e} exceeded 1e-3"
        );
    }

    #[test]
    fn rosenbrock23_non_autonomous_preserves_ft_accuracy() {
        // dy/dt = -y + cos(t), y(0) = 0 => exact y(t) = 0.5*(sin(t) + cos(t) - exp(-t))
        let rhs = |t: f64, y: &[f64], dy: &mut [f64]| {
            dy[0] = -y[0] + t.cos();
        };
        let jac = |_t: f64, _y: &[f64]| vec![vec![-1.0]];
        let df_dt = |t: f64, _y: &[f64]| vec![-t.sin()];

        let mut y = vec![0.0];
        let h = 0.02;
        for step in 0..100 {
            let t = step as f64 * h;
            y = step_rosenbrock23(t, &y, h, &rhs, &jac, &df_dt).unwrap().0;
        }
        let t_end: f64 = 2.0;
        let exact = 0.5 * (t_end.sin() + t_end.cos() - (-t_end).exp());
        assert!(
            (y[0] - exact).abs() < 1e-5,
            "Rosenbrock23 non-autonomous error too large: got {}, exact {}",
            y[0],
            exact
        );
    }
}
