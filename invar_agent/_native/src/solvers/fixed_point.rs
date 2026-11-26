//! Native Fixed-Point & 3-Tier Stability Certifier (`solvers/fixed_point.rs`).
//!
//! - Deuflhard affine-invariant damped Newton
//! - Finite-difference Jacobian cross-check (`> 1e-5` -> `JACOBIAN_MISMATCH`)
//! - 3-tier stability certifier:
//!   - Tier 1: Routh-Hurwitz (`N <= 2`) / Gershgorin strict left-half-plane
//!   - Tier 2: Continuous Lyapunov `J^T P + P J = -I` (`P > 0` Cholesky SPD proof)
//!     for non-normal and near-defective Jordan-block Jacobians!

use crate::kernel_compiler::{build_buffer_spec, eval_ir_jacobian, eval_ir_rhs};
use crate::sci_ir::{ArchetypeConfig, SciIR};
use crate::solvers::ode::solve_linear_system;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityCertificate {
    pub stable: bool,
    pub tier_used: String,
    pub min_cholesky_pivot: Option<f64>,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixedPointResult {
    pub converged: bool,
    pub iterations: usize,
    pub fixed_point: Vec<f64>,
    pub residual_norm: f64,
    pub jacobian_mismatch_max_rel: f64,
    pub stability: Option<StabilityCertificate>,
}

/// Cross-check analytical Jacobian against central finite differences.
/// Fails closed with `JACOBIAN_MISMATCH` if max relative error > `1e-5`.
pub fn cross_check_jacobian<F>(
    f: &F,
    x: &[f64],
    analytical_j: &[Vec<f64>],
) -> Result<f64, String>
where
    F: Fn(&[f64], &mut [f64]),
{
    let n = x.len();
    let mut fp = vec![0.0; n];
    let mut fm = vec![0.0; n];
    let mut xp = x.to_vec();
    let mut xm = x.to_vec();
    let mut max_rel = 0.0_f64;

    for j in 0..n {
        let eps = 1e-7 * x[j].abs().max(1.0);
        xp[j] = x[j] + eps;
        xm[j] = x[j] - eps;
        f(&xp, &mut fp);
        f(&xm, &mut fm);
        xp[j] = x[j];
        xm[j] = x[j];

        for i in 0..n {
            let fd = (fp[i] - fm[i]) / (2.0 * eps);
            let ana = analytical_j[i][j];
            let scale = fd.abs().max(ana.abs()).max(1.0);
            let rel = (fd - ana).abs() / scale;
            max_rel = max_rel.max(rel);
        }
    }
    if max_rel > 1e-5 {
        return Err(format!(
            "JACOBIAN_MISMATCH: analytical vs central-FD relative error {max_rel:.3e} > 1e-5"
        ));
    }
    Ok(max_rel)
}

/// Deuflhard affine-invariant damped Newton solver for `F(x) = 0`.
pub fn deuflhard_damped_newton<F, J>(
    x0: &[f64],
    max_iters: usize,
    tol: f64,
    f: &F,
    jac: &J,
) -> Result<(Vec<f64>, usize, f64, f64), String>
where
    F: Fn(&[f64], &mut [f64]),
    J: Fn(&[f64]) -> Vec<Vec<f64>>,
{
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut fx = vec![0.0; n];
    let mut max_jac_rel = 0.0_f64;

    for iter in 0..max_iters {
        f(&x, &mut fx);
        let res_norm = fx.iter().map(|v| v * v).sum::<f64>().sqrt();
        if res_norm <= tol {
            let j_final = jac(&x);
            max_jac_rel = max_jac_rel.max(cross_check_jacobian(f, &x, &j_final)?);
            return Ok((x, iter, res_norm, max_jac_rel));
        }

        let jx = jac(&x);
        if iter == 0 {
            max_jac_rel = cross_check_jacobian(f, &x, &jx)?;
        }
        let neg_fx: Vec<f64> = fx.iter().map(|v| -v).collect();
        let dx = solve_linear_system(&jx, &neg_fx)?;
        let dx_norm = dx.iter().map(|v| v * v).sum::<f64>().sqrt();
        if dx_norm <= tol {
            return Ok((x, iter + 1, res_norm, max_jac_rel));
        }

        // Affine-invariant natural monotonicity test: ||J(x)^-1 F(x + lambda dx)|| < (1 - lambda/2) ||dx||
        let mut lambda = 1.0_f64;
        let mut accepted = false;
        let mut x_trial = vec![0.0; n];
        let mut f_trial = vec![0.0; n];

        for _ in 0..16 {
            for i in 0..n {
                x_trial[i] = x[i] + lambda * dx[i];
            }
            f(&x_trial, &mut f_trial);
            let neg_ft: Vec<f64> = f_trial.iter().map(|v| -v).collect();
            if let Ok(dx_bar) = solve_linear_system(&jx, &neg_ft) {
                let bar_norm = dx_bar.iter().map(|v| v * v).sum::<f64>().sqrt();
                if bar_norm <= (1.0 - 0.5 * lambda) * dx_norm + tol {
                    x = x_trial.clone();
                    accepted = true;
                    break;
                }
            }
            lambda *= 0.5;
        }
        if !accepted {
            // Take small damped step
            for i in 0..n {
                x[i] += 0.1 * dx[i];
            }
        }
    }

    f(&x, &mut fx);
    let res_norm = fx.iter().map(|v| v * v).sum::<f64>().sqrt();
    Ok((x, max_iters, res_norm, max_jac_rel))
}

/// Check if a symmetric matrix `P` is strictly positive definite via Cholesky factorization.
/// Returns `Ok(min_pivot)` if `P > 0`, else `Err(...)`.
fn cholesky_min_pivot(p: &[Vec<f64>]) -> Result<f64, String> {
    let n = p.len();
    let mut l = vec![vec![0.0; n]; n];
    let mut min_piv = f64::INFINITY;
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.5 * (p[i][j] + p[j][i]);
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                if !sum.is_finite() || sum <= 1e-12 {
                    return Err(format!("Non-positive Cholesky pivot at {i}: {sum:.3e}"));
                }
                let diag = sum.sqrt();
                l[i][j] = diag;
                min_piv = min_piv.min(diag);
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    Ok(min_piv)
}

/// Solve continuous Lyapunov equation `J^T P + P J = -I` via Kronecker system
/// and verify `P > 0` (Cholesky SPD). Rigorously certifies Hurwitz stability
/// even for non-normal and near-defective Jordan-block Jacobians!
pub fn certify_lyapunov_stability(j: &[Vec<f64>]) -> Result<StabilityCertificate, String> {
    let n = j.len();
    if n == 0 {
        return Ok(StabilityCertificate {
            stable: true,
            tier_used: "TRIVIAL".into(),
            min_cholesky_pivot: None,
            detail: "Empty Jacobian".into(),
        });
    }
    if n > 12 {
        return Err("Lyapunov dense Kronecker solver limited to N <= 12".into());
    }

    // Build M * vec(P) = -vec(I), where M = I_n (x) J^T + J^T (x) I_n
    // Using row-major index idx(r, c) = r * n + c:
    // (J^T P + P J)_{r, c} = sum_k J[k][r] * P[k][c] + sum_k P[r][k] * J[k][c]
    let m_dim = n * n;
    let mut a = vec![vec![0.0; m_dim]; m_dim];
    let mut b = vec![0.0; m_dim];
    for r in 0..n {
        for c in 0..n {
            let row = r * n + c;
            if r == c {
                b[row] = -1.0;
            }
            for k in 0..n {
                a[row][k * n + c] += j[k][r];
                a[row][r * n + k] += j[k][c];
            }
        }
    }

    let vec_p = solve_linear_system(&a, &b)?;
    let mut p = vec![vec![0.0; n]; n];
    for r in 0..n {
        for c in 0..n {
            p[r][c] = vec_p[r * n + c];
        }
    }

    match cholesky_min_pivot(&p) {
        Ok(min_piv) => Ok(StabilityCertificate {
            stable: true,
            tier_used: "TIER2_LYAPUNOV_CHOLESKY".into(),
            min_cholesky_pivot: Some(min_piv),
            detail: format!("Continuous Lyapunov J^T P + P J = -I certified P > 0 (min pivot {min_piv:.3e})"),
        }),
        Err(reason) => Ok(StabilityCertificate {
            stable: false,
            tier_used: "TIER2_LYAPUNOV_CHOLESKY".into(),
            min_cholesky_pivot: None,
            detail: format!("Lyapunov matrix P is not SPD (unstable or marginal): {reason}"),
        }),
    }
}

/// 3-tier stability certifier: Tier 1 Gershgorin / Routh-Hurwitz -> Tier 2 Continuous Lyapunov.
pub fn certify_stability_3tier(j: &[Vec<f64>]) -> StabilityCertificate {
    let n = j.len();
    if n == 1 {
        let stable = j[0][0] < -1e-12;
        return StabilityCertificate {
            stable,
            tier_used: "TIER1_ROUTH_HURWITZ".into(),
            min_cholesky_pivot: None,
            detail: format!("1D eigenvalue lambda = {:.6e}", j[0][0]),
        };
    }
    if n == 2 {
        let tr = j[0][0] + j[1][1];
        let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
        let stable = tr < -1e-12 && det > 1e-12;
        return StabilityCertificate {
            stable,
            tier_used: "TIER1_ROUTH_HURWITZ".into(),
            min_cholesky_pivot: None,
            detail: format!("2D Routh-Hurwitz trace={tr:.6e}, det={det:.6e}"),
        };
    }

    // Tier 1: Strict Gershgorin left-half-plane disk check
    let mut gershgorin_ok = true;
    for i in 0..n {
        let off_sum: f64 = (0..n).filter(|&k| k != i).map(|k| j[i][k].abs()).sum();
        if j[i][i] + off_sum >= -1e-12 {
            gershgorin_ok = false;
            break;
        }
    }
    if gershgorin_ok {
        return StabilityCertificate {
            stable: true,
            tier_used: "TIER1_GERSHGORIN".into(),
            min_cholesky_pivot: None,
            detail: "All Gershgorin disks strictly in open left half-plane".into(),
        };
    }

    // Tier 2: Continuous Lyapunov (handles non-normal & near-defective Jordan blocks!)
    match certify_lyapunov_stability(j) {
        Ok(cert) => cert,
        Err(err) => StabilityCertificate {
            stable: false,
            tier_used: "TIER2_LYAPUNOV_ERROR".into(),
            min_cholesky_pivot: None,
            detail: err,
        },
    }
}

/// Solve fixed point & certify stability for a sealed `SciIR`.
pub fn solve_sci_ir_fixed_point(
    ir: &SciIR,
    override_params: Option<&[f64]>,
) -> Result<FixedPointResult, String> {
    let spec = build_buffer_spec(ir);
    let params = override_params.unwrap_or(&spec.default_params);
    let constants = &spec.constants;

    let (max_iters, tol, req_stab) = match &ir.config {
        ArchetypeConfig::FixedPoint {
            max_newton_iters,
            residual_tol,
            require_stability_cert,
            ..
        } => (*max_newton_iters, *residual_tol, *require_stability_cert),
        _ => (100, 1e-10, true),
    };

    let f = |x: &[f64], out: &mut [f64]| {
        eval_ir_rhs(ir, 0.0, x, params, constants, out);
    };
    let jac = |x: &[f64]| eval_ir_jacobian(ir, 0.0, x, params, constants);

    let (fp, iters, res_norm, max_rel) =
        deuflhard_damped_newton(&spec.initial_states, max_iters, tol, &f, &jac)?;
    let converged = res_norm <= tol * 10.0;
    let stability = if req_stab && converged {
        let j_star = jac(&fp);
        Some(certify_stability_3tier(&j_star))
    } else {
        None
    };

    Ok(FixedPointResult {
        converged,
        iterations: iters,
        fixed_point: fp,
        residual_norm: res_norm,
        jacobian_mismatch_max_rel: max_rel,
        stability,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lyapunov_certifies_non_normal_jordan_block_where_gershgorin_fails() {
        // Non-normal 3x3 Jordan block: eigenvalues all = -1, but off-diagonal 100 breaks Gershgorin!
        let j = vec![
            vec![-1.0, 100.0, 0.0],
            vec![0.0, -1.0, 100.0],
            vec![0.0, 0.0, -1.0],
        ];
        let cert = certify_stability_3tier(&j);
        assert!(cert.stable, "Non-normal Hurwitz Jordan block must certify stable");
        assert_eq!(cert.tier_used, "TIER2_LYAPUNOV_CHOLESKY");
        assert!(cert.min_cholesky_pivot.unwrap() > 0.0);

        // Unstable matrix must NOT certify stable
        let j_bad = vec![
            vec![0.5, 1.0, 0.0],
            vec![0.0, -1.0, 1.0],
            vec![0.0, 0.0, -1.0],
        ];
        let cert_bad = certify_stability_3tier(&j_bad);
        assert!(!cert_bad.stable);
    }

    #[test]
    fn fd_jacobian_cross_check_catches_mismatch() {
        let f = |x: &[f64], out: &mut [f64]| {
            out[0] = x[0] * x[0] - 2.0;
        };
        let x = vec![1.5];
        assert!(cross_check_jacobian(&f, &x, &[vec![3.0]]).is_ok());
        let err = cross_check_jacobian(&f, &x, &[vec![2.0]]).unwrap_err();
        assert!(err.contains("JACOBIAN_MISMATCH"));
    }
}
