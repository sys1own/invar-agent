//! Reverse-Mode Adjoint HIN Symbolic Differentiation & Jacobian Spectrum Analysis.
//!
//! Reflects the Holographic Interaction Net along its boundary ports to compute
//! exact symbolic partial derivatives df_i / dx_j, evaluates the Jacobian matrix
//! J(x, p) with the Python GIL released, and computes the eigenvalue spectrum,
//! spectral radius rho(J), and stiffness ratio using `nalgebra`.

use std::collections::HashMap;

use nalgebra::DMatrix;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use crate::holographic_hin::CselExpr;

// ---------------------------------------------------------------------------
// Symbolic Reverse-Mode Adjoint Differentiation over CselExpr
// ---------------------------------------------------------------------------

fn lit(v: f64) -> CselExpr {
    CselExpr::Literal { value: v }
}

fn add(a: CselExpr, b: CselExpr) -> CselExpr {
    match (&a, &b) {
        (CselExpr::Literal { value: 0.0 }, _) => b,
        (_, CselExpr::Literal { value: 0.0 }) => a,
        (CselExpr::Literal { value: x }, CselExpr::Literal { value: y }) => lit(x + y),
        _ => CselExpr::Binary {
            op: "+".into(),
            left: Box::new(a),
            right: Box::new(b),
        },
    }
}

fn sub(a: CselExpr, b: CselExpr) -> CselExpr {
    match (&a, &b) {
        (_, CselExpr::Literal { value: 0.0 }) => a,
        (CselExpr::Literal { value: 0.0 }, _) => neg(b),
        (CselExpr::Literal { value: x }, CselExpr::Literal { value: y }) => lit(x - y),
        _ => CselExpr::Binary {
            op: "-".into(),
            left: Box::new(a),
            right: Box::new(b),
        },
    }
}

fn mul(a: CselExpr, b: CselExpr) -> CselExpr {
    match (&a, &b) {
        (CselExpr::Literal { value: 0.0 }, _) | (_, CselExpr::Literal { value: 0.0 }) => lit(0.0),
        (CselExpr::Literal { value: 1.0 }, _) => b,
        (_, CselExpr::Literal { value: 1.0 }) => a,
        (CselExpr::Literal { value: x }, CselExpr::Literal { value: y }) => lit(x * y),
        _ => CselExpr::Binary {
            op: "*".into(),
            left: Box::new(a),
            right: Box::new(b),
        },
    }
}

fn neg(a: CselExpr) -> CselExpr {
    match a {
        CselExpr::Literal { value: 0.0 } => lit(0.0),
        CselExpr::Literal { value: v } => lit(-v),
        CselExpr::Unary { op, arg } if op == "-" || op == "neg" => *arg,
        other => CselExpr::Unary {
            op: "-".into(),
            arg: Box::new(other),
        },
    }
}

fn sdiv(a: CselExpr, b: CselExpr) -> CselExpr {
    match (&a, &b) {
        (CselExpr::Literal { value: 0.0 }, _) => lit(0.0),
        (_, CselExpr::Literal { value: 1.0 }) => a,
        _ => CselExpr::Call {
            func: "safe_div".into(),
            args: vec![a, b],
        },
    }
}

fn call1(func: &str, a: CselExpr) -> CselExpr {
    CselExpr::Call {
        func: func.into(),
        args: vec![a],
    }
}

/// Reverse-mode adjoint accumulation on a CselExpr:
/// Propagates `adjoint` from the output port backward to every `Symbol` input port,
/// naturally summing contributions across `!` promotion (duplication) sites.
pub fn reverse_adjoint(
    expr: &CselExpr,
    adjoint: CselExpr,
    grads: &mut HashMap<String, CselExpr>,
) {
    if matches!(adjoint, CselExpr::Literal { value: 0.0 }) {
        return;
    }
    match expr {
        CselExpr::Literal { .. } => {}
        CselExpr::Symbol { name } => {
            let entry = grads.entry(name.clone()).or_insert_with(|| lit(0.0));
            let prev = entry.clone();
            *entry = add(prev, adjoint);
        }
        CselExpr::Unary { op, arg } => {
            let next_adj = match op.as_str() {
                "-" | "neg" => neg(adjoint),
                _ => adjoint,
            };
            reverse_adjoint(arg, next_adj, grads);
        }
        CselExpr::Binary { op, left, right } => match op.as_str() {
            "+" => {
                reverse_adjoint(left, adjoint.clone(), grads);
                reverse_adjoint(right, adjoint, grads);
            }
            "-" => {
                reverse_adjoint(left, adjoint.clone(), grads);
                reverse_adjoint(right, neg(adjoint), grads);
            }
            "*" => {
                reverse_adjoint(left, mul(adjoint.clone(), *right.clone()), grads);
                reverse_adjoint(right, mul(adjoint, *left.clone()), grads);
            }
            "/" => {
                // d/dl (l / r) = 1 / r ; d/dr (l / r) = -l / r^2
                let dl = sdiv(adjoint.clone(), *right.clone());
                let r2 = mul(*right.clone(), *right.clone());
                let dr = neg(sdiv(mul(adjoint, *left.clone()), r2));
                reverse_adjoint(left, dl, grads);
                reverse_adjoint(right, dr, grads);
            }
            "^" | "**" => {
                if let CselExpr::Literal { value: p } = right.as_ref() {
                    let pow_m1 = CselExpr::Call {
                        func: "safe_pow".into(),
                        args: vec![*left.clone(), lit(p - 1.0)],
                    };
                    let dl = mul(adjoint, mul(lit(*p), pow_m1));
                    reverse_adjoint(left, dl, grads);
                } else {
                    // d/dl (l^r) = r * l^(r-1)
                    let pow_m1 = CselExpr::Call {
                        func: "safe_pow".into(),
                        args: vec![*left.clone(), sub(*right.clone(), lit(1.0))],
                    };
                    let dl = mul(adjoint.clone(), mul(*right.clone(), pow_m1));
                    let full = CselExpr::Call {
                        func: "safe_pow".into(),
                        args: vec![*left.clone(), *right.clone()],
                    };
                    let dr = mul(adjoint, mul(full, call1("safe_log", call1("abs", *left.clone()))));
                    reverse_adjoint(left, dl, grads);
                    reverse_adjoint(right, dr, grads);
                }
            }
            _ => {}
        },
        CselExpr::Call { func, args } => {
            if args.is_empty() {
                return;
            }
            let a0 = args[0].clone();
            match func.as_str() {
                "safe_div" if args.len() >= 2 => {
                    let a1 = args[1].clone();
                    let dl = sdiv(adjoint.clone(), a1.clone());
                    let dr = neg(sdiv(mul(adjoint, a0), mul(a1.clone(), a1)));
                    reverse_adjoint(&args[0], dl, grads);
                    reverse_adjoint(&args[1], dr, grads);
                }
                "safe_exp" | "exp" => {
                    let da = mul(adjoint, call1("safe_exp", a0));
                    reverse_adjoint(&args[0], da, grads);
                }
                "safe_log" | "log" => {
                    let da = sdiv(adjoint, a0);
                    reverse_adjoint(&args[0], da, grads);
                }
                "log1p" => {
                    let da = sdiv(adjoint, add(lit(1.0), a0));
                    reverse_adjoint(&args[0], da, grads);
                }
                "expm1" => {
                    let da = mul(adjoint, call1("safe_exp", a0));
                    reverse_adjoint(&args[0], da, grads);
                }
                "safe_sqrt" | "sqrt" => {
                    let da = sdiv(adjoint, mul(lit(2.0), call1("safe_sqrt", a0)));
                    reverse_adjoint(&args[0], da, grads);
                }
                "safe_sin" | "sin" => {
                    let da = mul(adjoint, call1("safe_cos", a0));
                    reverse_adjoint(&args[0], da, grads);
                }
                "safe_cos" | "cos" => {
                    let da = neg(mul(adjoint, call1("safe_sin", a0)));
                    reverse_adjoint(&args[0], da, grads);
                }
                "safe_tanh" | "tanh" => {
                    let t = call1("safe_tanh", a0);
                    let da = mul(adjoint, sub(lit(1.0), mul(t.clone(), t)));
                    reverse_adjoint(&args[0], da, grads);
                }
                "safe_pow" | "pow" if args.len() >= 2 => {
                    let bin = CselExpr::Binary {
                        op: "^".into(),
                        left: Box::new(args[0].clone()),
                        right: Box::new(args[1].clone()),
                    };
                    reverse_adjoint(&bin, adjoint, grads);
                }
                "abs" | "fabs" => {
                    let sgn = CselExpr::Call {
                        func: "where".into(),
                        args: vec![a0, lit(1.0), lit(-1.0)],
                    };
                    reverse_adjoint(&args[0], mul(adjoint, sgn), grads);
                }
                "smooth_clamp" if args.len() >= 3 => {
                    // d/dx smooth_clamp(x, a, b) = (b-a) * 10 * sig * (1 - sig)
                    let lo = args[1].clone();
                    let hi = args[2].clone();
                    let mid = mul(lit(0.5), add(lo.clone(), hi.clone()));
                    let sig = sdiv(
                        lit(1.0),
                        add(lit(1.0), call1("safe_exp", mul(lit(-10.0), sub(a0, mid)))),
                    );
                    let deriv = mul(
                        lit(10.0),
                        mul(sub(hi, lo), mul(sig.clone(), sub(lit(1.0), sig))),
                    );
                    reverse_adjoint(&args[0], mul(adjoint, deriv), grads);
                }
                "where" if args.len() >= 3 => {
                    let dt = CselExpr::Call {
                        func: "where".into(),
                        args: vec![a0.clone(), adjoint.clone(), lit(0.0)],
                    };
                    let df = CselExpr::Call {
                        func: "where".into(),
                        args: vec![a0, lit(0.0), adjoint],
                    };
                    reverse_adjoint(&args[1], dt, grads);
                    reverse_adjoint(&args[2], df, grads);
                }
                _ => {}
            }
        }
    }
}

pub fn csel_to_string(expr: &CselExpr) -> String {
    match expr {
        CselExpr::Literal { value } => {
            if (*value - value.round()).abs() < 1e-12 && value.abs() < 1e15 {
                format!("{:.1}", value)
            } else {
                format!("{}", value)
            }
        }
        CselExpr::Symbol { name } => name.clone(),
        CselExpr::Unary { op, arg } => format!("({}{})", op, csel_to_string(arg)),
        CselExpr::Binary { op, left, right } => {
            format!("({} {} {})", csel_to_string(left), op, csel_to_string(right))
        }
        CselExpr::Call { func, args } => {
            let parts: Vec<String> = args.iter().map(csel_to_string).collect();
            format!("{}({})", func, parts.join(", "))
        }
    }
}

// ---------------------------------------------------------------------------
// Jacobian Spectrum & Stiffness Analysis
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JacobianAnalysisResult {
    pub state_vars: Vec<String>,
    pub jacobian: Vec<Vec<f64>>,
    pub symbolic_jacobian: Vec<Vec<String>>,
    pub eigenvalues: Vec<[f64; 2]>,
    pub spectral_radius: f64,
    pub stiffness_ratio: f64,
    pub recommended_solver: String,
}

pub fn analyze_jacobian_spectrum(
    state_vars: &[String],
    equations: &[(String, CselExpr)],
    env: &HashMap<String, f64>,
) -> JacobianAnalysisResult {
    let n = state_vars.len();
    let mut jacobian = vec![vec![0.0; n]; n];
    let mut symbolic_jacobian = vec![vec!["0.0".to_string(); n]; n];

    for (row, (_, expr)) in equations.iter().enumerate() {
        if row >= n {
            break;
        }
        let mut grads: HashMap<String, CselExpr> = HashMap::new();
        reverse_adjoint(expr, lit(1.0), &mut grads);
        for (col, var) in state_vars.iter().enumerate() {
            if let Some(g) = grads.get(var) {
                symbolic_jacobian[row][col] = csel_to_string(g);
                let val = g.eval(env);
                jacobian[row][col] = if val.is_finite() { val } else { 0.0 };
            }
        }
    }

    if n == 0 {
        return JacobianAnalysisResult {
            state_vars: vec![],
            jacobian,
            symbolic_jacobian,
            eigenvalues: vec![],
            spectral_radius: 0.0,
            stiffness_ratio: 1.0,
            recommended_solver: "rk4".into(),
        };
    }

    let mut flat = Vec::with_capacity(n * n);
    for row in &jacobian {
        flat.extend_from_slice(row);
    }
    let mat = DMatrix::from_row_slice(n, n, &flat);
    let eigs = mat.complex_eigenvalues();

    let mut eigenvalues = Vec::with_capacity(n);
    let mut spectral_radius = 0.0_f64;
    let mut max_re = 0.0_f64;
    let mut min_nonzero_re = f64::INFINITY;

    for c in eigs.iter() {
        let re = if c.re.is_finite() { c.re } else { 0.0 };
        let im = if c.im.is_finite() { c.im } else { 0.0 };
        eigenvalues.push([re, im]);
        let mag = (re * re + im * im).sqrt();
        if mag > spectral_radius {
            spectral_radius = mag;
        }
        let abs_re = re.abs();
        if abs_re > max_re {
            max_re = abs_re;
        }
        if abs_re > 1e-12 && abs_re < min_nonzero_re {
            min_nonzero_re = abs_re;
        }
    }

    let stiffness_ratio = if min_nonzero_re.is_finite() && min_nonzero_re > 0.0 {
        (max_re / min_nonzero_re).max(1.0)
    } else if spectral_radius > 1e-12 {
        spectral_radius.max(1.0)
    } else {
        1.0
    };

    let recommended_solver = if spectral_radius > 1e3 || stiffness_ratio > 1e4 {
        "ros3pr".to_string()
    } else if spectral_radius > 50.0 || stiffness_ratio > 100.0 {
        "semi_implicit_euler".to_string()
    } else if spectral_radius > 10.0 {
        "dopri5".to_string()
    } else {
        "rk4".to_string()
    };

    JacobianAnalysisResult {
        state_vars: state_vars.to_vec(),
        jacobian,
        symbolic_jacobian,
        eigenvalues,
        spectral_radius,
        stiffness_ratio,
        recommended_solver,
    }
}

// ---------------------------------------------------------------------------
// PyO3 Wrapper
// ---------------------------------------------------------------------------

#[pyclass(module = "invar_agent._native", name = "PyAdjointHin")]
pub struct PyAdjointHin {
    state_vars: Vec<String>,
    equations: Vec<(String, CselExpr)>,
}

#[pymethods]
impl PyAdjointHin {
    #[new]
    pub fn new(state_vars: Vec<String>, equations_json: &str) -> PyResult<Self> {
        let eq_map: HashMap<String, CselExpr> = serde_json::from_str(equations_json)
            .map_err(|e| PyValueError::new_err(format!("Invalid equations JSON: {}", e)))?;
        let mut equations = Vec::with_capacity(state_vars.len());
        for var in &state_vars {
            if let Some(expr) = eq_map.get(var) {
                equations.push((var.clone(), expr.clone()));
            } else {
                equations.push((var.clone(), lit(0.0)));
            }
        }
        Ok(Self {
            state_vars,
            equations,
        })
    }

    /// Computes symbolic Jacobian via Reverse-Mode Adjoint HIN and evaluates
    /// the Jacobian matrix, eigenvalues, spectral radius, and stiffness ratio
    /// at `state_env_json` with the Python GIL released.
    pub fn analyze(&self, py: Python<'_>, state_env_json: &str) -> PyResult<String> {
        let env: HashMap<String, f64> = serde_json::from_str(state_env_json)
            .map_err(|e| PyValueError::new_err(format!("Invalid state_env JSON: {}", e)))?;
        let state_vars = self.state_vars.clone();
        let equations = self.equations.clone();

        let result = py.allow_threads(move || {
            analyze_jacobian_spectrum(&state_vars, &equations, &env)
        });

        serde_json::to_string(&result)
            .map_err(|e| PyValueError::new_err(format!("Serialization error: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stiff_robertson_jacobian_spectrum() {
        // Stiff linear decay: dx/dt = -1e6 * x + y, dy/dt = x - y
        let state_vars = vec!["x".to_string(), "y".to_string()];
        let eq_x = CselExpr::Binary {
            op: "+".into(),
            left: Box::new(CselExpr::Binary {
                op: "*".into(),
                left: Box::new(CselExpr::Literal { value: -1e6 }),
                right: Box::new(CselExpr::Symbol { name: "x".into() }),
            }),
            right: Box::new(CselExpr::Symbol { name: "y".into() }),
        };
        let eq_y = CselExpr::Binary {
            op: "-".into(),
            left: Box::new(CselExpr::Symbol { name: "x".into() }),
            right: Box::new(CselExpr::Symbol { name: "y".into() }),
        };
        let env = HashMap::from([("x".into(), 1.0), ("y".into(), 0.0)]);
        let res = analyze_jacobian_spectrum(
            &state_vars,
            &[("x".into(), eq_x), ("y".into(), eq_y)],
            &env,
        );
        assert!(res.spectral_radius >= 1e6 - 1.0);
        assert!(res.stiffness_ratio >= 1e5);
        assert_eq!(res.recommended_solver, "ros3pr");
    }
}
