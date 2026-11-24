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

pub fn next_up_compat(x: f64) -> f64 {
    if x.is_nan() || x == f64::INFINITY { return x; }
    if x == 0.0 { return f64::from_bits(1); }
    let bits = x.to_bits();
    f64::from_bits(if x > 0.0 { bits + 1 } else { bits - 1 })
}

pub fn next_down_compat(x: f64) -> f64 { -next_up_compat(-x) }

#[cfg(test)]
mod interval_tests {
    use super::*;

    #[test]
    fn portable_ieee_neighbors() {
        for zero in [0.0, -0.0] {
            assert_eq!(next_up_compat(zero).to_bits(), 1);
            assert_eq!(next_down_compat(zero), -f64::from_bits(1));
        }
        assert_eq!(next_up_compat(-f64::from_bits(1)), -0.0);
        assert_eq!(next_up_compat(f64::MAX), f64::INFINITY);
        assert_eq!(next_down_compat(-f64::MAX), f64::NEG_INFINITY);
        assert_eq!(next_up_compat(f64::NEG_INFINITY), -f64::MAX);
        assert_eq!(next_down_compat(f64::INFINITY), f64::MAX);
        assert_eq!(next_up_compat(f64::INFINITY), f64::INFINITY);
        assert!(next_up_compat(f64::NAN).is_nan());
    }

    #[test]
    fn interval_arithmetic_and_singularities() {
        let a = IntervalF64::point(0.1).add(IntervalF64::point(0.2));
        assert!(a.lo <= 0.3 && a.hi > 0.3);
        let b = IntervalF64 { lo: -2.0, hi: 3.0 }.mul(IntervalF64 { lo: -4.0, hi: 5.0 });
        assert!(b.lo <= -12.0 && b.hi >= 15.0);
        let c = a.div(IntervalF64 { lo: -1.0, hi: 1.0 });
        assert_eq!((c.lo, c.hi), (f64::NEG_INFINITY, f64::INFINITY));
        let d = IntervalF64::point(0.0).mul(IntervalF64::point(f64::INFINITY));
        assert_eq!((d.lo, d.hi), (f64::NEG_INFINITY, f64::INFINITY));
        let missing = CselExpr::Symbol { name: "missing".into() };
        assert!(eval_interval(&missing, &HashMap::new()).is_err());
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct IntervalF64 { pub lo: f64, pub hi: f64 }

impl IntervalF64 {
    pub fn point(x: f64) -> Self { Self { lo: x, hi: x } }
    fn outward(lo: f64, hi: f64) -> Self {
        if lo.is_nan() || hi.is_nan() {
            return Self { lo: f64::NEG_INFINITY, hi: f64::INFINITY };
        }
        Self { lo: next_down_compat(lo), hi: next_up_compat(hi) }
    }
    pub fn add(self, rhs: Self) -> Self { Self::outward(self.lo + rhs.lo, self.hi + rhs.hi) }
    pub fn sub(self, rhs: Self) -> Self { Self::outward(self.lo - rhs.hi, self.hi - rhs.lo) }
    pub fn mul(self, rhs: Self) -> Self {
        let p = [self.lo * rhs.lo, self.lo * rhs.hi, self.hi * rhs.lo, self.hi * rhs.hi];
        if p.iter().any(|x| x.is_nan()) {
            return Self { lo: f64::NEG_INFINITY, hi: f64::INFINITY };
        }
        Self::outward(p.iter().copied().fold(f64::INFINITY, f64::min),
                      p.iter().copied().fold(f64::NEG_INFINITY, f64::max))
    }
    pub fn div(self, rhs: Self) -> Self {
        if rhs.lo <= 0.0 && rhs.hi >= 0.0 {
            return Self { lo: f64::NEG_INFINITY, hi: f64::INFINITY };
        }
        self.mul(Self::outward(1.0 / rhs.hi, 1.0 / rhs.lo))
    }
}

pub fn eval_interval(expr: &CselExpr, env: &HashMap<String, IntervalF64>) -> Result<IntervalF64, String> {
    eval_interval_with_deadline(expr, env, None)
}

fn eval_interval_with_deadline(expr: &CselExpr, env: &HashMap<String, IntervalF64>, deadline: Option<std::time::Instant>) -> Result<IntervalF64, String> {
    if deadline.map_or(false, |d| std::time::Instant::now() >= d) {
        return Err("interval evaluation time budget exhausted".into());
    }
    let result = match expr {
        CselExpr::Literal { value } => IntervalF64::point(*value),
        CselExpr::Symbol { name } => *env.get(name).ok_or_else(|| format!("missing symbol: {name}"))?,
        CselExpr::Unary { op, arg } => {
            let a = eval_interval_with_deadline(arg, env, deadline)?;
            match op.as_str() { "-" | "neg" => IntervalF64 { lo: -a.hi, hi: -a.lo },
                "+" => a, _ => return Err(format!("unsupported unary operator: {op}")) }
        }
        CselExpr::Binary { op, left, right } => {
            let a = eval_interval_with_deadline(left, env, deadline)?;
            let b = eval_interval_with_deadline(right, env, deadline)?;
            match op.as_str() { "+" => a.add(b), "-" => a.sub(b), "*" => a.mul(b),
                "/" => a.div(b), _ => return Err(format!("unsupported binary operator: {op}")) }
        }
        CselExpr::Call { func, args } => {
            if args.len() != 1 { return Err("interval functions require one argument".into()); }
            let a = eval_interval_with_deadline(&args[0], env, deadline)?;
            let (lo, hi) = match func.as_str() {
                "exp" => (a.lo.exp(), a.hi.exp()),
                "log" if a.lo > 0.0 => (a.lo.ln(), a.hi.ln()),
                "sqrt" if a.lo >= 0.0 => (a.lo.sqrt(), a.hi.sqrt()),
                // Full codomain avoids platform-dependent argument reduction assumptions.
                "sin" | "cos" => return Ok(IntervalF64 { lo: -1.0, hi: 1.0 }),
                _ => return Err(format!("unsupported function or invalid domain: {func}")),
            };
            // Empirical libm boundary: allow 16 ULP per monotone endpoint.
            let mut out = IntervalF64 { lo, hi };
            for _ in 0..16 { out = IntervalF64::outward(out.lo, out.hi); }
            out
        }
    };
    Ok(result)
}

#[pyfunction]
pub fn evaluate_interval_f64(expr_json: &str, bounds: HashMap<String, (f64, f64)>) -> PyResult<(f64, f64)> {
    let expr: CselExpr = serde_json::from_str(expr_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut env = HashMap::new();
    for (name, (lo, hi)) in bounds {
        if lo.is_nan() || hi.is_nan() || lo > hi { return Err(PyValueError::new_err("invalid interval bounds")); }
        env.insert(name, IntervalF64 { lo, hi });
    }
    let out = eval_interval(&expr, &env).map_err(PyValueError::new_err)?;
    Ok((out.lo, out.hi))
}

// ---------------------------------------------------------------------------
// Symbolic Reverse-Mode Adjoint Differentiation over CselExpr
// ---------------------------------------------------------------------------

fn checked_matrix(rows: &[Vec<f64>]) -> Result<DMatrix<f64>, String> {
    let cols = rows.first().ok_or("matrix must not be empty")?.len();
    if cols == 0 || rows.iter().any(|r| r.len() != cols || r.iter().any(|x| !x.is_finite())) {
        return Err("matrix must be finite, rectangular and nonempty".into());
    }
    Ok(DMatrix::from_fn(rows.len(), cols, |i, j| rows[i][j]))
}

fn matrix_rows(m: &DMatrix<f64>) -> Vec<Vec<f64>> {
    (0..m.nrows()).map(|i| (0..m.ncols()).map(|j| m[(i, j)]).collect()).collect()
}

#[derive(Debug, Serialize)]
pub struct IdentifiabilityReport {
    pub singular_values: Vec<f64>,
    pub right_singular_vectors: Vec<Vec<f64>>,
    pub column_norms: Vec<f64>,
    pub dead_parameters: Vec<usize>,
    pub rank: usize,
    pub fim_condition_number: Option<f64>,
    pub brun_gamma_k: Option<f64>,
    pub dimensionless_group_collinearity: bool,
    pub active_coupling: bool,
}

// Two-sided Jacobi diagonalization of the symmetric SVD dilation [0 S; S^T 0].
fn jacobi_svd(s: &DMatrix<f64>) -> Result<(Vec<f64>, DMatrix<f64>), String> {
    let (m, n) = s.shape();
    let size = m + n;
    if size > 256 { return Err("Jacobi dilation dimension exceeds 256".into()); }
    let scale = s.iter().fold(0.0_f64, |a, x| a.max(x.abs()));
    if scale == 0.0 { return Ok((vec![0.0; n], DMatrix::identity(n, n))); }
    let mut a = DMatrix::zeros(size, size);
    for i in 0..m { for j in 0..n { a[(i, m+j)] = s[(i,j)] / scale; a[(m+j,i)] = a[(i,m+j)]; } }
    let mut v = DMatrix::<f64>::identity(size, size);
    let mut converged = false;
    for _ in 0..100 {
        let mut max_off = 0.0_f64;
        for p in 0..size { for q in p+1..size {
            let apq: f64 = a[(p,q)]; max_off = max_off.max(apq.abs());
            if apq.abs() <= 4.0 * f64::EPSILON { continue; }
            let tau: f64 = (a[(q,q)] - a[(p,p)]) / (2.0 * apq);
            let t = if tau == 0.0 { 1.0 } else { tau.signum() / (tau.abs() + tau.hypot(1.0)) };
            let c = 1.0 / (1.0 + t*t).sqrt(); let sn = t*c;
            let app = a[(p,p)]; let aqq = a[(q,q)];
            a[(p,p)] = app - t*apq; a[(q,q)] = aqq + t*apq; a[(p,q)] = 0.0; a[(q,p)] = 0.0;
            for k in 0..size {
                if k != p && k != q {
                    let kp = a[(k,p)]; let kq = a[(k,q)];
                    a[(k,p)] = c*kp - sn*kq; a[(p,k)] = a[(k,p)];
                    a[(k,q)] = sn*kp + c*kq; a[(q,k)] = a[(k,q)];
                }
                let vp = v[(k,p)]; let vq = v[(k,q)];
                v[(k,p)] = c*vp - sn*vq; v[(k,q)] = sn*vp + c*vq;
            }
        } }
        if max_off <= 8.0 * f64::EPSILON { converged = true; break; }
    }
    if !converged { return Err("Jacobi SVD did not converge".into()); }
    let mut order: Vec<_> = (0..size).collect();
    order.sort_by(|&i, &j| a[(j,j)].total_cmp(&a[(i,i)]).then(i.cmp(&j)));
    let cutoff = 64.0 * f64::EPSILON * size as f64;
    let mut sigma = Vec::new(); let mut cols: Vec<nalgebra::DVector<f64>> = Vec::new();
    for &k in &order {
        if a[(k,k)] <= cutoff { continue; }
        let mut col = v.view((m,k), (n,1)).column(0).into_owned();
        let norm = col.norm(); if norm < 0.1 { continue; } col /= norm;
        if let Some(first) = col.iter().find(|x| x.abs() > cutoff) { if *first < 0.0 { col *= -1.0; } }
        sigma.push(a[(k,k)] * scale); cols.push(col);
        if cols.len() == n { break; }
    }
    for &k in &order {
        if cols.len() == n { break; }
        if a[(k,k)].abs() > cutoff { continue; }
        let mut col = v.view((m,k), (n,1)).column(0).into_owned();
        for old in &cols { col -= old * old.dot(&col); }
        let norm = col.norm();
        if norm > cutoff { col /= norm; cols.push(col); sigma.push(0.0); }
    }
    if cols.len() != n || sigma.iter().any(|x| !x.is_finite()) { return Err("SVD basis or range failure".into()); }
    Ok((sigma, DMatrix::from_columns(&cols)))
}

pub fn compute_sensitivity_svd(s_stack: &[Vec<f64>], dead_tol: f64, svd_tol: f64) -> Result<IdentifiabilityReport, String> {
    if !dead_tol.is_finite() || !svd_tol.is_finite() || dead_tol < 0.0 || svd_tol < 0.0 { return Err("invalid SVD tolerances".into()); }
    let s = checked_matrix(s_stack)?;
    let (sigma, v) = jacobi_svd(&s)?;
    let norms: Vec<_> = s.column_iter().map(|c| c.iter().fold(0.0_f64, |a, x| a.hypot(*x))).collect();
    let dead: Vec<_> = norms.iter().enumerate().filter_map(|(j,c)| if *c <= dead_tol { Some(j) } else { None }).collect();
    let rank = sigma.iter().filter(|x| **x > svd_tol * sigma[0].max(f64::MIN_POSITIVE)).count();
    let fim = if rank == s.ncols() { Some((sigma[0] / sigma[sigma.len()-1]).powi(2)).filter(|x| x.is_finite()) } else { None };
    let normalized = DMatrix::from_fn(s.nrows(), s.ncols(), |i,j| if norms[j] > 0.0 { s[(i,j)] / norms[j] } else { 0.0 });
    let (ns, _) = jacobi_svd(&normalized)?;
    let gamma = if rank == s.ncols() { Some(1.0 / ns[ns.len()-1]).filter(|x| x.is_finite()) } else { None };
    Ok(IdentifiabilityReport { singular_values: sigma, right_singular_vectors: matrix_rows(&v), column_norms: norms,
        active_coupling: dead.is_empty(), dimensionless_group_collinearity: dead.is_empty() && rank < s.ncols(),
        dead_parameters: dead, rank, fim_condition_number: fim, brun_gamma_k: gamma })
}

#[derive(Debug, Serialize)]
pub struct EquilibriumSensitivityReport { pub dx_dtheta: Vec<Vec<f64>>, pub dy_dtheta: Vec<Vec<f64>>, pub residual_norm: f64 }

pub fn compute_ift_equilibrium_sensitivity(jx: &[Vec<f64>], jth: &[Vec<f64>], dy_dx: &[Vec<f64>], dy_dth: &[Vec<f64>])
    -> Result<EquilibriumSensitivityReport, String> {
    let x = checked_matrix(jx)?; let t = checked_matrix(jth)?;
    let yx = checked_matrix(dy_dx)?; let yt = checked_matrix(dy_dth)?;
    if !x.is_square() || t.nrows() != x.nrows() || yx.ncols() != x.ncols() || yt.shape() != (yx.nrows(), t.ncols()) {
        return Err("IFT matrix dimension mismatch".into());
    }
    let dx = x.clone().lu().solve(&(-&t)).ok_or("singular equilibrium Jacobian")?;
    let residual = (&x * &dx + &t).norm();
    let dy = yx * &dx + yt;
    if dx.iter().chain(dy.iter()).any(|x| !x.is_finite()) { return Err("nonfinite IFT solution".into()); }
    Ok(EquilibriumSensitivityReport { dx_dtheta: matrix_rows(&dx), dy_dtheta: matrix_rows(&dy), residual_norm: residual })
}

fn lit(v: f64) -> CselExpr {
    CselExpr::Literal { value: v }
}

fn binary(op: &str, a: CselExpr, b: CselExpr) -> CselExpr {
    CselExpr::Binary { op: op.into(), left: Box::new(a), right: Box::new(b) }
}

/// Mathematical (not totalized/clamped) CSEL fragment used by certificates.
pub fn certificate_derivative(expr: &CselExpr, var: &str) -> Result<CselExpr, String> {
    certificate_derivative_with_deadline(expr,var,None)
}

fn certificate_derivative_with_deadline(expr: &CselExpr, var: &str, deadline: Option<std::time::Instant>) -> Result<CselExpr,String> {
    if deadline.map_or(false,|d| std::time::Instant::now()>=d) { return Err("certificate derivative time budget exhausted".into()); }
    let derivative=|e: &CselExpr| certificate_derivative_with_deadline(e,var,deadline);
    Ok(match expr {
        CselExpr::Literal { .. } => lit(0.0),
        CselExpr::Symbol { name } => lit(if name == var { 1.0 } else { 0.0 }),
        CselExpr::Unary { op, arg } if op == "-" || op == "neg" => neg(derivative(arg)?),
        CselExpr::Unary { op, arg } if op == "+" => derivative(arg)?,
        CselExpr::Binary { op, left, right } => {
            let da = derivative(left)?; let db = derivative(right)?;
            match op.as_str() {
                "+" => add(da,db), "-" => sub(da,db),
                "*" => add(mul(da,*right.clone()),mul(*left.clone(),db)),
                "/" => binary("/",sub(mul(da,*right.clone()),mul(*left.clone(),db)),mul(*right.clone(),*right.clone())),
                _ => return Err(format!("unsupported certificate operator: {op}; express integer powers as products")),
            }
        }
        CselExpr::Call { func, args } if args.len() == 1 => {
            let a = args[0].clone(); let da = derivative(&a)?;
            let factor = match func.as_str() {
                "exp" => call1("exp",a), "log" => binary("/",lit(1.0),a),
                "sqrt" => binary("/",lit(1.0),mul(lit(2.0),call1("sqrt",a))),
                "sin" => call1("cos",a), "cos" => neg(call1("sin",a)),
                _ => return Err(format!("unsupported certificate function: {func}")),
            };
            mul(factor,da)
        }
        _ => return Err("unsupported certificate expression".into()),
    })
}

fn parse_certificate_system(exprs: &[String], n: usize) -> Result<(Vec<CselExpr>, Vec<String>, Vec<Vec<CselExpr>>), String> {
    let (exprs,symbols)=parse_certificate_expressions(exprs,n)?;
    let derivatives=certificate_jacobian_with_deadline(&exprs,&symbols,None)?;
    Ok((exprs,symbols,derivatives))
}

fn parse_certificate_expressions(exprs: &[String], n: usize) -> Result<(Vec<CselExpr>,Vec<String>),String> {
    let exprs: Vec<CselExpr> = exprs.iter().map(|e| serde_json::from_str(e).map_err(|e| e.to_string())).collect::<Result<_,_>>()?;
    let mut symbols = std::collections::BTreeSet::new();
    for expr in &exprs {
        let mut found = HashMap::new(); expr.collect_symbols(&mut found);
        symbols.extend(found.into_keys());
    }
    let mut symbols: Vec<_> = symbols.into_iter().collect();
    if symbols.len() != n {
        if symbols.iter().all(|s| s.strip_prefix('x').and_then(|i| i.parse::<usize>().ok()).map_or(false,|i| i < n)) {
            symbols = (0..n).map(|i| format!("x{i}")).collect();
        } else { return Err("box/center dimension must equal the number of symbols (lexicographic order)".into()); }
    }
    Ok((exprs,symbols))
}

fn certificate_jacobian_with_deadline(exprs: &[CselExpr], symbols: &[String], deadline: Option<std::time::Instant>) -> Result<Vec<Vec<CselExpr>>,String> {
    if deadline.map_or(false,|d| std::time::Instant::now()>=d) { return Err("certificate derivative time budget exhausted".into()); }
    let derivatives=exprs.iter().map(|e| symbols.iter().map(|s| certificate_derivative_with_deadline(e,s,deadline)).collect()).collect::<Result<_,_>>()?;
    if deadline.map_or(false,|d| std::time::Instant::now()>=d) { return Err("certificate derivative time budget exhausted".into()); }
    Ok(derivatives)
}

#[cfg(test)]
mod deadline_tests {
    use super::*;

    #[test]
    fn derivative_construction_observes_deadline_and_preserves_unbounded_api() {
        let expression=mul(CselExpr::Symbol {name:"x0".into()},CselExpr::Symbol {name:"x0".into()});
        let symbols=vec!["x0".into()];
        let expired=Some(std::time::Instant::now());
        assert!(certificate_jacobian_with_deadline(&[expression.clone()],&symbols,expired).unwrap_err().contains("time budget"));
        assert!(certificate_derivative_with_deadline(&expression,"x0",expired).is_err());
        let derivative=certificate_jacobian_with_deadline(&[expression],&symbols,None).unwrap();
        assert_eq!(derivative[0][0].eval(&HashMap::from([("x0".into(),3.0)])),6.0);
        assert!(certificate_jacobian_with_deadline(&[],&[],None).unwrap().is_empty());
    }
}

fn interval_env(symbols: &[String], box_iv: &[IntervalF64]) -> HashMap<String, IntervalF64> {
    symbols.iter().cloned().zip(box_iv.iter().copied()).collect()
}

fn interval_jacobian(derivatives: &[Vec<CselExpr>], env: &HashMap<String, IntervalF64>) -> Result<Vec<Vec<IntervalF64>>, String> {
    derivatives.iter().map(|row| row.iter().map(|e| eval_interval(e,env)).collect()).collect()
}

#[derive(Debug, Serialize)]
pub struct KrawczykCertificate {
    pub certified: bool,
    pub excluded: bool,
    pub symbols: Vec<String>,
    pub box_z: Vec<IntervalF64>,
    pub krawczyk_box: Vec<IntervalF64>,
    pub interval_jacobian: Vec<Vec<IntervalF64>>,
    pub krawczyk_jacobian: Option<Vec<Vec<IntervalF64>>>,
}

pub fn certify_krawczyk_fixed_point(exprs: &[String], z_center: &[f64], rad: f64) -> Result<KrawczykCertificate, String> {
    let n = z_center.len();
    if n == 0 || n > 32 || exprs.len() != n || !rad.is_finite() || rad <= 0.0 || z_center.iter().any(|x| !x.is_finite()) {
        return Err("invalid Krawczyk dimensions, center or radius".into());
    }
    let (exprs,symbols,derivatives) = parse_certificate_system(exprs,n)?;
    let z: Vec<_> = z_center.iter().map(|x| IntervalF64::outward(x-rad,x+rad)).collect();
    let points: Vec<_> = z_center.iter().map(|x| IntervalF64::point(*x)).collect();
    let env = interval_env(&symbols,&z); let center_env = interval_env(&symbols,&points);
    let jac = interval_jacobian(&derivatives,&env)?;
    let center_jac = interval_jacobian(&derivatives,&center_env)?;
    let midpoint = DMatrix::from_fn(n,n,|i,j| 0.5*center_jac[i][j].lo + 0.5*center_jac[i][j].hi);
    let c = midpoint.try_inverse().ok_or("singular Krawczyk preconditioner")?;
    if c.iter().any(|x| !x.is_finite()) { return Err("nonfinite Krawczyk preconditioner".into()); }
    let g = exprs.iter().map(|e| eval_interval(e,&center_env)).collect::<Result<Vec<_>,_>>()?;
    let mut kbox = Vec::new();
    for i in 0..n {
        let mut base = points[i];
        for j in 0..n { base = base.sub(IntervalF64::point(c[(i,j)]).mul(g[j])); }
        for j in 0..n {
            let mut r = IntervalF64::point(if i == j { 1.0 } else { 0.0 });
            for l in 0..n { r = r.sub(IntervalF64::point(c[(i,l)]).mul(jac[l][j])); }
            base = base.add(r.mul(z[j].sub(points[j])));
        }
        kbox.push(base);
    }
    let certified = kbox.iter().zip(&z).all(|(k,z)| k.lo > z.lo && k.hi < z.hi);
    let excluded = kbox.iter().zip(&z).any(|(k,z)| k.hi < z.lo || k.lo > z.hi);
    let kj = if certified { Some(interval_jacobian(&derivatives,&interval_env(&symbols,&kbox))?) } else { None };
    Ok(KrawczykCertificate { certified, excluded, symbols, box_z: z, krawczyk_box: kbox,
        interval_jacobian: jac, krawczyk_jacobian: kj })
}

#[derive(Debug, Serialize)]
pub struct ExhaustiveRefutationResult {
    pub refuted: bool,
    pub budget_exhausted: bool,
    pub certified_refuted_volume_fraction: f64,
    pub boxes_evaluated: usize,
    pub certified_subbox_count: usize,
    pub max_unresolved_upper_bound: Option<f64>,
    pub unresolved_bound_unbounded: bool,
    pub effective_max_boxes: usize,
    pub symbols: Vec<String>,
}

pub fn dimension_scaled_box_budget(n: usize) -> usize {
    2048usize.saturating_mul(1usize << n.saturating_sub(3).min(4)).min(32768)
}

#[derive(Debug, Serialize)]
pub struct StabilityCertificate {
    pub certified: bool,
    pub method: String,
    pub hurwitz_determinants: Vec<IntervalF64>,
    pub reason: Option<String>,
    pub bauer_fike: Option<BauerFikeBound>,
}

#[derive(Debug, Serialize)]
pub struct BauerFikeBound {
    pub eigenvector_condition_upper: f64,
    pub radius_norm_upper: f64,
    pub eigenpair_residual_upper: f64,
    pub spectral_abscissa_upper: f64,
}

#[derive(Clone, Copy)]
struct ComplexInterval { re: IntervalF64, im: IntervalF64 }

impl ComplexInterval {
    fn point(z: nalgebra::Complex<f64>) -> Self { Self { re: IntervalF64::point(z.re), im: IntervalF64::point(z.im) } }
    fn add(self, b: Self) -> Self { Self { re: self.re.add(b.re), im: self.im.add(b.im) } }
    fn sub(self, b: Self) -> Self { Self { re: self.re.sub(b.re), im: self.im.sub(b.im) } }
    fn mul(self, b: Self) -> Self {
        Self { re: self.re.mul(b.re).sub(self.im.mul(b.im)), im: self.re.mul(b.im).add(self.im.mul(b.re)) }
    }
    fn abs_squared_upper(self) -> f64 {
        let a=self.re.lo.abs().max(self.re.hi.abs()); let b=self.im.lo.abs().max(self.im.hi.abs());
        next_up_compat(next_up_compat(a*a)+next_up_compat(b*b))
    }
}

fn frobenius_upper(values: impl Iterator<Item=ComplexInterval>) -> f64 {
    next_up_compat(values.fold(0.0,|sum,z| next_up_compat(sum+z.abs_squared_upper())).sqrt())
}

fn bauer_fike_bound(j: &[Vec<IntervalF64>]) -> Result<BauerFikeBound,String> {
    use nalgebra::{Complex, linalg::Schur};
    let n=j.len(); let zero=Complex::new(0.0,0.0);
    let midpoint=DMatrix::from_fn(n,n,|i,k| Complex::new(0.5*j[i][k].lo+0.5*j[i][k].hi,0.0));
    let (q,t)=Schur::try_new(midpoint.clone(),f64::EPSILON,10000).ok_or("midpoint Schur decomposition did not converge")?.unpack();
    let lambda=t.diagonal();
    let mut u=DMatrix::from_element(n,n,zero);
    for k in 0..n {
        u[(k,k)]=Complex::new(1.0,0.0);
        for i in (0..k).rev() {
            let rhs=(i+1..=k).fold(zero,|sum,l| sum+t[(i,l)]*u[(l,k)]);
            let divisor=t[(i,i)]-lambda[k];
            if divisor.norm() <= 64.0*f64::EPSILON*(1.0+lambda[k].norm()) {
                if rhs.norm() > 64.0*f64::EPSILON { return Err("midpoint is defective or eigenvectors are unresolved".into()); }
                u[(i,k)]=zero;
            } else { u[(i,k)]=-rhs/divisor; }
        }
        let norm=u.column(k).norm();
        if !norm.is_finite() || norm==0.0 { return Err("nonfinite eigenvector".into()); }
        for i in 0..n { u[(i,k)]/=Complex::new(norm,0.0); }
    }
    let v=q*u;
    let b=v.clone().try_inverse().ok_or("singular midpoint eigenbasis")?;
    if v.iter().chain(b.iter()).chain(lambda.iter()).any(|z| !z.re.is_finite() || !z.im.is_finite()) { return Err("nonfinite eigenbasis".into()); }
    let point=ComplexInterval::point;
    let mut inverse_residual=Vec::new(); let mut eigenpair_residual=Vec::new(); let mut radii=Vec::new();
    for i in 0..n { for k in 0..n {
        let mut bv=point(zero); let mut av=point(zero);
        for l in 0..n {
            bv=bv.add(point(b[(i,l)]).mul(point(v[(l,k)])));
            av=av.add(point(midpoint[(i,l)]).mul(point(v[(l,k)])));
        }
        inverse_residual.push(point(Complex::new(if i==k {1.0} else {0.0},0.0)).sub(bv));
        eigenpair_residual.push(av.sub(point(v[(i,k)]).mul(point(lambda[k]))));
        let mid=midpoint[(i,k)].re;
        let radius=next_up_compat((j[i][k].hi-mid).max(mid-j[i][k].lo));
        radii.push(point(Complex::new(radius,0.0)));
    } }
    // The Neumann bound verifies invertibility of the *computed* basis. The
    // eigenpair residual also covers all eigensolver and Schur rounding errors.
    let eta=frobenius_upper(inverse_residual.into_iter());
    if eta>=1.0 { return Err("eigenbasis inverse could not be verified".into()); }
    let inv_norm=next_up_compat(frobenius_upper(b.iter().copied().map(point))/next_down_compat(1.0-eta));
    let v_norm=frobenius_upper(v.iter().copied().map(point));
    let condition=next_up_compat(inv_norm*v_norm);
    if !condition.is_finite() || condition>1e10 { return Err("midpoint eigenbasis is ill-conditioned".into()); }
    let radius=frobenius_upper(radii.into_iter()); let residual=frobenius_upper(eigenpair_residual.into_iter());
    let perturbation=next_up_compat(inv_norm*next_up_compat(residual+next_up_compat(radius*v_norm)));
    let max_real=lambda.iter().map(|z| z.re).fold(f64::NEG_INFINITY,f64::max);
    let upper=next_up_compat(max_real+perturbation);
    if !upper.is_finite() { return Err("Bauer-Fike bound overflow".into()); }
    Ok(BauerFikeBound { eigenvector_condition_upper:condition, radius_norm_upper:radius,
        eigenpair_residual_upper:residual,spectral_abscissa_upper:upper })
}

fn interval_determinant(a: &[Vec<IntervalF64>]) -> IntervalF64 {
    if a.is_empty() { return IntervalF64::point(1.0); }
    let mut det = IntervalF64::point(0.0);
    for j in 0..a.len() {
        let minor: Vec<Vec<_>> = a.iter().skip(1).map(|r| r.iter().enumerate().filter_map(|(k,x)| if k != j { Some(*x) } else { None }).collect()).collect();
        let term = a[0][j].mul(interval_determinant(&minor));
        det = if j % 2 == 0 { det.add(term) } else { det.sub(term) };
    }
    det
}

pub fn certify_interval_stability(j_iv: &[Vec<IntervalF64>]) -> Result<StabilityCertificate,String> {
    let n = j_iv.len();
    if n == 0 || n > 64 || j_iv.iter().any(|r| r.len() != n || r.iter().any(|x| !x.lo.is_finite() || !x.hi.is_finite() || x.lo > x.hi)) {
        return Err("stability requires a finite square interval matrix of order 1..64".into());
    }
    if n > 4 {
        let stable = (0..n).all(|i| {
            let mut bound = j_iv[i][i].hi;
            for j in 0..n { if i != j { bound = next_up_compat(bound + j_iv[i][j].lo.abs().max(j_iv[i][j].hi.abs())); } }
            bound < 0.0
        });
        let bound=bauer_fike_bound(j_iv);
        let bf_stable=bound.as_ref().map_or(false,|b| b.spectral_abscissa_upper<0.0);
        let reason=match &bound { Err(e) if !stable => Some(e.clone()), Ok(_) if !stable && !bf_stable => Some("Bauer-Fike and Gershgorin bounds are inconclusive".into()), _ => None };
        return Ok(StabilityCertificate { certified:stable || bf_stable,
            method:if bf_stable { "INTERVAL_BAUER_FIKE".into() } else { "INTERVAL_GERSHGORIN".into() },
            hurwitz_determinants:Vec::new(), reason, bauer_fike:bound.ok() });
    }
    let mut coeff = vec![IntervalF64::point(1.0)];
    for k in 1..=n {
        let mut sum = IntervalF64::point(0.0);
        for mask in 1usize..1usize<<n {
            if mask.count_ones() as usize != k { continue; }
            let ids: Vec<_> = (0..n).filter(|i| mask & (1<<i) != 0).collect();
            let minor: Vec<Vec<_>> = ids.iter().map(|&i| ids.iter().map(|&j| j_iv[i][j]).collect()).collect();
            sum = sum.add(interval_determinant(&minor));
        }
        if k % 2 == 1 { sum = IntervalF64 { lo: -sum.hi, hi: -sum.lo }; }
        coeff.push(sum);
    }
    let h: Vec<Vec<_>> = (0..n).map(|i| (0..n).map(|j| {
        let k = 2*j as isize - i as isize + 1;
        if k >= 0 && k <= n as isize { coeff[k as usize] } else { IntervalF64::point(0.0) }
    }).collect()).collect();
    let determinants: Vec<_> = (1..=n).map(|k| interval_determinant(&h[..k].iter().map(|r| r[..k].to_vec()).collect::<Vec<_>>())).collect();
    let certified = determinants.iter().all(|d| d.lo > 0.0);
    Ok(StabilityCertificate { certified, method: "INTERVAL_ROUTH_HURWITZ".into(), hurwitz_determinants: determinants,
        reason: if certified { None } else { Some("Hurwitz determinant not strictly positive".into()) }, bauer_fike:None })
}

/// Cooperative 1500 ms total-call budget, including symbolic differentiation.
/// Parsing (<=64 KiB, serde's depth bound), individual AST clones/allocations,
/// result cleanup/serialization and OS scheduling cannot be preempted. A hard
/// real-time bound requires an external worker/process deadline.
pub fn branch_and_bound_refute(margin_expr: &str, box_lo: &[f64], box_hi: &[f64], eps_total: f64, max_boxes: usize)
    -> Result<ExhaustiveRefutationResult, String> {
    let start = std::time::Instant::now();
    let deadline = start + std::time::Duration::from_millis(1500);
    let n = box_lo.len();
    if n == 0 || n > 64 || margin_expr.len() > 65536 || box_hi.len() != n || !eps_total.is_finite() || eps_total < 0.0
        || box_lo.iter().zip(box_hi).any(|(l,h)| !l.is_finite() || !h.is_finite() || l > h) {
        return Err("invalid refutation box or uncertainty".into());
    }
    let limit = max_boxes.min(dimension_scaled_box_budget(n));
    let (exprs,symbols)=parse_certificate_expressions(&[margin_expr.into()],n)?;
    let derivatives=certificate_jacobian_with_deadline(&exprs,&symbols,Some(deadline));
    let derivatives=match derivatives {
        Ok(d) if std::time::Instant::now()<deadline => d,
        Err(e) if std::time::Instant::now()<deadline => return Err(e),
        _ => return Ok(ExhaustiveRefutationResult { refuted:false,budget_exhausted:true,
            certified_refuted_volume_fraction:0.0,boxes_evaluated:0,certified_subbox_count:0,
            max_unresolved_upper_bound:None,unresolved_bound_unbounded:true,effective_max_boxes:limit,symbols }),
    };
    let root: Vec<_> = box_lo.iter().zip(box_hi).map(|(l,h)| IntervalF64 { lo: *l, hi: *h }).collect();
    let mut queue = vec![(root,1.0_f64,f64::INFINITY)]; let mut unresolved = Vec::new();
    let mut evaluated = 0; let mut certified_count = 0; let mut volume = 0.0_f64;
    while !queue.is_empty() && evaluated < limit && start.elapsed().as_millis() < 1500 {
        let (box_iv,weight,inherited_upper) = queue.pop().unwrap();
        let env = interval_env(&symbols,&box_iv);
        let mid: Vec<_> = box_iv.iter().map(|iv| IntervalF64::point(0.5*iv.lo + 0.5*iv.hi)).collect();
        let bound = || -> Result<f64,String> {
            let natural = eval_interval_with_deadline(&exprs[0],&env,Some(deadline))?;
            let mut centered = eval_interval_with_deadline(&exprs[0],&interval_env(&symbols,&mid),Some(deadline))?;
            for j in 0..n { centered = centered.add(eval_interval_with_deadline(&derivatives[0][j],&env,Some(deadline))?.mul(box_iv[j].sub(mid[j]))); }
            Ok(natural.hi.min(centered.hi).min(inherited_upper))
        };
        let upper = match bound() {
            Ok(upper) => upper,
            Err(_) if std::time::Instant::now() >= deadline => { queue.push((box_iv,weight,inherited_upper)); break; }
            Err(error) => return Err(error),
        };
        evaluated += 1;
        if next_up_compat(upper + eps_total) < 0.0 {
            certified_count += 1; volume = next_down_compat(volume + weight).max(0.0); continue;
        }
        let split = (0..n).max_by(|&i,&j| (box_iv[i].hi-box_iv[i].lo).total_cmp(&(box_iv[j].hi-box_iv[j].lo)).then(j.cmp(&i))).unwrap();
        let center = mid[split].lo;
        if center <= box_iv[split].lo || center >= box_iv[split].hi { unresolved.push(upper); continue; }
        let width = IntervalF64::point(box_iv[split].hi).sub(IntervalF64::point(box_iv[split].lo));
        let left_width = IntervalF64::point(center).sub(IntervalF64::point(box_iv[split].lo));
        let right_width = IntervalF64::point(box_iv[split].hi).sub(IntervalF64::point(center));
        let left_weight = next_down_compat(weight * left_width.div(width).lo).max(0.0);
        let right_weight = next_down_compat(weight * right_width.div(width).lo).max(0.0);
        let mut left = box_iv.clone(); let mut right = box_iv;
        left[split].hi = center; right[split].lo = center;
        queue.push((right,right_weight,upper)); queue.push((left,left_weight,upper));
    }
    let exhausted = !queue.is_empty();
    unresolved.extend(queue.into_iter().map(|(_,_,bound)| bound));
    let max_upper = unresolved.into_iter().reduce(f64::max);
    let refuted = max_upper.is_none();
    let unbounded = max_upper.map_or(false, |x| !x.is_finite());
    Ok(ExhaustiveRefutationResult { refuted, budget_exhausted: exhausted,
        certified_refuted_volume_fraction: if refuted { 1.0 } else { volume.min(1.0) }, boxes_evaluated: evaluated,
        certified_subbox_count: certified_count, max_unresolved_upper_bound: max_upper.filter(|x| x.is_finite()),
        unresolved_bound_unbounded: unbounded,
        effective_max_boxes: limit, symbols })
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
