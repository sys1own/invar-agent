//! Geometry of Interaction (GoI) matrix repair-isolation solver.
//!
//! Encodes a DAG dependency graph as routing matrix ``U`` and adjacency matrix
//! ``M``.  Given a base matrix pair and a perturbation ``ΔM`` the module
//! evaluates whether the spectral radius of ``U (M + ΔM)`` stays strictly
//! below the unit boundary, and computes the support set of the perturbation.

use std::collections::{HashMap, HashSet};

use ndarray::{Array1, Array2};
use pyo3::exceptions::{PyArithmeticError, PyValueError};
use pyo3::prelude::*;
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Bundle of GoI matrices for a single workspace snapshot.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GoIMatrixData {
    pub u: Vec<Vec<f64>>,
    pub m: Vec<Vec<f64>>,
    pub size: usize,
}

/// Result of a spectral-radius repair-isolation check.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RepairRadiusResult {
    pub radius: f64,
    pub bound: f64,
    pub isolated: bool,
    pub support_rows: Vec<usize>,
    pub support_cols: Vec<usize>,
}

fn matrix_from_json(v: &Value) -> PyResult<Array2<f64>> {
    let rows = v
        .as_array()
        .ok_or_else(|| PyValueError::new_err("matrix JSON is not an array"))?
        .iter()
        .map(|row| {
            row.as_array()
                .ok_or_else(|| PyValueError::new_err("matrix row is not an array"))?
                .iter()
                .map(|x| {
                    x.as_f64()
                        .ok_or_else(|| PyValueError::new_err("matrix entry is not a number"))
                })
                .collect::<PyResult<Vec<f64>>>()
        })
        .collect::<PyResult<Vec<Vec<f64>>>>()?;
    if rows.is_empty() {
        return Err(PyValueError::new_err("matrix is empty"));
    }
    let n = rows.len();
    let m = rows[0].len();
    let flat: Vec<f64> = rows.into_iter().flatten().collect();
    Array2::from_shape_vec((n, m), flat).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Estimate the dominant eigenvalue magnitude (spectral radius) of a square
/// matrix using power iteration.  Returns ``0.0`` for zero matrices.
fn spectral_radius(a: &Array2<f64>) -> f64 {
    let (n, m) = a.dim();
    if n == 0 || m == 0 {
        return 0.0;
    }
    let mut b = Array1::from_vec(vec![1.0; n]);
    let mut prev = 0.0;
    for _ in 0..80 {
        let mut v = a.dot(&b);
        let norm = v.iter().map(|x| x.abs()).fold(0.0, f64::max);
        if norm < 1e-15 {
            return 0.0;
        }
        v /= norm;
        b = v;
        if (norm - prev).abs() < 1e-12 {
            return norm;
        }
        prev = norm;
    }
    prev
}

/// Compute the row and column support sets of ``ΔM`` (non-zero entries).
fn perturbation_support(delta: &Array2<f64>) -> (Vec<usize>, Vec<usize>) {
    let mut rows = HashSet::new();
    let mut cols = HashSet::new();
    for ((r, c), v) in delta.indexed_iter() {
        if *v != 0.0 {
            rows.insert(r);
            cols.insert(c);
        }
    }
    let mut row_vec: Vec<usize> = rows.into_iter().collect();
    let mut col_vec: Vec<usize> = cols.into_iter().collect();
    row_vec.sort_unstable();
    col_vec.sort_unstable();
    (row_vec, col_vec)
}

/// Verify that an execution matrix ``M`` is non-zero.
///
/// ``matrix_json`` is a JSON 2-D array. Returns ``true`` when the matrix has at
/// least one non-zero entry and its spectral radius is above the zero threshold.
/// This is used as a post-synthesis GoI proof-net gate: a "hollow" artifact
/// produces an all-zero execution matrix and fails.
#[pyfunction]
pub fn execution_matrix_nonzero(matrix_json: &str) -> PyResult<bool> {
    let value: Value = serde_json::from_str(matrix_json)
        .map_err(|e| PyValueError::new_err(format!("invalid matrix JSON: {}", e)))?;
    let mat = matrix_from_json(&value)?;
    if mat.is_empty() {
        return Ok(false);
    }
    // A non-zero execution matrix must contain at least one non-zero entry.
    // Spectral radius can be zero for nilpotent (acyclic) graphs, so the Frobenius
    // norm is the correct hollow/empty check.
    let norm = mat.iter().map(|x| x * x).sum::<f64>().sqrt();
    Ok(norm > 1e-12)
}

/// Evaluate whether a perturbation ``ΔM`` keeps the GoI matrix ``U (M + ΔM)``
/// inside the unit spectral-radius boundary.
///
/// ``base_matrix_json`` may be a JSON object ``{"U": ..., "M": ...}`` or just
/// a matrix ``M`` (in which case ``U`` is taken as the identity).
/// ``delta_matrix_json`` is a matrix ``ΔM``.
#[pyfunction]
pub fn enforce_repair_isolation_py(
    base_matrix_json: &str,
    delta_matrix_json: &str,
) -> PyResult<String> {
    let base: Value = serde_json::from_str(base_matrix_json)
        .map_err(|e| PyValueError::new_err(format!("invalid base matrix JSON: {}", e)))?;
    let delta: Value = serde_json::from_str(delta_matrix_json)
        .map_err(|e| PyValueError::new_err(format!("invalid delta matrix JSON: {}", e)))?;

    let (u_mat, m_mat) = if let Some(u) = base.get("U").or_else(|| base.get("u")) {
        let m = base
            .get("M")
            .or_else(|| base.get("m"))
            .ok_or_else(|| PyValueError::new_err("base object missing M matrix"))?;
        (matrix_from_json(u)?, matrix_from_json(m)?)
    } else {
        let m = matrix_from_json(&base)?;
        let n = m.dim().0;
        let u = Array2::eye(n);
        (u, m)
    };

    let delta_mat = matrix_from_json(&delta)?;

    if u_mat.dim() != m_mat.dim() {
        return Err(PyValueError::new_err(
            "U and M must have the same dimensions",
        ));
    }
    if m_mat.dim() != delta_mat.dim() {
        return Err(PyValueError::new_err(
            "base M and delta M must have the same dimensions",
        ));
    }

    let m_new = &m_mat + &delta_mat;
    let product = u_mat.dot(&m_new);
    let radius = spectral_radius(&product);
    let bound = 1.0;
    let isolated = radius < bound;
    let (support_rows, support_cols) = perturbation_support(&delta_mat);

    let result = RepairRadiusResult {
        radius,
        bound,
        isolated,
        support_rows,
        support_cols,
    };
    serde_json::to_string(&result)
        .map_err(|e| PyValueError::new_err(format!("repair result serialization failed: {}", e)))
}

/// Girard Geometry-of-Interaction proof net for deadlock-free concurrency checking.
#[pyclass(name = "GoIProofNet")]
pub struct GoIProofNet {
    pub dimension: usize,
    pub axiom_matrix_m: DMatrix<f64>,
    pub cut_matrix_sigma: DMatrix<f64>,
    pub channel_map: HashMap<String, usize>,
}

#[pymethods]
impl GoIProofNet {
    #[new]
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            axiom_matrix_m: DMatrix::zeros(dimension, dimension),
            cut_matrix_sigma: DMatrix::zeros(dimension, dimension),
            channel_map: HashMap::new(),
        }
    }

    /// Compute Girard's Execution Formula:
    ///   EX(M, sigma) = (I - sigma^2) * M * (I - sigma * M)^(-1) * (I - sigma^2)
    pub fn compute_execution_formula(&self) -> PyResult<Vec<Vec<f64>>> {
        let identity = DMatrix::identity(self.dimension, self.dimension);
        let sigma_m = &self.cut_matrix_sigma * &self.axiom_matrix_m;
        let resolvent = &identity - &sigma_m;
        let resolvent_inv = resolvent.try_inverse().ok_or_else(|| {
            PyValueError::new_err(
                "GoI Execution Error: Operator (1 - sigma*M) is singular. Deadlock detected.",
            )
        })?;

        let sigma_sq = &self.cut_matrix_sigma * &self.cut_matrix_sigma;
        let proj = &identity - &sigma_sq;
        let ex = (&proj * &self.axiom_matrix_m) * resolvent_inv * &proj;
        Ok((0..ex.nrows())
            .map(|r| ex.row(r).iter().copied().collect())
            .collect())
    }

    /// Verify that (sigma * M) is nilpotent up to max_iterations.
    #[pyo3(signature = (max_iterations=1000))]
    pub fn verify_nilpotency(&self, max_iterations: usize) -> bool {
        let sigma_m = &self.cut_matrix_sigma * &self.axiom_matrix_m;
        let mut current_power = sigma_m.clone();
        for _ in 1..max_iterations {
            if current_power.norm() < 1e-9 {
                return true;
            }
            current_power = &current_power * &sigma_m;
        }
        false
    }

    /// Populate the axiom link matrix from a flat row-major list.
    pub fn set_axiom_matrix(&mut self, data: Vec<f64>) -> PyResult<()> {
        if data.len() != self.dimension * self.dimension {
            return Err(PyValueError::new_err("axiom matrix data length mismatch"));
        }
        self.axiom_matrix_m = DMatrix::from_row_slice(self.dimension, self.dimension, &data);
        Ok(())
    }

    /// Populate the cut link matrix from a flat row-major list.
    pub fn set_cut_matrix(&mut self, data: Vec<f64>) -> PyResult<()> {
        if data.len() != self.dimension * self.dimension {
            return Err(PyValueError::new_err("cut matrix data length mismatch"));
        }
        self.cut_matrix_sigma = DMatrix::from_row_slice(self.dimension, self.dimension, &data);
        Ok(())
    }
}

/// Convenience Python entry point: build a proof net from flat matrices, verify
/// nilpotency, and return false when the execution formula is singular (deadlock).
#[pyfunction(signature = (dimension, m_data, sigma_data, max_iterations=1000))]
pub fn verify_goi_proof_net(
    dimension: usize,
    m_data: Vec<f64>,
    sigma_data: Vec<f64>,
    max_iterations: usize,
) -> PyResult<bool> {
    let mut net = GoIProofNet::new(dimension);
    net.set_axiom_matrix(m_data)?;
    net.set_cut_matrix(sigma_data)?;

    // Deadlock detection: a singular resolvent means no execution formula exists.
    match net.compute_execution_formula() {
        Err(_) => return Ok(false),
        Ok(_) => {}
    }

    Ok(net.verify_nilpotency(max_iterations))
}

/// Wavefront scheduler based on the GoI operator (I - U * M).
///
/// Given an adjacency matrix M and a routing matrix U, the operator
/// K = I - U * M is inverted to compute the execution wave operator.
/// A singular K (determinant near zero) indicates a cyclic dependency
/// in the HIN graph.
#[pyclass(name = "GoIWavefrontSolverNative")]
pub struct GoIWavefrontSolverNative {
    labels: Vec<String>,
    dimension: usize,
    m: DMatrix<f64>,
    u: DMatrix<f64>,
}

#[pymethods]
impl GoIWavefrontSolverNative {
    #[new]
    fn new(labels: Vec<String>, m_data: Vec<Vec<f64>>, u_data: Vec<Vec<f64>>) -> PyResult<Self> {
        let n = labels.len();
        if n == 0 {
            return Err(PyValueError::new_err("labels must be non-empty"));
        }
        let m = matrix_to_dmatrix(&m_data, n)?;
        let u = matrix_to_dmatrix(&u_data, n)?;
        Ok(Self { labels, dimension: n, m, u })
    }

    /// Compute K = I - U * M and return its determinant.
    fn operator_determinant(&self) -> f64 {
        let identity = DMatrix::<f64>::identity(self.dimension, self.dimension);
        let k = identity - &self.u * &self.m;
        k.determinant()
    }

    /// Return the execution wave operator EX(M, U) = (I - U * M)^-1 * U.
    fn ex_operator(&self) -> PyResult<Vec<Vec<f64>>> {
        let identity = DMatrix::<f64>::identity(self.dimension, self.dimension);
        let k = identity - &self.u * &self.m;
        let det = k.determinant();
        if det.abs() < 1e-12 {
            return Err(PyArithmeticError::new_err(
                "Cyclic dependency in G_HIN: (I - U*M) matrix operator is singular."
            ));
        }
        let k_inv = k.try_inverse().ok_or_else(|| {
            PyArithmeticError::new_err(
                "Cyclic dependency in G_HIN: (I - U*M) matrix operator is singular."
            )
        })?;
        let ex = k_inv * &self.u;
        let mut rows = Vec::with_capacity(self.dimension);
        for i in 0..self.dimension {
            let mut row = Vec::with_capacity(self.dimension);
            for j in 0..self.dimension {
                row.push(ex[(i, j)]);
            }
            rows.push(row);
        }
        Ok(rows)
    }

    /// Extract parallel wavefront stages using in-degree reduction on M.
    fn wavefront_stages(&self) -> PyResult<Vec<Vec<String>>> {
        let identity = DMatrix::<f64>::identity(self.dimension, self.dimension);
        let k = identity - &self.u * &self.m;
        let det = k.determinant();
        if det.abs() < 1e-12 {
            return Err(PyArithmeticError::new_err(
                "Cyclic dependency in G_HIN: (I - U*M) matrix operator is singular."
            ));
        }

        // Use M as the dependency matrix: M[i][j] != 0 means source j must
        // complete before target i can run (edge j -> i).
        let mut in_degree = vec![0usize; self.dimension];
        let mut outgoing: Vec<Vec<usize>> = vec![Vec::new(); self.dimension];
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                if self.m[(i, j)].abs() > 1e-12 {
                    in_degree[i] += 1;
                    outgoing[j].push(i);
                }
            }
        }

        let mut remaining = in_degree.clone();
        let mut seen = vec![false; self.dimension];
        let mut stages: Vec<Vec<String>> = Vec::new();

        loop {
            let mut stage = Vec::new();
            for i in 0..self.dimension {
                if !seen[i] && remaining[i] == 0 {
                    stage.push(self.labels[i].clone());
                    seen[i] = true;
                }
            }
            if stage.is_empty() {
                if seen.iter().all(|&x| x) {
                    break;
                }
                return Err(PyArithmeticError::new_err(
                    "Cyclic dependency in G_HIN: no wavefront stage could be extracted."
                ));
            }

            // Decrement in-degrees of neighbours for the nodes advanced this stage.
            for i in 0..self.dimension {
                if seen[i] {
                    for &j in &outgoing[i] {
                        if !seen[j] {
                            remaining[j] = remaining[j].saturating_sub(1);
                        }
                    }
                }
            }

            stage.sort();
            stages.push(stage);
        }

        Ok(stages)
    }
}

fn matrix_to_dmatrix(data: &Vec<Vec<f64>>, n: usize) -> PyResult<DMatrix<f64>> {
    let mut flat = Vec::with_capacity(n * n);
    for row in data.iter().take(n) {
        if row.len() != n {
            return Err(PyValueError::new_err("matrix must be square"));
        }
        for &v in row.iter().take(n) {
            flat.push(v);
        }
    }
    if flat.len() != n * n {
        return Err(PyValueError::new_err("matrix size mismatch"));
    }
    Ok(DMatrix::from_row_slice(n, n, &flat))
}
