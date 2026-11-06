//! GoI wavefront scheduler with Sherman-Morrison-Woodbury low-rank updates.
//!
//! Models the research loop as `EX(M, U) = (I - U·M)^{-1} · U`. Termination
//! requires the spectral radius ρ(U·M) < 1 (strict nilpotency). Mid-research
//! parameter adjustments apply rank-k updates `ΔM = P·Qᵀ` in O(n²k) instead of
//! recomputing the full inverse in O(n³).

use nalgebra::DMatrix;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub struct GoIWavefrontScheduler {
    dim: usize,
    m_matrix: DMatrix<f64>,
    u_matrix: DMatrix<f64>,
    k_cached: Option<DMatrix<f64>>,
}

impl GoIWavefrontScheduler {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            m_matrix: DMatrix::zeros(dim, dim),
            u_matrix: DMatrix::identity(dim, dim),
            k_cached: None,
        }
    }

    pub fn set_edge(&mut self, from: usize, to: usize, weight: f64) {
        if from < self.dim && to < self.dim {
            self.m_matrix[(to, from)] = weight;
            self.k_cached = None;
        }
    }

    /// Returns the execution operator `K·U`; errors when `I - U·M` is singular
    /// (non-nilpotent cycle: spectral radius >= 1).
    pub fn compute_execution_operator(&mut self) -> Result<DMatrix<f64>, &'static str> {
        if let Some(ref k) = self.k_cached {
            return Ok(k.clone() * &self.u_matrix);
        }
        let um = &self.u_matrix * &self.m_matrix;
        let id = DMatrix::identity(self.dim, self.dim);
        let resolvent = id - um;
        match resolvent.try_inverse() {
            Some(inv) => {
                self.k_cached = Some(inv.clone());
                Ok(inv * &self.u_matrix)
            }
            None => Err("GoI Non-Nilpotent Cycle Detected: Spectral radius >= 1.0"),
        }
    }

    /// Spectral radius estimate of `U·M` via power iteration.
    pub fn spectral_radius(&self) -> f64 {
        if self.dim == 0 {
            return 0.0;
        }
        let um = &self.u_matrix * &self.m_matrix;
        let mut v = DMatrix::from_element(self.dim, 1, 1.0 / (self.dim as f64).sqrt());
        for _ in 0..200 {
            let mut w = &um * &v;
            let norm = w.norm();
            if norm < 1e-12 {
                return 0.0;
            }
            w /= norm;
            v = w;
        }
        let mut w = &um * &v;
        w.norm()
    }

    pub fn is_nilpotent(&self) -> bool {
        self.spectral_radius() < 1.0
    }

    /// Sherman-Morrison-Woodbury update for `ΔM = P·Qᵀ` (rank k = P.ncols()).
    pub fn apply_incremental_update(
        &mut self,
        p: &DMatrix<f64>,
        q: &DMatrix<f64>,
    ) -> Result<DMatrix<f64>, &'static str> {
        let k = match self.k_cached {
            Some(ref cached) => cached.clone(),
            None => {
                self.compute_execution_operator()?;
                self.k_cached.as_ref().unwrap().clone()
            }
        };
        let kup = &k * &self.u_matrix * p;
        let k_dim = p.ncols();
        let mid_id = DMatrix::identity(k_dim, k_dim);
        let inner = mid_id - (q.transpose() * &kup);
        let inner_inv = inner
            .try_inverse()
            .ok_or("Singular perturbation in delta update")?;
        let delta_k = &kup * inner_inv * (q.transpose() * &k);
        let k_prime = k + delta_k;
        self.k_cached = Some(k_prime.clone());
        self.m_matrix += p * q.transpose();
        Ok(k_prime * &self.u_matrix)
    }
}

#[pyclass(module = "invar_agent._native", name = "GoIWavefrontScheduler")]
pub struct PyGoIWavefrontScheduler {
    inner: GoIWavefrontScheduler,
}

fn mat_from_rows(rows: Vec<Vec<f64>>) -> PyResult<DMatrix<f64>> {
    let n = rows.len();
    if n == 0 {
        return Err(PyValueError::new_err("empty matrix"));
    }
    let cols = rows[0].len();
    if cols == 0 {
        return Err(PyValueError::new_err("empty matrix"));
    }
    let mut flat = Vec::with_capacity(n * cols);
    for row in &rows {
        if row.len() != cols {
            return Err(PyValueError::new_err("ragged matrix"));
        }
        flat.extend_from_slice(row);
    }
    Ok(DMatrix::from_row_slice(n, cols, &flat))
}

#[pymethods]
impl PyGoIWavefrontScheduler {
    #[new]
    pub fn new(dim: usize) -> Self {
        Self {
            inner: GoIWavefrontScheduler::new(dim),
        }
    }

    pub fn set_edge(&mut self, from: usize, to: usize, weight: f64) {
        self.inner.set_edge(from, to, weight);
    }

    pub fn spectral_radius(&self) -> f64 {
        self.inner.spectral_radius()
    }

    pub fn is_nilpotent(&self) -> bool {
        self.inner.is_nilpotent()
    }

    pub fn compute_execution_operator(&mut self) -> PyResult<Vec<Vec<f64>>> {
        let ex = self
            .inner
            .compute_execution_operator()
            .map_err(PyValueError::new_err)?;
        let n = ex.nrows();
        Ok((0..n)
            .map(|i| (0..n).map(|j| ex[(i, j)]).collect())
            .collect())
    }

    pub fn apply_incremental_update(
        &mut self,
        p_rows: Vec<Vec<f64>>,
        q_rows: Vec<Vec<f64>>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let p = mat_from_rows(p_rows).map_err(|_| PyValueError::new_err("invalid P matrix"))?;
        let q = mat_from_rows(q_rows).map_err(|_| PyValueError::new_err("invalid Q matrix"))?;
        let ex = self
            .inner
            .apply_incremental_update(&p, &q)
            .map_err(PyValueError::new_err)?;
        let n = ex.nrows();
        Ok((0..n)
            .map(|i| (0..n).map(|j| ex[(i, j)]).collect())
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acyclic_graph_is_nilpotent() {
        let mut s = GoIWavefrontScheduler::new(3);
        s.set_edge(0, 1, 1.0);
        s.set_edge(1, 2, 1.0);
        assert!(s.is_nilpotent());
        assert!(s.compute_execution_operator().is_ok());
    }

    #[test]
    fn cyclic_graph_fails_nilpotency() {
        let mut s = GoIWavefrontScheduler::new(2);
        s.set_edge(0, 1, 1.0);
        s.set_edge(1, 0, 1.0);
        assert!(!s.is_nilpotent());
        assert!(s.compute_execution_operator().is_err());
    }

    #[test]
    fn sherman_morrison_matches_full_recompute() {
        let mut s = GoIWavefrontScheduler::new(4);
        s.set_edge(0, 1, 0.2);
        s.set_edge(1, 2, 0.3);
        s.set_edge(2, 3, 0.1);
        let before = s.compute_execution_operator().unwrap();

        // rank-1 update ΔM = p·qᵀ with p = e0, q = 0.05·e3
        let p = DMatrix::from_row_slice(4, 1, &[1.0, 0.0, 0.0, 0.0]);
        let q = DMatrix::from_row_slice(4, 1, &[0.0, 0.0, 0.0, 0.05]);
        let updated = s.apply_incremental_update(&p, &q).unwrap();

        // Full recompute on a fresh scheduler with the same final M.
        let mut s2 = GoIWavefrontScheduler::new(4);
        s2.set_edge(0, 1, 0.2);
        s2.set_edge(1, 2, 0.3);
        s2.set_edge(2, 3, 0.1);
        s2.m_matrix[(0, 3)] += 1.0 * 0.05; // ΔM = p·qᵀ → M[to,from] += p[to]*q[from]
        let full = s2.compute_execution_operator().unwrap();

        let diff = (&updated - &full).norm();
        assert!(diff < 1e-9, "SM update diverged: {diff}");
        assert_ne!(before, updated);
    }
}
