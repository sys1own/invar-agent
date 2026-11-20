//! GoI wavefront scheduler with Sherman-Morrison-Woodbury low-rank updates.
//!
//! Models the research loop as `EX(M, U) = (I - U·M)^{-1} · U`. Termination
//! requires the spectral radius ρ(U·M) < 1 (strict nilpotency). Mid-research
//! parameter adjustments apply rank-k updates `ΔM = P·Qᵀ` in O(n²k) instead of
//! recomputing the full inverse in O(n³).

use nalgebra::DMatrix;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use crate::builder_uast::{ProjectGraphUAST, SmwDeltaResult, WavefrontScheduleResult};
use std::collections::{BTreeMap, BTreeSet};

fn row_major(matrix: &DMatrix<f64>) -> Vec<f64> {
    (0..matrix.nrows()).flat_map(|r| (0..matrix.ncols()).map(move |c| matrix[(r, c)])).collect()
}

#[cfg(test)]
mod builder_tests {
    use super::*;
    use crate::builder_uast::tests::{graph, module};
    use serde_json::{json, Value};

    fn edge(from: &str, to: &str) -> Value {
        json!({"from_node":from, "to_node":to, "edge_kind":"verification_dep", "port_id":null})
    }

    #[test]
    fn test_goi_smw_rank2_exactness() {
        let ids = ["kernel_core", "port_core", "mod_sdk", "mod_cli", "mod_sweep", "mod_data", "slot_custom_viz", "mod_workbench_api", "mod_workbench_ui", "mod_tests", "mod_receipt"];
        let links = [(0,1),(1,2),(2,3),(2,4),(2,5),(2,6),(6,7),(7,8),(2,9),(3,10),(4,10),(5,10),(8,10),(9,10)];
        let g = graph(ids.iter().map(|id| module(id)).collect(), links.iter().map(|&(u,v)| edge(ids[u],ids[v])).collect());
        let scheduled: WavefrontScheduleResult = serde_json::from_str(&schedule_project_graph_json(&g.to_string(), None).unwrap()).unwrap();
        assert_eq!(scheduled.spectral_radius, 0.0);
        assert_eq!(scheduled.wavefronts[0], vec!["kernel_core"]);
        assert_eq!(scheduled.wavefronts.last().unwrap(), &vec!["mod_receipt"]);
        let order = &scheduled.node_order;
        let r_json = serde_json::to_string(&scheduled.resolvent_flat).unwrap();
        let order_json = serde_json::to_string(order).unwrap();
        let delta = json!([{"from_node":"mod_cli","to_node":"mod_tests","weight":1.0}, {"from_node":"mod_data","to_node":"mod_workbench_api","weight":1.0}]);
        let result: SmwDeltaResult = serde_json::from_str(&smw_rank_k_update_json(&r_json, &order_json, &delta.to_string(), "[]").unwrap()).unwrap();
        let mut full = g.clone();
        full["edges"].as_array_mut().unwrap().extend([edge("mod_cli","mod_tests"), edge("mod_data","mod_workbench_api")]);
        let direct: WavefrontScheduleResult = serde_json::from_str(&schedule_project_graph_json(&full.to_string(), None).unwrap()).unwrap();
        assert!((result.capacitance_det - 1.0).abs() < 1e-12);
        assert!(result.updated_resolvent_flat.iter().zip(&direct.resolvent_flat).all(|(a,b)| (a-b).abs() < 1e-12));
        let cycle_delta = json!([{"from_node":"mod_workbench_api", "to_node":"slot_custom_viz", "weight":1.0}]);
        let err: Value = serde_json::from_str(&smw_rank_k_update_json(&r_json, &order_json, &cycle_delta.to_string(), "[]").unwrap_err()).unwrap();
        assert!(err["capacitance_det"].as_f64().unwrap().abs() < 1e-9);
        assert!(err["spectral_radius"].as_f64().unwrap() >= 1.0 - 1e-12);
        assert_eq!(err["cycle_witness"], json!(["mod_workbench_api", "slot_custom_viz"]));
        assert_eq!(err["llm_retries"], 0);
        let mut cyclic = g;
        cyclic["edges"].as_array_mut().unwrap().push(edge("mod_workbench_api", "slot_custom_viz"));
        assert!(schedule_project_graph_json(&cyclic.to_string(), None).unwrap_err().contains("GOI_IRREDUCIBLE_CYCLE"));
        let all: SmwDeltaResult = serde_json::from_str(&smw_rank_k_update_json(&r_json, &order_json, "[]", r#"["kernel_core"]"#).unwrap()).unwrap();
        assert_eq!(all.affected_node_ids.len(), 11);
        let removal = json!([{"from_node":"mod_data", "to_node":"mod_workbench_api", "weight":-1.0}, {"from_node":"mod_cli", "to_node":"mod_tests", "weight":-1.0}]);
        let reverted: SmwDeltaResult = serde_json::from_str(&smw_rank_k_update_json(&serde_json::to_string(&result.updated_resolvent_flat).unwrap(), &order_json, &removal.to_string(), "[]").unwrap()).unwrap();
        assert!(reverted.updated_resolvent_flat.iter().zip(&scheduled.resolvent_flat).all(|(a,b)| (a-b).abs() < 1e-12));
    }

    #[test]
    fn builder_goi_empty_and_invalid_inputs() {
        let empty = graph(vec![], vec![]).to_string();
        let result: WavefrontScheduleResult = serde_json::from_str(&schedule_project_graph_json(&empty, None).unwrap()).unwrap();
        assert!(result.wavefronts.is_empty());
        assert!(smw_rank_k_update_json("[]", "[]", "[]", "[]").is_ok());
        assert!(smw_rank_k_update_json("[1]", "[]", "[]", "[]").is_err());
        assert!(smw_rank_k_update_json("[0]", r#"["a"]"#, "[]", "[]").is_err());
        assert!(smw_rank_k_update_json("[1]", r#"["a"]"#, "[]", r#"["missing"]"#).is_err());
        let g = graph(vec![module("a"),module("b")],vec![edge("a","b")]).to_string();
        assert!(schedule_project_graph_json(&g, Some("[-1,1]")).is_err());
        assert!(schedule_project_graph_json(&g, Some("[1]")).is_err());
        assert!(schedule_project_graph_json(&g, Some("[[1,1],[0,1]]")).unwrap_err().contains("GOI_IRREDUCIBLE_CYCLE"));
        assert!(schedule_project_graph_json(&g, Some(r#"{"missing":1}"#)).is_err());
        let self_cycle = graph(vec![module("a")], vec![edge("a","a")]).to_string();
        assert!(schedule_project_graph_json(&self_cycle, None).unwrap_err().contains("cycle_witness"));
    }
}

fn graph_analysis(matrix: &DMatrix<f64>, order: &[String]) -> Result<(Vec<Vec<String>>, Vec<String>, f64), String> {
    let n = order.len();
    let mut graph = petgraph::graph::DiGraph::<(), ()>::new();
    let vertices: Vec<_> = (0..n).map(|_| graph.add_node(())).collect();
    let mut indegree = vec![0usize; n];
    for to in 0..n {
        for from in 0..n {
            if matrix[(to, from)].abs() > 1e-9 {
                graph.add_edge(vertices[from], vertices[to], ());
                indegree[to] += 1;
            }
        }
    }
    let mut cycle = BTreeSet::new();
    for component in petgraph::algo::tarjan_scc(&graph) {
        if component.len() > 1 || graph.contains_edge(component[0], component[0]) {
            for node in component { cycle.insert(order[node.index()].clone()); }
        }
    }
    if !cycle.is_empty() {
        let schur = nalgebra::linalg::Schur::try_new(matrix.clone(), 1e-14, 10000)
            .ok_or("GoI spectral decomposition did not converge")?;
        let rho = schur.complex_eigenvalues().iter().map(|x| x.norm()).fold(0.0f64, f64::max);
        return Ok((vec![], cycle.into_iter().collect(), rho));
    }
    let mut ready: Vec<_> = (0..n).filter(|&i| indegree[i] == 0).collect();
    let mut waves = Vec::new();
    while !ready.is_empty() {
        waves.push(ready.iter().map(|&i| order[i].clone()).collect());
        let mut next = BTreeSet::new();
        for from in ready {
            for to in 0..n {
                if matrix[(to, from)].abs() > 1e-9 {
                    indegree[to] -= 1;
                    if indegree[to] == 0 { next.insert(to); }
                }
            }
        }
        ready = next.into_iter().collect();
    }
    Ok((waves, vec![], 0.0))
}

fn cycle_error(matrix: &DMatrix<f64>, order: &[String], witness: &[String], rho: f64, det: f64) -> String {
    let mut edges = Vec::new();
    for to in 0..order.len() {
        for from in 0..order.len() {
            if matrix[(to, from)].abs() > 1e-9 && witness.contains(&order[from]) && witness.contains(&order[to]) {
                edges.push(serde_json::json!({"from_node":order[from], "to_node":order[to]}));
            }
        }
    }
    serde_json::json!({"error_code":"GOI_IRREDUCIBLE_CYCLE", "tier":"CRITICAL_NON_RETRYABLE", "action":"ABORT_BUILD_CRITICAL", "llm_retries":0, "spectral_radius":rho, "capacitance_det":det, "cycle_witness":witness, "cycle_edges":edges}).to_string()
}

pub fn schedule_project_graph_json(uast_json: &str, feedback_weights_json: Option<&str>) -> Result<String, String> {
    let graph = ProjectGraphUAST::from_json(uast_json)?;
    let mut order: Vec<_> = graph.nodes.iter().map(|n| n.node_id().to_owned()).collect();
    order.sort();
    let index: BTreeMap<_, _> = order.iter().enumerate().map(|(i, id)| (id.as_str(), i)).collect();
    let n = order.len();
    let mut m = DMatrix::zeros(n, n);
    for edge in &graph.edges { m[(index[edge.to_node.as_str()], index[edge.from_node.as_str()])] = 1.0; }
    let mut u = DMatrix::identity(n, n);
    if let Some(json) = feedback_weights_json {
        #[derive(serde::Deserialize)]
        #[serde(untagged)]
        enum Weights { Diagonal(Vec<f64>), Matrix(Vec<Vec<f64>>), Named(BTreeMap<String, f64>) }
        match serde_json::from_str(json).map_err(|e| e.to_string())? {
            Weights::Diagonal(weights) if weights.len() == n => {
                for (i, w) in weights.into_iter().enumerate() { u[(i, i)] = w; }
            }
            Weights::Matrix(rows) if rows.len() == n && rows.iter().all(|r| r.len() == n) => {
                for r in 0..n { for c in 0..n { u[(r, c)] = rows[r][c]; } }
            }
            Weights::Named(weights) => {
                for (id, w) in weights {
                    let i = *index.get(id.as_str()).ok_or("Unknown feedback node")?;
                    u[(i, i)] = w;
                }
            }
            _ => return Err("GoI feedback matrix shape mismatch".into()),
        }
    }
    if u.iter().any(|x| !x.is_finite() || *x < 0.0) { return Err("GoI feedback weights must be finite and nonnegative".into()); }
    let um = u * &m;
    let (_, witness, rho) = graph_analysis(&um, &order)?;
    let (waves, structural_cycle, _) = graph_analysis(&m, &order)?;
    let a = DMatrix::identity(n, n) - &um;
    let det = a.determinant();
    if rho >= 1.0 || det.abs() < 1e-9 || !witness.is_empty() || !structural_cycle.is_empty() {
        return Err(cycle_error(&um, &order, if witness.is_empty() { &structural_cycle } else { &witness }, rho, det));
    }
    let r = a.try_inverse().ok_or("GoI resolvent is singular")?;
    serde_json::to_string(&WavefrontScheduleResult { is_nilpotent:true, spectral_radius:rho, wavefronts:waves, resolvent_flat:row_major(&r), node_order:order, cycle_witness:vec![] }).map_err(|e| e.to_string())
}

pub fn smw_rank_k_update_json(resolvent_json: &str, node_order_json: &str, delta_edges_json: &str, changed_his_nodes_json: &str) -> Result<String, String> {
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct DeltaEdge {
        from_node: String,
        to_node: String,
        #[serde(default = "unit_weight", alias = "delta_weight")]
        weight: f64,
    }
    fn unit_weight() -> f64 { 1.0 }
    let order: Vec<String> = serde_json::from_str(node_order_json).map_err(|e| e.to_string())?;
    let flat: Vec<f64> = serde_json::from_str(resolvent_json).map_err(|e| e.to_string())?;
    let deltas: Vec<DeltaEdge> = serde_json::from_str(delta_edges_json).map_err(|e| e.to_string())?;
    let changed: Vec<String> = serde_json::from_str(changed_his_nodes_json).map_err(|e| e.to_string())?;
    let n = order.len();
    let index: BTreeMap<_, _> = order.iter().enumerate().map(|(i, id)| (id.as_str(), i)).collect();
    if index.len() != n || order.iter().any(|s| s.is_empty()) || n.checked_mul(n) != Some(flat.len()) || flat.iter().any(|x| !x.is_finite()) {
        return Err("GoI invalid node order or resolvent shape".into());
    }
    let mut roots = BTreeSet::new();
    for id in changed { roots.insert(*index.get(id.as_str()).ok_or("Unknown changed HIS node")?); }
    let r = DMatrix::from_row_slice(n, n, &flat);
    let signed_update = deltas.iter().any(|edge| edge.weight < 0.0);
    let mut updated_m = if signed_update {
        DMatrix::identity(n, n) - r.clone().try_inverse().ok_or("GoI input resolvent is singular")?
    } else {
        if (0..n).any(|i| (r[(i, i)] - 1.0).abs() > 1e-9) || r.iter().any(|x| *x < -1e-9) {
            return Err("GoI input resolvent is not a DAG resolvent".into());
        }
        // Positive off-diagonal entries of a DAG resolvent are its transitive closure.
        DMatrix::from_fn(n, n, |to, from| if to != from && r[(to, from)] > 1e-9 { 1.0 } else { 0.0 })
    };
    let (_, old_cycle, old_rho) = graph_analysis(&updated_m, &order)?;
    if !old_cycle.is_empty() || old_rho >= 1.0 { return Err(cycle_error(&updated_m, &order, &old_cycle, old_rho, 0.0)); }
    let k = deltas.len();
    let mut p = DMatrix::zeros(n, k);
    let mut q = DMatrix::zeros(n, k);
    // Inputs are delta-M edges; A = I - M changes by P*Q^T = -delta-M.
    for (j, edge) in deltas.iter().enumerate() {
        if !edge.weight.is_finite() { return Err("GoI nonfinite delta weight".into()); }
        let from = *index.get(edge.from_node.as_str()).ok_or("Unknown delta source")?;
        let to = *index.get(edge.to_node.as_str()).ok_or("Unknown delta target")?;
        p[(to, j)] = -edge.weight;
        q[(from, j)] = 1.0;
        updated_m[(to, from)] += edge.weight;
        if edge.weight != 0.0 { roots.insert(to); }
    }
    if updated_m.iter().any(|x| !x.is_finite() || *x < -1e-9) { return Err("GoI delta removes a nonexistent dependency".into()); }
    let cap = DMatrix::identity(k, k) + q.transpose() * &r * &p;
    let det = cap.determinant();
    let (_, mut witness, mut rho) = graph_analysis(&updated_m, &order)?;
    if !witness.is_empty() && !signed_update {
        // Reconstruct exact weights only on rejection, not on the O(n²k) forward path.
        updated_m = DMatrix::identity(n, n) - r.clone().try_inverse().ok_or("GoI input resolvent is singular")?;
        for edge in &deltas { updated_m[(index[edge.to_node.as_str()], index[edge.from_node.as_str()])] += edge.weight; }
        let (_, exact_witness, exact_rho) = graph_analysis(&updated_m, &order)?;
        witness = exact_witness;
        rho = exact_rho;
    }
    if !det.is_finite() || det.abs() < 1e-9 || rho >= 1.0 || !witness.is_empty() {
        return Err(cycle_error(&updated_m, &order, &witness, rho, det));
    }
    let updated = if k == 0 { r } else {
        let cap_inv = cap.try_inverse().ok_or("GoI capacitance is singular")?;
        &r - (&r * p) * cap_inv * (q.transpose() * &r)
    };
    if updated.iter().any(|x| !x.is_finite()) { return Err("GoI nonfinite SMW result".into()); }
    let affected = (0..n).filter(|to| roots.iter().any(|from| updated[(*to, *from)].abs() > 1e-9)).map(|i| order[i].clone()).collect();
    serde_json::to_string(&SmwDeltaResult { is_valid_dag:true, capacitance_det:det, updated_resolvent_flat:row_major(&updated), affected_node_ids:affected }).map_err(|e| e.to_string())
}

/// 2-norm condition number κ(A) = σ_max / σ_min via SVD; infinite when the
/// matrix is rank-deficient (or SVD fails to converge).
fn inner_condition_number(a: &DMatrix<f64>) -> f64 {
    if a.nrows() == 0 || a.ncols() == 0 {
        return f64::INFINITY;
    }
    let svd = nalgebra::SVD::new(a.clone(), false, false);
    let s = svd.singular_values;
    if s.is_empty() {
        return f64::INFINITY;
    }
    let mut smax = 0.0f64;
    let mut smin = f64::INFINITY;
    for &v in s.iter() {
        if v > smax {
            smax = v;
        }
        if v > 0.0 && v < smin {
            smin = v;
        }
    }
    if smin.is_infinite() || smin == 0.0 {
        return f64::INFINITY;
    }
    smax / smin
}

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
    ///
    /// The inner system `I - Qᵀ·K·U·P` must be well conditioned; when it is
    /// singular or ill-conditioned (κ > 1e8) the low-rank update would
    /// accumulate floating-point drift, so we fall back to a full
    /// recomputation `EX(M, U) = (I - U·M)^{-1} · U` from the updated M.
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
        if inner_condition_number(&inner) > 1e8 {
            self.m_matrix += p * q.transpose();
            self.k_cached = None;
            return self.compute_execution_operator();
        }
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
