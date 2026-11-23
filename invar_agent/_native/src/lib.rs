//! PyO3 bindings for the invar-agent native acceleration core.
//!
//! Exposes four zero-allocation, thread-safe structs:
//!   * `PyGoiSolver`             — Girard GoI nilpotency & wavefront solver
//!   * `PyHinEngine`             — arena-allocated Lafont Interaction Net
//!   * `PyDeterministicHealer`   — sub-ms 0-token diagnostic repair transducers
//!   * `PyHisRegister`           — 256-bit BLAKE3 Holographic Boundary Register

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};

mod adjoint;
mod builder_uast;
mod buckingham_pi;
mod csel_egraph;
mod deterministic_healer;
mod egraph_healer;
mod goi_matrix;
mod goi_solver;
mod goi_wavefront;
mod hin_arena;
mod hin_engine;
mod his;
mod holographic_hin;
mod uast;
#[cfg(test)]
mod epistemic_tests;

use adjoint::PyAdjointHin;
use adjoint::evaluate_interval_f64;
use csel_egraph::{csel_egraph_optimize, desugar_math_to_csel, resolve_symbol_fuzzy};
use deterministic_healer::DeterministicHealerCore;
use egraph_healer::{repair_uast_expression, align_return_type, ASTRewritePatch, PreWriteHealer};
use goi_matrix::{GoIProofNet, GoIWavefrontSolverNative};
use goi_solver::{GoiEdge, GoiNode, GoiSolverCore};
use goi_wavefront::PyGoIWavefrontScheduler;
use hin_arena::{reduce_hin_uast, PyHinEngine as PyHinArenaEngine};
use hin_engine::HinEngineCore;
use his::HisBoundaryCore;
use holographic_hin::PyHolographicHin;
use uast::{PyGraphPolyglotMaterializer, PyUASTGraph, PyUASTKind};

fn to_set(items: Vec<String>) -> HashSet<String> {
    items.into_iter().collect()
}

fn epistemic_json<T: serde::Serialize>(value: T) -> Result<String,String> {
    serde_json::to_string(&value).map_err(|e| e.to_string())
}

#[pyfunction]
fn py_compute_ihcg_digest(py: Python<'_>, ihcg_json: &str) -> PyResult<String> {
    py.allow_threads(|| his::compute_ihcg_digest(ihcg_json)).map_err(PyValueError::new_err)
}

#[pyfunction]
fn py_compute_ihcg_stage_chain(py: Python<'_>, prev_hash: &str, delta_json: &str) -> PyResult<String> {
    py.allow_threads(|| his::compute_ihcg_stage_chain(prev_hash,delta_json)).map_err(PyValueError::new_err)
}

#[pyfunction]
fn py_compute_cds_digests(py: Python<'_>, canon_model: &str, canon_claim: &str, domain_and_verdict: &str) -> PyResult<String> {
    py.allow_threads(|| epistemic_json(his::compute_cds_digests(canon_model,canon_claim,domain_and_verdict))).map_err(PyValueError::new_err)
}

#[pyfunction]
fn py_compute_certificate_receipt(py: Python<'_>, payload_json: &str) -> PyResult<String> {
    py.allow_threads(|| his::compute_certificate_receipt(payload_json)).map_err(PyValueError::new_err)
}

#[pyfunction]
fn py_verify_observational_purity(py: Python<'_>, wires_json: &str, writes_json: &str) -> PyResult<String> {
    py.allow_threads(|| -> Result<String,String> {
        let wires = serde_json::from_str(wires_json).map_err(|e| e.to_string())?;
        let writes: Vec<holographic_hin::SlotWriteOp> = serde_json::from_str(writes_json).map_err(|e| e.to_string())?;
        match holographic_hin::verify_observational_purity(&wires,&writes) {
            Ok(report) => epistemic_json(report),
            Err(violations) => epistemic_json(serde_json::json!({"valid":false,"violations":violations})),
        }
    }).map_err(PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (lhs, rhs, iter_limit=30, node_limit=20000, time_limit_ms=1500))]
fn py_check_algebraic_equivalence(py: Python<'_>, lhs: &str, rhs: &str, iter_limit: usize, node_limit: usize, time_limit_ms: u64) -> PyResult<String> {
    py.allow_threads(|| csel_egraph::check_algebraic_equivalence(lhs,rhs,iter_limit,node_limit,time_limit_ms).and_then(epistemic_json)).map_err(PyValueError::new_err)
}

#[pyfunction]
fn py_canonicalize_csel_expr(py: Python<'_>, expr: &str) -> PyResult<String> {
    py.allow_threads(|| csel_egraph::canonicalize_csel_expr(expr)).map_err(PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (eqs_json, max_aut_branches=64))]
fn py_compute_alpha_canonical_system(py: Python<'_>, eqs_json: &str, max_aut_branches: usize) -> PyResult<String> {
    py.allow_threads(|| -> Result<String,String> {
        let eqs: Vec<csel_egraph::TaggedCselEq> = serde_json::from_str(eqs_json).map_err(|e| e.to_string())?;
        csel_egraph::compute_alpha_canonical_system(&eqs,max_aut_branches).and_then(epistemic_json)
    }).map_err(PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (s_stack, dead_tol=1e-10, svd_tol=1e-10))]
fn py_compute_sensitivity_svd(py: Python<'_>, s_stack: Vec<Vec<f64>>, dead_tol: f64, svd_tol: f64) -> PyResult<String> {
    py.allow_threads(|| adjoint::compute_sensitivity_svd(&s_stack,dead_tol,svd_tol).and_then(epistemic_json)).map_err(PyValueError::new_err)
}

#[pyfunction]
fn py_compute_ift_equilibrium_sensitivity(py: Python<'_>, jx: Vec<Vec<f64>>, jth: Vec<Vec<f64>>, dy_dx: Vec<Vec<f64>>, dy_dth: Vec<Vec<f64>>) -> PyResult<String> {
    py.allow_threads(|| adjoint::compute_ift_equilibrium_sensitivity(&jx,&jth,&dy_dx,&dy_dth).and_then(epistemic_json)).map_err(PyValueError::new_err)
}

#[pyfunction]
fn py_certify_krawczyk_fixed_point(py: Python<'_>, exprs: Vec<String>, z_center: Vec<f64>, rad: f64) -> PyResult<String> {
    py.allow_threads(|| adjoint::certify_krawczyk_fixed_point(&exprs,&z_center,rad).and_then(epistemic_json)).map_err(PyValueError::new_err)
}

#[pyfunction]
fn py_certify_interval_stability(py: Python<'_>, j_iv_json: &str) -> PyResult<String> {
    py.allow_threads(|| -> Result<String,String> {
        let matrix: Vec<Vec<adjoint::IntervalF64>> = serde_json::from_str(j_iv_json).map_err(|e| e.to_string())?;
        adjoint::certify_interval_stability(&matrix).and_then(epistemic_json)
    }).map_err(PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (margin_expr, box_lo, box_hi, eps_total, max_boxes=32768))]
fn py_branch_and_bound_refute(py: Python<'_>, margin_expr: &str, box_lo: Vec<f64>, box_hi: Vec<f64>, eps_total: f64, max_boxes: usize) -> PyResult<String> {
    py.allow_threads(|| adjoint::branch_and_bound_refute(margin_expr,&box_lo,&box_hi,eps_total,max_boxes).and_then(epistemic_json)).map_err(PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (uast_json))]
pub fn validate_builder_uast(py: Python<'_>, uast_json: &str) -> PyResult<String> {
    py.allow_threads(|| crate::builder_uast::validate_uast_json(uast_json)).map_err(PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (uast_json, feedback_weights_json = None))]
pub fn schedule_builder_wavefronts(py: Python<'_>, uast_json: &str, feedback_weights_json: Option<&str>) -> PyResult<String> {
    py.allow_threads(|| crate::goi_wavefront::schedule_project_graph_json(uast_json, feedback_weights_json)).map_err(PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (resolvent_json, node_order_json, delta_edges_json, changed_his_nodes_json))]
pub fn compute_builder_smw_delta(py: Python<'_>, resolvent_json: &str, node_order_json: &str, delta_edges_json: &str, changed_his_nodes_json: &str) -> PyResult<String> {
    py.allow_threads(|| crate::goi_wavefront::smw_rank_k_update_json(resolvent_json, node_order_json, delta_edges_json, changed_his_nodes_json)).map_err(PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (domain_context, canonical_payload_json))]
pub fn compute_his_digest_v2(domain_context: &str, canonical_payload_json: &str) -> PyResult<String> {
    Ok(crate::his::domain_separated_blake3_hex(domain_context, canonical_payload_json.as_bytes()))
}

#[pyfunction]
#[pyo3(signature = (fields_json))]
pub fn compute_fault_fingerprint(fields_json: &str) -> PyResult<String> {
    crate::his::compute_fault_fingerprint_json(fields_json).map_err(PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (csel_expr_json, symbol_dims_json))]
pub fn verify_equation_dimensions_hin(py: Python<'_>, csel_expr_json: &str, symbol_dims_json: &str) -> PyResult<String> {
    py.allow_threads(|| crate::hin_engine::check_equation_dimensional_homogeneity_json(csel_expr_json, symbol_dims_json)).map_err(PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (uast_json, fault_node_id, max_hops = 3))]
pub fn extract_hin_causal_slice(py: Python<'_>, uast_json: &str, fault_node_id: &str, max_hops: usize) -> PyResult<String> {
    py.allow_threads(|| crate::hin_engine::extract_builder_causal_slice_json(uast_json, fault_node_id, max_hops)).map_err(PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (source_code, module_kind = "llm_extension_slot"))]
pub fn heal_builder_tier0(py: Python<'_>, source_code: &str, module_kind: &str) -> PyResult<String> {
    py.allow_threads(|| crate::deterministic_healer::heal_builder_module_tier0_json(source_code, module_kind)).map_err(PyValueError::new_err)
}

/// Geometry of Interaction task scheduler.
#[pyclass(module = "invar_agent._native", name = "PyGoiSolver")]
pub struct PyGoiSolver {
    inner: GoiSolverCore,
}

#[pymethods]
impl PyGoiSolver {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: GoiSolverCore::new(),
        }
    }

    /// Registers a task node with read/write path sets and port polarities.
    pub fn add_task_node(
        &mut self,
        node_id: String,
        read_set: Vec<String>,
        write_set: Vec<String>,
        in_ports: Vec<String>,
        out_ports: Vec<String>,
    ) -> PyResult<()> {
        self.inner
            .add_node(GoiNode {
                node_id,
                read_set: to_set(read_set),
                write_set: to_set(write_set),
                input_ports: in_ports,
                output_ports: out_ports,
            })
            .map_err(PyValueError::new_err)
    }

    /// Connects an output port of `src_node` to an input port of `dst_node`.
    pub fn add_dependency_edge(
        &mut self,
        src_node: String,
        src_port: String,
        dst_node: String,
        dst_port: String,
    ) -> PyResult<()> {
        self.inner
            .add_edge(GoiEdge {
                src_node,
                src_port,
                dst_node,
                dst_port,
            })
            .map_err(PyValueError::new_err)
    }

    /// Verifies Girard's GoI nilpotency condition `(Pi_22 * sigma)^k == 0`.
    pub fn check_nilpotency(&self) -> bool {
        self.inner.verify_nilpotency()
    }

    /// Partitions task nodes into parallel wavefront generations enforcing
    /// strict Bernstein read/write set disjointness within each wavefront.
    pub fn solve_wavefronts(&self) -> PyResult<Vec<Vec<String>>> {
        self.inner
            .compute_disjoint_wavefronts()
            .map_err(PyRuntimeError::new_err)
    }
}

/// Arena-allocated Lafont Interaction Net engine.
#[pyclass(module = "invar_agent._native", name = "PyHinEngine")]
pub struct PyHinEngine {
    inner: HinEngineCore,
}

#[pymethods]
impl PyHinEngine {
    #[new]
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: HinEngineCore::with_capacity(capacity),
        }
    }

    /// Allocates an Interaction Agent with 1 principal port and N auxiliary
    /// ports.  Returns the arena agent id.
    pub fn spawn_agent(
        &mut self,
        symbol_name: String,
        principal_port: String,
        aux_ports: Vec<String>,
    ) -> u64 {
        self.inner.spawn_agent(symbol_name, principal_port, aux_ports)
    }

    /// Connects two ports; principal-principal meetings register an Active
    /// Pair for annihilation/commutation reduction.
    pub fn connect_ports(
        &mut self,
        agent_a: u64,
        a_is_principal: bool,
        agent_b: u64,
        b_is_principal: bool,
    ) -> PyResult<()> {
        self.inner
            .connect_ports(agent_a, a_is_principal, agent_b, b_is_principal)
            .map_err(PyValueError::new_err)
    }

    /// Reduces all active pairs via deterministic annihilation and
    /// commutation rules, freeing annihilated arena slots in O(1) time.
    pub fn reduce_to_normal_form(&mut self) -> usize {
        self.inner.reduce_all()
    }

    /// Traces backward from a failing agent along causal wires to return the
    /// minimal root-cause symbol list.
    pub fn trace_causal_fault_slice(&self, fault_agent_id: u64, max_hops: usize) -> Vec<String> {
        self.inner.trace_fault_slice(fault_agent_id, max_hops)
    }

    pub fn active_agent_count(&self) -> usize {
        self.inner.active_count()
    }
}

/// Zero-token deterministic diagnostic & syntax/import healer.
#[pyclass(module = "invar_agent._native", name = "PyDeterministicHealer")]
pub struct PyDeterministicHealer {
    inner: DeterministicHealerCore,
}

#[pymethods]
impl PyDeterministicHealer {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: DeterministicHealerCore::new_with_builtin_rules(),
        }
    }

    /// Whether a diagnostic matches a deterministic Tier-0 repair rule.
    pub fn can_heal(&self, error_code: &str, diagnostic_msg: &str) -> bool {
        self.inner.matches_rule(error_code, diagnostic_msg)
    }

    /// Applies the deterministic repair transducer.  Returns
    /// `(healed_source, applied_rule_id)` or `None`.
    pub fn match_and_heal(
        &self,
        error_code: &str,
        diagnostic_msg: &str,
        source_code: &str,
        line_number: usize,
    ) -> Option<(String, String)> {
        self.inner
            .apply_reflex_heal(error_code, diagnostic_msg, source_code, line_number)
    }
}

/// 256-bit BLAKE3 Holographic Boundary Register.
#[pyclass(module = "invar_agent._native", name = "PyHisRegister")]
pub struct PyHisRegister {
    inner: HisBoundaryCore,
}

#[pymethods]
impl PyHisRegister {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: HisBoundaryCore::new(),
        }
    }

    /// Updates a module's canonicalized public interface signature.
    pub fn update_module_boundary(&mut self, module_path: String, canonical_signature: String) {
        self.inner.update_module(module_path, canonical_signature);
    }

    /// Projects the bulk module map onto the 256-bit boundary checksum.
    pub fn project_boundary_digest(&self) -> String {
        self.inner.compute_xor_blake3_digest()
    }

    /// Verifies the live register against an expected boundary digest.
    pub fn verify_boundary_parity(&self, expected_digest: &str) -> bool {
        self.inner.verify_parity(expected_digest)
    }
}

#[pymodule]
fn _native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_compute_ihcg_digest, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_ihcg_stage_chain, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_cds_digests, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_certificate_receipt, m)?)?;
    m.add_function(wrap_pyfunction!(py_verify_observational_purity, m)?)?;
    m.add_function(wrap_pyfunction!(py_check_algebraic_equivalence, m)?)?;
    m.add_function(wrap_pyfunction!(py_canonicalize_csel_expr, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_alpha_canonical_system, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_sensitivity_svd, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_ift_equilibrium_sensitivity, m)?)?;
    m.add_function(wrap_pyfunction!(py_certify_krawczyk_fixed_point, m)?)?;
    m.add_function(wrap_pyfunction!(py_certify_interval_stability, m)?)?;
    m.add_function(wrap_pyfunction!(py_branch_and_bound_refute, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_interval_f64, m)?)?;
    m.add_function(wrap_pyfunction!(validate_builder_uast, m)?)?;
    m.add_function(wrap_pyfunction!(schedule_builder_wavefronts, m)?)?;
    m.add_function(wrap_pyfunction!(compute_builder_smw_delta, m)?)?;
    m.add_function(wrap_pyfunction!(compute_his_digest_v2, m)?)?;
    m.add_function(wrap_pyfunction!(compute_fault_fingerprint, m)?)?;
    m.add_function(wrap_pyfunction!(verify_equation_dimensions_hin, m)?)?;
    m.add_function(wrap_pyfunction!(extract_hin_causal_slice, m)?)?;
    m.add_function(wrap_pyfunction!(heal_builder_tier0, m)?)?;
    m.add_class::<PyGoiSolver>()?;
    m.add_class::<PyHinEngine>()?;
    m.add_class::<PyDeterministicHealer>()?;
    m.add_class::<PyHisRegister>()?;
    // Consolidated aero-forge research-agent surface.
    m.add_class::<PyUASTKind>()?;
    m.add_class::<PyUASTGraph>()?;
    m.add_class::<PyHinArenaEngine>()?;
    m.add_class::<PyGoIWavefrontScheduler>()?;
    m.add_class::<GoIProofNet>()?;
    m.add_class::<GoIWavefrontSolverNative>()?;
    m.add_class::<PyGraphPolyglotMaterializer>()?;
    m.add_function(wrap_pyfunction!(repair_uast_expression, m)?)?;
    m.add_function(wrap_pyfunction!(align_return_type, m)?)?;
    m.add_class::<ASTRewritePatch>()?;
    m.add_class::<PreWriteHealer>()?;
    m.add_function(wrap_pyfunction!(reduce_hin_uast, m)?)?;
    // UAST/HIN Research Compute Engine v2 (Holographic HIN, Adjoint HIN, CSEL E-Graph)
    m.add_class::<PyHolographicHin>()?;
    m.add_class::<PyAdjointHin>()?;
    m.add_function(wrap_pyfunction!(csel_egraph_optimize, m)?)?;
    m.add_function(wrap_pyfunction!(desugar_math_to_csel, m)?)?;
    m.add_function(wrap_pyfunction!(resolve_symbol_fuzzy, m)?)?;
    Ok(())
}
