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

use adjoint::PyAdjointHin;
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
