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

mod deterministic_healer;
mod egraph_healer;
mod goi_matrix;
mod goi_solver;
mod goi_wavefront;
mod hin_arena;
mod hin_engine;
mod his;
mod uast;

use deterministic_healer::DeterministicHealerCore;
use egraph_healer::{repair_uast_expression, align_return_type, ASTRewritePatch, PreWriteHealer};
use goi_matrix::{GoIProofNet, GoIWavefrontSolverNative};
use goi_solver::{GoiEdge, GoiNode, GoiSolverCore};
use goi_wavefront::PyGoIWavefrontScheduler;
use hin_arena::{reduce_hin_uast, PyHinEngine as PyHinArenaEngine};
use hin_engine::HinEngineCore;
use his::HisBoundaryCore;
use uast::{PyGraphPolyglotMaterializer, PyUASTGraph, PyUASTKind};

fn to_set(items: Vec<String>) -> HashSet<String> {
    items.into_iter().collect()
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
    Ok(())
}
