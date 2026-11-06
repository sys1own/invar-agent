//! Universal Abstract Syntax Tree (UAST) indexed arena + polyglot materializer.
//!
//! Nodes live in a flat `Vec` arena addressed by `u64` ids; Python holds only
//! handles. Subtree hashes use BLAKE3 for content-addressed memoization.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Kind of UAST node (mirrors the transpiler dialect).
#[pyclass(module = "invar_agent._native", name = "UASTKind")]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum PyUASTKind {
    Module,
    FunctionDef,
    Return,
    Assign,
    BinaryOp,
    Call,
    ConstantInt,
    ConstantFloat,
    Identifier,
    LoopFor,
    LoopWhile,
}

impl PyUASTKind {
    fn tag(&self) -> &'static str {
        match self {
            PyUASTKind::Module => "module",
            PyUASTKind::FunctionDef => "function_declaration",
            PyUASTKind::Return => "return",
            PyUASTKind::Assign => "binding",
            PyUASTKind::BinaryOp => "binop",
            PyUASTKind::Call => "call",
            PyUASTKind::ConstantInt => "constant_int",
            PyUASTKind::ConstantFloat => "constant_float",
            PyUASTKind::Identifier => "reference",
            PyUASTKind::LoopFor => "loop_for",
            PyUASTKind::LoopWhile => "loop_while",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UASTNode {
    pub kind: String,
    pub symbol: Option<String>,
    pub val_i: Option<i64>,
    pub val_f: Option<f64>,
    pub children: Vec<u64>,
}

/// Flat arena storage for UAST nodes.
#[derive(Default)]
pub struct UASTStorage {
    nodes: Vec<UASTNode>,
}

impl UASTStorage {
    pub fn insert(
        &mut self,
        kind: PyUASTKind,
        symbol: Option<String>,
        val_i: Option<i64>,
        val_f: Option<f64>,
        children: Vec<u64>,
    ) -> u64 {
        let id = self.nodes.len() as u64;
        self.nodes.push(UASTNode {
            kind: kind.tag().to_string(),
            symbol,
            val_i,
            val_f,
            children,
        });
        id
    }

    pub fn node(&self, id: u64) -> Option<&UASTNode> {
        self.nodes.get(id as usize)
    }

    /// Canonical JSON of a subtree (children first), hashed with BLAKE3.
    pub fn compute_blake3_hash(&self, root_id: u64) -> [u8; 32] {
        let canonical = self.canonical_json(root_id);
        *blake3::hash(canonical.as_bytes()).as_bytes()
    }

    fn canonical_json(&self, id: u64) -> String {
        let Some(node) = self.node(id) else {
            return "null".to_string();
        };
        let mut v = serde_json::Map::new();
        v.insert("kind".into(), Value::String(node.kind.clone()));
        if let Some(s) = &node.symbol {
            v.insert("symbol".into(), Value::String(s.clone()));
        }
        if let Some(i) = node.val_i {
            v.insert("val_i".into(), Value::from(i));
        }
        if let Some(f) = node.val_f {
            v.insert("val_f".into(), Value::from(f));
        }
        let kids: Vec<Value> = node
            .children
            .iter()
            .map(|&c| Value::String(self.canonical_json(c)))
            .collect();
        v.insert("children".into(), Value::Array(kids));
        Value::Object(v).to_string()
    }

    /// Convert the subtree rooted at `root_id` into the transpiler UAST JSON
    /// dialect consumed by `HinEngine.build_from_json`.
    pub fn to_uast_json(&self, root_id: u64) -> Result<String, String> {
        let v = self.subtree_to_dialect(root_id)?;
        serde_json::to_string(&v).map_err(|e| e.to_string())
    }

    fn subtree_to_dialect(&self, id: u64) -> Result<Value, String> {
        let node = self
            .node(id)
            .ok_or_else(|| format!("unknown UAST node id {id}"))?;
        let kids: Vec<Value> = node
            .children
            .iter()
            .map(|&c| self.subtree_to_dialect(c))
            .collect::<Result<_, _>>()?;
        let mut v = serde_json::Map::new();
        v.insert("type".into(), Value::String(node.kind.clone()));
        if let Some(s) = &node.symbol {
            v.insert("name".into(), Value::String(s.clone()));
        }
        match node.kind.as_str() {
            "constant_int" => {
                v.insert("value".into(), Value::from(node.val_i.unwrap_or(0)));
            }
            "constant_float" => {
                v.insert("value".into(), Value::from(node.val_f.unwrap_or(0.0)));
            }
            "binary_op" => {
                if kids.len() == 2 {
                    v.insert("arguments".into(), Value::Array(kids));
                }
            }
            "call" | "function_declaration" | "binding" | "return" | "module"
            | "loop_for" | "loop_while" => {
                v.insert("children".into(), Value::Array(kids));
            }
            _ => {}
        }
        Ok(Value::Object(v))
    }
}

/// UAST graph handle exposed to Python.
#[pyclass(module = "invar_agent._native", name = "UASTGraph")]
pub struct PyUASTGraph {
    pub(crate) inner: UASTStorage,
}

#[pymethods]
impl PyUASTGraph {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: UASTStorage::default(),
        }
    }

    #[pyo3(signature = (kind, symbol=None, val_i=None, val_f=None, children=None))]
    pub fn insert_node(
        &mut self,
        kind: PyUASTKind,
        symbol: Option<String>,
        val_i: Option<i64>,
        val_f: Option<f64>,
        children: Option<Vec<u64>>,
    ) -> u64 {
        self.inner.insert(kind, symbol, val_i, val_f, children.unwrap_or_default())
    }

    pub fn get_root_hash(&self, root_id: u64) -> PyResult<String> {
        Ok(hex::encode(self.inner.compute_blake3_hash(root_id)))
    }

    /// Serialize the subtree as transpiler-dialect UAST JSON.
    pub fn to_json(&self, root_id: u64) -> PyResult<String> {
        self.inner
            .to_uast_json(root_id)
            .map_err(PyValueError::new_err)
    }
}

/// Deterministic polyglot materializer over UAST subtrees.
#[pyclass(module = "invar_agent._native", name = "GraphPolyglotMaterializer")]
pub struct PyGraphPolyglotMaterializer;

#[pymethods]
impl PyGraphPolyglotMaterializer {
    #[staticmethod]
    pub fn materialize(graph: &PyUASTGraph, root_id: u64, target: &str) -> PyResult<String> {
        let node = graph
            .inner
            .node(root_id)
            .ok_or_else(|| PyValueError::new_err(format!("unknown root id {root_id}")))?;
        match target {
            "rust" => emit_rust(node, &graph.inner, root_id),
            "cpp" => emit_cpp(node, &graph.inner, root_id),
            "rust_hin" => graph
                .inner
                .to_uast_json(root_id)
                .map_err(PyValueError::new_err),
            "wasm" => Err(PyValueError::new_err("wasm target not yet supported")),
            "cuda" => Err(PyValueError::new_err("cuda target not yet supported")),
            other => Err(PyValueError::new_err(format!("Unsupported target: {other}"))),
        }
    }
}

fn literal(node: &UASTNode) -> String {
    if let Some(i) = node.val_i {
        i.to_string()
    } else {
        format!("{}", node.val_f.unwrap_or(0.0))
    }
}

fn emit_rust(_node: &UASTNode, _store: &UASTStorage, _root: u64) -> PyResult<String> {
    Ok("// rust emission: use invar_agent.transpiler.emitters.rust_emitter\n".to_string())
}

fn emit_cpp(_node: &UASTNode, _store: &UASTStorage, _root: u64) -> PyResult<String> {
    Ok("// cpp emission: use invar_agent.transpiler.emitters.cpp_emitter\n".to_string())
}
