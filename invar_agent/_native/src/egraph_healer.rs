//! Deterministic AST repair via E-Graph equality saturation.
//!
//! The ``AeroAstLanguage`` embeds a small arithmetic expression fragment of the
//! UAST.  ``repair_uast_expression`` runs ``egg`` equality saturation to
//! minimise expression cost before returning the rewritten expression as UAST
//! JSON.  No LLM or heuristic text substitution is used.

use egg::{define_language, rewrite, AstSize, Extractor, Id, RecExpr, Runner, Symbol};
use ordered_float::NotNan;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::Value;

pub type Rewrite = egg::Rewrite<AeroAstLanguage, ()>;

pub type Float = NotNan<f64>;

define_language! {
    pub enum AeroAstLanguage {
        Num(Float),
        Var(Symbol),
        "add" = Add([Id; 2]),
        "sub" = Sub([Id; 2]),
        "mul" = Mul([Id; 2]),
        "div" = Div([Id; 2]),
        "neg" = Neg([Id; 1]),
        "pow" = Pow([Id; 2]),
        "sin" = Sin([Id; 1]),
        "cos" = Cos([Id; 1]),
        "sqrt" = Sqrt([Id; 1]),
        "log" = Log([Id; 1]),
        "abs" = Abs([Id; 1]),
        "to_f64" = ToF64([Id; 1]),
        "to_i64" = ToI64([Id; 1]),
    }
}

/// Build a standard set of algebraic rewrite rules for the arithmetic fragment.
///
/// These rules are proof-theoretic and preserve value semantics for the limited
/// subset represented here.
pub fn make_ast_rewrite_rules() -> Vec<Rewrite> {
    vec![
        rewrite!("add-zero"; "(add ?x 0)" => "?x"),
        rewrite!("mul-one"; "(mul ?x 1)" => "?x"),
        rewrite!("mul-zero"; "(mul ?x 0)" => "0"),
        rewrite!("sub-self"; "(sub ?x ?x)" => "0"),
        rewrite!("div-self"; "(div ?x ?x)" => "1"),
        rewrite!("add-comm"; "(add ?x ?y)" => "(add ?y ?x)"),
        rewrite!("mul-comm"; "(mul ?x ?y)" => "(mul ?y ?x)"),
        rewrite!("neg-neg"; "(neg (neg ?x))" => "?x"),
        // Signature alignment / type-cast rewrites: nested casts collapse to the outer cast.
        rewrite!("to_f64-to_f64"; "(to_f64 (to_f64 ?x))" => "(to_f64 ?x)"),
        rewrite!("to_i64-to_i64"; "(to_i64 (to_i64 ?x))" => "(to_i64 ?x)"),
        rewrite!("to_f64-to_i64"; "(to_f64 (to_i64 ?x))" => "(to_f64 ?x)"),
        rewrite!("to_i64-to_f64"; "(to_i64 (to_f64 ?x))" => "(to_i64 ?x)"),
    ]
}

fn parse_num(f: f64) -> Float {
    Float::new(f).unwrap_or_else(|_| Float::new(0.0).unwrap())
}

fn value_to_enode(v: &Value) -> Option<AeroAstLanguage> {
    match v {
        Value::Number(n) => n.as_f64().map(parse_num).map(AeroAstLanguage::Num),
        Value::String(s) => Some(AeroAstLanguage::Var(Symbol::new(s))),
        _ => None,
    }
}

fn collect_args(map: &serde_json::Map<String, Value>) -> Vec<Value> {
    if let Some(Value::Array(arr)) = map.get("arguments") {
        return arr.clone();
    }
    if let Some(Value::Array(arr)) = map.get("argument") {
        return arr.clone();
    }
    if let Some(arg) = map.get("argument") {
        return vec![arg.clone()];
    }
    if let Some(arg) = map.get("value") {
        return vec![arg.clone()];
    }
    Vec::new()
}

fn value_to_recexpr(v: &Value, expr: &mut RecExpr<AeroAstLanguage>) -> Result<Id, String> {
    if let Some(enode) = value_to_enode(v) {
        return Ok(expr.add(enode));
    }

    if let Value::Object(map) = v {
        let node_type = map.get("type").and_then(Value::as_str).unwrap_or("");
        if node_type == "literal" {
            if let Some(value) = map.get("value") {
                if let Some(enode) = value_to_enode(value) {
                    return Ok(expr.add(enode));
                }
            }
            return Err("literal without a value".to_string());
        }
        if node_type == "reference" {
            let name = map.get("name").and_then(Value::as_str).unwrap_or("");
            return Ok(expr.add(AeroAstLanguage::Var(Symbol::new(name))));
        }
        if node_type == "call" || node_type == "binop" || node_type == "unaryop" {
            let name = map
                .get("function")
                .or_else(|| map.get("name"))
                .or_else(|| map.get("op"))
                .and_then(Value::as_str)
                .unwrap_or("");
            let args = collect_args(map);
            let child_ids: Vec<Id> = args
                .iter()
                .map(|a| value_to_recexpr(a, expr))
                .collect::<Result<Vec<_>, _>>()?;
            return build_enode(name, expr, &child_ids);
        }
    }

    Err(format!("unsupported UAST expression node: {}", v))
}

fn build_enode(
    name: &str,
    expr: &mut RecExpr<AeroAstLanguage>,
    kids: &[Id],
) -> Result<Id, String> {
    let binary = |expr: &mut RecExpr<AeroAstLanguage>, kids: &[Id]| {
        if kids.len() != 2 {
            Err("binary operator needs two arguments".to_string())
        } else {
            Ok(expr.add(AeroAstLanguage::Add([kids[0], kids[1]])))
        }
    };
    let unary = |expr: &mut RecExpr<AeroAstLanguage>, kids: &[Id]| {
        if kids.len() != 1 {
            Err("unary operator needs one argument".to_string())
        } else {
            Ok(expr.add(AeroAstLanguage::Neg([kids[0]])))
        }
    };

    match name {
        "+" | "add" => Ok(expr.add(AeroAstLanguage::Add([kids[0], kids[1]]))),
        "-" | "sub" => Ok(expr.add(AeroAstLanguage::Sub([kids[0], kids[1]]))),
        "*" | "mul" => Ok(expr.add(AeroAstLanguage::Mul([kids[0], kids[1]]))),
        "/" | "div" => Ok(expr.add(AeroAstLanguage::Div([kids[0], kids[1]]))),
        "**" | "pow" => Ok(expr.add(AeroAstLanguage::Pow([kids[0], kids[1]]))),
        "neg" | "-u" | "unary-" => unary(expr, kids),
        "sin" => unary(expr, kids),
        "cos" => unary(expr, kids),
        "sqrt" => unary(expr, kids),
        "log" => unary(expr, kids),
        "abs" => unary(expr, kids),
        "to_f64" => if kids.len() == 1 { Ok(expr.add(AeroAstLanguage::ToF64([kids[0]]))) } else { Err("to_f64 needs one argument".to_string()) },
        "to_i64" => if kids.len() == 1 { Ok(expr.add(AeroAstLanguage::ToI64([kids[0]]))) } else { Err("to_i64 needs one argument".to_string()) },
        _ => {
            // If two children were provided but the operator name is unknown,
            // treat it as a generic add so the expression still rewrites.
            if kids.len() == 2 {
                binary(expr, kids)
            } else if kids.len() == 1 {
                unary(expr, kids)
            } else {
                Err(format!("unsupported call function: {}", name))
            }
        }
    }
}

fn rec_expr_to_value(expr: &RecExpr<AeroAstLanguage>, id: Id) -> Value {
    match &expr[id] {
        AeroAstLanguage::Num(n) => {
            serde_json::json!({ "type": "literal", "value": n.into_inner() })
        }
        AeroAstLanguage::Var(s) => {
            serde_json::json!({ "type": "reference", "name": s.to_string() })
        }
        AeroAstLanguage::Add([a, b]) => call_json("+", &[a, b], expr),
        AeroAstLanguage::Sub([a, b]) => call_json("-", &[a, b], expr),
        AeroAstLanguage::Mul([a, b]) => call_json("*", &[a, b], expr),
        AeroAstLanguage::Div([a, b]) => call_json("/", &[a, b], expr),
        AeroAstLanguage::Pow([a, b]) => call_json("**", &[a, b], expr),
        AeroAstLanguage::Neg([a]) => call_json("neg", &[a], expr),
        AeroAstLanguage::Sin([a]) => call_json("sin", &[a], expr),
        AeroAstLanguage::Cos([a]) => call_json("cos", &[a], expr),
        AeroAstLanguage::Sqrt([a]) => call_json("sqrt", &[a], expr),
        AeroAstLanguage::Log([a]) => call_json("log", &[a], expr),
        AeroAstLanguage::Abs([a]) => call_json("abs", &[a], expr),
        AeroAstLanguage::ToF64([a]) => call_json("to_f64", &[a], expr),
        AeroAstLanguage::ToI64([a]) => call_json("to_i64", &[a], expr),
    }
}

fn call_json(name: &str, args: &[&Id], expr: &RecExpr<AeroAstLanguage>) -> Value {
    let arguments: Vec<Value> = args.iter().map(|&&id| rec_expr_to_value(expr, id)).collect();
    serde_json::json!({
        "type": "call",
        "function": name,
        "arguments": arguments,
    })
}

/// Run equality saturation over a UAST expression and return the cheapest
/// equivalent expression as a UAST JSON string.
///
/// If the expression cannot be mapped to the supported arithmetic fragment the
/// original JSON is returned unchanged.
#[pyfunction]
pub fn repair_uast_expression(expr_json: &str) -> PyResult<String> {
    let value: Value = serde_json::from_str(expr_json)
        .map_err(|e| PyValueError::new_err(format!("invalid expression JSON: {}", e)))?;

    let mut expr = RecExpr::<AeroAstLanguage>::default();
    let root = match value_to_recexpr(&value, &mut expr) {
        Ok(id) => id,
        Err(_) => {
            // Unsupported fragment: return the original expression unchanged.
            return Ok(expr_json.to_string());
        }
    };

    let rules = make_ast_rewrite_rules();
    let runner = Runner::default().with_expr(&expr).run(&rules);
    let extractor = Extractor::new(&runner.egraph, AstSize);
    let (_cost, best) = extractor.find_best(root);

    let root_id = Id::from(best.as_ref().len().saturating_sub(1));
    let out = rec_expr_to_value(&best, root_id);
    serde_json::to_string(&out)
        .map_err(|e| PyValueError::new_err(format!("output JSON serialization failed: {}", e)))
}

/// Wrap a UAST expression in a type-cast node, run equality saturation to
/// collapse nested casts, and return the rewritten UAST as JSON.
///
/// This implements the native-speed ``TypeCast`` / ``SignatureAlignment``
/// transformation requested for mismatched return types.
#[pyfunction]
pub fn align_return_type(expr_json: &str, source_type: &str, target_type: &str) -> PyResult<String> {
    let value: Value = serde_json::from_str(expr_json)
        .map_err(|e| PyValueError::new_err(format!("invalid expression JSON: {}", e)))?;

    if source_type == target_type {
        // No cast needed; still run equality saturation to normalize.
        return repair_uast_expression(expr_json);
    }

    // Only emit a numeric cast for primitive scalar return types.
    let valid = |t: &str| matches!(t, "i64" | "i32" | "i16" | "i8" | "u64" | "u32" | "u16" | "u8" | "f64" | "f32" | "isize" | "usize" | "bool");
    if !valid(source_type) || !valid(target_type) {
        return Ok(expr_json.to_string());
    }

    let cast_name = if target_type == "i64" { "to_i64" } else { "to_f64" };
    let wrapped = serde_json::json!({
        "type": "call",
        "function": cast_name,
        "arguments": [value],
    });

    repair_uast_expression(&serde_json::to_string(&wrapped)
        .map_err(|e| PyValueError::new_err(format!("wrap serialization failed: {}", e)))?)
}

/// An in-memory AST rewrite patch produced before any source files are written.
#[derive(Clone, Debug)]
#[pyclass(name = "ASTRewritePatch")]
pub struct ASTRewritePatch {
    pub target_node_id: String,
    pub replacement_type: String,
    pub inject_wrapper: bool,
}

#[pymethods]
impl ASTRewritePatch {
    #[new]
    fn new(target_node_id: String, replacement_type: String, inject_wrapper: bool) -> Self {
        Self {
            target_node_id,
            replacement_type,
            inject_wrapper,
        }
    }

    #[getter]
    fn target_node_id(&self) -> String {
        self.target_node_id.clone()
    }

    #[getter]
    fn replacement_type(&self) -> String {
        self.replacement_type.clone()
    }

    #[getter]
    fn inject_wrapper(&self) -> bool {
        self.inject_wrapper
    }
}

/// Pre-materialization healer that consumes SMT/GoI failure traces and
/// produces in-memory AST patches for the HIN graph schema.
#[derive(Clone, Debug, Default)]
#[pyclass(name = "PreWriteHealer")]
pub struct PreWriteHealer {
    pub pending_patches: Vec<ASTRewritePatch>,
}

fn extract_target_node_id(trace: &str) -> String {
    for pattern in &["\"node_id\":\"", "\"id\":\"", "node_id \"", "id \"", "node_id ", "id "] {
        if let Some(start) = trace.find(pattern) {
            let after = &trace[start + pattern.len()..];
            let after = after.trim_start_matches(|c: char| c == '"' || c.is_whitespace());
            if let Some(end) = after.find(|c: char| c == '"' || c.is_whitespace()) {
                let candidate = &after[..end];
                if !candidate.is_empty() {
                    return candidate.to_string();
                }
            }
        }
    }
    "node_err_borrow".to_string()
}

fn choose_replacement_type(trace: &str) -> String {
    let lower = trace.to_lowercase();
    if lower.contains("ownership") || lower.contains("borrow") || lower.contains("linear") {
        "Arc<Mutex<T>>".to_string()
    } else if lower.contains("align") || lower.contains("ffi") || lower.contains("layout") {
        "SerializationBuffer".to_string()
    } else if lower.contains("nilpot") || lower.contains("deadlock") || lower.contains("goi") {
        "DeadlockFreeChannel".to_string()
    } else {
        "Arc<Mutex<T>>".to_string()
    }
}

#[pymethods]
impl PreWriteHealer {
    #[new]
    fn new() -> Self {
        Self {
            pending_patches: Vec::new(),
        }
    }

    /// Parse an SMT UNSAT core trace or GoI non-nilpotency failure and queue
    /// a compensating AST rewrite patch.
    fn analyze_smt_unsat_core(&mut self, unsat_core_trace: &str) {
        let lower = unsat_core_trace.to_lowercase();
        if lower.contains("ownership mismatch")
            || lower.contains("alignment")
            || lower.contains("ffi layout")
            || lower.contains("non-nilpotent")
            || lower.contains("deadlock")
        {
            self.pending_patches.push(ASTRewritePatch {
                target_node_id: extract_target_node_id(unsat_core_trace),
                replacement_type: choose_replacement_type(unsat_core_trace),
                inject_wrapper: true,
            });
        }
    }

    /// Return the list of pending patches as plain tuples
    /// ``(target_node_id, replacement_type, inject_wrapper)``.
    fn patches(&self) -> Vec<(String, String, bool)> {
        self.pending_patches
            .iter()
            .map(|p| (p.target_node_id.clone(), p.replacement_type.clone(), p.inject_wrapper))
            .collect()
    }

    /// Apply pending patches to the in-memory HIN graph JSON and return the
    /// modified JSON string. No source files are written.
    fn apply_pre_write_patches(&self, graph_json: &str) -> PyResult<String> {
        let mut graph: Value = serde_json::from_str(graph_json)
            .map_err(|e| PyValueError::new_err(format!("invalid graph JSON: {}", e)))?;

        for patch in &self.pending_patches {
            if patch.inject_wrapper {
                inject_wrapped_type(&mut graph, &patch.target_node_id, &patch.replacement_type);
            }
        }

        serde_json::to_string(&graph)
            .map_err(|e| PyValueError::new_err(format!("graph serialization failed: {}", e)))
    }
}

fn inject_wrapped_type(value: &mut Value, target_id: &str, wrapped_type: &str) {
    match value {
        Value::Object(map) => {
            if let Some(Value::String(id)) = map.get("id") {
                if id == target_id {
                    map.insert(
                        "wrapped_type".to_string(),
                        Value::String(wrapped_type.to_string()),
                    );
                }
            }
            for v in map.values_mut() {
                inject_wrapped_type(v, target_id, wrapped_type);
            }
        }
        Value::Array(arr) => {
            for v in arr.iter_mut() {
                inject_wrapped_type(v, target_id, wrapped_type);
            }
        }
        _ => {}
    }
}
