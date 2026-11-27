//! Unified mathematical function registry mirror (UPG-006).
//!
//! Keeps function names, arities, dimension rules, and lowering names synchronized
//! with `invar_agent/ir/function_registry.py` (verified via differential test).

use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DimRule {
    /// All arguments must be dimensionless; returns dimensionless.
    Dimensionless,
    /// Returns the dimension of arg 0 (all args must share the same dimension).
    SameAsArgs,
    /// Single arg; exponents halved (error if odd numerator).
    Sqrt,
}

#[derive(Debug, Clone, Serialize)]
pub struct NativeFunctionSpec {
    pub parse_name: &'static str,
    pub min_arity: usize,
    pub max_arity: Option<usize>,
    pub dim_rule: DimRule,
    pub py_name: &'static str,
    pub cpp_name: &'static str,
    pub rust_name: &'static str,
    pub csel_name: Option<&'static str>,
    pub formal_supported: bool,
}

pub const NATIVE_FUNCTIONS: &[NativeFunctionSpec] = &[
    NativeFunctionSpec {
        parse_name: "exp",
        min_arity: 1,
        max_arity: Some(1),
        dim_rule: DimRule::Dimensionless,
        py_name: "math.exp",
        cpp_name: "std::exp",
        rust_name: "f64::exp",
        csel_name: Some("exp"),
        formal_supported: true,
    },
    NativeFunctionSpec {
        parse_name: "log",
        min_arity: 1,
        max_arity: Some(1),
        dim_rule: DimRule::Dimensionless,
        py_name: "math.log",
        cpp_name: "std::log",
        rust_name: "f64::ln",
        csel_name: Some("log"),
        formal_supported: true,
    },
    NativeFunctionSpec {
        parse_name: "log10",
        min_arity: 1,
        max_arity: Some(1),
        dim_rule: DimRule::Dimensionless,
        py_name: "math.log10",
        cpp_name: "std::log10",
        rust_name: "f64::log10",
        csel_name: Some("log10"),
        formal_supported: true,
    },
    NativeFunctionSpec {
        parse_name: "sqrt",
        min_arity: 1,
        max_arity: Some(1),
        dim_rule: DimRule::Sqrt,
        py_name: "math.sqrt",
        cpp_name: "std::sqrt",
        rust_name: "f64::sqrt",
        csel_name: Some("sqrt"),
        formal_supported: true,
    },
    NativeFunctionSpec {
        parse_name: "sin",
        min_arity: 1,
        max_arity: Some(1),
        dim_rule: DimRule::Dimensionless,
        py_name: "math.sin",
        cpp_name: "std::sin",
        rust_name: "f64::sin",
        csel_name: Some("sin"),
        formal_supported: true,
    },
    NativeFunctionSpec {
        parse_name: "cos",
        min_arity: 1,
        max_arity: Some(1),
        dim_rule: DimRule::Dimensionless,
        py_name: "math.cos",
        cpp_name: "std::cos",
        rust_name: "f64::cos",
        csel_name: Some("cos"),
        formal_supported: true,
    },
    NativeFunctionSpec {
        parse_name: "tan",
        min_arity: 1,
        max_arity: Some(1),
        dim_rule: DimRule::Dimensionless,
        py_name: "math.tan",
        cpp_name: "std::tan",
        rust_name: "f64::tan",
        csel_name: Some("tan"),
        formal_supported: true,
    },
    NativeFunctionSpec {
        parse_name: "tanh",
        min_arity: 1,
        max_arity: Some(1),
        dim_rule: DimRule::Dimensionless,
        py_name: "math.tanh",
        cpp_name: "std::tanh",
        rust_name: "f64::tanh",
        csel_name: Some("tanh"),
        formal_supported: true,
    },
    NativeFunctionSpec {
        parse_name: "sinh",
        min_arity: 1,
        max_arity: Some(1),
        dim_rule: DimRule::Dimensionless,
        py_name: "math.sinh",
        cpp_name: "std::sinh",
        rust_name: "f64::sinh",
        csel_name: Some("sinh"),
        formal_supported: true,
    },
    NativeFunctionSpec {
        parse_name: "cosh",
        min_arity: 1,
        max_arity: Some(1),
        dim_rule: DimRule::Dimensionless,
        py_name: "math.cosh",
        cpp_name: "std::cosh",
        rust_name: "f64::cosh",
        csel_name: Some("cosh"),
        formal_supported: true,
    },
    NativeFunctionSpec {
        parse_name: "abs",
        min_arity: 1,
        max_arity: Some(1),
        dim_rule: DimRule::SameAsArgs,
        py_name: "abs",
        cpp_name: "std::abs",
        rust_name: "f64::abs",
        csel_name: Some("abs"),
        formal_supported: true,
    },
    NativeFunctionSpec {
        parse_name: "min",
        min_arity: 2,
        max_arity: None,
        dim_rule: DimRule::SameAsArgs,
        py_name: "min",
        cpp_name: "std::min",
        rust_name: "f64::min",
        csel_name: Some("min"),
        formal_supported: true,
    },
    NativeFunctionSpec {
        parse_name: "max",
        min_arity: 2,
        max_arity: None,
        dim_rule: DimRule::SameAsArgs,
        py_name: "max",
        cpp_name: "std::max",
        rust_name: "f64::max",
        csel_name: Some("max"),
        formal_supported: true,
    },
];

pub fn lookup_function(name: &str) -> Option<&'static NativeFunctionSpec> {
    NATIVE_FUNCTIONS.iter().find(|f| f.parse_name == name)
}

pub fn registry_json() -> Result<String, String> {
    serde_json::to_string(NATIVE_FUNCTIONS).map_err(|e| e.to_string())
}
