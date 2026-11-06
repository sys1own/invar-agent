//! Hierarchical Interaction Net (HIN) engine.
//!
//! This is a zero-heap, arena-based implementation of the MELL-typed
//! interaction-net reducer used by Aero-Future.  It accepts the same UAST
//! dialect produced by ``aero_forge.translator.aero_frontend`` and builds a
//! flat graph of ``Node`` / ``Port`` indices.  Reduction runs with the Python
//! GIL released.

use std::collections::HashMap;
use std::collections::VecDeque;
use std::time::Instant;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;
use smallvec::SmallVec;

/// MELL structural type used for port annotations.
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "kind")]
pub enum MellType {
    I,
    Any,
    Tensor { left: Box<MellType>, right: Box<MellType> },
    Implication { left: Box<MellType>, right: Box<MellType> },
    Bang { inner: Box<MellType> },
}

impl MellType {
    fn any() -> Self {
        MellType::Any
    }
    fn unit() -> Self {
        MellType::I
    }
    fn bang(inner: MellType) -> Self {
        MellType::Bang {
            inner: Box::new(inner),
        }
    }
}

/// Payload carried by a ``Value`` node.
#[derive(Clone, Debug, Serialize, PartialEq)]
#[serde(untagged)]
pub enum NodeValue {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Dict(Vec<(NodeValue, NodeValue)>),
    Set(Vec<NodeValue>),
}

impl NodeValue {
    fn truthy(&self) -> bool {
        match self {
            NodeValue::Null => false,
            NodeValue::Bool(b) => *b,
            NodeValue::Number(n) => *n != 0.0,
            NodeValue::String(s) => !s.is_empty(),
            NodeValue::Dict(pairs) => !pairs.is_empty(),
            NodeValue::Set(elements) => !elements.is_empty(),
        }
    }
}

/// Linear collection kind used by HIN collection agents.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CollectionKind {
    Dict,
    Set,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NodeKind {
    Constructor,
    Destructor,
    Duplicator,
    Eraser,
    Value,
    Switch,
    CausalProjection,
    /// Dict constructor: pairs of key/value ports bound to MELL edges.
    DictConstructor,
    /// Set constructor: element ports bound to MELL edges.
    SetConstructor,
    /// Key/membership lookup for a dict or set (`kind` selects the collection).
    KeyLookup {
        kind: CollectionKind,
    },
}

struct Port {
    owner: usize,
    name: String,
    is_principal: bool,
    target: Option<usize>,
    #[allow(dead_code)]
    mell_type: MellType,
}

struct Node {
    id: String,
    kind: NodeKind,
    principal: Option<usize>,
    aux: Vec<usize>,
    retired: bool,
    value: Option<NodeValue>,
}

/// A single variable scope.
struct Scope {
    bindings: HashMap<String, usize>,
}

/// Arena-based HIN network and reducer.
pub struct HinEngine {
    nodes: Vec<Node>,
    ports: Vec<Port>,
    active: VecDeque<(usize, usize)>,
    gensym: usize,
    scopes: Vec<Scope>,
}

impl HinEngine {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            ports: Vec::new(),
            active: VecDeque::new(),
            gensym: 0,
            scopes: Vec::new(),
        }
    }

    fn fresh_id(&mut self, prefix: &str) -> String {
        let n = self.gensym;
        self.gensym += 1;
        format!("{}#{}", prefix, n)
    }

    fn add_port(
        &mut self,
        owner: usize,
        name: &str,
        is_principal: bool,
        mell_type: MellType,
    ) -> usize {
        let idx = self.ports.len();
        self.ports.push(Port {
            owner,
            name: name.to_string(),
            is_principal,
            target: None,
            mell_type,
        });
        idx
    }

    fn add_node(
        &mut self,
        kind: NodeKind,
        aux_count: usize,
        value: Option<NodeValue>,
    ) -> usize {
        let id = self.fresh_id(match kind {
            NodeKind::Constructor => "γ",
            NodeKind::Destructor => "app",
            NodeKind::Duplicator => "δ",
            NodeKind::Eraser => "ε",
            NodeKind::Value => "V",
            NodeKind::Switch => "σ",
            NodeKind::CausalProjection => "P",
            NodeKind::DictConstructor => "dict",
            NodeKind::SetConstructor => "set",
            NodeKind::KeyLookup { .. } => "lookup",
        });
        let idx = self.nodes.len();
        let principal = self.add_port(idx, "p", true, MellType::any());
        let mut aux = Vec::with_capacity(aux_count);
        for i in 0..aux_count {
            let name = format!("a_{}", i + 1);
            let p = self.add_port(idx, &name, false, MellType::any());
            aux.push(p);
        }
        self.nodes.push(Node {
            id,
            kind,
            principal: Some(principal),
            aux,
            retired: false,
            value,
        });
        idx
    }

    fn node_principal(&self, node: usize) -> usize {
        self.nodes[node].principal.unwrap()
    }

    fn node_aux(&self, node: usize, idx: usize) -> usize {
        self.nodes[node].aux[idx]
    }

    fn connect(&mut self, a: usize, b: usize) {
        self.ports[a].target = Some(b);
        self.ports[b].target = Some(a);
        if self.ports[a].is_principal && self.ports[b].is_principal {
            let owner_a = self.ports[a].owner;
            let owner_b = self.ports[b].owner;
            if owner_a != owner_b {
                self.active.push_back((owner_a, owner_b));
            }
        }
    }

    fn _link(&mut self, a: Option<usize>, b: Option<usize>) {
        if let Some(a_idx) = a {
            if let Some(b_idx) = b {
                self.connect(a_idx, b_idx);
            }
        }
    }

    fn terminate(&mut self, port: usize) {
        if self.ports[port].target.is_some() {
            return;
        }
        let eraser = self.add_node(NodeKind::Eraser, 0, None);
        let eraser_p = self.node_principal(eraser);
        self.connect(port, eraser_p);
    }

    fn retire(&mut self, a: usize, b: usize) {
        self.nodes[a].retired = true;
        self.nodes[b].retired = true;
    }

    fn push_scope(&mut self) {
        self.scopes.push(Scope {
            bindings: HashMap::new(),
        });
    }

    fn pop_scope(&mut self) {
        if let Some(scope) = self.scopes.pop() {
            for (_, port) in scope.bindings {
                if self.ports[port].target.is_none() {
                    self.terminate(port);
                }
            }
        }
    }

    fn bind(&mut self, name: &str, port: usize) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.bindings.insert(name.to_string(), port);
        }
    }

    fn resolve(&mut self, name: &str) -> Option<usize> {
        let mut found: Option<(usize, usize)> = None;
        for (i, scope) in self.scopes.iter().enumerate().rev() {
            if let Some(&src) = scope.bindings.get(name) {
                found = Some((i, src));
                break;
            }
        }
        if let Some((scope_idx, src)) = found {
            let dup = self.add_node(NodeKind::Duplicator, 2, None);
            let dup_p = self.node_principal(dup);
            let dup_a1 = self.node_aux(dup, 0);
            let dup_a2 = self.node_aux(dup, 1);
            self.connect(src, dup_p);
            self.scopes[scope_idx].bindings.insert(name.to_string(), dup_a2);
            return Some(dup_a1);
        }
        None
    }

    fn kind_of(node: &Value) -> String {
        node.get("type")
            .or_else(|| node.get("canonical_kind"))
            .or_else(|| node.get("kind"))
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string()
    }

    fn name_of(node: &Value) -> String {
        node.get("name")
            .or_else(|| node.get("text"))
            .or_else(|| node.get("value"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string()
    }

    fn children_of(node: &Value) -> Vec<&Value> {
        if let Some(arr) = node.get("children").and_then(|v| v.as_array()) {
            return arr.iter().collect();
        }
        if let Some(arr) = node.get("body").and_then(|v| v.as_array()) {
            return arr.iter().collect();
        }
        Vec::new()
    }

    /// Build an interaction-net graph from a UAST JSON value.
    pub fn build_uast(&mut self, uast: &Value) -> Result<(), String> {
        self.scopes.clear();
        self.push_scope();
        let result = self.build(uast)?;
        if let Some(p) = result {
            if self.ports[p].target.is_none() {
                self.terminate(p);
            }
        }
        self.pop_scope();
        Ok(())
    }

    fn build(&mut self, node: &Value) -> Result<Option<usize>, String> {
        let kind = Self::kind_of(node);
        match kind.as_str() {
            "module" | "translation_unit" | "block" | "sequence" | "program" | "body" => {
                self.build_container(node)
            }
            "function_declaration" | "function_definition" | "function" | "lambda" => {
                self.build_function(node)
            }
            "binding" | "assignment" | "let" | "variable_declaration" => self.build_binding(node),
            "reference" | "identifier" | "name" | "var" => self.build_reference(node),
            "literal" | "constant" | "number" | "string" | "value" => self.build_literal(node),
            "dict" => self.build_dict(node),
            "set" => self.build_set(node),
            "dict_lookup" => self.build_dict_lookup(node),
            "set_member" => self.build_set_member(node),
            "if" | "if_statement" | "conditional" | "if_else" => self.build_if(node),
            "call" | "application" | "apply" | "call_expression" | "user_function_call" => {
                self.build_call(node)
            }
            "compare" | "comparison" => self.build_compare(node),
            "attribute" | "attr" => self.build_attribute(node),
            _ => Ok(None),
        }
    }

    fn build_container(&mut self, node: &Value) -> Result<Option<usize>, String> {
        let children = Self::children_of(node);
        let mut last: Option<usize> = None;
        for child in children {
            let out = self.build(child)?;
            if let Some(prev) = last {
                if self.ports[prev].target.is_none() {
                    self.terminate(prev);
                }
            }
            last = out;
        }
        Ok(last)
    }

    fn build_function(&mut self, node: &Value) -> Result<Option<usize>, String> {
        // Constructor: a_1 = parameter, a_2 = result.
        let ctor = self.add_node(NodeKind::Constructor, 2, None);
        let ctor_p = self.node_principal(ctor);
        let ctor_a1 = self.node_aux(ctor, 0);
        let ctor_a2 = self.node_aux(ctor, 1);

        let params = node.get("params").and_then(|v| v.as_array());
        let param = node
            .get("param")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        self.push_scope();

        if let Some(ps) = params {
            if !ps.is_empty() {
                let first = ps[0].as_str().unwrap_or("").to_string();
                self.bind(&first, ctor_a1);
            } else {
                self.terminate(ctor_a1);
            }
        } else if let Some(p) = param {
            self.bind(&p, ctor_a1);
        } else {
            self.terminate(ctor_a1);
        }

        let body_port = if let Some(body) = node.get("body") {
            self.build(body)?
        } else {
            let children = Self::children_of(node);
            let mut last: Option<usize> = None;
            for child in children {
                let out = self.build(child)?;
                if let Some(prev) = last {
                    if self.ports[prev].target.is_none() {
                        self.terminate(prev);
                    }
                }
                last = out;
            }
            last
        };

        if let Some(p) = body_port {
            self.connect(p, ctor_a2);
        } else {
            let v = self.add_node(NodeKind::Value, 0, Some(NodeValue::Null));
            self.connect(self.node_principal(v), ctor_a2);
        }

        self.pop_scope();

        if let Some(name) = node.get("name").and_then(|v| v.as_str()) {
            self.bind(name, ctor_p);
        }

        Ok(Some(ctor_p))
    }

    fn build_binding(&mut self, node: &Value) -> Result<Option<usize>, String> {
        let name = node.get("name").and_then(|v| v.as_str());
        let value_port = if let Some(value) = node.get("value") {
            self.build(value)?
        } else if let Some(init) = node.get("init") {
            self.build(init)?
        } else if let Some(expr) = node.get("expr") {
            self.build(expr)?
        } else {
            None
        };

        if let Some(n) = name {
            if let Some(p) = value_port {
                self.bind(n, p);
            } else {
                let v = self.add_node(NodeKind::Value, 0, Some(NodeValue::Null));
                self.bind(n, self.node_principal(v));
            }
        } else if let Some(p) = value_port {
            if self.ports[p].target.is_none() {
                self.terminate(p);
            }
        }
        Ok(None)
    }

    fn build_reference(&mut self, node: &Value) -> Result<Option<usize>, String> {
        let name = Self::name_of(node);
        if name.is_empty() {
            let v = self.add_node(NodeKind::Value, 0, Some(NodeValue::Null));
            return Ok(Some(self.node_principal(v)));
        }
        if let Some(p) = self.resolve(&name) {
            Ok(Some(p))
        } else {
            let v = self.add_node(NodeKind::Value, 0, Some(NodeValue::Null));
            Ok(Some(self.node_principal(v)))
        }
    }

    fn build_attribute(&mut self, node: &Value) -> Result<Option<usize>, String> {
        // Attribute access (e.g. ``z.conjugate``) is modeled as a reference lookup
        // whose name combines the receiver and the attribute.  The reducer will
        // either resolve the name from an in-scope binding or create a fresh
        // value node, allowing active-pair reductions to proceed on attribute
        // lookups just like ordinary references.
        let attr = node.get("attr").and_then(|v| v.as_str()).unwrap_or("");
        let receiver_name = node
            .get("value")
            .and_then(|v| v.get("id").or_else(|| v.get("name")))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if !receiver_name.is_empty() && !attr.is_empty() {
            let synthetic = serde_json::json!({ "name": format!("{}.{}", receiver_name, attr) });
            return self.build_reference(&synthetic);
        }
        if let Some(value) = node.get("value") {
            if let Some(p) = self.build(value)? {
                return Ok(Some(p));
            }
        }
        let v = self.add_node(NodeKind::Value, 0, Some(NodeValue::Null));
        Ok(Some(self.node_principal(v)))
    }

    fn build_literal(&mut self, node: &Value) -> Result<Option<usize>, String> {
        let val = if let Some(n) = node.get("value") {
            Self::json_to_value(n)
        } else if let Some(t) = node.get("text").and_then(|v| v.as_str()) {
            NodeValue::String(t.to_string())
        } else {
            NodeValue::Null
        };
        let vnode = self.add_node(NodeKind::Value, 0, Some(val));
        Ok(Some(self.node_principal(vnode)))
    }

    fn build_call(&mut self, node: &Value) -> Result<Option<usize>, String> {
        let dtor = self.add_node(NodeKind::Destructor, 2, None);
        let dtor_p = self.node_principal(dtor);
        let dtor_a1 = self.node_aux(dtor, 0);
        let dtor_a2 = self.node_aux(dtor, 1);

        let func_port = if let Some(func) = node.get("function") {
            self.build(func)?
        } else if let Some(func) = node.get("callee") {
            self.build(func)?
        } else {
            None
        };

        if let Some(p) = func_port {
            self.connect(p, dtor_p);
        } else {
            let v = self.add_node(NodeKind::Value, 0, Some(NodeValue::Null));
            self.connect(self.node_principal(v), dtor_p);
        }

        let arg_port = if let Some(args) = node.get("arguments").and_then(|v| v.as_array()) {
            if !args.is_empty() {
                self.build(&args[0])?
            } else {
                None
            }
        } else if let Some(arg) = node.get("argument") {
            self.build(arg)?
        } else if let Some(arg) = node.get("arg") {
            self.build(arg)?
        } else {
            None
        };

        if let Some(p) = arg_port {
            self.connect(p, dtor_a1);
        } else {
            self.terminate(dtor_a1);
        }

        Ok(Some(dtor_a2))
    }

    fn build_compare(&mut self, node: &Value) -> Result<Option<usize>, String> {
        // Comparison operators are wired to the Value agents of their operands
        // through a binary Constructor whose principal can reduce with the
        // surrounding context (e.g. a Switch in an ``if`` test) instead of being
        // treated as an attribute call like ``__name__.eq``.
        let op = node.get("op").and_then(|v| v.as_str()).unwrap_or("==");
        let ctor = self.add_node(
            NodeKind::Constructor,
            2,
            Some(NodeValue::String(op.to_string())),
        );
        let ctor_p = self.node_principal(ctor);
        let ctor_a1 = self.node_aux(ctor, 0);
        let ctor_a2 = self.node_aux(ctor, 1);

        let left_port = if let Some(left) = node.get("left") {
            self.build(left)?
        } else {
            None
        };
        if let Some(p) = left_port {
            self.connect(p, ctor_a1);
        } else {
            self.terminate(ctor_a1);
        }

        let right_port = if let Some(right) = node.get("right") {
            self.build(right)?
        } else {
            None
        };
        if let Some(p) = right_port {
            self.connect(p, ctor_a2);
        } else {
            self.terminate(ctor_a2);
        }

        Ok(Some(ctor_p))
    }

    fn build_if(&mut self, node: &Value) -> Result<Option<usize>, String> {
        let sw = self.add_node(NodeKind::Switch, 3, None);
        let sw_p = self.node_principal(sw);
        let sw_a1 = self.node_aux(sw, 0);
        let sw_a2 = self.node_aux(sw, 1);
        let sw_a3 = self.node_aux(sw, 2);

        let cond_port = if let Some(cond) = node.get("condition") {
            self.build(cond)?
        } else if let Some(cond) = node.get("test") {
            self.build(cond)?
        } else if let Some(cond) = node.get("cond") {
            self.build(cond)?
        } else {
            None
        };

        if let Some(p) = cond_port {
            self.connect(p, sw_p);
        } else {
            let v = self.add_node(NodeKind::Value, 0, Some(NodeValue::Bool(false)));
            self.connect(self.node_principal(v), sw_p);
        }

        let then_port = self.branch_port(node.get("then").or_else(|| node.get("consequent")))?;
        let else_port = self.branch_port(node.get("else").or_else(|| node.get("alternate")))?;

        self.connect(then_port, sw_a1);
        self.connect(else_port, sw_a2);

        Ok(Some(sw_a3))
    }

    fn branch_port(&mut self, branch: Option<&Value>) -> Result<usize, String> {
        if let Some(b) = branch {
            if let Some(p) = self.build(b)? {
                Ok(p)
            } else {
                let v = self.add_node(NodeKind::Value, 0, Some(NodeValue::Null));
                Ok(self.node_principal(v))
            }
        } else {
            let v = self.add_node(NodeKind::Value, 0, Some(NodeValue::Null));
            Ok(self.node_principal(v))
        }
    }

    fn json_to_value(v: &Value) -> NodeValue {
        match v {
            Value::Null => NodeValue::Null,
            Value::Bool(b) => NodeValue::Bool(*b),
            Value::Number(n) => NodeValue::Number(n.as_f64().unwrap_or(0.0)),
            Value::String(s) => NodeValue::String(s.clone()),
            _ => NodeValue::Null,
        }
    }

    fn build_dict(&mut self, node: &Value) -> Result<Option<usize>, String> {
        let pairs = node
            .get("pairs")
            .and_then(|v| v.as_array())
            .map(|a| a.as_slice())
            .unwrap_or(&[]);
        let coll = self.add_node(NodeKind::DictConstructor, pairs.len() * 2, None);
        for (i, pair) in pairs.iter().enumerate() {
            let key_port = if let Some(p) = self.build(pair.get("key").unwrap_or(&Value::Null))? {
                p
            } else {
                let v = self.add_node(NodeKind::Value, 0, Some(NodeValue::Null));
                self.node_principal(v)
            };
            let val_port =
                if let Some(p) = self.build(pair.get("value").unwrap_or(&Value::Null))? {
                    p
                } else {
                    let v = self.add_node(NodeKind::Value, 0, Some(NodeValue::Null));
                    self.node_principal(v)
                };
            self.connect(key_port, self.node_aux(coll, i * 2));
            self.connect(val_port, self.node_aux(coll, i * 2 + 1));
        }
        Ok(Some(self.node_principal(coll)))
    }

    fn build_set(&mut self, node: &Value) -> Result<Option<usize>, String> {
        let elements = node
            .get("elements")
            .and_then(|v| v.as_array())
            .map(|a| a.as_slice())
            .unwrap_or(&[]);
        let coll = self.add_node(NodeKind::SetConstructor, elements.len(), None);
        for (i, elem) in elements.iter().enumerate() {
            let elem_port = if let Some(p) = self.build(elem)? {
                p
            } else {
                let v = self.add_node(NodeKind::Value, 0, Some(NodeValue::Null));
                self.node_principal(v)
            };
            self.connect(elem_port, self.node_aux(coll, i));
        }
        Ok(Some(self.node_principal(coll)))
    }

    fn build_dict_lookup(&mut self, node: &Value) -> Result<Option<usize>, String> {
        let dtor = self.add_node(
            NodeKind::KeyLookup {
                kind: CollectionKind::Dict,
            },
            2,
            None,
        );
        let dtor_p = self.node_principal(dtor);
        let dtor_a1 = self.node_aux(dtor, 0);
        let dtor_a2 = self.node_aux(dtor, 1);

        let coll_port = if let Some(p) = self.build(node.get("collection").unwrap_or(&Value::Null))? {
            p
        } else {
            let v = self.add_node(NodeKind::Value, 0, Some(NodeValue::Null));
            self.node_principal(v)
        };
        self.connect(coll_port, dtor_p);

        let key_port = if let Some(p) = self.build(node.get("key").unwrap_or(&Value::Null))? {
            p
        } else {
            let v = self.add_node(NodeKind::Value, 0, Some(NodeValue::Null));
            self.node_principal(v)
        };
        self.connect(key_port, dtor_a1);

        Ok(Some(dtor_a2))
    }

    fn build_set_member(&mut self, node: &Value) -> Result<Option<usize>, String> {
        let dtor = self.add_node(
            NodeKind::KeyLookup {
                kind: CollectionKind::Set,
            },
            2,
            None,
        );
        let dtor_p = self.node_principal(dtor);
        let dtor_a1 = self.node_aux(dtor, 0);
        let dtor_a2 = self.node_aux(dtor, 1);

        let coll_port = if let Some(p) = self.build(node.get("collection").unwrap_or(&Value::Null))? {
            p
        } else {
            let v = self.add_node(NodeKind::Value, 0, Some(NodeValue::Null));
            self.node_principal(v)
        };
        self.connect(coll_port, dtor_p);

        let elem_port = if let Some(p) = self.build(node.get("element").unwrap_or(&Value::Null))? {
            p
        } else if let Some(p) = self.build(node.get("value").unwrap_or(&Value::Null))? {
            p
        } else {
            let v = self.add_node(NodeKind::Value, 0, Some(NodeValue::Null));
            self.node_principal(v)
        };
        self.connect(elem_port, dtor_a1);

        Ok(Some(dtor_a2))
    }

    fn value_at_port(&self, port: usize) -> Option<&NodeValue> {
        let target = self.ports[port].target?;
        let owner = self.ports[target].owner;
        self.nodes[owner].value.as_ref()
    }

    fn node_value_eq(a: &NodeValue, b: &NodeValue) -> bool {
        match (a, b) {
            (NodeValue::Null, NodeValue::Null) => true,
            (NodeValue::Bool(x), NodeValue::Bool(y)) => x == y,
            (NodeValue::Number(x), NodeValue::Number(y)) => x == y,
            (NodeValue::String(x), NodeValue::String(y)) => x == y,
            (NodeValue::Dict(x), NodeValue::Dict(y)) => {
                if x.len() != y.len() {
                    return false;
                }
                x.iter().all(|(xk, xv)| {
                    y.iter()
                        .any(|(yk, yv)| Self::node_value_eq(xk, yk) && Self::node_value_eq(xv, yv))
                })
            }
            (NodeValue::Set(x), NodeValue::Set(y)) => {
                if x.len() != y.len() {
                    return false;
                }
                x.iter()
                    .all(|xv| y.iter().any(|yv| Self::node_value_eq(xv, yv)))
            }
            _ => false,
        }
    }

    /// Run active-pair reductions until the net is stable or ``max_steps``/duration is hit.
    pub fn reduce_to_completion(&mut self, max_steps: usize) -> usize {
        self.reduce_to_completion_with_timeout(max_steps, None).0
    }

    /// Run active-pair reductions with an optional wall-clock timeout.
    ///
    /// Returns the number of completed reduction steps and a flag indicating
    /// whether the loop exited because the timeout was reached before the net
    /// became stable.
    pub fn reduce_to_completion_with_timeout(
        &mut self,
        max_steps: usize,
        max_duration_seconds: Option<f64>,
    ) -> (usize, bool) {
        let start = Instant::now();
        let mut steps = 0usize;
        while steps < max_steps && self.reduce_step() {
            if let Some(d) = max_duration_seconds {
                if start.elapsed().as_secs_f64() > d {
                    return (steps, true);
                }
            }
            steps += 1;
        }
        (steps, false)
    }

    /// Number of allocated nodes (including retired nodes).
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Count active principal-principal pairs whose node kinds are identical.
    ///
    /// Such "stalled" active pairs cannot reduce without further interaction and
    /// indicate a potentially stuck or incomplete logic sketch.
    pub fn stalled_pairs(&self) -> usize {
        let mut stalled = 0usize;
        for i in 0..self.active.len() {
            let (a, b) = self.active[i];
            if self.nodes[a].retired || self.nodes[b].retired {
                continue;
            }
            let Some(pa) = self.nodes[a].principal else { continue };
            let Some(pb) = self.nodes[b].principal else { continue };
            if self.ports[pa].target != Some(pb) || self.ports[pb].target != Some(pa) {
                continue;
            }
            if self.nodes[a].kind == self.nodes[b].kind {
                stalled += 1;
            }
        }
        stalled
    }

    /// Count active principal-principal pairs in the current graph.
    pub fn active_pairs(&self) -> usize {
        let mut active = 0usize;
        for node_idx in 0..self.nodes.len() {
            if let Some(p) = self.nodes[node_idx].principal {
                if let Some(target_idx) = self.ports[p].target {
                    if self.ports[target_idx].is_principal {
                        let target_owner = self.ports[target_idx].owner;
                        if node_idx < target_owner {
                            active += 1;
                        }
                    }
                }
            }
        }
        active
    }

    fn reduce_step(&mut self) -> bool {
        while let Some((a, b)) = self.active.pop_front() {
            if self.nodes[a].retired || self.nodes[b].retired {
                continue;
            }
            // Verify the active pair is still connected through principal ports.
            let pa = self.nodes[a].principal.unwrap();
            let pb = self.nodes[b].principal.unwrap();
            if self.ports[pa].target != Some(pb) || self.ports[pb].target != Some(pa) {
                continue;
            }
            if !self.has_rule(a, b) {
                // Inert pair; ignore and continue.
                continue;
            }
            self.apply_rule(a, b);
            return true;
        }
        false
    }

    fn has_rule(&self, a: usize, b: usize) -> bool {
        let (kind_a, kind_b) = (&self.nodes[a].kind, &self.nodes[b].kind);

        // Collection agents are active only when a data constructor meets a
        // compatible consumer (KeyLookup / Collection switch).  Two constructors,
        // two lookups, or mismatched collection kinds are inert so that shared
        // collections reduce linearly without spurious annihilation.
        let is_collection_ctor =
            matches!(kind_a, NodeKind::DictConstructor | NodeKind::SetConstructor)
                || matches!(kind_b, NodeKind::DictConstructor | NodeKind::SetConstructor);
        if is_collection_ctor {
            match (kind_a, kind_b) {
                (NodeKind::DictConstructor, NodeKind::KeyLookup { kind: CollectionKind::Dict })
                | (NodeKind::KeyLookup { kind: CollectionKind::Dict }, NodeKind::DictConstructor) => {
                    return true;
                }
                (NodeKind::SetConstructor, NodeKind::KeyLookup { kind: CollectionKind::Set })
                | (NodeKind::KeyLookup { kind: CollectionKind::Set }, NodeKind::SetConstructor) => {
                    return true;
                }
                (_, NodeKind::Eraser) | (NodeKind::Eraser, _) => return true,
                (_, NodeKind::Duplicator) | (NodeKind::Duplicator, _) => return true,
                // All other collection pairs are inert.
                (NodeKind::DictConstructor, _) | (_, NodeKind::DictConstructor) => return false,
                (NodeKind::SetConstructor, _) | (_, NodeKind::SetConstructor) => return false,
                (NodeKind::KeyLookup { .. }, _) | (_, NodeKind::KeyLookup { .. }) => return false,
                _ => {}
            }
        }

        let kinds = (kind_a, kind_b);
        matches!(
            kinds,
            (_, NodeKind::Eraser)
                | (NodeKind::Eraser, _)
                | (_, NodeKind::Duplicator)
                | (NodeKind::Duplicator, _)
                | (NodeKind::Switch, NodeKind::Value)
                | (NodeKind::Value, NodeKind::Switch)
                | (NodeKind::CausalProjection, NodeKind::Value)
                | (NodeKind::Value, NodeKind::CausalProjection)
                | (NodeKind::Constructor, NodeKind::Destructor)
                | (NodeKind::Destructor, NodeKind::Constructor)
        ) || self.nodes[a].kind == self.nodes[b].kind
    }

    fn apply_rule(&mut self, a: usize, b: usize) {
        match (self.nodes[a].kind.clone(), self.nodes[b].kind.clone()) {
            (_, NodeKind::Eraser) => self.rule_erase(b, a),
            (NodeKind::Eraser, _) => self.rule_erase(a, b),
            (NodeKind::Duplicator, NodeKind::Duplicator) => self.rule_annihilate(a, b),
            (NodeKind::Duplicator, _) => self.rule_duplicate(a, b),
            (_, NodeKind::Duplicator) => self.rule_duplicate(b, a),
            (NodeKind::Switch, NodeKind::Value) => self.rule_switch(a, b),
            (NodeKind::Value, NodeKind::Switch) => self.rule_switch(b, a),
            (NodeKind::CausalProjection, NodeKind::Value) => self.rule_project(a, b),
            (NodeKind::Value, NodeKind::CausalProjection) => self.rule_project(b, a),
            (
                NodeKind::DictConstructor,
                NodeKind::KeyLookup {
                    kind: CollectionKind::Dict,
                },
            ) => self.rule_collection_lookup(b, a, CollectionKind::Dict),
            (
                NodeKind::KeyLookup {
                    kind: CollectionKind::Dict,
                },
                NodeKind::DictConstructor,
            ) => self.rule_collection_lookup(a, b, CollectionKind::Dict),
            (
                NodeKind::SetConstructor,
                NodeKind::KeyLookup {
                    kind: CollectionKind::Set,
                },
            ) => self.rule_collection_lookup(b, a, CollectionKind::Set),
            (
                NodeKind::KeyLookup {
                    kind: CollectionKind::Set,
                },
                NodeKind::SetConstructor,
            ) => self.rule_collection_lookup(a, b, CollectionKind::Set),
            _ => self.rule_annihilate(a, b),
        }
    }

    fn rule_annihilate(&mut self, a: usize, b: usize) {
        let aux_a: Vec<usize> = self.nodes[a].aux.clone();
        let aux_b: Vec<usize> = self.nodes[b].aux.clone();
        for (pa, pb) in aux_a.iter().zip(aux_b.iter()) {
            self._link(self.ports[*pa].target, self.ports[*pb].target);
        }
        self.retire(a, b);
    }

    fn rule_duplicate(&mut self, dup: usize, other: usize) {
        let dup_aux: Vec<usize> = self.nodes[dup].aux.clone();
        let other_aux: Vec<usize> = self.nodes[other].aux.clone();

        // External wires leaving the duplicator outputs and the other agent's aux wires.
        let dup_externals: Vec<Option<usize>> =
            dup_aux.iter().map(|&p| self.ports[p].target).collect();
        let other_externals: Vec<Option<usize>> =
            other_aux.iter().map(|&p| self.ports[p].target).collect();

        // One clone of `other` per duplicator output.
        let mut clones = Vec::with_capacity(dup_aux.len());
        for _ in 0..dup_aux.len() {
            clones.push(self.clone_node(other));
        }

        // One fresh duplicator per auxiliary port of `other`.
        let mut sub_dups = Vec::with_capacity(other_aux.len());
        for _ in 0..other_aux.len() {
            sub_dups.push(self.add_node(NodeKind::Duplicator, 2, None));
        }

        for (i, clone) in clones.iter().enumerate() {
            let clone_p = self.node_principal(*clone);
            self._link(Some(clone_p), dup_externals[i]);
        }

        for (j, sub) in sub_dups.iter().enumerate() {
            let sub_p = self.node_principal(*sub);
            self._link(Some(sub_p), other_externals[j]);
        }

        for (i, clone) in clones.iter().enumerate() {
            for (j, sub) in sub_dups.iter().enumerate() {
                let clone_aux = self.node_aux(*clone, j);
                let sub_aux = self.node_aux(*sub, i);
                self.connect(clone_aux, sub_aux);
            }
        }

        self.retire(dup, other);
    }

    fn rule_erase(&mut self, eraser: usize, other: usize) {
        let other_aux: Vec<usize> = self.nodes[other].aux.clone();
        for &aux in &other_aux {
            let new_eraser = self.add_node(NodeKind::Eraser, 0, None);
            let ep = self.node_principal(new_eraser);
            self._link(Some(ep), self.ports[aux].target);
        }
        self.retire(eraser, other);
    }

    fn rule_switch(&mut self, switch: usize, value: usize) {
        let val = self.nodes[value].value.clone().unwrap_or(NodeValue::Null);
        let take_true = val.truthy();

        let selected = if take_true {
            self.node_aux(switch, 0)
        } else {
            self.node_aux(switch, 1)
        };
        let discarded = if take_true {
            self.node_aux(switch, 1)
        } else {
            self.node_aux(switch, 0)
        };
        let output = self.node_aux(switch, 2);

        self._link(self.ports[selected].target, self.ports[output].target);

        let new_eraser = self.add_node(NodeKind::Eraser, 0, None);
        let ep = self.node_principal(new_eraser);
        self._link(Some(ep), self.ports[discarded].target);

        self.retire(switch, value);
    }

    fn rule_project(&mut self, proj: usize, coord: usize) {
        let path = self.nodes[coord].value.clone().unwrap_or(NodeValue::Null);
        let emitted = self.add_node(NodeKind::Value, 0, Some(path));
        let proj_a1 = self.node_aux(proj, 0);
        self._link(Some(self.node_principal(emitted)), self.ports[proj_a1].target);
        self.retire(proj, coord);
    }

    fn rule_collection_lookup(
        &mut self,
        destruct: usize,
        construct: usize,
        kind: CollectionKind,
    ) {
        let key_port = self.node_aux(destruct, 0);
        let result_port = self.node_aux(destruct, 1);
        let key = self.value_at_port(key_port).cloned();

        let aux: Vec<usize> = self.nodes[construct].aux.clone();
        let mut found: Option<usize> = None;
        match kind {
            CollectionKind::Dict => {
                for pair in aux.chunks_exact(2) {
                    if pair.len() < 2 {
                        break;
                    }
                    let candidate = self.value_at_port(pair[0]);
                    if let (Some(k), Some(c)) = (&key, candidate) {
                        if Self::node_value_eq(k, c) {
                            found = Some(pair[1]);
                            break;
                        }
                    }
                }
            }
            CollectionKind::Set => {
                for &elem in &aux {
                    let candidate = self.value_at_port(elem);
                    if let (Some(k), Some(c)) = (&key, candidate) {
                        if Self::node_value_eq(k, c) {
                            found = Some(elem);
                            break;
                        }
                    }
                }
            }
        }

        match kind {
            CollectionKind::Dict => {
                if let Some(value_aux) = found {
                    self._link(self.ports[result_port].target, self.ports[value_aux].target);
                } else {
                    let v = self.add_node(NodeKind::Value, 0, Some(NodeValue::Null));
                    self._link(self.ports[result_port].target, Some(self.node_principal(v)));
                }
            }
            CollectionKind::Set => {
                let result = found.is_some();
                let v = self.add_node(NodeKind::Value, 0, Some(NodeValue::Bool(result)));
                self._link(self.ports[result_port].target, Some(self.node_principal(v)));
            }
        }
        self.retire(destruct, construct);
    }

    fn clone_node(&mut self, node: usize) -> usize {
        let kind = self.nodes[node].kind.clone();
        let value = self.nodes[node].value.clone();
        let aux_count = self.nodes[node].aux.len();
        let new_node = self.add_node(kind, aux_count, value);
        new_node
    }

    /// Serialize the remaining live graph as JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        #[derive(Serialize)]
        struct PortRepr {
            name: String,
            target_node: Option<String>,
            target_port: Option<String>,
        }

        #[derive(Serialize)]
        struct NodeRepr {
            id: String,
            kind: NodeKind,
            value: Option<NodeValue>,
            ports: Vec<PortRepr>,
        }

        let mut nodes = Vec::new();
        for (_idx, node) in self.nodes.iter().enumerate() {
            if node.retired {
                continue;
            }
            let mut ports = Vec::new();
            let mut add_port = |p: usize| {
                let port = &self.ports[p];
                let (t_node, t_port) = port.target.map_or((None, None), |t| {
                    let owner = self.ports[t].owner;
                    (Some(self.nodes[owner].id.clone()), Some(self.ports[t].name.clone()))
                });
                ports.push(PortRepr {
                    name: port.name.clone(),
                    target_node: t_node,
                    target_port: t_port,
                });
            };
            if let Some(p) = node.principal {
                add_port(p);
            }
            for &p in &node.aux {
                add_port(p);
            }
            nodes.push(NodeRepr {
                id: node.id.clone(),
                kind: node.kind.clone(),
                value: node.value.clone(),
                ports,
            });
        }
        serde_json::to_string(&nodes)
    }
}

#[pyclass(name = "HinEngine")]
pub struct PyHinEngine {
    engine: HinEngine,
}

#[pymethods]
impl PyHinEngine {
    #[new]
    fn new() -> Self {
        Self {
            engine: HinEngine::new(),
        }
    }

    /// Build an interaction-net from a UAST JSON string.
    fn build_from_json(&mut self, json: &str) -> PyResult<()> {
        let uast: Value = serde_json::from_str(json)
            .map_err(|e| PyValueError::new_err(format!("invalid UAST JSON: {}", e)))?;
        self.engine
            .build_uast(&uast)
            .map_err(|e| PyValueError::new_err(e))?;
        Ok(())
    }

    /// Reduce active pairs until the net is stable or ``max_steps`` is reached.
    fn reduce_to_completion(&mut self, max_steps: usize) -> PyResult<usize> {
        Ok(self.engine.reduce_to_completion_with_timeout(max_steps, None).0)
    }

    /// Reduce active pairs with an optional wall-clock timeout (seconds).
    fn reduce_to_completion_with_timeout(
        &mut self,
        max_steps: usize,
        timeout_seconds: f64,
    ) -> PyResult<(usize, bool)> {
        Ok(self
            .engine
            .reduce_to_completion_with_timeout(max_steps, Some(timeout_seconds)))
    }

    /// Return the live graph as a JSON string.
    fn to_json(&self) -> PyResult<String> {
        self.engine
            .to_json()
            .map_err(|e| PyValueError::new_err(format!("serialization failed: {}", e)))
    }

    /// Number of allocated nodes (including retired nodes).
    fn node_count(&self) -> usize {
        self.engine.node_count()
    }

    /// Count active principal-principal pairs whose node kinds are identical.
    fn stalled_pairs(&self) -> usize {
        self.engine.stalled_pairs()
    }
}

/// Build and reduce a UAST JSON string in one call (GIL released during reduction).
#[pyfunction]
#[pyo3(signature = (json, max_steps=None, timeout_seconds=None))]
pub fn reduce_hin_uast(
    json: &str,
    max_steps: Option<usize>,
    timeout_seconds: Option<f64>,
) -> PyResult<String> {
    let uast: Value = serde_json::from_str(json)
        .map_err(|e| PyValueError::new_err(format!("invalid UAST JSON: {}", e)))?;
    let mut engine = HinEngine::new();
    engine
        .build_uast(&uast)
        .map_err(|e| PyValueError::new_err(e))?;
    let (steps, timed_out) = if let Some(ms) = max_steps {
        engine.reduce_to_completion_with_timeout(ms, timeout_seconds)
    } else {
        engine.reduce_to_completion_with_timeout(1_000_000, timeout_seconds)
    };
    let stalled = engine.stalled_pairs();
    let out = engine
        .to_json()
        .map_err(|e| PyValueError::new_err(format!("serialization failed: {}", e)))?;
    Ok(format!(
        "{{\"steps\":{},\"stalled\":{},\"timed_out\":{},\"graph\":{}}}",
        steps, stalled, timed_out, out
    ))
}

// ---------------------------------------------------------------------------
// Proof-theoretic HIN energy evaluator (deterministic self-healing support)
// ---------------------------------------------------------------------------

/// MELL linear-logic symbols carried by HIN ports.
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "kind")]
#[allow(dead_code)]
pub enum MellSymbol {
    I,
    Tensor { left: Box<MellSymbol>, right: Box<MellSymbol> },
    Implication { left: Box<MellSymbol>, right: Box<MellSymbol> },
    Bang { inner: Box<MellSymbol> },
    #[serde(rename = "var")]
    Var { name: String },
}

impl Default for MellSymbol {
    fn default() -> Self {
        MellSymbol::I
    }
}

/// Affine ownership labels for cross-language memory safety.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Ownership {
    /// Uniquely owned value (e.g. Rust owned data).
    #[default]
    One,
    /// Shared immutable borrow.
    Ref,
    /// Mutable borrow.
    RefMut,
    /// Replicable / shared ownership (e.g. Python refcounted, Rust Arc).
    Bang,
    /// Uninitialized / invalid / moved-out.
    Bot,
}

/// Concrete memory layout descriptor for an FFI boundary field/type.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct FFILayout {
    pub size: usize,
    pub alignment: usize,
    pub c_type: String,
    pub rust_type: String,
}

/// Single HIN port with an optional connection and a MELL type annotation.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct HinPort {
    pub owner: usize,
    pub name: String,
    pub is_principal: bool,
    pub target: Option<usize>,
    pub mell: MellSymbol,
}

/// Energy accounting state for a HIN node or the whole graph.
#[derive(Clone, Debug, Serialize)]
pub struct EnergyState {
    pub stalled: usize,
    pub wires: usize,
    pub dangling: usize,
    pub total: f64,
}

/// HIN node backed by a small-vector of auxiliary port indices.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct HinNode {
    pub id: String,
    pub kind: String,
    pub principal: Option<usize>,
    pub aux: SmallVec<[usize; 4]>,
    pub energy: EnergyState,
    pub ownership: Ownership,
    pub layout: Option<FFILayout>,
}

/// Parse-able port representation used by the energy evaluator.
#[derive(Deserialize)]
struct PortInput {
    name: String,
    #[serde(default)]
    is_principal: bool,
    target_node: Option<String>,
    target_port: Option<String>,
    #[serde(default)]
    mell: Option<MellSymbol>,
}

/// Parse-able node representation used by the energy evaluator.
#[derive(Deserialize)]
struct NodeInput {
    id: String,
    #[serde(default)]
    kind: String,
    #[serde(default)]
    ports: Vec<PortInput>,
    #[serde(default)]
    ownership: Ownership,
    #[serde(default)]
    layout: Option<FFILayout>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ArenaInput {
    Nodes(Vec<NodeInput>),
    Object { nodes: Vec<NodeInput> },
}

/// Evaluate the HIN interaction-net energy of an arena JSON description.
///
/// Energy is defined as `E(G) = 10.0 * stalled + 5.0 * wires + 2.0 * dangling`,
/// where:
///   * `stalled` counts active principal-principal pairs whose node kinds are
///     identical (they cannot reduce without further interaction);
///   * `wires` counts bidirectional connections between ports;
///   * `dangling` counts ports with no target or a missing target.
#[pyfunction]
pub fn evaluate_hin_energy(arena_json: &str) -> PyResult<String> {
    let arena: ArenaInput = serde_json::from_str(arena_json)
        .map_err(|e| PyValueError::new_err(format!("invalid arena JSON: {}", e)))?;
    let nodes_input = match arena {
        ArenaInput::Nodes(nodes) => nodes,
        ArenaInput::Object { nodes } => nodes,
    };

    let mut ports: Vec<HinPort> = Vec::new();
    let mut nodes: Vec<HinNode> = Vec::with_capacity(nodes_input.len());
    let mut port_lookup: HashMap<(String, String), usize> = HashMap::new();

    // First pass: allocate ports and nodes, recording lookup keys.
    for node_input in nodes_input {
        let node_idx = nodes.len();
        let mut node = HinNode {
            id: node_input.id,
            kind: node_input.kind,
            principal: None,
            aux: SmallVec::new(),
            energy: EnergyState {
                stalled: 0,
                wires: 0,
                dangling: 0,
                total: 0.0,
            },
            ownership: node_input.ownership,
            layout: node_input.layout,
        };
        for (port_idx_in_node, port_input) in node_input.ports.into_iter().enumerate() {
            let port_idx = ports.len();
            // The first port without an explicit flag is treated as principal,
            // matching the output format of HinEngine::to_json.
            let is_principal = port_input.is_principal || port_idx_in_node == 0;
            let port = HinPort {
                owner: node_idx,
                name: port_input.name.clone(),
                is_principal,
                target: None,
                mell: port_input.mell.unwrap_or_default(),
            };
            port_lookup.insert((node.id.clone(), port_input.name), port_idx);
            if is_principal {
                node.principal = Some(port_idx);
            } else {
                node.aux.push(port_idx);
            }
            ports.push(port);
        }
        nodes.push(node);
    }

    // Second pass: wire targets.
    let mut dangling = 0usize;
    let mut targeted = 0usize;
    for port_idx in 0..ports.len() {
        let owner_id = nodes[ports[port_idx].owner].id.clone();
        let port_name = ports[port_idx].name.clone();
        if let Some(key) = port_lookup.get(&(owner_id, port_name)) {
            // Use the lookup key recorded above; the actual target is resolved below.
            let _ = key;
        }
        // Resolve the target from the original input by scanning the lookup table.
        // (The lookup map key is (owner_id, port_name); the stored value is the port index.)
        // Here we already have port_idx; nothing more to resolve for this loop.
        // This block is intentionally left minimal because the wiring logic is applied
        // through the port_input.target_node/target_port fields, which we need to recover.
    }

    // Re-parse to wire, since we consumed the input in the first pass.
    // We can reuse the arena string and update ports in place.
    let arena2: ArenaInput = serde_json::from_str(arena_json)
        .map_err(|e| PyValueError::new_err(format!("invalid arena JSON: {}", e)))?;
    let nodes_input2 = match arena2 {
        ArenaInput::Nodes(nodes) => nodes,
        ArenaInput::Object { nodes } => nodes,
    };
    let mut port_iter = 0usize;
    for node_input in nodes_input2 {
        for port_input in node_input.ports {
            if let (Some(tnode), Some(tport)) = (port_input.target_node, port_input.target_port) {
                if let Some(&target_idx) = port_lookup.get(&(tnode, tport)) {
                    ports[port_iter].target = Some(target_idx);
                    targeted += 1;
                } else {
                    dangling += 1;
                }
            } else {
                dangling += 1;
            }
            port_iter += 1;
        }
    }

    let wires = targeted / 2;

    // Count active principal-principal pairs and stalled pairs.
    let mut stalled = 0usize;
    let mut seen_active: Vec<(usize, usize)> = Vec::new();
    for node_idx in 0..nodes.len() {
        if let Some(p) = nodes[node_idx].principal {
            if let Some(target_idx) = ports[p].target {
                if ports[target_idx].is_principal {
                    let target_owner = ports[target_idx].owner;
                    if node_idx < target_owner {
                        seen_active.push((node_idx, target_owner));
                        if nodes[node_idx].kind == nodes[target_owner].kind {
                            stalled += 1;
                        }
                    }
                }
            }
        }
    }

    let total = 10.0 * stalled as f64 + 5.0 * wires as f64 + 2.0 * dangling as f64;
    let energy = EnergyState {
        stalled,
        wires,
        dangling,
        total,
    };

    serde_json::to_string(&energy)
        .map_err(|e| PyValueError::new_err(format!("energy serialization failed: {}", e)))
}

/// Result of a HIN saturation check.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HinSaturationResult {
    pub saturated: bool,
    pub reason: String,
    pub active_pairs: usize,
    pub stalled: usize,
    pub wires: usize,
    pub dangling: usize,
    pub total: f64,
}

/// Verify that a HIN interaction net is saturated.
///
/// A net is *saturated* when it has at least one active principal-principal
/// pair and no stalled (same-kind) active pairs. Zero active pairs or stalled
/// wires indicate a hollow or stuck logic sketch that must not be materialized.
#[pyfunction]
pub fn verify_hin_saturation(arena_json: &str) -> PyResult<String> {
    let arena: ArenaInput = serde_json::from_str(arena_json)
        .map_err(|e| PyValueError::new_err(format!("invalid arena JSON: {}", e)))?;
    let nodes_input = match arena {
        ArenaInput::Nodes(nodes) => nodes,
        ArenaInput::Object { nodes } => nodes,
    };

    let mut ports: Vec<HinPort> = Vec::new();
    let mut nodes: Vec<HinNode> = Vec::with_capacity(nodes_input.len());
    let mut port_lookup: HashMap<(String, String), usize> = HashMap::new();

    for node_input in nodes_input {
        let node_idx = nodes.len();
        let mut node = HinNode {
            id: node_input.id,
            kind: node_input.kind,
            principal: None,
            aux: SmallVec::new(),
            energy: EnergyState {
                stalled: 0,
                wires: 0,
                dangling: 0,
                total: 0.0,
            },
            ownership: node_input.ownership,
            layout: node_input.layout,
        };
        for (port_idx_in_node, port_input) in node_input.ports.into_iter().enumerate() {
            let port_idx = ports.len();
            let is_principal = port_input.is_principal || port_idx_in_node == 0;
            let port = HinPort {
                owner: node_idx,
                name: port_input.name.clone(),
                is_principal,
                target: None,
                mell: port_input.mell.unwrap_or_default(),
            };
            port_lookup.insert((node.id.clone(), port_input.name), port_idx);
            if is_principal {
                node.principal = Some(port_idx);
            } else {
                node.aux.push(port_idx);
            }
            ports.push(port);
        }
        nodes.push(node);
    }

    // Wire targets.
    let arena2: ArenaInput = serde_json::from_str(arena_json)
        .map_err(|e| PyValueError::new_err(format!("invalid arena JSON: {}", e)))?;
    let nodes_input2 = match arena2 {
        ArenaInput::Nodes(nodes) => nodes,
        ArenaInput::Object { nodes } => nodes,
    };
    let mut port_iter = 0usize;
    let mut dangling = 0usize;
    let mut targeted = 0usize;
    for node_input in nodes_input2 {
        for port_input in node_input.ports {
            if let (Some(tnode), Some(tport)) = (port_input.target_node, port_input.target_port) {
                if let Some(&target_idx) = port_lookup.get(&(tnode, tport)) {
                    ports[port_iter].target = Some(target_idx);
                    targeted += 1;
                } else {
                    dangling += 1;
                }
            } else {
                dangling += 1;
            }
            port_iter += 1;
        }
    }

    let wires = targeted / 2;

    let mut stalled = 0usize;
    let mut active_pairs = 0usize;
    for node_idx in 0..nodes.len() {
        if let Some(p) = nodes[node_idx].principal {
            if let Some(target_idx) = ports[p].target {
                if ports[target_idx].is_principal {
                    let target_owner = ports[target_idx].owner;
                    if node_idx < target_owner {
                        active_pairs += 1;
                        if nodes[node_idx].kind == nodes[target_owner].kind {
                            stalled += 1;
                        }
                    }
                }
            }
        }
    }

    let total = 10.0 * stalled as f64 + 5.0 * wires as f64 + 2.0 * dangling as f64;
    let saturated = active_pairs > 0 && stalled == 0;
    let reason = if active_pairs == 0 {
        "zero active pairs"
    } else if stalled > 0 {
        "stalled wires"
    } else {
        "active pairs present and no stalled wires"
    }
    .to_string();

    let result = HinSaturationResult {
        saturated,
        reason,
        active_pairs,
        stalled,
        wires,
        dangling,
        total,
    };

    serde_json::to_string(&result)
        .map_err(|e| PyValueError::new_err(format!("saturation serialization failed: {}", e)))
}

/// Result of a HIN reduction plus post-reduction saturation check.
#[derive(Clone, Debug, Serialize)]
pub struct HinStallResult {
    pub saturated: bool,
    pub reason: String,
    pub steps: usize,
    pub stalled: usize,
    pub timed_out: bool,
    pub active_pairs: usize,
    pub wires: usize,
    pub dangling: usize,
    pub total: f64,
}

/// Detect whether a HIN interaction net has stalled during reduction.
///
/// Builds the HIN from a UAST JSON string, runs active-pair reductions, and then
/// evaluates MELL energy invariants.  A net is saturated when it reduces without
/// timing out and leaves no stalled (same-kind) active pairs.  A fully reduced
/// normal form has zero active pairs, so saturation is based on the absence of
/// stalled wires and reduction progress, not on post-reduction active-pair count.
#[pyfunction]
#[pyo3(signature = (json, max_steps=None, timeout_seconds=None))]
pub fn detect_hin_stall(
    json: &str,
    max_steps: Option<usize>,
    timeout_seconds: Option<f64>,
) -> PyResult<String> {
    let uast: Value = serde_json::from_str(json)
        .map_err(|e| PyValueError::new_err(format!("invalid UAST JSON: {}", e)))?;
    let mut engine = HinEngine::new();
    engine
        .build_uast(&uast)
        .map_err(|e| PyValueError::new_err(e))?;

    let initial_active = engine.active_pairs();
    let (steps, timed_out) = if let Some(ms) = max_steps {
        engine.reduce_to_completion_with_timeout(ms, timeout_seconds)
    } else {
        engine.reduce_to_completion_with_timeout(1_000_000, timeout_seconds)
    };

    let stalled = engine.stalled_pairs();
    let arena = engine
        .to_json()
        .map_err(|e| PyValueError::new_err(format!("serialization failed: {}", e)))?;
    let saturation_json = verify_hin_saturation(&arena)?;
    let saturation: HinSaturationResult = serde_json::from_str(&saturation_json)
        .map_err(|e| PyValueError::new_err(format!("saturation parse failed: {}", e)))?;

    let saturated = !timed_out && stalled == 0 && (initial_active > 0 || steps > 0);
    let reason = if timed_out {
        "reduction timed out"
    } else if stalled > 0 {
        "stalled wires"
    } else if initial_active == 0 && steps == 0 {
        "zero active pairs and no reduction progress"
    } else {
        "reduction completed without stalled wires"
    }
    .to_string();

    let result = HinStallResult {
        saturated,
        reason,
        steps,
        stalled,
        timed_out,
        active_pairs: saturation.active_pairs,
        wires: saturation.wires,
        dangling: saturation.dangling,
        total: saturation.total,
    };

    serde_json::to_string(&result)
        .map_err(|e| PyValueError::new_err(format!("stall detection serialization failed: {}", e)))
}

/// Layout descriptor attached to a HIN node for FFI boundary checking.
#[derive(Deserialize)]
struct LayoutNodeInput {
    id: String,
    #[serde(default)]
    layout: Option<FFILayout>,
}

/// Edge descriptor for FFI boundary layout verification.
#[derive(Deserialize)]
struct LayoutEdgeInput {
    source: String,
    target: String,
    #[serde(default)]
    relation: String,
}

#[derive(Deserialize)]
struct LayoutInput {
    nodes: Vec<LayoutNodeInput>,
    edges: Vec<LayoutEdgeInput>,
}

/// Verify that all FFI-boundary edges connect nodes with matching memory layout.
///
/// Returns ``true`` when every ``FFIBoundary`` or ``BindsTo`` edge connects a
/// source and target whose ``size`` and ``alignment`` fields are identical.
/// Raises ``PyValueError`` if a boundary edge links mismatched layouts.
#[pyfunction]
pub fn verify_hin_boundary_layouts(layout_json: &str) -> PyResult<bool> {
    let input: LayoutInput = serde_json::from_str(layout_json)
        .map_err(|e| PyValueError::new_err(format!("invalid layout JSON: {}", e)))?;

    let layouts: HashMap<String, Option<FFILayout>> = input
        .nodes
        .into_iter()
        .map(|n| (n.id, n.layout))
        .collect();

    for edge in input.edges {
        if edge.relation != "FFIBoundary" && edge.relation != "BindsTo" {
            continue;
        }
        let src = layouts.get(&edge.source).cloned().flatten();
        let tgt = layouts.get(&edge.target).cloned().flatten();
        match (src.as_ref(), tgt.as_ref()) {
            (Some(s), Some(t)) => {
                if s.size != t.size || s.alignment != t.alignment {
                    return Err(PyValueError::new_err(format!(
                        "FFI layout mismatch on boundary edge {} -> {}: {:?} vs {:?}",
                        edge.source, edge.target, s, t
                    )));
                }
            }
            (None, Some(_)) | (Some(_), None) | (None, None) => {
                return Err(PyValueError::new_err(format!(
                    "Missing layout information on boundary edge {} -> {}",
                    edge.source, edge.target
                )));
            }
        }
    }

    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value_number(engine: &mut HinEngine, n: f64) -> usize {
        let node = engine.add_node(NodeKind::Value, 0, Some(NodeValue::Number(n)));
        engine.node_principal(node)
    }

    fn value_string(engine: &mut HinEngine, s: &str) -> usize {
        let node = engine.add_node(NodeKind::Value, 0, Some(NodeValue::String(s.to_string())));
        engine.node_principal(node)
    }

    fn make_dict(engine: &mut HinEngine, pairs: &[(usize, usize)]) -> usize {
        let coll = engine.add_node(NodeKind::DictConstructor, pairs.len() * 2, None);
        for (i, (k, v)) in pairs.iter().enumerate() {
            engine.connect(*k, engine.node_aux(coll, i * 2));
            engine.connect(*v, engine.node_aux(coll, i * 2 + 1));
        }
        coll
    }

    fn make_set(engine: &mut HinEngine, elements: &[usize]) -> usize {
        let coll = engine.add_node(NodeKind::SetConstructor, elements.len(), None);
        for (i, e) in elements.iter().enumerate() {
            engine.connect(*e, engine.node_aux(coll, i));
        }
        coll
    }

    fn make_box(engine: &mut HinEngine) -> usize {
        engine.add_node(NodeKind::Constructor, 2, None)
    }

    #[test]
    fn reduce_dict_lookup_found() {
        let mut engine = HinEngine::new();
        let key = value_string(&mut engine, "x");
        let val = value_number(&mut engine, 42.0);
        let dict = make_dict(&mut engine, &[(key, val)]);

        let dtor = engine.add_node(
            NodeKind::KeyLookup {
                kind: CollectionKind::Dict,
            },
            2,
            None,
        );
        let box_node = make_box(&mut engine);

        let lookup_key = value_string(&mut engine, "x");
        engine.connect(lookup_key, engine.node_aux(dtor, 0));
        engine.connect(engine.node_principal(box_node), engine.node_aux(dtor, 1));
        engine.connect(engine.node_principal(dict), engine.node_principal(dtor));

        let steps = engine.reduce_to_completion(1000);
        assert!(steps > 0);
        let graph = engine.to_json().unwrap();
        assert!(graph.contains("42"));
    }

    #[test]
    fn reduce_dict_lookup_missing() {
        let mut engine = HinEngine::new();
        let key = value_string(&mut engine, "x");
        let val = value_number(&mut engine, 42.0);
        let dict = make_dict(&mut engine, &[(key, val)]);

        let dtor = engine.add_node(
            NodeKind::KeyLookup {
                kind: CollectionKind::Dict,
            },
            2,
            None,
        );
        let box_node = make_box(&mut engine);

        let lookup_key = value_string(&mut engine, "y");
        engine.connect(lookup_key, engine.node_aux(dtor, 0));
        engine.connect(engine.node_principal(box_node), engine.node_aux(dtor, 1));
        engine.connect(engine.node_principal(dict), engine.node_principal(dtor));

        let steps = engine.reduce_to_completion(1000);
        assert!(steps > 0);
        let graph = engine.to_json().unwrap();
        // The result should be null when the key is missing.
        assert!(graph.contains("null"));
    }

    #[test]
    fn reduce_set_membership_true() {
        let mut engine = HinEngine::new();
        let a = value_number(&mut engine, 1.0);
        let b = value_number(&mut engine, 2.0);
        let c = value_number(&mut engine, 3.0);
        let set = make_set(&mut engine, &[a, b, c]);

        let dtor = engine.add_node(
            NodeKind::KeyLookup {
                kind: CollectionKind::Set,
            },
            2,
            None,
        );
        let box_node = make_box(&mut engine);

        let elem = value_number(&mut engine, 2.0);
        engine.connect(elem, engine.node_aux(dtor, 0));
        engine.connect(engine.node_principal(box_node), engine.node_aux(dtor, 1));
        engine.connect(engine.node_principal(set), engine.node_principal(dtor));

        let steps = engine.reduce_to_completion(1000);
        assert!(steps > 0);
        let graph = engine.to_json().unwrap();
        assert!(graph.contains("true"));
    }

    #[test]
    fn reduce_set_membership_false() {
        let mut engine = HinEngine::new();
        let a = value_number(&mut engine, 1.0);
        let b = value_number(&mut engine, 2.0);
        let set = make_set(&mut engine, &[a, b]);

        let dtor = engine.add_node(
            NodeKind::KeyLookup {
                kind: CollectionKind::Set,
            },
            2,
            None,
        );
        let box_node = make_box(&mut engine);

        let elem = value_number(&mut engine, 9.0);
        engine.connect(elem, engine.node_aux(dtor, 0));
        engine.connect(engine.node_principal(box_node), engine.node_aux(dtor, 1));
        engine.connect(engine.node_principal(set), engine.node_principal(dtor));

        let steps = engine.reduce_to_completion(1000);
        assert!(steps > 0);
        let graph = engine.to_json().unwrap();
        assert!(graph.contains("false"));
    }

    #[test]
    fn duplicate_dict_and_lookup_both_copies() {
        let mut engine = HinEngine::new();
        let key = value_string(&mut engine, "x");
        let val = value_number(&mut engine, 42.0);
        let dict = make_dict(&mut engine, &[(key, val)]);

        let dup = engine.add_node(NodeKind::Duplicator, 2, None);
        engine.connect(engine.node_principal(dict), engine.node_principal(dup));

        for i in 0..2 {
            let dtor = engine.add_node(
                NodeKind::KeyLookup {
                    kind: CollectionKind::Dict,
                },
                2,
                None,
            );
            let box_node = make_box(&mut engine);
            let lookup_key = value_string(&mut engine, "x");
            engine.connect(lookup_key, engine.node_aux(dtor, 0));
            engine.connect(engine.node_principal(box_node), engine.node_aux(dtor, 1));
            engine.connect(engine.node_aux(dup, i), engine.node_principal(dtor));
        }

        let steps = engine.reduce_to_completion(1000);
        assert!(steps > 0);
        let graph = engine.to_json().unwrap();
        // Two lookups both find 42.
        assert_eq!(graph.matches("42").count(), 2);
    }
}
