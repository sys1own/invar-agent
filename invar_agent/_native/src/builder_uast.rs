use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SIDimensionSpec {
    pub exponents: [i8; 7],
    #[serde(default)]
    pub semantic_tag: u8,
    pub canonical_unit: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct IntervalBoundSpec {
    pub lower: Option<f64>,
    pub upper: Option<f64>,
    #[serde(default = "default_true")]
    pub lower_inclusive: bool,
    #[serde(default = "default_true")]
    pub upper_inclusive: bool,
}

fn default_true() -> bool {
    true
}
fn default_slot_retry_cap() -> u8 {
    2
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TypedPortField {
    pub name: String,
    pub scalar_type: String,
    pub dimension: SIDimensionSpec,
    pub bounds: Option<IntervalBoundSpec>,
    pub mell_linearity: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct InterfacePortNode {
    pub node_id: String,
    pub port_name: String,
    pub inputs: Vec<TypedPortField>,
    pub outputs: Vec<TypedPortField>,
    pub z3_preconditions: Vec<String>,
    pub z3_postconditions: Vec<String>,
    pub his_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct KernelNode {
    pub node_id: String,
    pub kernel_name: String,
    pub research_compute_uast_json: String,
    pub csel_expressions: Vec<String>,
    pub manifold_projectors: Vec<String>,
    pub export_adjoint_jacobian: bool,
    pub output_port_id: String,
    pub his_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DeterministicModuleNode {
    pub node_id: String,
    pub module_role: String,
    pub emitter_id: String,
    pub target_relpaths: Vec<String>,
    pub required_port_ids: Vec<String>,
    pub provided_port_ids: Vec<String>,
    pub emitter_config: BTreeMap<String, serde_json::Value>,
    pub his_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LlmExtensionSlotNode {
    pub node_id: String,
    pub slot_kind: String,
    pub target_relpath: String,
    pub intent_summary: String,
    pub frozen_port_id: String,
    pub frozen_his_digest: String,
    pub fallback_emitter_id: String,
    #[serde(default = "default_slot_retry_cap")]
    pub max_tier2_retries: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum BuilderNode {
    Kernel(KernelNode),
    InterfacePort(InterfacePortNode),
    DeterministicModule(DeterministicModuleNode),
    LlmExtensionSlot(LlmExtensionSlotNode),
}

impl BuilderNode {
    pub fn node_id(&self) -> &str {
        match self {
            Self::Kernel(n) => &n.node_id,
            Self::InterfacePort(n) => &n.node_id,
            Self::DeterministicModule(n) => &n.node_id,
            Self::LlmExtensionSlot(n) => &n.node_id,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DependencyEdge {
    pub from_node: String,
    pub to_node: String,
    pub edge_kind: String,
    pub port_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ProjectGraphUAST {
    pub schema_version: String,
    pub project_id: String,
    pub canonical_ir_digest: String,
    pub active_profiles: Vec<String>,
    pub nodes: Vec<BuilderNode>,
    pub edges: Vec<DependencyEdge>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WavefrontScheduleResult {
    pub is_nilpotent: bool,
    pub spectral_radius: f64,
    pub wavefronts: Vec<Vec<String>>,
    pub resolvent_flat: Vec<f64>,
    pub node_order: Vec<String>,
    pub cycle_witness: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmwDeltaResult {
    pub is_valid_dag: bool,
    pub capacitance_det: f64,
    pub updated_resolvent_flat: Vec<f64>,
    pub affected_node_ids: Vec<String>,
}

pub fn is_digest(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn nonempty(s: &str) -> Result<(), String> {
    if s.trim().is_empty() {
        Err("BuilderUAST: empty identifier".into())
    } else {
        Ok(())
    }
}

fn safe_path(s: &str) -> Result<(), String> {
    if s.is_empty()
        || s.contains(['\\', ':', '\0'])
        || s.split('/').any(|p| p.is_empty() || p == "." || p == "..")
    {
        return Err(format!("BuilderUAST: unsafe target path {s:?}"));
    }
    Ok(())
}

impl ProjectGraphUAST {
    pub fn from_json(json: &str) -> Result<Self, String> {
        let graph: Self = serde_json::from_str(json).map_err(|e| e.to_string())?;
        graph.validate()?;
        Ok(graph)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != "builder_uast.v1" || !is_digest(&self.canonical_ir_digest) {
            return Err("BuilderUAST: invalid schema version or IR digest".into());
        }
        nonempty(&self.project_id)?;
        for profile in &self.active_profiles {
            if ![
                "hybrid_compute_library",
                "interactive_workbench",
                "executable_paper_audit",
                "calibration_inverse_design",
            ]
            .contains(&profile.as_str())
            {
                return Err(format!("BuilderUAST: unknown profile {profile}"));
            }
        }
        let mut nodes = BTreeMap::new();
        for node in &self.nodes {
            nonempty(node.node_id())?;
            if nodes.insert(node.node_id(), node).is_some() {
                return Err(format!("BuilderUAST: duplicate node {}", node.node_id()));
            }
        }
        let port = |id: &str| -> Result<&InterfacePortNode, String> {
            match nodes.get(id) {
                Some(BuilderNode::InterfacePort(p)) => Ok(p),
                _ => Err(format!("BuilderUAST: missing interface port {id}")),
            }
        };
        let mut paths = BTreeSet::new();
        for node in &self.nodes {
            let digest = match node {
                BuilderNode::InterfacePort(n) => {
                    nonempty(&n.port_name)?;
                    for fields in [&n.inputs, &n.outputs] {
                        let mut names = BTreeSet::new();
                        for f in fields {
                            nonempty(&f.name)?;
                            if !names.insert(&f.name)
                                || !["f64", "i64", "bool", "vec_f64", "str", "table"]
                                    .contains(&f.scalar_type.as_str())
                                || !["Linear", "Bang", "Tensor"]
                                    .contains(&f.mell_linearity.as_str())
                            {
                                return Err("BuilderUAST: invalid or duplicate typed field".into());
                            }
                            nonempty(&f.dimension.canonical_unit)?;
                            if let Some(b) = &f.bounds {
                                if b.lower.into_iter().chain(b.upper).any(|x| !x.is_finite())
                                    || matches!((b.lower, b.upper), (Some(l), Some(u)) if l > u || (l == u && (!b.lower_inclusive || !b.upper_inclusive)))
                                {
                                    return Err("BuilderUAST: empty or nonfinite interval".into());
                                }
                            }
                        }
                    }
                    &n.his_digest
                }
                BuilderNode::Kernel(n) => {
                    nonempty(&n.kernel_name)?;
                    crate::uast::canonical_json_bytes(&n.research_compute_uast_json)?;
                    port(&n.output_port_id)?;
                    &n.his_digest
                }
                BuilderNode::DeterministicModule(n) => {
                    if ![
                        "package_metadata",
                        "cpp_cabi_bridge",
                        "python_fallback_kernel",
                        "python_sdk",
                        "cli_runner",
                        "sweep_pareto_runner",
                        "data_serializer",
                        "fastapi_workbench",
                        "calibration_engine",
                        "audit_report",
                        "pytest_z3_suite",
                        "receipt_builder",
                    ]
                    .contains(&n.module_role.as_str())
                    {
                        return Err(format!(
                            "BuilderUAST: invalid module role {}",
                            n.module_role
                        ));
                    }
                    nonempty(&n.emitter_id)?;
                    for id in n.required_port_ids.iter().chain(&n.provided_port_ids) {
                        port(id)?;
                    }
                    for path in &n.target_relpaths {
                        safe_path(path)?;
                        if !paths.insert(path) {
                            return Err(format!("BuilderUAST: duplicate target {path}"));
                        }
                    }
                    &n.his_digest
                }
                BuilderNode::LlmExtensionSlot(n) => {
                    let expected = match n.slot_kind.as_str() {
                        "bespoke_data_adapter" => "fb_csv_json_adapter",
                        "domain_visualization" => "fb_svg_trajectory_viz",
                        "workflow_glue" => "fb_workflow_glue",
                        "narrative_documentation" => "fb_markdown_narrative_doc",
                        _ => return Err("BuilderUAST: invalid slot kind".into()),
                    };
                    if n.fallback_emitter_id != expected
                        || n.max_tier2_retries > 2
                        || port(&n.frozen_port_id)?.his_digest != n.frozen_his_digest
                    {
                        return Err(
                            "BuilderUAST: invalid fallback, retry cap, or frozen HIS".into()
                        );
                    }
                    nonempty(&n.intent_summary)?;
                    safe_path(&n.target_relpath)?;
                    if !paths.insert(&n.target_relpath) {
                        return Err("BuilderUAST: duplicate target".into());
                    }
                    &n.frozen_his_digest
                }
            };
            if !is_digest(digest) {
                return Err(format!("BuilderUAST: invalid HIS for {}", node.node_id()));
            }
        }
        let mut edges = BTreeSet::new();
        for edge in &self.edges {
            if !nodes.contains_key(edge.from_node.as_str())
                || !nodes.contains_key(edge.to_node.as_str())
                || !["data_port", "abi_link", "verification_dep", "receipt_dep"]
                    .contains(&edge.edge_kind.as_str())
                || !edges.insert((
                    &edge.from_node,
                    &edge.to_node,
                    &edge.edge_kind,
                    &edge.port_id,
                ))
            {
                return Err("BuilderUAST: invalid or duplicate dependency edge".into());
            }
            if let Some(id) = &edge.port_id {
                port(id)?;
            }
        }
        Ok(())
    }
}

pub fn validate_uast_json(json: &str) -> Result<String, String> {
    let graph = ProjectGraphUAST::from_json(json)?;
    let json = serde_json::to_string(&graph).map_err(|e| e.to_string())?;
    String::from_utf8(crate::uast::canonical_json_bytes(&json)?).map_err(|e| e.to_string())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use serde_json::{json, Value};

    pub fn module(id: &str) -> Value {
        json!({"kind":"deterministic_module", "node_id":id, "module_role":"python_sdk", "emitter_id":"sdk", "target_relpaths":[format!("{id}.py")], "required_port_ids":[], "provided_port_ids":[], "emitter_config":{}, "his_digest":"a".repeat(64)})
    }

    pub fn graph(nodes: Vec<Value>, edges: Vec<Value>) -> Value {
        json!({"schema_version":"builder_uast.v1", "project_id":"test", "canonical_ir_digest":"b".repeat(64), "active_profiles":[], "nodes":nodes, "edges":edges})
    }

    #[test]
    fn test_builder_uast_serde_roundtrip() {
        let port = json!({"kind":"interface_port", "node_id":"port", "port_name":"result", "inputs":[], "outputs":[], "z3_preconditions":[], "z3_postconditions":[], "his_digest":"a".repeat(64)});
        let kernel = json!({"kind":"kernel", "node_id":"kernel", "kernel_name":"core", "research_compute_uast_json":"{}", "csel_expressions":[], "manifold_projectors":[], "export_adjoint_jacobian":true, "output_port_id":"port", "his_digest":"a".repeat(64)});
        let slot = json!({"kind":"llm_extension_slot", "node_id":"slot", "slot_kind":"domain_visualization", "target_relpath":"viz.py", "intent_summary":"plot", "frozen_port_id":"port", "frozen_his_digest":"a".repeat(64), "fallback_emitter_id":"fb_svg_trajectory_viz", "max_tier2_retries":2});
        let value = graph(vec![kernel, port, module("sdk"), slot], vec![]);
        let parsed = ProjectGraphUAST::from_json(&value.to_string()).unwrap();
        assert_eq!(
            parsed,
            ProjectGraphUAST::from_json(&validate_uast_json(&value.to_string()).unwrap()).unwrap()
        );
        for field in ["fallback_emitter_id", "frozen_his_digest"] {
            let mut bad = value.clone();
            bad["nodes"][3].as_object_mut().unwrap().remove(field);
            assert!(validate_uast_json(&bad.to_string()).is_err());
        }
        for (field, replacement) in [
            ("max_tier2_retries", json!(3)),
            ("target_relpath", json!("../escape.py")),
            ("frozen_his_digest", json!("b".repeat(64))),
        ] {
            let mut bad = value.clone();
            bad["nodes"][3][field] = replacement;
            assert!(validate_uast_json(&bad.to_string()).is_err());
        }
        assert!(validate_uast_json(&graph(vec![], vec![]).to_string()).is_ok());
        assert!(validate_uast_json("{}").is_err());
        assert!(
            validate_uast_json(&graph(vec![module("x"), module("x")], vec![]).to_string()).is_err()
        );
    }
}
