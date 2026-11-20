//! Arena-allocated Lafont Interaction Net & causal fault slicer (`hin_engine.rs`).
//!
//! Every entity — AST symbol definition, staged overlay patch, SMT proof
//! obligation, or test execution result — is an Interaction Agent with exactly
//! one Principal Port (the causal driver) and zero or more Auxiliary Ports
//! (consumers).  Principal-principal meetings form Active Pairs governed by
//! deterministic annihilation (`alpha >< alpha`) and commutation (`alpha >< beta`)
//! rewrite rules.  All state nodes live in a generational arena with a free
//! list, giving zero-leak topological memory semantics, while
//! `trace_fault_slice` performs backward BFS along causal wires for pinpoint
//! fault isolation.

use std::collections::{HashMap, HashSet, VecDeque};

fn strict_builder_dimensions(expr: &crate::holographic_hin::CselExpr, symbols: &HashMap<String, ([i8; 7], u8)>) -> Result<([i8; 7], u8), String> {
    use crate::holographic_hin::CselExpr;
    type Dim = ([i8; 7], u8);
    fn same(a: Dim, b: Dim) -> Result<Dim, String> {
        if a.1 != 0 && b.1 != 0 && a.1 != b.1 { return Err("SEMANTIC_TAG_MISMATCH".into()); }
        if a.0 != b.0 { return Err("HIN_SI_DIM_MISMATCH".into()); }
        Ok((a.0, a.1.max(b.1)))
    }
    fn dimensionless(d: Dim) -> Result<(), String> {
        if d.0 != [0; 7] { Err("HIN_SI_DIM_MISMATCH: dimensionless operand required".into()) } else { Ok(()) }
    }
    fn product(a: Dim, b: Dim, sign: i16) -> Result<Dim, String> {
        let mut out = [0; 7];
        for i in 0..7 { out[i] = i8::try_from(a.0[i] as i16 + sign * b.0[i] as i16).map_err(|_| "HIN_DIMENSION_OVERFLOW")?; }
        let tag = if sign == -1 && a.1 == b.1 { 0 } else if a.1 == 0 { b.1 } else if b.1 == 0 || a.1 == b.1 { a.1 } else { 0 };
        Ok((out, tag))
    }
    let zero = ([0; 7], 0);
    let recurse = |e: &CselExpr| strict_builder_dimensions(e, symbols);
    match expr {
        CselExpr::Literal { value } if value.is_finite() => Ok(zero),
        CselExpr::Literal { .. } => Err("HIN_NONFINITE_LITERAL".into()),
        CselExpr::Symbol { name } => match name.as_str() {
            "t" => Ok(([0, 0, 1, 0, 0, 0, 0], 0)),
            "pi" | "e" => Ok(zero),
            _ => symbols.get(name).copied().ok_or_else(|| format!("HIN_UNBOUND_SYMBOL: {name}")),
        },
        CselExpr::Unary { op, arg } => {
            if !["+", "-", "neg"].contains(&op.as_str()) { return Err(format!("HIN_UNKNOWN_OPERATOR: {op}")); }
            recurse(arg)
        }
        CselExpr::Binary { op, left, right } => {
            let a = recurse(left)?;
            let b = recurse(right)?;
            match op.as_str() {
                "+" | "-" => {
                    if matches!(left.as_ref(), CselExpr::Literal { value } if *value == 0.0) { return Ok(b); }
                    if matches!(right.as_ref(), CselExpr::Literal { value } if *value == 0.0) { return Ok(a); }
                    same(a, b)
                }
                "*" => product(a, b, 1),
                "/" => product(a, b, -1),
                "^" | "**" => {
                    dimensionless(b)?;
                    match right.as_ref() {
                        CselExpr::Literal { value } if value.fract() == 0.0 && value.abs() <= 12.0 => {
                            let mut out = [0; 7];
                            for i in 0..7 { out[i] = i8::try_from(a.0[i] as i16 * *value as i16).map_err(|_| "HIN_DIMENSION_OVERFLOW")?; }
                            Ok((out, if *value == 1.0 { a.1 } else { 0 }))
                        }
                        _ => { dimensionless(a)?; Ok(zero) }
                    }
                }
                _ => Err(format!("HIN_UNKNOWN_OPERATOR: {op}")),
            }
        }
        CselExpr::Call { func, args } => {
            let dims: Vec<_> = args.iter().map(recurse).collect::<Result<_, _>>()?;
            let arity = match func.as_str() {
                "safe_div" | "min" | "max" | "kl_term" | "gammainc_reg" => 2,
                "smooth_clamp" | "where" | "hill_eq" | "michaelis_menten" | "betainc_reg" => 3,
                "safe_sqrt" | "sqrt" | "safe_exp" | "exp" | "safe_log" | "log" | "log1p" | "expm1" | "safe_sin" | "sin" | "safe_cos" | "cos" | "safe_tanh" | "tanh" | "safe_acos" | "acos" | "safe_asin" | "asin" | "abs" | "fabs" | "norm_cdf" | "norm_ppf" | "logistic_sigmoid" | "logit" | "shannon_term" => 1,
                _ => return Err(format!("HIN_UNKNOWN_FUNCTION: {func}")),
            };
            if dims.len() != arity { return Err(format!("HIN_FUNCTION_ARITY: {func}")); }
            let a = dims[0];
            match func.as_str() {
                "safe_div" => product(a, dims[1], -1),
                "safe_sqrt" | "sqrt" => {
                    if a.0.iter().any(|x| x % 2 != 0) { return Err("HIN_FRACTIONAL_DIMENSION".into()); }
                    Ok((a.0.map(|x| x / 2), 0))
                }
                "abs" | "fabs" => Ok(a),
                "min" | "max" => same(a, dims[1]),
                "smooth_clamp" => { same(a, dims[1])?; same(a, dims[2]) },
                "where" => { dimensionless(a)?; same(dims[1], dims[2]) },
                "michaelis_menten" => { same(a, dims[2])?; Ok(dims[1]) },
                "hill_eq" => { same(a, dims[1])?; dimensionless(dims[2])?; Ok(([0; 7], 1)) },
                _ => {
                    for d in dims { dimensionless(d)?; }
                    let tag = match func.as_str() {
                        "norm_cdf" | "logistic_sigmoid" | "gammainc_reg" | "betainc_reg" => 1,
                        "norm_ppf" | "logit" => 4,
                        "kl_term" | "shannon_term" => 2,
                        _ => 0,
                    };
                    Ok(([0; 7], tag))
                }
            }
        }
    }
}

pub fn check_equation_dimensional_homogeneity_json(csel_expr_json: &str, symbol_dims_json: &str) -> Result<String, String> {
    use crate::holographic_hin::{CselExpr, HolographicHinArena, QuantitySpec};
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct SymbolDimension {
        exponents: [i8; 7],
        #[serde(default)]
        semantic_tag: u8,
        #[serde(default)]
        canonical_unit: String,
        #[serde(default)]
        interval: Option<[f64; 2]>,
    }
    #[derive(serde::Deserialize)]
    #[serde(untagged)]
    enum Dimensions { Quantities(Vec<QuantitySpec>), Symbols(std::collections::BTreeMap<String, SymbolDimension>) }
    let expr: CselExpr = serde_json::from_str(csel_expr_json).map_err(|e| e.to_string())?;
    let quantities = match serde_json::from_str(symbol_dims_json).map_err(|e| e.to_string())? {
        Dimensions::Quantities(q) => q,
        Dimensions::Symbols(symbols) => symbols.into_iter().map(|(symbol, dim)| {
            let _ = dim.canonical_unit;
            QuantitySpec { symbol, si_dimension:dim.exponents, interval:dim.interval.unwrap_or([-1e300, 1e300]), is_state_variable:false, dimension_unspecified:false, semantic_tag:dim.semantic_tag }
        }).collect(),
    };
    let mut names = HashSet::new();
    for q in &quantities {
        if q.symbol.trim().is_empty() || !names.insert(&q.symbol) || q.dimension_unspecified
            || q.interval.iter().any(|x| !x.is_finite()) || q.interval[0] > q.interval[1] {
            return Err("HIN_INVALID_SYMBOL_DIMENSIONS".into());
        }
    }
    let symbols = quantities.iter().map(|q| (q.symbol.clone(), (q.si_dimension, q.semantic_tag))).collect();
    let expected = strict_builder_dimensions(&expr, &symbols)?;
    let mut arena = HolographicHinArena::new(quantities);
    let mut report = arena.verify_equation("builder_equation", &expr);
    report.output_dimensions = expected.0;
    report.output_semantic_tag = expected.1;
    let semantic = report.dimensional_warnings.iter().any(|w| w.contains("Semantic Tag Mismatch"));
    // Research permits inference warnings; Builder treats dimensional warnings as a hard gate.
    if semantic || !report.dimensional_warnings.is_empty() { report.valid = false; }
    let json = serde_json::to_string(&report).map_err(|e| e.to_string())?;
    if !report.valid {
        let code = if semantic { "SEMANTIC_TAG_MISMATCH" } else { "HIN_SI_DIM_MISMATCH" };
        return Err(format!("{code}: {json}"));
    }
    Ok(json)
}

pub fn extract_builder_causal_slice_json(uast_json: &str, fault_node_id: &str, max_hops: usize) -> Result<String, String> {
    let graph = crate::builder_uast::ProjectGraphUAST::from_json(uast_json)?;
    if !graph.nodes.iter().any(|n| n.node_id() == fault_node_id) { return Err("HIN_UNKNOWN_FAULT_NODE".into()); }
    let mut adjacent: HashMap<&str, Vec<&str>> = HashMap::new();
    for edge in &graph.edges {
        adjacent.entry(&edge.from_node).or_default().push(&edge.to_node);
        adjacent.entry(&edge.to_node).or_default().push(&edge.from_node);
    }
    let mut seen = std::collections::BTreeSet::new();
    let mut queue = VecDeque::from([(fault_node_id, 0usize)]);
    while let Some((node, hops)) = queue.pop_front() {
        if !seen.insert(node) || hops >= max_hops { continue; }
        if let Some(neighbors) = adjacent.get(node) {
            for neighbor in neighbors { queue.push_back((neighbor, hops + 1)); }
        }
    }
    let nodes: Vec<_> = graph.nodes.iter().filter(|n| seen.contains(n.node_id())).collect();
    let edges: Vec<_> = graph.edges.iter().filter(|e| seen.contains(e.from_node.as_str()) && seen.contains(e.to_node.as_str())).collect();
    Ok(serde_json::json!({"fault_node_id":fault_node_id, "max_hops":max_hops, "node_ids":seen, "nodes":nodes, "edges":edges}).to_string())
}

#[cfg(test)]
mod builder_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_hin_semantic_tag_mismatch() {
        let expr = json!({"kind":"binary", "op":"+", "left":{"kind":"symbol", "name":"p"}, "right":{"kind":"symbol", "name":"u"}}).to_string();
        let mut dims = json!({"p":{"exponents":[0,0,0,0,0,0,0], "semantic_tag":1}, "u":{"exponents":[0,0,0,0,0,0,0], "semantic_tag":3}});
        assert!(check_equation_dimensional_homogeneity_json(&expr, &dims.to_string()).unwrap_err().contains("SEMANTIC_TAG_MISMATCH"));
        dims["u"]["semantic_tag"] = json!(1);
        assert!(check_equation_dimensional_homogeneity_json(&expr, &dims.to_string()).is_ok());
        dims["u"]["exponents"][0] = json!(1);
        assert!(check_equation_dimensional_homogeneity_json(&expr, &dims.to_string()).is_err());
        assert!(check_equation_dimensional_homogeneity_json(&expr, "{}").is_err());
        assert!(check_equation_dimensional_homogeneity_json(r#"{"kind":"literal","value":1.0}"#, "{}").is_ok());
        assert!(check_equation_dimensional_homogeneity_json("bad", "{}").is_err());
    }

    #[test]
    fn builder_causal_slice_bounded_and_validated() {
        use crate::builder_uast::tests::{graph, module};
        let g = graph((0..5).map(|i| module(&format!("n{i}"))).collect(), (0..4).map(|i| json!({"from_node":format!("n{i}"), "to_node":format!("n{}", i+1), "edge_kind":"data_port", "port_id":null})).collect()).to_string();
        let result: serde_json::Value = serde_json::from_str(&extract_builder_causal_slice_json(&g, "n0", 3).unwrap()).unwrap();
        assert_eq!(result["node_ids"], json!(["n0","n1","n2","n3"]));
        let zero: serde_json::Value = serde_json::from_str(&extract_builder_causal_slice_json(&g, "n2", 0).unwrap()).unwrap();
        assert_eq!(zero["node_ids"], json!(["n2"]));
        assert!(extract_builder_causal_slice_json(&g, "missing", 3).is_err());
        assert!(extract_builder_causal_slice_json("{}", "n0", 3).is_err());
    }
}

/// A single Interaction Agent node in the net.
#[derive(Clone, Debug)]
pub struct HinAgentNode {
    pub id: u64,
    pub symbol_name: String,
    pub principal_port: String,
    pub aux_ports: Vec<String>,
    pub alive: bool,
}

/// Core Holographic Interaction Net engine.
pub struct HinEngineCore {
    arena: Vec<Option<HinAgentNode>>,
    free_list: Vec<usize>,
    /// Per-agent cursor into `aux_ports` for auxiliary connections.
    aux_cursor: HashMap<u64, usize>,
    /// Symmetric wire table: (agent_id, port_name) <-> (agent_id, port_name).
    wires: HashMap<(u64, String), (u64, String)>,
    /// Pending principal-principal active pairs.
    active_pairs: Vec<(u64, u64)>,
    /// Monotonic count of completed rewrite steps.
    reductions: usize,
}

impl HinEngineCore {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            arena: Vec::with_capacity(capacity),
            free_list: Vec::new(),
            aux_cursor: HashMap::new(),
            wires: HashMap::new(),
            active_pairs: Vec::new(),
            reductions: 0,
        }
    }

    pub fn new() -> Self {
        Self::with_capacity(1024)
    }

    /// Allocate an Interaction Agent with 1 principal port and N auxiliary
    /// ports.  Returns the arena agent id.
    pub fn spawn_agent(
        &mut self,
        symbol_name: String,
        principal_port: String,
        aux_ports: Vec<String>,
    ) -> u64 {
        let slot = if let Some(slot) = self.free_list.pop() {
            slot
        } else {
            let slot = self.arena.len();
            self.arena.push(None);
            slot
        };
        let id = slot as u64;
        self.arena[slot] = Some(HinAgentNode {
            id,
            symbol_name,
            principal_port,
            aux_ports,
            alive: true,
        });
        self.aux_cursor.insert(id, 0);
        id
    }

    fn agent(&self, id: u64) -> Result<&HinAgentNode, String> {
        let slot = id as usize;
        self.arena
            .get(slot)
            .and_then(|a| a.as_ref())
            .filter(|a| a.alive)
            .ok_or_else(|| format!("agent {} is not active", id))
    }

    fn agent_mut(&mut self, id: u64) -> Result<&mut HinAgentNode, String> {
        let slot = id as usize;
        self.arena
            .get_mut(slot)
            .and_then(|a| a.as_mut())
            .filter(|a| a.alive)
            .ok_or_else(|| format!("agent {} is not active", id))
    }

    /// Resolve the concrete port name for a connection endpoint.  Auxiliary
    /// connections consume auxiliary ports in declaration order.
    fn resolve_port(&mut self, agent_id: u64, is_principal: bool) -> Result<String, String> {
        if is_principal {
            return Ok(self.agent(agent_id)?.principal_port.clone());
        }
        let agent = self.agent(agent_id)?;
        let cursor = self
            .aux_cursor
            .get(&agent_id)
            .copied()
            .unwrap_or(0);
        let port = agent
            .aux_ports
            .get(cursor)
            .cloned()
            .ok_or_else(|| format!("agent {} has no free auxiliary port", agent_id))?;
        *self.aux_cursor.get_mut(&agent_id).unwrap() += 1;
        Ok(port)
    }

    fn wire_key(agent_id: u64, port: &str) -> (u64, String) {
        (agent_id, port.to_string())
    }

    fn insert_wire(&mut self, a: (u64, String), b: (u64, String)) -> Result<(), String> {
        if a == b {
            return Err("cannot wire a port to itself".to_string());
        }
        if self.wires.contains_key(&a) || self.wires.contains_key(&b) {
            return Err("port already wired (degree > 1)".to_string());
        }
        self.wires.insert(a.clone(), b.clone());
        self.wires.insert(b, a);
        Ok(())
    }

    fn remove_wire(&mut self, a: &(u64, String), b: &(u64, String)) {
        self.wires.remove(a);
        self.wires.remove(b);
    }

    /// Connect two ports.  Principal-principal connections register an Active
    /// Pair; auxiliary connections splice causal wires directly.
    pub fn connect_ports(
        &mut self,
        agent_a: u64,
        a_is_principal: bool,
        agent_b: u64,
        b_is_principal: bool,
    ) -> Result<(), String> {
        if agent_a == agent_b {
            return Err("cannot connect an agent to itself".to_string());
        }
        self.agent(agent_a)?;
        self.agent(agent_b)?;
        let port_a = self.resolve_port(agent_a, a_is_principal)?;
        let port_b = self.resolve_port(agent_b, b_is_principal)?;
        let key_a = Self::wire_key(agent_a, &port_a);
        let key_b = Self::wire_key(agent_b, &port_b);
        self.insert_wire(key_a, key_b)?;
        if a_is_principal && b_is_principal {
            self.active_pairs.push((agent_a, agent_b));
        }
        Ok(())
    }

    /// Reduce all pending active pairs via annihilation and commutation.
    /// Returns the number of rewrite steps performed.
    pub fn reduce_all(&mut self) -> usize {
        let pairs = std::mem::take(&mut self.active_pairs);
        let mut steps = 0usize;
        for (a, b) in pairs {
            if !self.is_alive(a) || !self.is_alive(b) {
                continue;
            }
            let pa = self
                .agent(a)
                .map(|x| x.principal_port.clone())
                .unwrap_or_default();
            let pb = self
                .agent(b)
                .map(|x| x.principal_port.clone())
                .unwrap_or_default();
            if pa == pb {
                self.annihilate(a, b);
            } else {
                self.commute(a, b);
            }
            steps += 1;
        }
        self.reductions += steps;
        steps
    }

    fn is_alive(&self, id: u64) -> bool {
        self.agent(id).is_ok()
    }

    /// Annihilation rule (`alpha >< alpha`): both agents are consumed in O(1)
    /// arena time, their auxiliary wires are spliced pairwise, and both slots
    /// are reclaimed immediately.
    fn annihilate(&mut self, a: u64, b: u64) {
        let aux_a = self.agent(a).map(|x| x.aux_ports.clone()).unwrap_or_default();
        let aux_b = self.agent(b).map(|x| x.aux_ports.clone()).unwrap_or_default();
        let principal_a = Self::wire_key(a, &self.agent(a).unwrap().principal_port.clone());
        let principal_b = Self::wire_key(b, &self.agent(b).unwrap().principal_port.clone());

        // Splice auxiliary wires pairwise before tearing down.
        let common = aux_a.len().min(aux_b.len());
        let mut splice: Vec<((u64, String), (u64, String))> = Vec::new();
        for i in 0..common {
            let ka = Self::wire_key(a, &aux_a[i]);
            let kb = Self::wire_key(b, &aux_b[i]);
            let ea = self.wires.get(&ka).cloned();
            let eb = self.wires.get(&kb).cloned();
            // Remove the old aux wires (both directions).
            if let Some(ref x) = ea {
                self.remove_wire(&ka, x);
            }
            if let Some(ref x) = eb {
                self.remove_wire(&kb, x);
            }
            if let (Some(x), Some(y)) = (ea, eb) {
                // x and y are the external endpoints; splice them together.
                if x != ka && x != kb && y != ka && y != kb {
                    splice.push((x, y));
                }
            }
        }

        // Remove every remaining wire touching either agent (principal pair and
        // any dangling auxiliaries).
        let keys: Vec<(u64, String)> = self
            .wires
            .keys()
            .filter(|(id, _)| *id == a || *id == b)
            .cloned()
            .collect();
        for key in keys {
            if let Some(other) = self.wires.get(&key).cloned() {
                self.remove_wire(&key, &other);
            }
        }

        // Reconnect spliced pairs.
        for (x, y) in splice {
            let _ = self.insert_wire(x, y);
        }

        self.free_agent(a);
        self.free_agent(b);
        let _ = (principal_a, principal_b);
    }

    /// Commutation rule (`alpha >< beta`): a mutation agent meeting a
    /// composite distributes across dependent symbol ports.  Both agents
    /// remain alive; their auxiliary consumers are cross-wired so the
    /// disturbance propagates to the dependent subgraph.
    fn commute(&mut self, a: u64, b: u64) {
        let aux_a = self.agent(a).map(|x| x.aux_ports.clone()).unwrap_or_default();
        let aux_b = self.agent(b).map(|x| x.aux_ports.clone()).unwrap_or_default();
        let common = aux_a.len().min(aux_b.len());
        let mut splice: Vec<((u64, String), (u64, String))> = Vec::new();
        for i in 0..common {
            let ka = Self::wire_key(a, &aux_a[i]);
            let kb = Self::wire_key(b, &aux_b[i]);
            let ea = self.wires.get(&ka).cloned();
            let eb = self.wires.get(&kb).cloned();
            if let Some(ref x) = ea {
                self.remove_wire(&ka, x);
            }
            if let Some(ref x) = eb {
                self.remove_wire(&kb, x);
            }
            if let (Some(x), Some(y)) = (ea, eb) {
                if x != ka && x != kb && y != ka && y != kb {
                    splice.push((x, y));
                }
            }
        }
        for (x, y) in splice {
            let _ = self.insert_wire(x, y);
        }
    }

    fn free_agent(&mut self, id: u64) {
        if let Some(slot) = self.arena.get_mut(id as usize) {
            *slot = None;
        }
        self.free_list.push(id as usize);
        self.aux_cursor.remove(&id);
    }

    /// Deterministically collect disconnected dead subgraphs (agents with no
    /// wires and no pending active pairs).  Returns the number reclaimed.
    pub fn prune_dead_subgraphs(&mut self) -> usize {
        let wired: HashSet<u64> = self
            .wires
            .keys()
            .map(|(id, _)| *id)
            .collect();
        let paired: HashSet<u64> = self.active_pairs.iter().flat_map(|(a, b)| [*a, *b]).collect();
        let dead: Vec<u64> = self
            .arena
            .iter()
            .enumerate()
            .filter(|(slot, agent)| {
                agent.as_ref().map_or(false, |a| a.alive && !wired.contains(&(a.id)) && !paired.contains(&(a.id)))
                    && !self.free_list.contains(slot)
            })
            .map(|(_, agent)| agent.as_ref().unwrap().id)
            .collect();
        let count = dead.len();
        for id in dead {
            self.free_agent(id);
        }
        count
    }

    /// Number of live agents in the arena.
    pub fn active_count(&self) -> usize {
        self.arena.iter().filter(|a| a.as_ref().map_or(false, |x| x.alive)).count()
    }

    /// Total arena slots ever allocated (live + free).
    pub fn arena_len(&self) -> usize {
        self.arena.len()
    }

    /// Total completed rewrite steps.
    pub fn total_reductions(&self) -> usize {
        self.reductions
    }

    /// Backward BFS along causal wires from a failing agent, returning the
    /// minimal root-cause symbol list within `max_hops` principal-to-auxiliary
    /// causal distance.  The fault agent itself is excluded.
    pub fn trace_fault_slice(&self, fault_agent_id: u64, max_hops: usize) -> Vec<String> {
        // Build undirected agent adjacency from the wire table.
        let mut adj: HashMap<u64, Vec<u64>> = HashMap::new();
        for ((a, _), (b, _)) in self.wires.iter() {
            if a == b {
                continue;
            }
            adj.entry(*a).or_default().push(*b);
            adj.entry(*b).or_default().push(*a);
        }

        let mut result: Vec<String> = Vec::new();
        let mut visited: HashSet<u64> = HashSet::new();
        let mut queue: VecDeque<(u64, usize)> = VecDeque::new();
        visited.insert(fault_agent_id);
        if let Some(neighbors) = adj.get(&fault_agent_id) {
            for &n in neighbors {
                queue.push_back((n, 1));
            }
        }
        while let Some((agent_id, hops)) = queue.pop_front() {
            if hops > max_hops {
                continue;
            }
            if !visited.insert(agent_id) {
                continue;
            }
            if let Ok(agent) = self.agent(agent_id) {
                result.push(agent.symbol_name.clone());
            }
            if let Some(neighbors) = adj.get(&agent_id) {
                for &n in neighbors {
                    if !visited.contains(&n) {
                        queue.push_back((n, hops + 1));
                    }
                }
            }
        }
        result
    }
}

impl Default for HinEngineCore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn annihilation_frees_arena_slots() {
        let mut net = HinEngineCore::new();
        let mut ids = Vec::new();
        for i in 0..100 {
            ids.push(net.spawn_agent(
                format!("agent_{}", i),
                "mutate".to_string(),
                vec!["consumed".to_string()],
            ));
        }
        assert_eq!(net.active_count(), 100);
        // Pair up agents principal-to-principal with matching principal ports.
        for pair in ids.chunks(2) {
            net.connect_ports(pair[0], true, pair[1], true).unwrap();
        }
        let steps = net.reduce_all();
        assert_eq!(steps, 50);
        assert_eq!(net.active_count(), 0);
        // Arena slots are reclaimed through the free list.
        assert_eq!(net.arena_len(), 100);
    }

    #[test]
    fn commutation_keeps_agents_alive() {
        let mut net = HinEngineCore::new();
        let a = net.spawn_agent("A".into(), "mutate".into(), vec!["x".into()]);
        let b = net.spawn_agent("B".into(), "verify".into(), vec!["y".into()]);
        net.connect_ports(a, true, b, true).unwrap();
        net.reduce_all();
        assert_eq!(net.active_count(), 2);
    }

    #[test]
    fn causal_fault_slice_traces_three_hops() {
        let mut net = HinEngineCore::new();
        // Chain A -> B -> C -> D(fault), each link principal-to-auxiliary.
        let a = net.spawn_agent("A".into(), "mutate".into(), vec!["a_out".into()]);
        let b = net.spawn_agent("B".into(), "mutate".into(), vec!["b_in".into()]);
        let c = net.spawn_agent("C".into(), "mutate".into(), vec!["c_in".into()]);
        let d = net.spawn_agent("D_fault".into(), "verify".into(), vec!["d_in".into()]);
        net.connect_ports(a, true, b, false).unwrap();
        net.connect_ports(b, true, c, false).unwrap();
        net.connect_ports(c, true, d, false).unwrap();

        let slice = net.trace_fault_slice(d, 3);
        assert_eq!(slice, vec!["C".to_string(), "B".to_string(), "A".to_string()]);
        // A tighter hop budget truncates the slice.
        assert_eq!(net.trace_fault_slice(d, 1), vec!["C".to_string()]);
    }
}
