//! Geometry of Interaction (GoI) nilpotency & wavefront solver (`goi_solver.rs`).
//!
//! Encodes task dependencies and resource claims as Girard's Geometry of
//! Interaction: each `TaskWavefrontNode` exposes typed input ports (read
//! dependencies) and output ports (mutated symbols), and dependency edges wire
//! output ports to input ports.  The global task network must be nilpotent —
//! `(Pi_22 * sigma)^k == 0` — for the parallel schedule to be deadlock-free
//! and strongly normalizing.  Nilpotency is decided by depth-bounded
//! token-particle propagation along port polarities before any worker is
//! spawned.  Wavefronts are then partitioned under strict Bernstein
//! disjointness over read/write sets.

use std::collections::{HashMap, HashSet, VecDeque};

/// A single schedulable task node in the GoI execution matrix.
#[derive(Clone, Debug)]
pub struct GoiNode {
    pub node_id: String,
    pub read_set: HashSet<String>,
    pub write_set: HashSet<String>,
    pub input_ports: Vec<String>,
    pub output_ports: Vec<String>,
}

/// A wiring edge connecting an output port of `src_node` to an input port of
/// `dst_node`.
#[derive(Clone, Debug)]
pub struct GoiEdge {
    pub src_node: String,
    pub src_port: String,
    pub dst_node: String,
    pub dst_port: String,
}

/// Core GoI solver state.
#[derive(Default)]
pub struct GoiSolverCore {
    nodes: HashMap<String, GoiNode>,
    edges: Vec<GoiEdge>,
}

/// Maximum token-propagation steps before declaring non-nilpotency.  Generous
/// enough that any finite acyclic wiring terminates far below the bound.
const TOKEN_STEP_BOUND: usize = 1_000_000;

impl GoiSolverCore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn add_node(&mut self, node: GoiNode) -> Result<(), String> {
        if node.node_id.is_empty() {
            return Err("node_id must be non-empty".to_string());
        }
        if self.nodes.contains_key(&node.node_id) {
            return Err(format!("duplicate node_id '{}'", node.node_id));
        }
        let id = node.node_id.clone();
        self.nodes.insert(id, node);
        Ok(())
    }

    pub fn add_edge(&mut self, edge: GoiEdge) -> Result<(), String> {
        let src = self
            .nodes
            .get(&edge.src_node)
            .ok_or_else(|| format!("unknown src_node '{}'", edge.src_node))?;
        if !src.output_ports.contains(&edge.src_port) {
            return Err(format!(
                "node '{}' has no output port '{}'",
                edge.src_node, edge.src_port
            ));
        }
        let dst = self
            .nodes
            .get(&edge.dst_node)
            .ok_or_else(|| format!("unknown dst_node '{}'", edge.dst_node))?;
        if !dst.input_ports.contains(&edge.dst_port) {
            return Err(format!(
                "node '{}' has no input port '{}'",
                edge.dst_node, edge.dst_port
            ));
        }
        self.edges.push(edge);
        Ok(())
    }

    /// Girard nilpotency check: `(Pi_22 * sigma)^k == 0`.
    ///
    /// Implemented as depth-bounded token-particle propagation over the port
    /// wiring graph.  Tokens originate at every boundary input port (and at
    /// the outputs of source nodes with no input ports), propagate through
    /// node pass-throughs and dependency wirings, and are absorbed at sinks.
    /// A token that revisits a vertex witnesses a cyclic feedback loop, and
    /// exceeding the step bound witnesses non-termination; both imply the
    /// interaction operator is NOT nilpotent.
    pub fn verify_nilpotency(&self) -> bool {
        if self.nodes.is_empty() {
            return true;
        }

        // Vertices are (node_id, port_name).  Edges are wirings plus node
        // pass-throughs from input ports to output ports.
        let mut adj: HashMap<(String, String), Vec<(String, String)>> = HashMap::new();
        let mut ensure = |v: &(String, String)| {
            adj.entry(v.clone()).or_default();
        };

        for node in self.nodes.values() {
            for port in node.input_ports.iter().chain(node.output_ports.iter()) {
                ensure(&(node.node_id.clone(), port.clone()));
            }
        }
        for edge in &self.edges {
            let from = (edge.src_node.clone(), edge.src_port.clone());
            let to = (edge.dst_node.clone(), edge.dst_port.clone());
            adj.entry(from).or_default().push(to);
        }
        // Node pass-through: a token arriving on an input port may exit via
        // any output port of the same node.
        for node in self.nodes.values() {
            for ip in &node.input_ports {
                for op in &node.output_ports {
                    let from = (node.node_id.clone(), ip.clone());
                    let to = (node.node_id.clone(), op.clone());
                    adj.entry(from).or_default().push(to);
                }
            }
        }

        // Token origins: every port vertex is seeded.  In a nilpotent (DAG)
        // wiring every token is absorbed at a sink; seeding all vertices also
        // catches feedback cycles that are disconnected from any boundary.
        let mut origins: Vec<(String, String)> = Vec::new();
        for node in self.nodes.values() {
            for port in node.input_ports.iter().chain(node.output_ports.iter()) {
                origins.push((node.node_id.clone(), port.clone()));
            }
        }

        // Depth-bounded token propagation with per-token visited sets.
        // (vertex, token_id) dedup keeps the search polynomial on DAGs.
        let mut steps: usize = 0;
        let mut queue: VecDeque<((String, String), usize, HashSet<(String, String)>)> =
            VecDeque::new();
        for (i, origin) in origins.into_iter().enumerate() {
            let mut visited = HashSet::new();
            visited.insert(origin.clone());
            queue.push_back((origin, i, visited));
        }

        while let Some((vertex, token_id, visited)) = queue.pop_front() {
            steps += 1;
            if steps > TOKEN_STEP_BOUND {
                return false;
            }
            let targets = match adj.get(&vertex) {
                Some(t) => t.clone(),
                None => continue, // absorbed at a sink
            };
            for target in targets {
                if visited.contains(&target) {
                    // Token returned to a vertex on its own path: cyclic
                    // feedback => the nilpotency exponent is unbounded.
                    return false;
                }
                let mut next_visited = visited.clone();
                next_visited.insert(target.clone());
                let _ = token_id;
                queue.push_back((target, token_id, next_visited));
            }
        }
        true
    }

    /// Topological wavefront partitioning enforcing Bernstein disjointness.
    ///
    /// Nodes are first leveled by dependency depth (Kahn).  Each node is then
    /// placed in the earliest wavefront generation at or after its dependency
    /// level that does not conflict with any node already placed there, where
    /// conflict means any non-empty intersection among
    /// `W(u) ∩ W(v)`, `W(u) ∩ R(v)`, or `R(u) ∩ W(v)`.
    pub fn compute_disjoint_wavefronts(&self) -> Result<Vec<Vec<String>>, String> {
        if self.nodes.is_empty() {
            return Ok(Vec::new());
        }

        // Dependency graph: edge src -> dst means dst depends on src.
        let mut deps: HashMap<String, HashSet<String>> = HashMap::new();
        let mut dependents: HashMap<String, Vec<String>> = HashMap::new();
        for node in self.nodes.keys() {
            deps.insert(node.clone(), HashSet::new());
            dependents.insert(node.clone(), Vec::new());
        }
        for edge in &self.edges {
            deps.entry(edge.dst_node.clone())
                .or_default()
                .insert(edge.src_node.clone());
            dependents
                .entry(edge.src_node.clone())
                .or_default()
                .push(edge.dst_node.clone());
        }

        // Kahn topological levels.
        let mut indegree: HashMap<String, usize> = HashMap::new();
        for (node, ds) in &deps {
            indegree.insert(node.clone(), ds.len());
        }
        let mut levels: HashMap<String, usize> = HashMap::new();
        let mut queue: VecDeque<String> = indegree
            .iter()
            .filter(|(_, d)| **d == 0)
            .map(|(n, _)| n.clone())
            .collect();
        let mut processed = 0usize;
        while let Some(node) = queue.pop_front() {
            processed += 1;
            let level = deps[&node]
                .iter()
                .map(|d| levels.get(d).copied().unwrap_or(0) + 1)
                .max()
                .unwrap_or(0);
            levels.insert(node.clone(), level);
            for dep in dependents.get(&node).cloned().unwrap_or_default() {
                let d = indegree.get_mut(&dep).unwrap();
                *d -= 1;
                if *d == 0 {
                    queue.push_back(dep);
                }
            }
        }
        if processed != self.nodes.len() {
            return Err(
                "GoI task network contains a cyclic dependency; cannot schedule wavefronts"
                    .to_string(),
            );
        }

        // Bernstein-disjoint packing: process nodes in (level, id) order and
        // assign each to the earliest non-conflicting wavefront generation.
        let mut order: Vec<String> = self.nodes.keys().cloned().collect();
        order.sort_by(|a, b| {
            levels[a.as_str()]
                .cmp(&levels[b.as_str()])
                .then_with(|| a.cmp(b))
        });

        let mut wavefronts: Vec<Vec<String>> = Vec::new();
        let mut assigned: HashMap<String, usize> = HashMap::new();
        for node_id in order {
            let node = &self.nodes[&node_id];
            let min_gen = levels[&node_id];
            let mut gen = min_gen;
            loop {
                if gen >= wavefronts.len() {
                    wavefronts.resize_with(gen + 1, Vec::new);
                }
                let mut conflict = false;
                for other_id in &wavefronts[gen] {
                    let other = &self.nodes[other_id];
                    if bernstein_conflict(node, other) {
                        conflict = true;
                        break;
                    }
                }
                if !conflict {
                    break;
                }
                gen += 1;
            }
            wavefronts[gen].push(node_id.clone());
            assigned.insert(node_id, gen);
        }

        for wf in wavefronts.iter_mut() {
            wf.sort();
        }
        // Drop trailing empty generations (should not occur, but stay total).
        while wavefronts.last().map_or(false, |w| w.is_empty()) {
            wavefronts.pop();
        }
        Ok(wavefronts)
    }
}

/// Bernstein disjointness predicate: two concurrent tasks conflict iff any of
/// `W(u) ∩ W(v)`, `W(u) ∩ R(v)`, `R(u) ∩ W(v)` is non-empty.
fn bernstein_conflict(a: &GoiNode, b: &GoiNode) -> bool {
    intersects(&a.write_set, &b.write_set)
        || intersects(&a.write_set, &b.read_set)
        || intersects(&a.read_set, &b.write_set)
}

fn intersects(a: &HashSet<String>, b: &HashSet<String>) -> bool {
    a.iter().any(|x| b.contains(x))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: &str, reads: &[&str], writes: &[&str]) -> GoiNode {
        GoiNode {
            node_id: id.to_string(),
            read_set: reads.iter().map(|s| s.to_string()).collect(),
            write_set: writes.iter().map(|s| s.to_string()).collect(),
            input_ports: vec!["in".to_string()],
            output_ports: vec!["out".to_string()],
        }
    }

    #[test]
    fn acyclic_graph_is_nilpotent_and_disjoint_wavefronts() {
        let mut solver = GoiSolverCore::new();
        solver.add_node(node("A", &["base.py"], &["a.py"])).unwrap();
        solver.add_node(node("B", &["base.py"], &["b.py"])).unwrap();
        solver.add_node(node("C", &["a.py"], &["c.py"])).unwrap();
        solver.add_node(node("D", &["b.py"], &["d.py"])).unwrap();
        solver.add_edge(GoiEdge {
            src_node: "A".into(),
            src_port: "out".into(),
            dst_node: "C".into(),
            dst_port: "in".into(),
        })
        .unwrap();
        solver.add_edge(GoiEdge {
            src_node: "B".into(),
            src_port: "out".into(),
            dst_node: "D".into(),
            dst_port: "in".into(),
        })
        .unwrap();
        assert!(solver.verify_nilpotency());
        let wfs = solver.compute_disjoint_wavefronts().unwrap();
        assert_eq!(wfs.len(), 2);
        assert_eq!(wfs[0].len(), 2); // A and B disjoint -> parallel
        assert_eq!(wfs[1].len(), 2); // C and D disjoint -> parallel
    }

    #[test]
    fn cyclic_port_dependency_is_not_nilpotent() {
        let mut solver = GoiSolverCore::new();
        solver.add_node(node("A", &[], &["x.py"])).unwrap();
        solver.add_node(node("B", &["x.py"], &["y.py"])).unwrap();
        solver.add_node(node("C", &["y.py"], &["x.py"])).unwrap();
        solver
            .add_edge(GoiEdge {
                src_node: "A".into(),
                src_port: "out".into(),
                dst_node: "B".into(),
                dst_port: "in".into(),
            })
            .unwrap();
        solver
            .add_edge(GoiEdge {
                src_node: "B".into(),
                src_port: "out".into(),
                dst_node: "C".into(),
                dst_port: "in".into(),
            })
            .unwrap();
        solver
            .add_edge(GoiEdge {
                src_node: "C".into(),
                src_port: "out".into(),
                dst_node: "A".into(),
                dst_port: "in".into(),
            })
            .unwrap();
        assert!(!solver.verify_nilpotency());
        assert!(solver.compute_disjoint_wavefronts().is_err());
    }

    #[test]
    fn overlapping_write_sets_are_serialized() {
        let mut solver = GoiSolverCore::new();
        solver.add_node(node("A", &[], &["shared.py"])).unwrap();
        solver.add_node(node("B", &[], &["shared.py"])).unwrap();
        let wfs = solver.compute_disjoint_wavefronts().unwrap();
        assert_eq!(wfs.len(), 2);
        assert_eq!(wfs[0].len(), 1);
        assert_eq!(wfs[1].len(), 1);
    }

    #[test]
    fn unknown_nodes_and_ports_are_rejected() {
        let mut solver = GoiSolverCore::new();
        solver.add_node(node("A", &[], &["a.py"])).unwrap();
        assert!(solver
            .add_edge(GoiEdge {
                src_node: "A".into(),
                src_port: "out".into(),
                dst_node: "MISSING".into(),
                dst_port: "in".into(),
            })
            .is_err());
        assert!(solver
            .add_edge(GoiEdge {
                src_node: "A".into(),
                src_port: "nope".into(),
                dst_node: "A".into(),
                dst_port: "in".into(),
            })
            .is_err());
    }
}
