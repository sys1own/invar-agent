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
