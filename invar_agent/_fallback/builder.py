"""JSON-compatible Builder operations when the Rust extension is unavailable."""
from __future__ import annotations

import ast
import copy
import io
import json
import math
import re
import struct
import time
import tokenize
from collections import deque
from functools import wraps

import blake3


def _json_contract(function):
    @wraps(function)
    def checked(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise ValueError(f"Invalid JSON contract: {exc}") from exc
    return checked


def _loads(text):
    return json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def compute_his_digest_v2(domain_context, canonical_payload_json):
    return blake3.blake3(canonical_payload_json.encode(), derive_key_context=domain_context).hexdigest()


def compute_fault_fingerprint(fields_json):
    fields = _loads(fields_json)
    if not isinstance(fields, list) or len(fields) != 6 or any(not isinstance(f, str) for f in fields):
        raise ValueError("Expected exactly six strings")
    data = b"".join(struct.pack("<Q", len(f.encode())) + f.encode() for f in fields)
    return blake3.blake3(data, derive_key_context="invar.builder.fault_fingerprint.v1").hexdigest()


def _digest(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value)


def _path(path):
    if (not isinstance(path, str) or not path or any(c in path for c in "\\:\0")
            or any(p in {"", ".", ".."} for p in path.split("/"))):
        raise ValueError("BuilderUAST: unsafe target path")


@_json_contract
def validate_builder_uast(uast_json):
    graph = _loads(uast_json)
    required = {"schema_version", "project_id", "canonical_ir_digest", "active_profiles", "nodes", "edges"}
    if not isinstance(graph, dict) or set(graph) != required or graph["schema_version"] != "builder_uast.v1" or not _digest(graph["canonical_ir_digest"]) or not graph["project_id"].strip():
        raise ValueError("BuilderUAST: invalid project graph")
    if set(graph["active_profiles"]) - {"hybrid_compute_library", "interactive_workbench", "executable_paper_audit", "calibration_inverse_design"}:
        raise ValueError("BuilderUAST: unknown profile")
    fields = {
        "kernel": {"node_id", "kernel_name", "research_compute_uast_json", "csel_expressions", "manifold_projectors", "export_adjoint_jacobian", "output_port_id", "his_digest"},
        "interface_port": {"node_id", "port_name", "inputs", "outputs", "z3_preconditions", "z3_postconditions", "his_digest"},
        "deterministic_module": {"node_id", "module_role", "emitter_id", "target_relpaths", "required_port_ids", "provided_port_ids", "emitter_config", "his_digest"},
        "llm_extension_slot": {"node_id", "slot_kind", "target_relpath", "intent_summary", "frozen_port_id", "frozen_his_digest", "fallback_emitter_id", "max_tier2_retries"},
    }
    nodes = {}
    for node in graph["nodes"]:
        kind = node.get("kind")
        if kind == "llm_extension_slot":
            node.setdefault("max_tier2_retries", 2)
        if kind not in fields or set(node) != fields[kind] | {"kind"} or not node["node_id"].strip() or node["node_id"] in nodes:
            raise ValueError("BuilderUAST: invalid or duplicate node")
        nodes[node["node_id"]] = node

    def port(identifier):
        if identifier not in nodes or nodes[identifier]["kind"] != "interface_port":
            raise ValueError("BuilderUAST: missing interface port")
        return nodes[identifier]

    paths = set()
    for node in nodes.values():
        kind = node["kind"]
        digest = node.get("his_digest", node.get("frozen_his_digest"))
        if not _digest(digest):
            raise ValueError("BuilderUAST: invalid HIS")
        targets = []
        if kind == "interface_port":
            if not node["port_name"].strip(): raise ValueError("Empty port name")
            for direction in ("inputs", "outputs"):
                names = set()
                for field in node[direction]:
                    if set(field) - {"name", "scalar_type", "dimension", "bounds", "mell_linearity"} or field["name"] in names or not field["name"].strip():
                        raise ValueError("BuilderUAST: invalid typed field")
                    names.add(field["name"])
                    if field["scalar_type"] not in {"f64", "i64", "bool", "vec_f64", "str", "table"} or field["mell_linearity"] not in {"Linear", "Bang", "Tensor"}:
                        raise ValueError("BuilderUAST: invalid port type")
                    dimension = field["dimension"]
                    dimension.setdefault("semantic_tag", 0)
                    if set(dimension) != {"exponents", "semantic_tag", "canonical_unit"} or len(dimension["exponents"]) != 7 or not dimension["canonical_unit"].strip() or any(not isinstance(x, int) or not -128 <= x <= 127 for x in dimension["exponents"]):
                        raise ValueError("BuilderUAST: invalid dimension")
                    field.setdefault("bounds", None)
                    if field["bounds"]:
                        bounds = field["bounds"]
                        bounds.setdefault("lower", None)
                        bounds.setdefault("upper", None)
                        bounds.setdefault("lower_inclusive", True)
                        bounds.setdefault("upper_inclusive", True)
                        lo, hi = bounds["lower"], bounds["upper"]
                        if any(x is not None and not math.isfinite(x) for x in (lo, hi)) or (lo is not None and hi is not None and (lo > hi or lo == hi and not (bounds["lower_inclusive"] and bounds["upper_inclusive"]))):
                            raise ValueError("BuilderUAST: invalid interval")
        elif kind == "kernel":
            if not node["kernel_name"].strip(): raise ValueError("Empty kernel name")
            _loads(node["research_compute_uast_json"])
            port(node["output_port_id"])
        elif kind == "deterministic_module":
            if node["module_role"] not in {"package_metadata", "cpp_cabi_bridge", "python_fallback_kernel", "python_sdk", "cli_runner", "sweep_pareto_runner", "data_serializer", "fastapi_workbench", "calibration_engine", "audit_report", "pytest_z3_suite", "receipt_builder"} or not node["emitter_id"].strip():
                raise ValueError("BuilderUAST: invalid module")
            for identifier in node["required_port_ids"] + node["provided_port_ids"]: port(identifier)
            targets = node["target_relpaths"]
        else:
            fallbacks = {"bespoke_data_adapter": "fb_csv_json_adapter", "domain_visualization": "fb_svg_trajectory_viz", "workflow_glue": "fb_workflow_glue", "narrative_documentation": "fb_markdown_narrative_doc"}
            if (fallbacks.get(node["slot_kind"]) != node["fallback_emitter_id"] or not 0 <= node["max_tier2_retries"] <= 2
                    or port(node["frozen_port_id"])["his_digest"] != digest or not node["intent_summary"].strip()):
                raise ValueError("BuilderUAST: invalid slot contract")
            targets = [node["target_relpath"]]
        for target in targets:
            _path(target)
            if target in paths: raise ValueError("BuilderUAST: duplicate target")
            paths.add(target)
    edges = set()
    for edge in graph["edges"]:
        edge.setdefault("port_id", None)
        if set(edge) != {"from_node", "to_node", "edge_kind", "port_id"} or edge["from_node"] not in nodes or edge["to_node"] not in nodes or edge["edge_kind"] not in {"data_port", "abi_link", "verification_dep", "receipt_dep"}:
            raise ValueError("BuilderUAST: invalid dependency")
        key = _json(edge)
        if key in edges: raise ValueError("BuilderUAST: duplicate dependency")
        edges.add(key)
        if edge["port_id"] is not None: port(edge["port_id"])
    return _json(graph)


def _analyze(matrix, order):
    n = len(order)
    adjacency = [[j for j in range(n) if abs(matrix[j, i]) > 1e-9] for i in range(n)]
    counter, stack, active, indices, low, witness = 0, [], set(), {}, {}, set()

    def visit(v):
        nonlocal counter
        indices[v] = low[v] = counter
        counter += 1
        stack.append(v)
        active.add(v)
        for w in adjacency[v]:
            if w not in indices:
                visit(w)
                low[v] = min(low[v], low[w])
            elif w in active:
                low[v] = min(low[v], indices[w])
        if low[v] == indices[v]:
            component = []
            while True:
                w = stack.pop()
                active.remove(w)
                component.append(w)
                if w == v: break
            if len(component) > 1 or v in adjacency[v]: witness.update(component)
    for v in range(n):
        if v not in indices: visit(v)
    degree = [sum(v in row for row in adjacency) for v in range(n)]
    ready = [i for i in range(n) if not degree[i]]
    waves = []
    while ready:
        waves.append([order[i] for i in ready])
        following = set()
        for v in ready:
            for w in adjacency[v]:
                degree[w] -= 1
                if not degree[w]: following.add(w)
        ready = sorted(following)
    return waves, sorted(order[i] for i in witness)


def _reject(matrix, order, witness, rho, determinant):
    raise ValueError(_json(dict(error_code="GOI_IRREDUCIBLE_CYCLE", tier="CRITICAL_NON_RETRYABLE", action="ABORT_BUILD_CRITICAL", llm_retries=0,
        spectral_radius=float(rho), capacitance_det=float(determinant), cycle_witness=witness,
        cycle_edges=[dict(from_node=a, to_node=b) for i, a in enumerate(order) for j, b in enumerate(order) if a in witness and b in witness and abs(matrix[j, i]) > 1e-9])))


@_json_contract
def schedule_builder_wavefronts(uast_json, feedback_weights_json=None):
    import numpy as np

    graph = _loads(validate_builder_uast(uast_json))
    order = sorted(n["node_id"] for n in graph["nodes"])
    index = {name: i for i, name in enumerate(order)}
    n = len(order)
    m, u = np.zeros((n, n)), np.eye(n)
    for edge in graph["edges"]: m[index[edge["to_node"]], index[edge["from_node"]]] = 1
    if feedback_weights_json is not None:
        weights = _loads(feedback_weights_json)
        if isinstance(weights, dict):
            for name, weight in weights.items():
                if name not in index: raise ValueError("Unknown feedback node")
                u[index[name], index[name]] = weight
        elif isinstance(weights, list) and len(weights) == n and all(isinstance(w, (float, int)) for w in weights):
            u = np.diag(weights)
        else:
            u = np.array(weights, dtype=float)
            if u.shape != (n, n): raise ValueError("Feedback matrix shape mismatch")
    if not np.all(np.isfinite(u)) or np.any(u < 0): raise ValueError("Invalid feedback weights")
    um = u @ m
    waves, structural = _analyze(m, order)
    _, witness = _analyze(um, order)
    rho = max(abs(np.linalg.eigvals(um)), default=0) if witness else 0
    a = np.eye(n) - um
    determinant = np.linalg.det(a)
    if rho >= 1 or abs(determinant) < 1e-9 or witness or structural:
        _reject(um, order, witness or structural, rho, determinant)
    return _json(dict(is_nilpotent=True, spectral_radius=float(rho), wavefronts=waves,
        resolvent_flat=np.linalg.inv(a).ravel().tolist(), node_order=order, cycle_witness=[]))


@_json_contract
def compute_builder_smw_delta(resolvent_json, node_order_json, delta_edges_json, changed_his_nodes_json):
    import numpy as np

    order, flat, deltas, changed = map(_loads, (node_order_json, resolvent_json, delta_edges_json, changed_his_nodes_json))
    n, k = len(order), len(deltas)
    index = {name: i for i, name in enumerate(order)}
    if len(index) != n or any(not name for name in order) or len(flat) != n * n or not np.all(np.isfinite(flat)):
        raise ValueError("Invalid node order or resolvent shape")
    if any(name not in index for name in changed): raise ValueError("Unknown changed HIS node")
    r = np.array(flat, dtype=float).reshape(n, n)
    try:
        m = np.eye(n) - np.linalg.inv(r)
    except np.linalg.LinAlgError as exc:
        raise ValueError("GoI input resolvent is singular") from exc
    _, old_cycle = _analyze(m, order)
    if old_cycle: _reject(m, order, old_cycle, max(abs(np.linalg.eigvals(m)), default=0), 0)
    p, q = np.zeros((n, k)), np.zeros((n, k))
    roots = {index[name] for name in changed}
    for j, edge in enumerate(deltas):
        if set(edge) - {"from_node", "to_node", "weight", "delta_weight"}: raise ValueError("Unknown delta fields")
        if edge["from_node"] not in index or edge["to_node"] not in index: raise ValueError("Unknown delta node")
        weight = edge.get("weight", edge.get("delta_weight", 1.0))
        if not math.isfinite(weight): raise ValueError("Nonfinite delta weight")
        a, b = index[edge["from_node"]], index[edge["to_node"]]
        p[b, j], q[a, j] = -weight, 1
        m[b, a] += weight
        if weight: roots.add(b)
    if np.any(m < -1e-9): raise ValueError("Delta removes a nonexistent dependency")
    cap = np.eye(k) + q.T @ r @ p
    determinant = np.linalg.det(cap)
    _, witness = _analyze(m, order)
    rho = max(abs(np.linalg.eigvals(m)), default=0) if witness else 0
    if abs(determinant) < 1e-9 or rho >= 1 or witness: _reject(m, order, witness, rho, determinant)
    updated = r - (r @ p) @ np.linalg.inv(cap) @ (q.T @ r) if k else r
    return _json(dict(is_valid_dag=True, capacitance_det=float(determinant), updated_resolvent_flat=updated.ravel().tolist(),
        affected_node_ids=[name for i, name in enumerate(order) if any(abs(updated[i, root]) > 1e-9 for root in roots)]))


@_json_contract
def verify_equation_dimensions_hin(csel_expr_json, symbol_dims_json):
    expression, dimensions = _loads(csel_expr_json), _loads(symbol_dims_json)
    if isinstance(dimensions, list):
        if len({q["symbol"] for q in dimensions}) != len(dimensions) or any(q.get("dimension_unspecified") for q in dimensions):
            raise ValueError("HIN_INVALID_SYMBOL_DIMENSIONS")
        dimensions = {q["symbol"]: dict(exponents=q["si_dimension"], semantic_tag=q.get("semantic_tag", 0), interval=q["interval"]) for q in dimensions}
    for name, dimension in dimensions.items():
        powers = dimension["exponents"]
        interval = dimension.get("interval", [-1e300, 1e300])
        if (set(dimension) - {"exponents", "semantic_tag", "interval", "canonical_unit"} or not 0 <= dimension.get("semantic_tag", 0) <= 255
                or not name.strip() or len(powers) != 7 or any(not isinstance(x, int) or not -128 <= x <= 127 for x in powers)
                or len(interval) != 2 or any(not math.isfinite(x) for x in interval) or interval[0] > interval[1]):
            raise ValueError("HIN_INVALID_SYMBOL_DIMENSIONS")
    zero = ((0,) * 7, 0)

    def same(a, b):
        if a[1] and b[1] and a[1] != b[1]: raise ValueError("SEMANTIC_TAG_MISMATCH")
        if a[0] != b[0]: raise ValueError("HIN_SI_DIM_MISMATCH")
        return a[0], max(a[1], b[1])

    def dimensionless(a):
        if a[0] != zero[0]: raise ValueError("HIN_SI_DIM_MISMATCH")

    def product(a, b, sign):
        powers = tuple(x + sign * y for x, y in zip(a[0], b[0]))
        if any(not -128 <= v <= 127 for v in powers): raise ValueError("HIN_DIMENSION_OVERFLOW")
        tag = 0 if sign == -1 and a[1] == b[1] else b[1] if not a[1] else a[1] if not b[1] or a[1] == b[1] else 0
        return powers, tag

    def visit(expr):
        kind = expr["kind"]
        if kind == "literal":
            if not math.isfinite(expr["value"]): raise ValueError("HIN_NONFINITE_LITERAL")
            return zero
        if kind == "symbol":
            name = expr["name"]
            if name == "t": return (0, 0, 1, 0, 0, 0, 0), 0
            if name in {"pi", "e"}: return zero
            if name not in dimensions: raise ValueError(f"HIN_UNBOUND_SYMBOL: {name}")
            return tuple(dimensions[name]["exponents"]), dimensions[name].get("semantic_tag", 0)
        if kind == "unary":
            if expr["op"] not in {"+", "-", "neg"}: raise ValueError("HIN_UNKNOWN_OPERATOR")
            return visit(expr["arg"])
        if kind == "binary":
            a, b = visit(expr["left"]), visit(expr["right"])
            op = expr["op"]
            if op in {"+", "-"}:
                if expr["left"].get("kind") == "literal" and expr["left"].get("value") == 0: return b
                if expr["right"].get("kind") == "literal" and expr["right"].get("value") == 0: return a
                return same(a, b)
            if op in {"*", "/"}: return product(a, b, 1 if op == "*" else -1)
            if op in {"^", "**"}:
                dimensionless(b)
                exponent = expr["right"].get("value")
                if expr["right"]["kind"] == "literal" and float(exponent).is_integer() and abs(exponent) <= 12:
                    powers = tuple(int(x * exponent) for x in a[0])
                    if any(not -128 <= x <= 127 for x in powers): raise ValueError("HIN_DIMENSION_OVERFLOW")
                    return powers, a[1] if exponent == 1 else 0
                dimensionless(a)
                return zero
            raise ValueError("HIN_UNKNOWN_OPERATOR")
        if kind != "call": raise ValueError("Unknown CSEL kind")
        name, args = expr["func"], expr["args"]
        arities = {**dict.fromkeys("safe_div min max kl_term gammainc_reg".split(), 2),
            **dict.fromkeys("smooth_clamp where hill_eq michaelis_menten betainc_reg".split(), 3),
            **dict.fromkeys("safe_sqrt sqrt safe_exp exp safe_log log log1p expm1 safe_sin sin safe_cos cos safe_tanh tanh safe_acos acos safe_asin asin abs fabs norm_cdf norm_ppf logistic_sigmoid logit shannon_term".split(), 1)}
        if name not in arities or len(args) != arities[name]: raise ValueError("HIN_FUNCTION_ARITY")
        dims = [visit(arg) for arg in args]
        a = dims[0]
        if name == "safe_div": return product(a, dims[1], -1)
        if name in {"sqrt", "safe_sqrt"}:
            if any(x % 2 for x in a[0]): raise ValueError("HIN_FRACTIONAL_DIMENSION")
            return tuple(x // 2 for x in a[0]), 0
        if name in {"abs", "fabs"}: return a
        if name in {"min", "max"}: return same(a, dims[1])
        if name == "smooth_clamp": same(a, dims[1]); return same(a, dims[2])
        if name == "where": dimensionless(a); return same(dims[1], dims[2])
        if name == "michaelis_menten": same(a, dims[2]); return dims[1]
        if name == "hill_eq": same(a, dims[1]); dimensionless(dims[2]); return zero[0], 1
        for dim in dims: dimensionless(dim)
        tags = {**dict.fromkeys(("norm_cdf", "logistic_sigmoid", "gammainc_reg", "betainc_reg"), 1),
                **dict.fromkeys(("norm_ppf", "logit"), 4), **dict.fromkeys(("kl_term", "shannon_term"), 2)}
        return zero[0], tags.get(name, 0)
    powers, tag = visit(expression)
    bound = [-float.fromhex("0x1.fffffffffffffp+1023"), float.fromhex("0x1.fffffffffffffp+1023")]
    if expression.get("kind") == "literal":
        bound = [expression["value"], expression["value"]]
    elif expression.get("kind") == "symbol" and expression["name"] in dimensions:
        bound = dimensions[expression["name"]].get("interval", [-1e300, 1e300])
    return _json(dict(equation_name="builder_equation", valid=True, output_dimensions=powers, output_semantic_tag=tag,
        output_bound=bound, promotion_agents=[], dereliction_agents=[], unbound_symbols=[], dimensional_warnings=[], error_message=None))


@_json_contract
def extract_hin_causal_slice(uast_json, fault_node_id, max_hops=3):
    graph = _loads(validate_builder_uast(uast_json))
    if max_hops < 0 or fault_node_id not in {n["node_id"] for n in graph["nodes"]}: raise ValueError("HIN_UNKNOWN_FAULT_NODE")
    adjacency = {}
    for edge in graph["edges"]:
        a, b = edge["from_node"], edge["to_node"]
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
    queue, seen = deque([(fault_node_id, 0)]), set()
    while queue:
        node, hops = queue.popleft()
        if node in seen: continue
        seen.add(node)
        if hops < max_hops: queue.extend((other, hops + 1) for other in adjacency.get(node, []))
    return _json(dict(fault_node_id=fault_node_id, max_hops=max_hops, node_ids=sorted(seen), nodes=[n for n in graph["nodes"] if n["node_id"] in seen],
        edges=[e for e in graph["edges"] if e["from_node"] in seen and e["to_node"] in seen]))


def heal_builder_tier0(source_code, module_kind="llm_extension_slot"):
    if module_kind not in {"kernel", "deterministic_module", "llm_extension_slot", "module"}: raise ValueError("Unknown builder module kind")
    started, code, notes = time.perf_counter(), source_code, []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        statements, current = [], []
        try:
            for token in tokenize.generate_tokens(io.StringIO(code).readline):
                if token.type in {tokenize.INDENT, tokenize.DEDENT, tokenize.COMMENT, tokenize.NL}:
                    continue
                if token.type in {tokenize.NEWLINE, tokenize.ENDMARKER}:
                    if current:
                        statements.append(current)
                        current = []
                    continue
                current.append(token)
        except (tokenize.TokenError, IndentationError):
            pass
        if current:
            statements.append(current)
        lines = code.splitlines(keepends=True)
        starts = [0]
        for line in lines: starts.append(starts[-1] + len(line))
        edits = []
        for index, statement in enumerate(statements):
            first, last = statement[0], statement[-1]
            if first.type != tokenize.NAME or first.string not in {"async", "def", "class", "if", "elif", "else", "for", "while", "try", "except", "finally", "with", "match", "case"}:
                continue
            depth, colon = 0, False
            for token in statement:
                if token.string in {"(", "[", "{"}: depth += 1
                elif token.string in {")", "]", "}"}: depth -= 1
                elif token.string == ":" and not depth: colon = True
            if depth or colon and last.string != ":":
                continue
            prefix = lines[first.start[0] - 1][:first.start[1]]
            empty = index + 1 == len(statements) or statements[index + 1][0].start[1] <= len(prefix)
            end = starts[last.end[0] - 1] + last.end[1]
            if empty:
                newline = code.find("\n", end)
                position = len(code) if newline == -1 else newline + 1
                unit = "\t" if "\t" in prefix else "    "
                separator = "\n" if position == len(code) and not code.endswith("\n") else ""
                edits.append((position, f"{separator}{prefix}{unit}pass\n"))
            if not colon: edits.append((end, ":"))
        for position, text in sorted(edits, key=lambda edit: edit[0], reverse=True):
            code = code[:position] + text + code[position:]
        try: tree = ast.parse(code)
        except SyntaxError as exc: raise ValueError(f"PY_SYNTAX_ERROR: {exc}") from exc
        notes.append("syntax_empty_block_repair")
    if module_kind == "kernel":
        class Algebra(ast.NodeTransformer):
            def visit_BinOp(self, node):
                self.generic_visit(node)
                a, b = node.left, node.right
                av = a.value if isinstance(a, ast.Constant) else None
                bv = b.value if isinstance(b, ast.Constant) else None
                if isinstance(node.op, (ast.Add, ast.Sub)) and bv == 0: return a
                if isinstance(node.op, ast.Add) and av == 0: return b
                if isinstance(node.op, ast.Mult):
                    if av == 0 or bv == 0: return ast.Constant(0.0)
                    if av == 1: return b
                    if bv == 1: return a
                if isinstance(node.op, ast.Sub) and ast.dump(a) == ast.dump(b): return ast.Constant(0.0)
                return node

            def visit_UnaryOp(self, node):
                self.generic_visit(node)
                if isinstance(node.op, ast.USub) and isinstance(node.operand, ast.UnaryOp) and isinstance(node.operand.op, ast.USub):
                    return node.operand.operand
                return node

            def visit_Call(self, node):
                self.generic_visit(node)
                if isinstance(node.func, ast.Name) and node.func.id == "kl_term" and len(node.args) == 2 and ast.dump(node.args[0]) == ast.dump(node.args[1]):
                    return ast.copy_location(ast.Constant(0.0), node)
                if isinstance(node.func, ast.Name) and node.args:
                    name, arg = node.func.id, node.args[0]
                    if name in {"shannon_term", "hill_eq", "michaelis_menten"} and isinstance(arg, ast.Constant) and arg.value == 0:
                        return ast.Constant(0.0)
                    inverse = {"logit": "logistic_sigmoid", "logistic_sigmoid": "logit", "norm_ppf": "norm_cdf"}
                    if name in inverse and isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name) and arg.func.id == inverse[name] and len(arg.args) == 1:
                        return arg.args[0]
                return node
        algebra = Algebra()
        edits = []
        lines = code.encode().splitlines(keepends=True)
        starts = [0]
        for line in lines: starts.append(starts[-1] + len(line))

        def collect(node):
            if isinstance(node, ast.expr):
                candidate = algebra.visit(copy.deepcopy(node))
                if ast.dump(node) != ast.dump(candidate):
                    edits.append((starts[node.lineno - 1] + node.col_offset,
                                  starts[node.end_lineno - 1] + node.end_col_offset,
                                  ast.unparse(ast.fix_missing_locations(candidate)).encode()))
                    return
            for child in ast.iter_child_nodes(node): collect(child)
        collect(tree)
        if edits:
            data = code.encode()
            for start, end, replacement in sorted(edits, reverse=True):
                data = data[:start] + replacement + data[end:]
            code = data.decode()
            tree = ast.parse(code)
            notes.append("csel_egraph_canonicalization")
    loaded = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    bound = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Load)}
    bound |= {n.arg for n in ast.walk(tree) if isinstance(n, ast.arg)}
    bound |= {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            bound.update(alias.asname or (alias.name.split(".")[0] if isinstance(node, ast.Import) else alias.name) for alias in node.names)
    imports = set()
    typing = {"List", "Dict", "Tuple", "Optional", "Any", "Set", "Union", "Callable", "Iterable", "Sequence", "Mapping", "TypeVar", "Generic"}
    for name in loaded - bound:
        if name in {"math", "json", "os", "re", "time", "pathlib", "typing", "itertools", "functools", "statistics", "collections", "csv"}: imports.add(f"import {name}")
        elif name in typing: imports.add(f"from typing import {name}")
        elif name == "Path": imports.add("from pathlib import Path")
        elif name == "dataclass": imports.add("from dataclasses import dataclass")
        elif name == "Enum": imports.add("from enum import Enum")
    if imports and "*" not in bound:
        offset = tree.body[0].lineno - 1 if tree.body else 0
        if tree.body and getattr(tree.body[0], "decorator_list", None):
            offset = min(d.lineno - 1 for d in tree.body[0].decorator_list)
        for index, node in enumerate(tree.body):
            if (index == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)) or (isinstance(node, ast.ImportFrom) and node.module == "__future__"):
                offset = node.end_lineno
            else: break
        lines = code.splitlines(keepends=True)
        lines.insert(offset, "\n".join(sorted(imports)) + "\n")
        code = "".join(lines)
        ast.parse(code)
        notes.append("missing_stdlib_imports")
    return _json(dict(code=code, changed=code != source_code, notes=notes, llm_tokens_used=0, elapsed_ms=(time.perf_counter() - started) * 1000))
