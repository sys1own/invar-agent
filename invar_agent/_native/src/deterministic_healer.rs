//! Sub-ms 0-token Rust diagnostic & syntax/import healer (`deterministic_healer.rs`).
//!
//! Deterministic regex/token transducers for recurring mechanical faults:
//!   * F401 / unused imports        -> deterministic line/symbol removal
//!   * NameError on stdlib symbols  -> deterministic import injection
//!   * SyntaxError "expected ':'"   -> deterministic colon completion
//!   * Rust E0382 moved values      -> simple `.clone()` coercion hint
//!
//! Every rule executes in microseconds at zero LLM tokens.

use regex::Regex;
use rustpython_parser::{ast, lexer, Mode, Parse, Tok};
use rustpython_ast::{Ranged, Visitor};
use std::collections::BTreeSet;

fn builder_syntax_repair(source: &str) -> String {
    let mut statements = Vec::new();
    let mut statement = Vec::new();
    for token in lexer::lex(source, Mode::Module) {
        let Ok((token, range)) = token else { break };
        match token {
            Tok::Newline | Tok::EndOfFile => {
                if !statement.is_empty() { statements.push(std::mem::take(&mut statement)); }
            }
            Tok::Indent | Tok::Dedent => {}
            _ => statement.push((token, range)),
        }
    }
    if !statement.is_empty() { statements.push(statement); }
    let indentation = |start: usize| -> String {
        let line = source[..start].rfind('\n').map_or(0, |i| i + 1);
        source[line..start].chars().take_while(|c| *c == ' ' || *c == '\t').collect()
    };
    let mut edits = Vec::new();
    for (i, tokens) in statements.iter().enumerate() {
        let Some((first, first_range)) = tokens.first() else { continue };
        if !matches!(first, Tok::Def | Tok::Class | Tok::If | Tok::Elif | Tok::Else | Tok::For | Tok::While | Tok::Try | Tok::Except | Tok::Finally | Tok::With | Tok::Async | Tok::Match | Tok::Case) { continue; }
        let mut depth = 0;
        let mut colon = false;
        for (tok, _) in tokens {
            match tok {
                Tok::Lpar | Tok::Lsqb | Tok::Lbrace => depth += 1,
                Tok::Rpar | Tok::Rsqb | Tok::Rbrace => depth -= 1,
                Tok::Colon if depth == 0 => colon = true,
                _ => {}
            }
        }
        if depth != 0 { continue; }
        let (last, range) = tokens.last().unwrap();
        if colon && !matches!(last, Tok::Colon) { continue; }
        let end = u32::from(range.end()) as usize;
        let prefix = indentation(u32::from(first_range.start()) as usize);
        let next_indent = statements.get(i + 1).map(|next| indentation(u32::from(next[0].1.start()) as usize));
        let empty = next_indent.as_ref().is_none_or(|next| next.len() <= prefix.len());
        let mut suffix = if colon { String::new() } else { ":".to_string() };
        if empty {
            let newline = source[end..].find('\n').map_or(source.len(), |j| end + j + 1);
            let unit = if prefix.contains('\t') { "\t" } else { "    " };
            let separator = if newline == source.len() && !source.ends_with('\n') { "\n" } else { "" };
            edits.push((newline, format!("{separator}{prefix}{unit}pass\n")));
        }
        if !suffix.is_empty() { edits.push((end, std::mem::take(&mut suffix))); }
    }
    edits.sort_by(|a, b| b.0.cmp(&a.0));
    let mut result = source.to_string();
    for (offset, text) in edits { result.insert_str(offset, &text); }
    result
}

#[derive(Default)]
struct BuilderNames {
    loaded: BTreeSet<String>,
    bound: BTreeSet<String>,
}

impl Visitor for BuilderNames {
    fn visit_expr_name(&mut self, node: ast::ExprName) {
        if node.ctx == ast::ExprContext::Load { self.loaded.insert(node.id.to_string()); }
        else { self.bound.insert(node.id.to_string()); }
    }
    fn visit_stmt_import(&mut self, node: ast::StmtImport) {
        for alias in node.names {
            self.bound.insert(alias.asname.map(|s| s.to_string()).unwrap_or_else(|| alias.name.split('.').next().unwrap().to_string()));
        }
    }
    fn visit_stmt_import_from(&mut self, node: ast::StmtImportFrom) {
        for alias in node.names { self.bound.insert(alias.asname.unwrap_or(alias.name).to_string()); }
    }
    fn visit_stmt_function_def(&mut self, node: ast::StmtFunctionDef) {
        self.bound.insert(node.name.to_string());
        self.generic_visit_stmt_function_def(node);
    }
    fn visit_stmt_async_function_def(&mut self, node: ast::StmtAsyncFunctionDef) {
        self.bound.insert(node.name.to_string());
        self.generic_visit_stmt_async_function_def(node);
    }
    fn visit_stmt_class_def(&mut self, node: ast::StmtClassDef) {
        self.bound.insert(node.name.to_string());
        self.generic_visit_stmt_class_def(node);
    }
    fn visit_arguments(&mut self, node: ast::Arguments) {
        for arg in node.posonlyargs.into_iter().chain(node.args).chain(node.kwonlyargs) {
            self.visit_arg(arg.def);
            if let Some(default) = arg.default { self.visit_expr(*default); }
        }
        for arg in node.vararg.into_iter().chain(node.kwarg) { self.visit_arg(*arg); }
    }
    fn visit_arg(&mut self, node: ast::Arg) {
        self.bound.insert(node.arg.to_string());
        if let Some(annotation) = node.annotation { self.visit_expr(*annotation); }
    }
}

fn builder_import(symbol: &str) -> Option<String> {
    if ["math", "json", "os", "re", "time", "pathlib", "typing", "itertools", "functools", "statistics", "collections", "csv"].contains(&symbol) {
        Some(format!("import {symbol}"))
    } else {
        symbol_to_import(symbol).map(|line| if line == "from typing import" { format!("{line} {symbol}") } else { line.to_string() })
    }
}

fn python_csel(expr: &ast::Expr) -> Option<crate::holographic_hin::CselExpr> {
    use crate::holographic_hin::CselExpr as C;
    Some(match expr {
        ast::Expr::Name(n) => C::Symbol { name:n.id.to_string() },
        ast::Expr::Constant(c) => C::Literal { value:match &c.value { ast::Constant::Float(v) => *v, ast::Constant::Int(v) => v.to_string().parse().ok()?, _ => return None } },
        ast::Expr::UnaryOp(u) if u.op == ast::UnaryOp::USub => C::Unary { op:"-".into(), arg:Box::new(python_csel(&u.operand)?) },
        ast::Expr::BinOp(b) => C::Binary {
            op:match b.op { ast::Operator::Add => "+", ast::Operator::Sub => "-", ast::Operator::Mult => "*", _ => return None }.into(),
            left:Box::new(python_csel(&b.left)?), right:Box::new(python_csel(&b.right)?),
        },
        ast::Expr::Call(c) if c.keywords.is_empty() => {
            let ast::Expr::Name(name) = c.func.as_ref() else { return None };
            if !["kl_term", "shannon_term", "hill_eq", "michaelis_menten", "logit", "logistic_sigmoid", "norm_ppf", "norm_cdf"].contains(&name.id.as_str()) { return None; }
            C::Call { func:name.id.to_string(), args:c.args.iter().map(python_csel).collect::<Option<_>>()? }
        }
        _ => return None,
    })
}

#[derive(Default)]
struct BuilderAlgebra { edits: Vec<(usize, usize, String)> }

impl Visitor for BuilderAlgebra {
    fn visit_expr(&mut self, expr: ast::Expr) {
        if let Some(csel) = python_csel(&expr) {
            let (optimized, _) = crate::csel_egraph::optimize_csel_egraph(&csel);
            if csel.canonical_bytes().ok() != optimized.canonical_bytes().ok() {
                self.edits.push((u32::from(expr.range().start()) as usize, u32::from(expr.range().end()) as usize, crate::adjoint::csel_to_string(&optimized)));
                return;
            }
        }
        self.generic_visit_expr(expr);
    }
}

pub fn heal_builder_module_tier0_json(source: &str, module_kind: &str) -> Result<String, String> {
    if !["kernel", "deterministic_module", "llm_extension_slot", "module"].contains(&module_kind) { return Err("Unknown builder module kind".into()); }
    let started = std::time::Instant::now();
    let mut code = source.to_string();
    let mut notes = Vec::new();
    if ast::Suite::parse(&code, "<builder>").is_err() {
        let candidate = builder_syntax_repair(&code);
        ast::Suite::parse(&candidate, "<builder>").map_err(|e| format!("PY_SYNTAX_ERROR: {e}"))?;
        code = candidate;
        notes.push("syntax_empty_block_repair".to_string());
    }
    let mut suite = ast::Suite::parse(&code, "<builder>").map_err(|e| e.to_string())?;
    if module_kind == "kernel" {
        let mut algebra = BuilderAlgebra::default();
        for stmt in suite.clone() { algebra.visit_stmt(stmt); }
        if !algebra.edits.is_empty() {
            algebra.edits.sort_by_key(|edit| std::cmp::Reverse(edit.0));
            let mut candidate = code.clone();
            for (start, end, text) in algebra.edits { candidate.replace_range(start..end, &text); }
            if let Ok(parsed) = ast::Suite::parse(&candidate, "<builder>") {
                code = candidate;
                suite = parsed;
                notes.push("csel_egraph_canonicalization".to_string());
            }
        }
    }
    let mut names = BuilderNames::default();
    for stmt in suite.clone() { names.visit_stmt(stmt); }
    let imports: BTreeSet<_> = names.loaded.difference(&names.bound).filter_map(|name| builder_import(name)).collect();
    if !imports.is_empty() && !names.bound.contains("*") {
        let mut offset = suite.first().map_or(code.len(), |s| u32::from(s.range().start()) as usize);
        for (i, stmt) in suite.iter().enumerate() {
            let docstring = i == 0 && matches!(stmt, ast::Stmt::Expr(e) if matches!(e.value.as_ref(), ast::Expr::Constant(c) if matches!(c.value, ast::Constant::Str(_))));
            let future = matches!(stmt, ast::Stmt::ImportFrom(s) if s.module.as_ref().is_some_and(|m| m.as_str() == "__future__"));
            if !docstring && !future { break; }
            offset = u32::from(stmt.range().end()) as usize;
            offset = code[offset..].find('\n').map_or(code.len(), |j| offset + j + 1);
        }
        let prefix = if offset > 0 && !code[..offset].ends_with('\n') { "\n" } else { "" };
        let mut candidate = code.clone();
        candidate.insert_str(offset, &format!("{prefix}{}\n", imports.into_iter().collect::<Vec<_>>().join("\n")));
        ast::Suite::parse(&candidate, "<builder>").map_err(|e| format!("PY_SYNTAX_ERROR: {e}"))?;
        code = candidate;
        notes.push("missing_stdlib_imports".to_string());
    }
    Ok(serde_json::json!({"code":code, "changed":code != source, "notes":notes, "llm_tokens_used":0, "elapsed_ms":started.elapsed().as_secs_f64()*1000.0}).to_string())
}

#[cfg(test)]
mod builder_tests {
    use super::*;
    use serde_json::Value;

    fn healed(source: &str, kind: &str) -> Value {
        serde_json::from_str(&heal_builder_module_tier0_json(source, kind).unwrap()).unwrap()
    }

    #[test]
    fn builder_tier0_syntax_imports_and_module_preservation() {
        let source = "\"\"\"Scientific module.\"\"\"\nfrom __future__ import annotations\nclass Plot:\n    def run(self, x: List[float])\n        if x:\n        return math.sqrt(x[0])\n";
        let result = healed(source, "llm_extension_slot");
        let code = result["code"].as_str().unwrap();
        assert!(code.contains("class Plot:"));
        assert!(code.contains("            pass"));
        assert!(code.contains("import math"));
        assert!(code.contains("from typing import List"));
        assert!(code.find("import math").unwrap() > code.find("from __future__").unwrap());
        assert_eq!(result["llm_tokens_used"], 0);
        assert_eq!(healed(code, "llm_extension_slot")["changed"], false);
        assert_eq!(healed("", "module")["code"], "");
        assert_eq!(healed("s = 'math.sqrt and if x'\n", "module")["changed"], false);
        assert_eq!(healed("def f(math):\n    return math.sqrt(1)\n", "module")["changed"], false);
        assert!(heal_builder_module_tier0_json("def f(:\n", "module").is_err());
        assert!(heal_builder_module_tier0_json("", "unknown").is_err());
        let eof = healed("if True:", "module");
        assert!(eof["code"].as_str().unwrap().contains("pass"));
    }

    #[test]
    fn builder_tier0_csel_reflex() {
        let result = healed("def f(p):\n    return kl_term(p, p)\n", "kernel");
        assert!(result["code"].as_str().unwrap().contains("return 0.0"));
        assert_eq!(healed("def f(p):\n    return kl_term(p, p)\n", "module")["changed"], false);
    }
}

/// A single deterministic Tier-0 repair transducer rule.
#[derive(Clone, Debug)]
pub struct ReflexFixRule {
    pub rule_id: String,
    pub error_code: String,
    pub diagnostic_pattern: Regex,
}

/// Built-in deterministic healer core.
pub struct DeterministicHealerCore {
    rules: Vec<ReflexFixRule>,
}

impl DeterministicHealerCore {
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
        }
    }

    pub fn new_with_builtin_rules() -> Self {
        let mut core = Self::new();
        core.add_rule(ReflexFixRule {
            rule_id: "syntax_colon_completion".to_string(),
            error_code: "SyntaxError".to_string(),
            diagnostic_pattern: Regex::new(r"expected ':'").unwrap(),
        });
        core.add_rule(ReflexFixRule {
            rule_id: "f401_unused_import_removal".to_string(),
            error_code: "F401".to_string(),
            diagnostic_pattern: Regex::new(r"imported but unused").unwrap(),
        });
        core.add_rule(ReflexFixRule {
            rule_id: "nameerror_stdlib_import_injection".to_string(),
            error_code: "NameError".to_string(),
            diagnostic_pattern: Regex::new(r"name '(\w+)' is not defined").unwrap(),
        });
        core.add_rule(ReflexFixRule {
            rule_id: "rust_e0382_moved_value_clone".to_string(),
            error_code: "E0382".to_string(),
            diagnostic_pattern: Regex::new(r"use of moved value").unwrap(),
        });
        core
    }

    pub fn add_rule(&mut self, rule: ReflexFixRule) {
        self.rules.push(rule);
    }

    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    fn find_rule(&self, error_code: &str, diagnostic_msg: &str) -> Option<&ReflexFixRule> {
        self.rules
            .iter()
            .find(|r| r.error_code == error_code && r.diagnostic_pattern.is_match(diagnostic_msg))
    }

    /// Whether a diagnostic matches a deterministic Tier-0 transducer.
    pub fn matches_rule(&self, error_code: &str, diagnostic_msg: &str) -> bool {
        self.find_rule(error_code, diagnostic_msg).is_some()
    }

    /// Apply the sub-millisecond deterministic repair transducer.
    /// Returns `Some((healed_source, applied_rule_id))` or `None`.
    pub fn apply_reflex_heal(
        &self,
        error_code: &str,
        diagnostic_msg: &str,
        source_code: &str,
        failing_line: usize,
    ) -> Option<(String, String)> {
        let rule = self.find_rule(error_code, diagnostic_msg)?;
        match rule.rule_id.as_str() {
            "syntax_colon_completion" => heal_colon_completion(source_code, failing_line)
                .map(|s| (s, rule.rule_id.clone())),
            "f401_unused_import_removal" => heal_unused_import(source_code, failing_line)
                .map(|s| (s, rule.rule_id.clone())),
            "nameerror_stdlib_import_injection" => {
                heal_missing_import(source_code, diagnostic_msg)
                    .map(|s| (s, rule.rule_id.clone()))
            }
            "rust_e0382_moved_value_clone" => {
                heal_moved_value_clone(source_code, diagnostic_msg, failing_line)
                    .map(|s| (s, rule.rule_id.clone()))
            }
            _ => None,
        }
    }
}

impl Default for DeterministicHealerCore {
    fn default() -> Self {
        Self::new_with_builtin_rules()
    }
}

// ---------------------------------------------------------------------------
// Rule implementations
// ---------------------------------------------------------------------------

static COLON_KEYWORDS: &str = r"^\s*(async\s+def|def|class|if|elif|else|for|while|try|except\b.*|finally|with|async\s+for|async\s+with|match|case)\b";

fn heal_colon_completion(source_code: &str, failing_line: usize) -> Option<String> {
    let keyword_re = Regex::new(COLON_KEYWORDS).ok()?;
    let lines: Vec<String> = source_code.lines().map(|s| s.to_string()).collect();
    if lines.is_empty() {
        return None;
    }

    // Locate the offending line: explicit line number first, then a scan for
    // the first keyword header lacking a trailing colon.
    let mut target: Option<usize> = None;
    if failing_line > 0 && failing_line <= lines.len() {
        let idx = failing_line - 1;
        if keyword_re.is_match(&lines[idx]) && !lines[idx].trim_end().ends_with(':') {
            target = Some(idx);
        }
    }
    if target.is_none() {
        for (i, line) in lines.iter().enumerate() {
            if keyword_re.is_match(line) && !line.trim_end().ends_with(':') {
                target = Some(i);
                break;
            }
        }
    }
    let idx = target?;
    let trimmed = lines[idx].trim_end().to_string();
    if trimmed.ends_with(':') || trimmed.ends_with('\\') || trimmed.starts_with('#') {
        return None;
    }
    let mut out = lines;
    out[idx] = format!("{}:", trimmed);
    Some(out.join("\n") + if source_code.ends_with('\n') { "\n" } else { "" })
}

fn heal_unused_import(source_code: &str, failing_line: usize) -> Option<String> {
    let mut lines: Vec<String> = source_code.lines().map(|s| s.to_string()).collect();
    if lines.is_empty() {
        return None;
    }
    let idx = if failing_line > 0 && failing_line <= lines.len() {
        failing_line - 1
    } else {
        // Scan for the first top-level import line.
        lines
            .iter()
            .position(|l| {
                let t = l.trim_start();
                (t.starts_with("import ") || t.starts_with("from ")) && !l.starts_with(' ')
            })?
    };
    let line = &lines[idx];
    let t = line.trim_start();
    if !(t.starts_with("import ") || t.starts_with("from ")) {
        return None;
    }
    lines.remove(idx);
    Some(lines.join("\n") + if source_code.ends_with('\n') { "\n" } else { "" })
}

fn symbol_to_import(symbol: &str) -> Option<&'static str> {
    match symbol {
        "Optional" | "List" | "Dict" | "Any" | "Tuple" | "Set" | "Union" | "Callable"
        | "Iterable" | "Sequence" | "Mapping" | "TypeVar" | "Generic" => {
            Some("from typing import")
        }
        "Path" => Some("from pathlib import Path"),
        "dataclass" => Some("from dataclasses import dataclass"),
        "Enum" => Some("from enum import Enum"),
        _ => None,
    }
}

fn heal_missing_import(source_code: &str, diagnostic_msg: &str) -> Option<String> {
    let cap = Regex::new(r"name '(\w+)' is not defined").ok()?.captures(diagnostic_msg)?;
    let symbol = cap.get(1)?.as_str();
    let import_line = symbol_to_import(symbol)?;

    // Already imported?  Nothing to do.
    let already = if import_line == "from typing import" {
        let re = Regex::new(&format!(r"from typing import [^#\n]*\b{}\b", symbol)).ok()?;
        re.is_match(source_code) || Regex::new(r"^import typing\b").ok()?.is_match(source_code)
    } else {
        source_code.contains(import_line)
    };
    if already {
        return None;
    }

    let new_import = if import_line == "from typing import" {
        format!("from typing import {}", symbol)
    } else {
        import_line.to_string()
    };

    let mut lines: Vec<&str> = source_code.lines().collect();
    // Insert after the last top-level import/from-import line, but never
    // before `from __future__` imports.
    let mut insert_at = 0usize;
    let mut future_end = 0usize;
    for (i, line) in lines.iter().enumerate() {
        if line.starts_with("from __future__") {
            future_end = i + 1;
        }
    }
    for (i, line) in lines.iter().enumerate() {
        if i < future_end {
            continue;
        }
        let t = line.trim_start();
        if !line.starts_with(' ') && (t.starts_with("import ") || t.starts_with("from ")) {
            insert_at = i + 1;
        }
    }
    let owned: Vec<String> = lines.iter().map(|s| s.to_string()).collect();
    let mut out: Vec<String> = owned;
    out.insert(insert_at, new_import);
    Some(out.join("\n") + if source_code.ends_with('\n') { "\n" } else { "" })
}

fn heal_moved_value_clone(
    source_code: &str,
    diagnostic_msg: &str,
    failing_line: usize,
) -> Option<String> {
    let cap = Regex::new(r"use of moved value: `?(\w+)`?").ok()?.captures(diagnostic_msg)?;
    let var = cap.get(1)?.as_str().to_string();
    let var_re = Regex::new(&format!(r"\b{}\b", regex::escape(&var))).ok()?;

    let mut lines: Vec<String> = source_code.lines().map(|s| s.to_string()).collect();
    if lines.is_empty() {
        return None;
    }
    let idx = if failing_line > 0 && failing_line <= lines.len() {
        failing_line - 1
    } else {
        lines
            .iter()
            .position(|l| var_re.is_match(l))
            .unwrap_or(0)
    };
    let line = &lines[idx];
    if !var_re.is_match(line) {
        return None;
    }
    // Append `.clone()` to the first bare occurrence of the moved variable.
    let replaced = var_re.replace(line, |m: &regex::Captures| {
        let matched = m.get(0).unwrap().as_str();
        // Avoid double-cloning or cloning field accesses.
        let after = line[m.get(0).unwrap().end()..].chars().next();
        if after == Some('.') {
            matched.to_string()
        } else {
            format!("{}.clone()", matched)
        }
    });
    if replaced == *line {
        return None;
    }
    lines[idx] = replaced.into_owned();
    Some(lines.join("\n") + if source_code.ends_with('\n') { "\n" } else { "" })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn colon_completion_repairs_def() {
        let src = "def foo(a, b)\n    return a + b\n";
        let healer = DeterministicHealerCore::new_with_builtin_rules();
        assert!(healer.matches_rule("SyntaxError", "invalid syntax: expected ':'"));
        let (healed, rule) = healer
            .apply_reflex_heal("SyntaxError", "invalid syntax: expected ':'", src, 1)
            .expect("should heal");
        assert_eq!(rule, "syntax_colon_completion");
        assert!(healed.starts_with("def foo(a, b):\n"));
    }

    #[test]
    fn unused_import_is_removed() {
        let src = "import os\n\nx = 1\n";
        let healer = DeterministicHealerCore::new_with_builtin_rules();
        assert!(healer.matches_rule("F401", "'os' imported but unused"));
        let (healed, _) = healer
            .apply_reflex_heal("F401", "'os' imported but unused", src, 1)
            .expect("should heal");
        assert!(!healed.contains("import os"));
        assert!(healed.contains("x = 1"));
    }

    #[test]
    fn missing_typing_import_is_injected() {
        let src = "def f(x: Optional[int]) -> int:\n    return 1\n";
        let healer = DeterministicHealerCore::new_with_builtin_rules();
        assert!(healer.matches_rule("NameError", "name 'Optional' is not defined"));
        let (healed, _) = healer
            .apply_reflex_heal("NameError", "name 'Optional' is not defined", src, 1)
            .expect("should heal");
        assert!(healed.contains("from typing import Optional"));
    }

    #[test]
    fn reflex_heals_are_sub_two_milliseconds() {
        let healer = DeterministicHealerCore::new_with_builtin_rules();
        let src = "def foo(a)\n    return a\n";
        let start = Instant::now();
        for _ in 0..1000 {
            let _ = healer.apply_reflex_heal(
                "SyntaxError",
                "invalid syntax: expected ':'",
                src,
                1,
            );
        }
        let elapsed = start.elapsed();
        assert!(elapsed.as_millis() < 2000, "1000 heals took {:?}", elapsed);
    }
}
