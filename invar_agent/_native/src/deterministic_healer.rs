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
