//! Unified Expression AST `invar.expr.v1` (UPG-006).
//!
//! Single canonical expression representation shared between Python and Rust,
//! with exact rational literals, Pratt parser, deterministic canonicalizer,
//! and multi-dialect renderer (`formal`, `exec_py`, `sci_text`, `csel`, `cpp`, `rust`, `latex`).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnaryNeg {
    Neg,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CmpOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoolOpKind {
    And,
    Or,
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RandDist {
    Uniform01,
    Normal01,
    Exponential1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Expr {
    Const {
        num: String,
        #[serde(default = "default_den")]
        den: String,
    },
    Sym {
        name: String,
    },
    Unary {
        op: UnaryNeg,
        arg: Box<Expr>,
    },
    Binary {
        op: ArithOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    Call {
        #[serde(rename = "fn")]
        func: String,
        args: Vec<Expr>,
    },
    Cond {
        pred: Box<Pred>,
        then_expr: Box<Expr>,
        else_expr: Box<Expr>,
    },
    Rand {
        dist: RandDist,
        #[serde(default)]
        draw_stream_id: u32,
    },
}

fn default_den() -> String {
    "1".to_string()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Pred {
    Cmp {
        op: CmpOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    Bool {
        op: BoolOpKind,
        args: Vec<Pred>,
    },
}

// ---------------------------------------------------------------------------
// Exact decimal/integer -> reduced rational (num, den) using i128 fast-path
// and arbitrary-precision decimal string reduction fallback.
// ---------------------------------------------------------------------------

fn gcd_u128(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

fn reduce_rational_i128(mut num: i128, mut den: i128) -> (String, String) {
    if den < 0 {
        num = -num;
        den = -den;
    }
    if num == 0 {
        return ("0".to_string(), "1".to_string());
    }
    let g = gcd_u128(num.unsigned_abs(), den.unsigned_abs()) as i128;
    ((num / g).to_string(), (den / g).to_string())
}

pub fn decimal_str_to_rational(raw: &str) -> Result<(String, String), String> {
    let s = raw.trim();
    if s.is_empty() {
        return Err("empty numeric literal".into());
    }
    let (neg, rest) = if let Some(r) = s.strip_prefix('-') {
        (true, r)
    } else if let Some(r) = s.strip_prefix('+') {
        (false, r)
    } else {
        (false, s)
    };
    let (mantissa, exp_val): (&str, i32) = if let Some(idx) = rest.find(['e', 'E']) {
        let m = &rest[..idx];
        let e = rest[idx + 1..]
            .parse::<i32>()
            .map_err(|_| format!("invalid exponent in literal: {raw}"))?;
        (m, e)
    } else {
        (rest, 0)
    };
    let (int_part, frac_part) = if let Some(dot) = mantissa.find('.') {
        (&mantissa[..dot], &mantissa[dot + 1..])
    } else {
        (mantissa, "")
    };
    if int_part.is_empty() && frac_part.is_empty() {
        return Err(format!("invalid numeric literal: {raw}"));
    }
    let digits = format!("{int_part}{frac_part}");
    if !digits.chars().all(|c| c.is_ascii_digit()) {
        return Err(format!("invalid digits in literal: {raw}"));
    }
    let trimmed = digits.trim_start_matches('0');
    if trimmed.is_empty() {
        return Ok(("0".to_string(), "1".to_string()));
    }
    let scale = exp_val - (frac_part.len() as i32);
    // Try exact i128 reduction when it fits
    if let Ok(base_val) = trimmed.parse::<i128>() {
        if scale >= 0 && scale <= 30 {
            if let Some(p10) = 10i128.checked_pow(scale as u32) {
                if let Some(num) = base_val.checked_mul(p10) {
                    let signed = if neg { -num } else { num };
                    return Ok(reduce_rational_i128(signed, 1));
                }
            }
        } else if scale < 0 && (-scale) <= 30 {
            if let Some(den) = 10i128.checked_pow((-scale) as u32) {
                let signed = if neg { -base_val } else { base_val };
                return Ok(reduce_rational_i128(signed, den));
            }
        }
    }
    // Arbitrary-precision fallback for very large/small exponents: reduce factors of 2 & 5 via string zeros
    if scale >= 0 {
        let mut n = String::from(if neg { "-" } else { "" });
        n.push_str(trimmed);
        for _ in 0..scale {
            n.push('0');
        }
        Ok((n, "1".to_string()))
    } else {
        let mut s_digits = trimmed.to_string();
        let mut neg_scale = (-scale) as usize;
        while neg_scale > 0 && s_digits.ends_with('0') {
            s_digits.pop();
            neg_scale -= 1;
        }
        let mut n = String::from(if neg { "-" } else { "" });
        n.push_str(&s_digits);
        let mut d = String::from("1");
        for _ in 0..neg_scale {
            d.push('0');
        }
        Ok((n, d))
    }
}

fn negate_rational(num: &str, den: &str) -> (String, String) {
    if num == "0" {
        return ("0".to_string(), "1".to_string());
    }
    if let Some(pos) = num.strip_prefix('-') {
        (pos.to_string(), den.to_string())
    } else {
        (format!("-{num}"), den.to_string())
    }
}

fn rational_to_f64(num: &str, den: &str) -> f64 {
    let n = num.parse::<f64>().unwrap_or(0.0);
    let d = den.parse::<f64>().unwrap_or(1.0);
    n / d
}

fn format_rational_for_code(num: &str, den: &str, force_float: bool) -> String {
    if den == "1" {
        if force_float {
            format!("{num}.0")
        } else {
            num.to_string()
        }
    } else {
        // Render exact division or float depending on dialect
        let v = rational_to_f64(num, den);
        if force_float {
            let s = format!("{v:?}");
            if !s.contains('.') && !s.contains('e') && !s.contains('E') {
                format!("{s}.0")
            } else {
                s
            }
        } else {
            format!("({num} / {den})")
        }
    }
}

// ---------------------------------------------------------------------------
// Balanced binary tree for n-ary min/max
// ---------------------------------------------------------------------------

fn lower_nary_binary_tree(func: &str, args: Vec<Expr>) -> Expr {
    assert!(!args.is_empty());
    if args.len() == 1 {
        return args.into_iter().next().unwrap();
    }
    if args.len() == 2 {
        return Expr::Call {
            func: func.to_string(),
            args,
        };
    }
    let mid = args.len() / 2;
    let mut left_vec = args;
    let right_vec = left_vec.split_off(mid);
    let lhs = lower_nary_binary_tree(func, left_vec);
    let rhs = lower_nary_binary_tree(func, right_vec);
    Expr::Call {
        func: func.to_string(),
        args: vec![lhs, rhs],
    }
}

// ---------------------------------------------------------------------------
// Canonicalization (identical semantics to Python `ir/expr_ast.py`)
// ---------------------------------------------------------------------------

pub fn canonicalize_pred(p: Pred) -> Pred {
    match p {
        Pred::Cmp { op, lhs, rhs } => Pred::Cmp {
            op,
            lhs: Box::new(canonicalize(*lhs)),
            rhs: Box::new(canonicalize(*rhs)),
        },
        Pred::Bool { op, args } => Pred::Bool {
            op,
            args: args.into_iter().map(canonicalize_pred).collect(),
        },
    }
}

pub fn canonicalize(expr: Expr) -> Expr {
    match expr {
        Expr::Const { num, den } => {
            if let (Ok(n), Ok(d)) = (num.parse::<i128>(), den.parse::<i128>()) {
                if d != 0 {
                    let (rn, rd) = reduce_rational_i128(n, d);
                    return Expr::Const { num: rn, den: rd };
                }
            }
            Expr::Const { num, den }
        }
        Expr::Sym { name } => Expr::Sym { name },
        Expr::Rand {
            dist,
            draw_stream_id,
        } => Expr::Rand {
            dist,
            draw_stream_id,
        },
        Expr::Unary { op, arg } => {
            let inner = canonicalize(*arg);
            match inner {
                Expr::Const { num, den } => {
                    let (rn, rd) = negate_rational(&num, &den);
                    Expr::Const { num: rn, den: rd }
                }
                other => Expr::Unary {
                    op,
                    arg: Box::new(other),
                },
            }
        }
        Expr::Binary { op, lhs, rhs } => {
            let cl = canonicalize(*lhs);
            let cr = canonicalize(*rhs);
            if op == ArithOp::Div {
                if let (
                    Expr::Const { num: n1, den: d1 },
                    Expr::Const { num: n2, den: d2 },
                ) = (&cl, &cr)
                {
                    if let (Ok(a), Ok(b), Ok(c), Ok(d)) = (
                        n1.parse::<i128>(),
                        d1.parse::<i128>(),
                        n2.parse::<i128>(),
                        d2.parse::<i128>(),
                    ) {
                        if b != 0 && c != 0 && d != 0 {
                            if let (Some(num), Some(den)) = (a.checked_mul(d), b.checked_mul(c)) {
                                let (rn, rd) = reduce_rational_i128(num, den);
                                return Expr::Const { num: rn, den: rd };
                            }
                        }
                    }
                }
            }
            Expr::Binary {
                op,
                lhs: Box::new(cl),
                rhs: Box::new(cr),
            }
        }
        Expr::Cond {
            pred,
            then_expr,
            else_expr,
        } => Expr::Cond {
            pred: Box::new(canonicalize_pred(*pred)),
            then_expr: Box::new(canonicalize(*then_expr)),
            else_expr: Box::new(canonicalize(*else_expr)),
        },
        Expr::Call { func, args } => {
            let mut cargs: Vec<Expr> = args.into_iter().map(canonicalize).collect();
            let norm_fn = match func.as_str() {
                "ln" => "log",
                "fabs" => "abs",
                other => other,
            };
            match norm_fn {
                "pow" if cargs.len() == 2 => {
                    let rhs = cargs.pop().unwrap();
                    let lhs = cargs.pop().unwrap();
                    Expr::Binary {
                        op: ArithOp::Pow,
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    }
                }
                "clamp" if cargs.len() == 3 => {
                    let hi = cargs.pop().unwrap();
                    let lo = cargs.pop().unwrap();
                    let x = cargs.pop().unwrap();
                    Expr::Call {
                        func: "min".to_string(),
                        args: vec![
                            hi,
                            Expr::Call {
                                func: "max".to_string(),
                                args: vec![lo, x],
                            },
                        ],
                    }
                }
                "step" if cargs.len() == 1 => {
                    let x = cargs.pop().unwrap();
                    Expr::Cond {
                        pred: Box::new(Pred::Cmp {
                            op: CmpOp::Ge,
                            lhs: Box::new(x),
                            rhs: Box::new(Expr::Const {
                                num: "0".to_string(),
                                den: "1".to_string(),
                            }),
                        }),
                        then_expr: Box::new(Expr::Const {
                            num: "1".to_string(),
                            den: "1".to_string(),
                        }),
                        else_expr: Box::new(Expr::Const {
                            num: "0".to_string(),
                            den: "1".to_string(),
                        }),
                    }
                }
                "min" | "max" if cargs.len() >= 2 => lower_nary_binary_tree(norm_fn, cargs),
                other => Expr::Call {
                    func: other.to_string(),
                    args: cargs,
                },
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Lexer & Recursive-Descent Pratt Parser for all dialects
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(String),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    LParen,
    RParen,
    Comma,
    Lt,
    Le,
    Gt,
    Ge,
    EqEq,
    Ne,
    And,
    Or,
    Not,
    If,
    Else,
    Eof,
}

fn sanitize_dialect_text(input: &str) -> String {
    let mut s = input
        .replace("**", "^")
        .replace('−', "-")
        .replace('–', "-")
        .replace('·', "*")
        .replace('×', "*")
        .replace('≥', ">=")
        .replace('≤', "<=")
        .replace('≠', "!=")
        .replace("\\cdot", "*")
        .replace("\\times", "*")
        .replace("\\left(", "(")
        .replace("\\right)", ")")
        .replace("math.", "")
        .replace("std::", "")
        .replace("f64::", "");
    // Longest names first so log10 is stripped before log, tanh before tan!
    for fn_name in [
        "log10", "tanh", "sinh", "cosh", "sqrt", "exp", "log", "tan", "sin", "cos", "abs", "min",
        "max", "ln",
    ] {
        s = s.replace(&format!("\\{fn_name}"), fn_name);
    }
    s
}

fn is_ident_start(c: char) -> bool {
    c.is_alphabetic() || c == '_' || ('\u{0370}'..='\u{03FF}').contains(&c)
}

fn is_ident_cont(c: char) -> bool {
    is_ident_start(c) || c.is_ascii_digit() || ('\u{2080}'..='\u{2089}').contains(&c)
}

fn lex(input: &str) -> Result<Vec<Tok>, String> {
    let s = sanitize_dialect_text(input);
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();
    let mut i = 0;
    let mut out = Vec::new();
    while i < n {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if c.is_ascii_digit() || (c == '.' && i + 1 < n && chars[i + 1].is_ascii_digit()) {
            let start = i;
            while i < n && chars[i].is_ascii_digit() {
                i += 1;
            }
            if i < n && chars[i] == '.' {
                i += 1;
                while i < n && chars[i].is_ascii_digit() {
                    i += 1;
                }
            }
            if i < n && (chars[i] == 'e' || chars[i] == 'E') {
                let save = i;
                i += 1;
                if i < n && (chars[i] == '+' || chars[i] == '-') {
                    i += 1;
                }
                let dstart = i;
                while i < n && chars[i].is_ascii_digit() {
                    i += 1;
                }
                if i == dstart {
                    i = save;
                }
            }
            out.push(Tok::Num(chars[start..i].iter().collect()));
            continue;
        }
        if is_ident_start(c) {
            let start = i;
            while i < n && is_ident_cont(chars[i]) {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            match word.as_str() {
                "and" => out.push(Tok::And),
                "or" => out.push(Tok::Or),
                "not" => out.push(Tok::Not),
                "if" => out.push(Tok::If),
                "else" => out.push(Tok::Else),
                _ => out.push(Tok::Ident(word)),
            }
            continue;
        }
        match c {
            '+' => {
                out.push(Tok::Plus);
                i += 1;
            }
            '-' => {
                out.push(Tok::Minus);
                i += 1;
            }
            '*' => {
                out.push(Tok::Star);
                i += 1;
            }
            '/' => {
                out.push(Tok::Slash);
                i += 1;
            }
            '^' => {
                out.push(Tok::Caret);
                i += 1;
            }
            '(' => {
                out.push(Tok::LParen);
                i += 1;
            }
            ')' => {
                out.push(Tok::RParen);
                i += 1;
            }
            ',' => {
                out.push(Tok::Comma);
                i += 1;
            }
            '<' => {
                if i + 1 < n && chars[i + 1] == '=' {
                    out.push(Tok::Le);
                    i += 2;
                } else {
                    out.push(Tok::Lt);
                    i += 1;
                }
            }
            '>' => {
                if i + 1 < n && chars[i + 1] == '=' {
                    out.push(Tok::Ge);
                    i += 2;
                } else {
                    out.push(Tok::Gt);
                    i += 1;
                }
            }
            '=' => {
                if i + 1 < n && chars[i + 1] == '=' {
                    out.push(Tok::EqEq);
                    i += 2;
                } else {
                    return Err("unexpected single '=' in expression".into());
                }
            }
            '!' => {
                if i + 1 < n && chars[i + 1] == '=' {
                    out.push(Tok::Ne);
                    i += 2;
                } else {
                    out.push(Tok::Not);
                    i += 1;
                }
            }
            '&' if i + 1 < n && chars[i + 1] == '&' => {
                out.push(Tok::And);
                i += 2;
            }
            '|' if i + 1 < n && chars[i + 1] == '|' => {
                out.push(Tok::Or);
                i += 2;
            }
            other => return Err(format!("unexpected character in expression: '{other}'")),
        }
    }
    out.push(Tok::Eof);
    Ok(out)
}

struct Parser {
    toks: Vec<Tok>,
    pos: usize,
}

impl Parser {
    fn new(toks: Vec<Tok>) -> Self {
        Self { toks, pos: 0 }
    }
    fn peek(&self) -> &Tok {
        self.toks.get(self.pos).unwrap_or(&Tok::Eof)
    }
    fn next(&mut self) -> Tok {
        let t = self.peek().clone();
        if self.pos < self.toks.len() {
            self.pos += 1;
        }
        t
    }
    fn expect(&mut self, expected: &Tok) -> Result<(), String> {
        let got = self.next();
        if &got == expected {
            Ok(())
        } else {
            Err(format!("expected {expected:?}, got {got:?}"))
        }
    }

    // expr := add_sub ["if" pred "else" expr]
    fn parse_expr(&mut self) -> Result<Expr, String> {
        let body = self.parse_add_sub()?;
        if self.peek() == &Tok::If {
            self.next();
            let pred = self.parse_pred()?;
            self.expect(&Tok::Else)?;
            let orelse = self.parse_expr()?;
            return Ok(Expr::Cond {
                pred: Box::new(pred),
                then_expr: Box::new(body),
                else_expr: Box::new(orelse),
            });
        }
        Ok(body)
    }

    fn parse_pred(&mut self) -> Result<Pred, String> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> Result<Pred, String> {
        let mut args = vec![self.parse_and()?];
        while self.peek() == &Tok::Or {
            self.next();
            args.push(self.parse_and()?);
        }
        if args.len() == 1 {
            Ok(args.pop().unwrap())
        } else {
            Ok(Pred::Bool {
                op: BoolOpKind::Or,
                args,
            })
        }
    }

    fn parse_and(&mut self) -> Result<Pred, String> {
        let mut args = vec![self.parse_not()?];
        while self.peek() == &Tok::And {
            self.next();
            args.push(self.parse_not()?);
        }
        if args.len() == 1 {
            Ok(args.pop().unwrap())
        } else {
            Ok(Pred::Bool {
                op: BoolOpKind::And,
                args,
            })
        }
    }

    fn parse_not(&mut self) -> Result<Pred, String> {
        if self.peek() == &Tok::Not {
            self.next();
            let inner = self.parse_not()?;
            return Ok(Pred::Bool {
                op: BoolOpKind::Not,
                args: vec![inner],
            });
        }
        // Parenthesized predicate vs parenthesized arithmetic expression
        if self.peek() == &Tok::LParen {
            let save = self.pos;
            self.next();
            if let Ok(p) = self.parse_pred() {
                if self.peek() == &Tok::RParen {
                    self.next();
                    // Make sure it wasn't followed by a comparison operator on an arith expr
                    if !matches!(
                        self.peek(),
                        Tok::Lt | Tok::Le | Tok::Gt | Tok::Ge | Tok::EqEq | Tok::Ne
                    ) {
                        return Ok(p);
                    }
                }
            }
            self.pos = save;
        }
        self.parse_cmp()
    }

    fn parse_cmp(&mut self) -> Result<Pred, String> {
        let lhs = self.parse_add_sub()?;
        let op = match self.peek() {
            Tok::Lt => CmpOp::Lt,
            Tok::Le => CmpOp::Le,
            Tok::Gt => CmpOp::Gt,
            Tok::Ge => CmpOp::Ge,
            Tok::EqEq => CmpOp::Eq,
            Tok::Ne => CmpOp::Ne,
            other => return Err(format!("expected comparison operator, got {other:?}")),
        };
        self.next();
        let rhs = self.parse_add_sub()?;
        // Support chained comparisons like `10 <= t <= 40`
        if matches!(
            self.peek(),
            Tok::Lt | Tok::Le | Tok::Gt | Tok::Ge | Tok::EqEq | Tok::Ne
        ) {
            let mut cmps = vec![Pred::Cmp {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs.clone()),
            }];
            let mut prev = rhs;
            while matches!(
                self.peek(),
                Tok::Lt | Tok::Le | Tok::Gt | Tok::Ge | Tok::EqEq | Tok::Ne
            ) {
                let op2 = match self.next() {
                    Tok::Lt => CmpOp::Lt,
                    Tok::Le => CmpOp::Le,
                    Tok::Gt => CmpOp::Gt,
                    Tok::Ge => CmpOp::Ge,
                    Tok::EqEq => CmpOp::Eq,
                    Tok::Ne => CmpOp::Ne,
                    _ => unreachable!(),
                };
                let rhs2 = self.parse_add_sub()?;
                cmps.push(Pred::Cmp {
                    op: op2,
                    lhs: Box::new(prev),
                    rhs: Box::new(rhs2.clone()),
                });
                prev = rhs2;
            }
            return Ok(Pred::Bool {
                op: BoolOpKind::And,
                args: cmps,
            });
        }
        Ok(Pred::Cmp {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        })
    }

    fn parse_add_sub(&mut self) -> Result<Expr, String> {
        let mut lhs = self.parse_mul_div()?;
        loop {
            match self.peek() {
                Tok::Plus => {
                    self.next();
                    let rhs = self.parse_mul_div()?;
                    lhs = Expr::Binary {
                        op: ArithOp::Add,
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    };
                }
                Tok::Minus => {
                    self.next();
                    let rhs = self.parse_mul_div()?;
                    lhs = Expr::Binary {
                        op: ArithOp::Sub,
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    };
                }
                _ => break,
            }
        }
        Ok(lhs)
    }

    fn parse_mul_div(&mut self) -> Result<Expr, String> {
        let mut lhs = self.parse_unary()?;
        loop {
            match self.peek() {
                Tok::Star => {
                    self.next();
                    let rhs = self.parse_unary()?;
                    lhs = Expr::Binary {
                        op: ArithOp::Mul,
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    };
                }
                Tok::Slash => {
                    self.next();
                    let rhs = self.parse_unary()?;
                    lhs = Expr::Binary {
                        op: ArithOp::Div,
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    };
                }
                _ => break,
            }
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> Result<Expr, String> {
        match self.peek() {
            Tok::Plus => {
                self.next();
                self.parse_unary()
            }
            Tok::Minus => {
                self.next();
                let inner = self.parse_unary()?;
                Ok(Expr::Unary {
                    op: UnaryNeg::Neg,
                    arg: Box::new(inner),
                })
            }
            _ => self.parse_power(),
        }
    }

    fn parse_power(&mut self) -> Result<Expr, String> {
        let base = self.parse_primary()?;
        if self.peek() == &Tok::Caret {
            self.next();
            let exp = self.parse_unary()?;
            return Ok(Expr::Binary {
                op: ArithOp::Pow,
                lhs: Box::new(base),
                rhs: Box::new(exp),
            });
        }
        Ok(base)
    }

    fn parse_primary(&mut self) -> Result<Expr, String> {
        match self.peek().clone() {
            Tok::Num(raw) => {
                self.next();
                let (num, den) = decimal_str_to_rational(&raw)?;
                Ok(Expr::Const { num, den })
            }
            Tok::If => {
                // `if(pred, then_expr, else_expr)` functional form
                self.next();
                self.expect(&Tok::LParen)?;
                let pred = self.parse_pred()?;
                self.expect(&Tok::Comma)?;
                let then_expr = self.parse_expr()?;
                self.expect(&Tok::Comma)?;
                let else_expr = self.parse_expr()?;
                self.expect(&Tok::RParen)?;
                Ok(Expr::Cond {
                    pred: Box::new(pred),
                    then_expr: Box::new(then_expr),
                    else_expr: Box::new(else_expr),
                })
            }
            Tok::Ident(name) => {
                self.next();
                if self.peek() == &Tok::LParen {
                    self.next();
                    // Special CSEL `where(cond_expr, t, f)` -> `Cond(cond_expr > 0, t, f)`
                    if name == "where" {
                        let cond_e = self.parse_expr()?;
                        self.expect(&Tok::Comma)?;
                        let t_e = self.parse_expr()?;
                        self.expect(&Tok::Comma)?;
                        let f_e = self.parse_expr()?;
                        self.expect(&Tok::RParen)?;
                        return Ok(Expr::Cond {
                            pred: Box::new(Pred::Cmp {
                                op: CmpOp::Gt,
                                lhs: Box::new(cond_e),
                                rhs: Box::new(Expr::Const {
                                    num: "0".to_string(),
                                    den: "1".to_string(),
                                }),
                            }),
                            then_expr: Box::new(t_e),
                            else_expr: Box::new(f_e),
                        });
                    }
                    let mut args = Vec::new();
                    if self.peek() != &Tok::RParen {
                        loop {
                            args.push(self.parse_expr()?);
                            if self.peek() == &Tok::Comma {
                                self.next();
                            } else {
                                break;
                            }
                        }
                    }
                    self.expect(&Tok::RParen)?;
                    Ok(Expr::Call { func: name, args })
                } else {
                    Ok(Expr::Sym { name })
                }
            }
            Tok::LParen => {
                self.next();
                let e = self.parse_expr()?;
                self.expect(&Tok::RParen)?;
                Ok(e)
            }
            other => Err(format!("unexpected token {other:?}")),
        }
    }
}

pub fn parse_expr_v1(text: &str, _dialect: &str) -> Result<Expr, String> {
    let toks = lex(text)?;
    let mut p = Parser::new(toks);
    let e = p.parse_expr()?;
    if p.peek() != &Tok::Eof {
        return Err(format!("trailing tokens starting at {:?}", p.peek()));
    }
    Ok(e)
}

// ---------------------------------------------------------------------------
// Multi-dialect Renderer (`formal`, `exec_py`, `sci_text`, `csel`, `cpp`, `rust`, `latex`)
// ---------------------------------------------------------------------------

fn render_pred(p: &Pred, dialect: &str) -> Result<String, String> {
    match p {
        Pred::Cmp { op, lhs, rhs } => {
            let op_s = match op {
                CmpOp::Lt => "<",
                CmpOp::Le => "<=",
                CmpOp::Gt => ">",
                CmpOp::Ge => ">=",
                CmpOp::Eq => "==",
                CmpOp::Ne => "!=",
            };
            Ok(format!(
                "({} {op_s} {})",
                render_expr(lhs, dialect)?,
                render_expr(rhs, dialect)?
            ))
        }
        Pred::Bool { op, args } => match op {
            BoolOpKind::Not => {
                if args.len() != 1 {
                    return Err("Bool::Not requires 1 arg".into());
                }
                let inner = render_pred(&args[0], dialect)?;
                match dialect {
                    "cpp" | "rust" => Ok(format!("(!{inner})")),
                    _ => Ok(format!("(not {inner})")),
                }
            }
            BoolOpKind::And | BoolOpKind::Or => {
                let joiner = match (op, dialect) {
                    (BoolOpKind::And, "cpp" | "rust") => " && ",
                    (BoolOpKind::Or, "cpp" | "rust") => " || ",
                    (BoolOpKind::And, _) => " and ",
                    (BoolOpKind::Or, _) => " or ",
                    _ => unreachable!(),
                };
                let parts: Vec<String> = args
                    .iter()
                    .map(|a| render_pred(a, dialect))
                    .collect::<Result<_, _>>()?;
                Ok(format!("({})", parts.join(joiner)))
            }
        },
    }
}

fn render_cond_csel(pred: &Pred, a: &Expr, b: &Expr) -> Result<String, String> {
    match pred {
        Pred::Bool { op, args } => match op {
            BoolOpKind::And => {
                if args.is_empty() {
                    return render_expr(a, "csel");
                }
                let mut acc = a.clone();
                for sub in args.iter().rev() {
                    acc = Expr::Cond {
                        pred: Box::new(sub.clone()),
                        then_expr: Box::new(acc),
                        else_expr: Box::new(b.clone()),
                    };
                }
                render_expr(&acc, "csel")
            }
            BoolOpKind::Or => {
                if args.is_empty() {
                    return render_expr(b, "csel");
                }
                let mut acc = b.clone();
                for sub in args.iter().rev() {
                    acc = Expr::Cond {
                        pred: Box::new(sub.clone()),
                        then_expr: Box::new(a.clone()),
                        else_expr: Box::new(acc),
                    };
                }
                render_expr(&acc, "csel")
            }
            BoolOpKind::Not => {
                if args.len() != 1 {
                    return Err("Bool::Not requires 1 arg".into());
                }
                render_cond_csel(&args[0], b, a)
            }
        },
        Pred::Cmp { op, lhs, rhs } => {
            let ls = render_expr(lhs, "csel")?;
            let rs = render_expr(rhs, "csel")?;
            let as_ = render_expr(a, "csel")?;
            let bs = render_expr(b, "csel")?;
            Ok(match op {
                CmpOp::Lt => format!("where(({rs} - {ls}), {as_}, {bs})"),
                CmpOp::Le => format!("where(({ls} - {rs}), {bs}, {as_})"),
                CmpOp::Gt => format!("where(({ls} - {rs}), {as_}, {bs})"),
                CmpOp::Ge => format!("where(({rs} - {ls}), {bs}, {as_})"),
                CmpOp::Eq => format!("where(abs({ls} - {rs}), {bs}, {as_})"),
                CmpOp::Ne => format!("where(abs({ls} - {rs}), {as_}, {bs})"),
            })
        }
    }
}

pub fn render_expr(expr: &Expr, dialect: &str) -> Result<String, String> {
    match expr {
        Expr::Const { num, den } => match dialect {
            "cpp" | "rust" => Ok(format_rational_for_code(num, den, true)),
            "latex" => {
                if den == "1" {
                    Ok(num.clone())
                } else {
                    Ok(format!("\\frac{{{num}}}{{{den}}}"))
                }
            }
            _ => Ok(format_rational_for_code(num, den, false)),
        },
        Expr::Sym { name } => Ok(name.clone()),
        Expr::Rand {
            dist,
            draw_stream_id,
        } => {
            let dname = match dist {
                RandDist::Uniform01 => "uniform01",
                RandDist::Normal01 => "normal01",
                RandDist::Exponential1 => "exponential1",
            };
            Ok(format!("rand_{dname}({draw_stream_id})"))
        }
        Expr::Unary { op: UnaryNeg::Neg, arg } => {
            Ok(format!("(-{})", render_expr(arg, dialect)?))
        }
        Expr::Binary { op, lhs, rhs } => {
            let l = render_expr(lhs, dialect)?;
            let r = render_expr(rhs, dialect)?;
            match op {
                ArithOp::Add => Ok(format!("({l} + {r})")),
                ArithOp::Sub => Ok(format!("({l} - {r})")),
                ArithOp::Mul => match dialect {
                    "latex" => Ok(format!("({l} \\cdot {r})")),
                    _ => Ok(format!("({l} * {r})")),
                },
                ArithOp::Div => match dialect {
                    "latex" => Ok(format!("\\frac{{{l}}}{{{r}}}")),
                    _ => Ok(format!("({l} / {r})")),
                },
                ArithOp::Pow => match dialect {
                    "formal" | "exec_py" | "csel" => Ok(format!("({l} ** {r})")),
                    "sci_text" => Ok(format!("({l} ^ {r})")),
                    "cpp" => Ok(format!("std::pow({l}, {r})")),
                    "rust" => Ok(format!("({l}).powf({r})")),
                    "latex" => Ok(format!("({l})^{{{r}}}")),
                    other => Err(format!("unknown render dialect: {other}")),
                },
            }
        }
        Expr::Call { func, args } => {
            let rargs: Vec<String> = args
                .iter()
                .map(|a| render_expr(a, dialect))
                .collect::<Result<_, _>>()?;
            match dialect {
                "exec_py" => {
                    let fname = match func.as_str() {
                        "abs" | "min" | "max" => func.clone(),
                        other => format!("math.{other}"),
                    };
                    Ok(format!("{fname}({})", rargs.join(", ")))
                }
                "cpp" => {
                    let fname = match func.as_str() {
                        "exp" | "log" | "log10" | "sqrt" | "sin" | "cos" | "tan" | "tanh"
                        | "sinh" | "cosh" | "abs" | "min" | "max" => format!("std::{func}"),
                        other => other.to_string(),
                    };
                    Ok(format!("{fname}({})", rargs.join(", ")))
                }
                "rust" => {
                    if rargs.len() == 1 {
                        let m = match func.as_str() {
                            "log" => "ln",
                            other => other,
                        };
                        Ok(format!("({}).{m}()", rargs[0]))
                    } else if rargs.len() == 2 && (func == "min" || func == "max") {
                        Ok(format!("({}).{func}({})", rargs[0], rargs[1]))
                    } else {
                        Ok(format!("{func}({})", rargs.join(", ")))
                    }
                }
                "latex" => {
                    if func == "sqrt" && rargs.len() == 1 {
                        Ok(format!("\\sqrt{{{}}}", rargs[0]))
                    } else if func == "abs" && rargs.len() == 1 {
                        Ok(format!("|{}|", rargs[0]))
                    } else {
                        Ok(format!("\\{func}({})", rargs.join(", ")))
                    }
                }
                "formal" | "sci_text" | "csel" => Ok(format!("{func}({})", rargs.join(", "))),
                other => Err(format!("unknown render dialect: {other}")),
            }
        }
        Expr::Cond {
            pred,
            then_expr,
            else_expr,
        } => match dialect {
            "formal" | "exec_py" => Ok(format!(
                "({} if {} else {})",
                render_expr(then_expr, dialect)?,
                render_pred(pred, dialect)?,
                render_expr(else_expr, dialect)?
            )),
            "sci_text" => Ok(format!(
                "if({}, {}, {})",
                render_pred(pred, dialect)?,
                render_expr(then_expr, dialect)?,
                render_expr(else_expr, dialect)?
            )),
            "csel" => render_cond_csel(pred, then_expr, else_expr),
            "cpp" => Ok(format!(
                "({} ? {} : {})",
                render_pred(pred, dialect)?,
                render_expr(then_expr, dialect)?,
                render_expr(else_expr, dialect)?
            )),
            "rust" => Ok(format!(
                "(if {} {{ {} }} else {{ {} }})",
                render_pred(pred, dialect)?,
                render_expr(then_expr, dialect)?,
                render_expr(else_expr, dialect)?
            )),
            "latex" => Ok(format!(
                "\\begin{{cases}} {} & \\text{{if }} {} \\\\ {} & \\text{{otherwise}} \\end{{cases}}",
                render_expr(then_expr, dialect)?,
                render_pred(pred, dialect)?,
                render_expr(else_expr, dialect)?
            )),
            other => Err(format!("unknown render dialect: {other}")),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_and_canonicalize_core_forms() {
        let e = parse_expr_v1("ln(x) + clamp(y, 0.1, 1.5) + step(t - 10)", "formal").unwrap();
        let c = canonicalize(e);
        for d in ["formal", "exec_py", "sci_text"] {
            let rendered = render_expr(&c, d).unwrap();
            let reparsed = canonicalize(parse_expr_v1(&rendered, d).unwrap());
            assert_eq!(c, reparsed, "dialect {d} failed round-trip: {rendered}");
        }
    }

    #[test]
    fn decimal_literal_exact_rational() {
        let e = canonicalize(parse_expr_v1("-0.1", "formal").unwrap());
        assert_eq!(
            e,
            Expr::Const {
                num: "-1".to_string(),
                den: "10".to_string()
            }
        );
    }
}
