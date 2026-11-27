//! Zero-eval Pratt / precedence-climbing expression parser + LaTeX/Unicode operator sanitizer.
//!
//! Produces a raw symbolic `RawExpr` AST that `scope_check.rs` resolves into typed `ExprNode`.
//! Python `eval()` and `exec()` never touch model expressions.

use crate::sci_ir::{BinaryOp, CondOp, RandDist, UnaryOp};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RawExpr {
    Const { value: f64 },
    Ident { name: String },
    Unary { op: UnaryOp, arg: Box<RawExpr> },
    Binary { op: BinaryOp, lhs: Box<RawExpr>, rhs: Box<RawExpr> },
    IfThenElse {
        cond_op: CondOp,
        lhs: Box<RawExpr>,
        rhs: Box<RawExpr>,
        then_branch: Box<RawExpr>,
        else_branch: Box<RawExpr>,
    },
    Rand { dist: RandDist, draw_stream_id: u32 },
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Num(f64),
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
    Eof,
}

/// Normalize LaTeX/Unicode operators (`−`, `·`, `×`, `**`, `≥`, `≤`, `\cdot`, `\times`, `\exp`, etc.)
/// into canonical Pratt input while preserving Unicode identifiers (e.g. `ΔΨ`, `k₁`) for Pass 3 alias resolution.
pub fn sanitize_expr_text(input: &str) -> String {
    let mut s = input
        .replace("**", "^")
        .replace('−', "-") // U+2212
        .replace('–', "-") // U+2013
        .replace('·', "*") // U+00B7
        .replace('×', "*") // U+00D7
        .replace('≥', ">=")
        .replace('≤', "<=")
        .replace("\\cdot", "*")
        .replace("\\times", "*")
        .replace("\\left(", "(")
        .replace("\\right)", ")")
        .replace("math.", "");
    // Strip leading backslash on known math functions (longest names first so log10 precedes log, tanh precedes tan).
    for fn_name in [
        "log10", "tanh", "sinh", "cosh", "sqrt", "exp", "log", "tan", "sin", "cos", "abs", "min", "max", "ln",
    ] {
        s = s.replace(&format!("\\{}", fn_name), fn_name);
    }
    s
}

fn is_ident_start(c: char) -> bool {
    c.is_alphabetic() || c == '_' || ('\u{0370}'..='\u{03FF}').contains(&c)
}

fn is_ident_cont(c: char) -> bool {
    is_ident_start(c) || c.is_ascii_digit() || ('\u{2080}'..='\u{2089}').contains(&c)
}

fn tokenize(input: &str) -> Result<Vec<Token>, String> {
    let sanitized = sanitize_expr_text(input);
    let chars: Vec<char> = sanitized.chars().collect();
    let n = chars.len();
    let mut i = 0;
    let mut tokens = Vec::new();

    while i < n {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if c.is_ascii_digit() || (c == '.' && i + 1 < n && chars[i + 1].is_ascii_digit()) {
            let start = i;
            let mut has_dot = false;
            while i < n && (chars[i].is_ascii_digit() || (!has_dot && chars[i] == '.')) {
                if chars[i] == '.' {
                    has_dot = true;
                }
                i += 1;
            }
            if i < n && (chars[i] == 'e' || chars[i] == 'E') {
                let exp_pos = i;
                i += 1;
                if i < n && (chars[i] == '+' || chars[i] == '-') {
                    i += 1;
                }
                let digit_start = i;
                while i < n && chars[i].is_ascii_digit() {
                    i += 1;
                }
                if i == digit_start {
                    i = exp_pos; // backtrack if 'e' wasn't an exponent
                }
            }
            let num_str: String = chars[start..i].iter().collect();
            let val: f64 = num_str
                .parse()
                .map_err(|e| format!("Invalid number literal {:?}: {}", num_str, e))?;
            if !val.is_finite() {
                return Err(format!("Non-finite numeric literal {:?}", num_str));
            }
            tokens.push(Token::Num(val));
            continue;
        }
        if is_ident_start(c) {
            let start = i;
            i += 1;
            while i < n && is_ident_cont(chars[i]) {
                i += 1;
            }
            let ident: String = chars[start..i].iter().collect();
            tokens.push(Token::Ident(ident));
            continue;
        }
        match c {
            '+' => { tokens.push(Token::Plus); i += 1; }
            '-' => { tokens.push(Token::Minus); i += 1; }
            '*' => { tokens.push(Token::Star); i += 1; }
            '/' => { tokens.push(Token::Slash); i += 1; }
            '^' => { tokens.push(Token::Caret); i += 1; }
            '(' => { tokens.push(Token::LParen); i += 1; }
            ')' => { tokens.push(Token::RParen); i += 1; }
            ',' => { tokens.push(Token::Comma); i += 1; }
            '<' => {
                if i + 1 < n && chars[i + 1] == '=' {
                    tokens.push(Token::Le);
                    i += 2;
                } else {
                    tokens.push(Token::Lt);
                    i += 1;
                }
            }
            '>' => {
                if i + 1 < n && chars[i + 1] == '=' {
                    tokens.push(Token::Ge);
                    i += 2;
                } else {
                    tokens.push(Token::Gt);
                    i += 1;
                }
            }
            '=' => {
                if i + 1 < n && chars[i + 1] == '=' {
                    tokens.push(Token::EqEq);
                    i += 2;
                } else {
                    return Err("Unexpected single '=' in expression; use '==' for equality comparison".into());
                }
            }
            other => return Err(format!("Unsupported character {:?} in expression", other)),
        }
    }
    tokens.push(Token::Eof);
    Ok(tokens)
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    depth: usize,
}

impl Parser {
    fn peek(&self) -> &Token {
        &self.tokens[self.pos]
    }

    fn next(&mut self) -> Token {
        let tok = self.tokens[self.pos].clone();
        if self.pos + 1 < self.tokens.len() {
            self.pos += 1;
        }
        tok
    }

    fn expect(&mut self, expected: Token) -> Result<(), String> {
        let got = self.next();
        if got == expected {
            Ok(())
        } else {
            Err(format!("Expected {:?}, got {:?}", expected, got))
        }
    }

    fn parse_expr(&mut self, min_bp: u8) -> Result<RawExpr, String> {
        self.depth += 1;
        if self.depth > 64 {
            return Err("Expression exceeds maximum nesting depth (64)".into());
        }
        let mut lhs = self.parse_prefix()?;
        loop {
            let (op, l_bp, r_bp) = match self.peek() {
                Token::Plus => (BinaryOp::Add, 10, 11),
                Token::Minus => (BinaryOp::Sub, 10, 11),
                Token::Star => (BinaryOp::Mul, 20, 21),
                Token::Slash => (BinaryOp::Div, 20, 21),
                Token::Caret => (BinaryOp::Pow, 31, 30), // right-associative
                _ => break,
            };
            if l_bp < min_bp {
                break;
            }
            self.next();
            let rhs = self.parse_expr(r_bp)?;
            lhs = RawExpr::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        self.depth -= 1;
        Ok(lhs)
    }

    fn parse_prefix(&mut self) -> Result<RawExpr, String> {
        match self.next() {
            Token::Num(v) => Ok(RawExpr::Const { value: v }),
            Token::Plus => self.parse_expr(25),
            Token::Minus => {
                let arg = self.parse_expr(25)?;
                Ok(RawExpr::Unary {
                    op: UnaryOp::Neg,
                    arg: Box::new(arg),
                })
            }
            Token::LParen => {
                let inner = self.parse_expr(0)?;
                self.expect(Token::RParen)?;
                Ok(inner)
            }
            Token::Ident(name) => {
                if *self.peek() == Token::LParen {
                    self.next(); // consume '('
                    self.parse_call(&name)
                } else {
                    Ok(RawExpr::Ident { name })
                }
            }
            other => Err(format!("Unexpected token {:?} in expression", other)),
        }
    }

    fn parse_call(&mut self, name: &str) -> Result<RawExpr, String> {
        let lower = name.to_ascii_lowercase();
        if lower == "if" || lower == "if_then_else" {
            let cond_lhs = self.parse_expr(0)?;
            let cond_op = match self.next() {
                Token::Lt => CondOp::Lt,
                Token::Le => CondOp::Le,
                Token::Gt => CondOp::Gt,
                Token::Ge => CondOp::Ge,
                Token::EqEq => CondOp::Eq,
                other => return Err(format!("Expected comparison operator in if(), got {:?}", other)),
            };
            let cond_rhs = self.parse_expr(0)?;
            self.expect(Token::Comma)?;
            let then_branch = self.parse_expr(0)?;
            self.expect(Token::Comma)?;
            let else_branch = self.parse_expr(0)?;
            self.expect(Token::RParen)?;
            return Ok(RawExpr::IfThenElse {
                cond_op,
                lhs: Box::new(cond_lhs),
                rhs: Box::new(cond_rhs),
                then_branch: Box::new(then_branch),
                else_branch: Box::new(else_branch),
            });
        }

        let mut args = Vec::new();
        if *self.peek() != Token::RParen {
            loop {
                args.push(self.parse_expr(0)?);
                if *self.peek() == Token::Comma {
                    self.next();
                } else {
                    break;
                }
            }
        }
        self.expect(Token::RParen)?;

        match lower.as_str() {
            "abs" | "fabs" => Self::unary(UnaryOp::Abs, args, name),
            "exp" => Self::unary(UnaryOp::Exp, args, name),
            "log" | "ln" => Self::unary(UnaryOp::Log, args, name),
            "sqrt" => Self::unary(UnaryOp::Sqrt, args, name),
            "sin" => Self::unary(UnaryOp::Sin, args, name),
            "cos" => Self::unary(UnaryOp::Cos, args, name),
            "tanh" => Self::unary(UnaryOp::Tanh, args, name),
            "step" | "heaviside" => Self::unary(UnaryOp::Step, args, name),
            "logistic_sigmoid" => {
                if args.len() != 1 {
                    return Err("logistic_sigmoid expects 1 argument".into());
                }
                let x = args.pop().unwrap();
                // 1.0 / (1.0 + exp(-x))
                Ok(RawExpr::Binary {
                    op: BinaryOp::Div,
                    lhs: Box::new(RawExpr::Const { value: 1.0 }),
                    rhs: Box::new(RawExpr::Binary {
                        op: BinaryOp::Add,
                        lhs: Box::new(RawExpr::Const { value: 1.0 }),
                        rhs: Box::new(RawExpr::Unary {
                            op: UnaryOp::Exp,
                            arg: Box::new(RawExpr::Unary {
                                op: UnaryOp::Neg,
                                arg: Box::new(x),
                            }),
                        }),
                    }),
                })
            }
            "min" => Self::binary(BinaryOp::Min, args, name),
            "max" => Self::binary(BinaryOp::Max, args, name),
            "pow" => Self::binary(BinaryOp::Pow, args, name),
            "clamp" => {
                if args.len() != 3 {
                    return Err("clamp(x, lo, hi) expects 3 arguments".into());
                }
                let hi = args.pop().unwrap();
                let lo = args.pop().unwrap();
                let x = args.pop().unwrap();
                Ok(RawExpr::Binary {
                    op: BinaryOp::Min,
                    lhs: Box::new(RawExpr::Binary {
                        op: BinaryOp::Max,
                        lhs: Box::new(x),
                        rhs: Box::new(lo),
                    }),
                    rhs: Box::new(hi),
                })
            }
            "hill_eq" => {
                if args.len() != 3 {
                    return Err("hill_eq(x, k, n) expects 3 arguments".into());
                }
                let n_exp = args.pop().unwrap();
                let k = args.pop().unwrap();
                let x = args.pop().unwrap();
                let x_n = RawExpr::Binary {
                    op: BinaryOp::Pow,
                    lhs: Box::new(x),
                    rhs: Box::new(n_exp.clone()),
                };
                let k_n = RawExpr::Binary {
                    op: BinaryOp::Pow,
                    lhs: Box::new(k),
                    rhs: Box::new(n_exp),
                };
                Ok(RawExpr::Binary {
                    op: BinaryOp::Div,
                    lhs: Box::new(x_n.clone()),
                    rhs: Box::new(RawExpr::Binary {
                        op: BinaryOp::Add,
                        lhs: Box::new(k_n),
                        rhs: Box::new(x_n),
                    }),
                })
            }
            "michaelis_menten" => {
                if args.len() != 3 {
                    return Err("michaelis_menten(s, vmax, km) expects 3 arguments".into());
                }
                let km = args.pop().unwrap();
                let vmax = args.pop().unwrap();
                let s = args.pop().unwrap();
                Ok(RawExpr::Binary {
                    op: BinaryOp::Div,
                    lhs: Box::new(RawExpr::Binary {
                        op: BinaryOp::Mul,
                        lhs: Box::new(vmax),
                        rhs: Box::new(s.clone()),
                    }),
                    rhs: Box::new(RawExpr::Binary {
                        op: BinaryOp::Add,
                        lhs: Box::new(km),
                        rhs: Box::new(s),
                    }),
                })
            }
            "rand_uniform01" | "uniform01" => Self::rand_call(RandDist::Uniform01, args),
            "rand_normal01" | "normal01" => Self::rand_call(RandDist::Normal01, args),
            "rand_exponential1" | "exponential1" => Self::rand_call(RandDist::Exponential1, args),
            _ => Err(format!("Unsupported function call {:?}", name)),
        }
    }

    fn unary(op: UnaryOp, mut args: Vec<RawExpr>, name: &str) -> Result<RawExpr, String> {
        if args.len() != 1 {
            return Err(format!("{} expects 1 argument, got {}", name, args.len()));
        }
        Ok(RawExpr::Unary {
            op,
            arg: Box::new(args.pop().unwrap()),
        })
    }

    fn binary(op: BinaryOp, mut args: Vec<RawExpr>, name: &str) -> Result<RawExpr, String> {
        if args.len() != 2 {
            return Err(format!("{} expects 2 arguments, got {}", name, args.len()));
        }
        let rhs = args.pop().unwrap();
        let lhs = args.pop().unwrap();
        Ok(RawExpr::Binary {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        })
    }

    fn rand_call(dist: RandDist, args: Vec<RawExpr>) -> Result<RawExpr, String> {
        let stream_id = match args.as_slice() {
            [] => 0u32,
            [RawExpr::Const { value }] if *value >= 0.0 && value.fract() == 0.0 => *value as u32,
            _ => return Err("Random draw expects 0 or 1 non-negative integer stream_id argument".into()),
        };
        Ok(RawExpr::Rand {
            dist,
            draw_stream_id: stream_id,
        })
    }
}

/// Parse a mathematical expression string into a closed `RawExpr` AST without `eval()`.
pub fn parse_expr(input: &str) -> Result<RawExpr, String> {
    if input.len() > 32_768 {
        return Err("Expression exceeds 32 KB parser budget".into());
    }
    let tokens = tokenize(input)?;
    let mut parser = Parser {
        tokens,
        pos: 0,
        depth: 0,
    };
    let expr = parser.parse_expr(0)?;
    if *parser.peek() != Token::Eof {
        return Err(format!("Trailing tokens after expression: {:?}", parser.peek()));
    }
    Ok(expr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_unicode_and_precedence() {
        let e = parse_expr("ΔΨ + (2.303 · R · T / F) × 0.5 − 1e-3^2").unwrap();
        assert!(matches!(e, RawExpr::Binary { op: BinaryOp::Sub, .. }));
    }

    #[test]
    fn parses_if_and_clamp() {
        let e = parse_expr("if(x >= 0.15, clamp(x, 0.0, 1.0), 0.0)").unwrap();
        assert!(matches!(e, RawExpr::IfThenElse { cond_op: CondOp::Ge, .. }));
    }

    #[test]
    fn rejects_assignment_and_eval() {
        assert!(parse_expr("x = 2").is_err());
        assert!(parse_expr("__import__('os')").is_err());
    }
}
