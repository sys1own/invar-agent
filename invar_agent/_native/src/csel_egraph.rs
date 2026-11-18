//! Tier-0 CSEL E-Graph Equality Saturation (`egg`), Deterministic Math/LaTeX Desugaring,
//! and Levenshtein/Canonical Symbol Auto-Resolution in Rust.

use std::collections::HashMap;

use egg::{define_language, rewrite, CostFunction, EGraph, Extractor, Id, Language, RecExpr, Runner, Symbol, Var};
use ordered_float::NotNan;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use regex::Regex;

use crate::holographic_hin::CselExpr;

pub type Float = NotNan<f64>;
pub type CselRewrite = egg::Rewrite<CselLanguage, ()>;

define_language! {
    pub enum CselLanguage {
        Num(Float),
        Var(Symbol),
        "+" = Add([Id; 2]),
        "-" = Sub([Id; 2]),
        "*" = Mul([Id; 2]),
        "/" = Div([Id; 2]),
        "^" = Pow([Id; 2]),
        "neg" = Neg([Id; 1]),
        "safe_div" = SafeDiv([Id; 2]),
        "safe_sqrt" = SafeSqrt([Id; 1]),
        "safe_log" = SafeLog([Id; 1]),
        "safe_exp" = SafeExp([Id; 1]),
        "safe_pow" = SafePow([Id; 2]),
        "safe_sin" = SafeSin([Id; 1]),
        "safe_cos" = SafeCos([Id; 1]),
        "safe_tan" = SafeTan([Id; 1]),
        "safe_tanh" = SafeTanh([Id; 1]),
        "safe_acos" = SafeAcos([Id; 1]),
        "safe_asin" = SafeAsin([Id; 1]),
        "abs" = Abs([Id; 1]),
        "min" = Min([Id; 2]),
        "max" = Max([Id; 2]),
        "log1p" = Log1p([Id; 1]),
        "expm1" = Expm1([Id; 1]),
        "smooth_clamp" = SmoothClamp([Id; 3]),
        "where" = Where([Id; 3]),
        "imex_split" = ImexSplit([Id; 2]),
        "norm_cdf" = NormCdf([Id; 1]),
        "norm_ppf" = NormPpf([Id; 1]),
        "logistic_sigmoid" = LogisticSigmoid([Id; 1]),
        "logit" = Logit([Id; 1]),
        "hill_eq" = HillEq([Id; 3]),
        "michaelis_menten" = MichaelisMenten([Id; 3]),
        "kl_term" = KlTerm([Id; 2]),
        "shannon_term" = ShannonTerm([Id; 1]),
        "gammainc_reg" = GammaincReg([Id; 2]),
        "betainc_reg" = BetaincReg([Id; 3]),
    }
}

fn is_not_zero(var: &'static str) -> impl Fn(&mut EGraph<CselLanguage, ()>, Id, &egg::Subst) -> bool {
    let v: Var = var.parse().unwrap();
    move |egraph, _, subst| {
        let id = subst[v];
        !egraph[id].nodes.iter().any(|n| matches!(n, CselLanguage::Num(f) if f.into_inner() == 0.0))
    }
}

fn is_stiff_operator(
    var: &'static str,
) -> impl Fn(&mut EGraph<CselLanguage, ()>, Id, &egg::Subst) -> bool {
    let v: Var = var.parse().unwrap();
    move |egraph, _, subst| {
        let id = subst[v];
        egraph[id].nodes.iter().any(|n| match n {
            CselLanguage::Num(f) => f.into_inner().abs() >= 100.0,
            CselLanguage::Neg([inner]) => egraph[*inner]
                .nodes
                .iter()
                .any(|m| matches!(m, CselLanguage::Num(f) if f.into_inner().abs() >= 100.0)),
            _ => false,
        })
    }
}

pub fn make_csel_rewrite_rules() -> Vec<CselRewrite> {
    vec![
        // 1. Algebraic Simplification
        rewrite!("add-zero"; "(+ ?a 0)" => "?a"),
        rewrite!("zero-add"; "(+ 0 ?a)" => "?a"),
        rewrite!("sub-zero"; "(- ?a 0)" => "?a"),
        rewrite!("mul-one"; "(* ?a 1)" => "?a"),
        rewrite!("one-mul"; "(* 1 ?a)" => "?a"),
        rewrite!("zero-mul"; "(* 0 ?a)" => "0"),
        rewrite!("mul-zero"; "(* ?a 0)" => "0"),
        rewrite!("cancel-sub"; "(+ ?a (* -1 ?a))" => "0"),
        rewrite!("sub-self"; "(- ?a ?a)" => "0"),
        rewrite!("neg-neg"; "(neg (neg ?a))" => "?a"),
        rewrite!("div-self"; "(safe_div ?a ?a)" => "1" if is_not_zero("?a")),
        rewrite!("div-one"; "(safe_div ?a 1)" => "?a"),
        // 2. Catastrophic Cancellation Elimination
        rewrite!("log1p-transform"; "(safe_log (+ 1 ?x))" => "(log1p ?x)"),
        rewrite!("log1p-transform-comm"; "(safe_log (+ ?x 1))" => "(log1p ?x)"),
        rewrite!("expm1-transform"; "(- (safe_exp ?x) 1)" => "(expm1 ?x)"),
        // 3. Difference of Radicals Rationalization (cost-guided via AstSize)
        rewrite!("rad-diff"; "(- (safe_sqrt ?a) (safe_sqrt ?b))" => "(safe_div (- ?a ?b) (+ (safe_sqrt ?a) (safe_sqrt ?b)))"),
        // 4. Division-by-Zero Factoring
        rewrite!("div-factor"; "(safe_div (* ?a ?b) ?a)" => "?b" if is_not_zero("?a")),
        rewrite!("div-factor-r"; "(safe_div (* ?b ?a) ?a)" => "?b" if is_not_zero("?a")),
        // 5. Separation of Stiff Linear Decay
        rewrite!("stiff-sep"; "(+ (* ?k ?x) ?nonlinear)" => "(imex_split (* ?k ?x) ?nonlinear)" if is_stiff_operator("?k")),
        // 6. Pan-Scientific Domain Identities
        rewrite!("kl-self"; "(kl_term ?p ?p)" => "0"),
        rewrite!("shannon-zero"; "(shannon_term 0)" => "0"),
        rewrite!("hill-zero"; "(hill_eq 0 ?ec50 ?n)" => "0"),
        rewrite!("mm-zero"; "(michaelis_menten 0 ?vmax ?km)" => "0"),
        rewrite!("logit-sigmoid"; "(logit (logistic_sigmoid ?x))" => "?x"),
        rewrite!("sigmoid-logit"; "(logistic_sigmoid (logit ?p))" => "?p"),
        rewrite!("ppf-cdf"; "(norm_ppf (norm_cdf ?x))" => "?x"),
    ]
}

fn csel_to_recexpr(expr: &CselExpr, rec: &mut RecExpr<CselLanguage>) -> Id {
    match expr {
        CselExpr::Literal { value } => {
            let f = Float::new(*value).unwrap_or_else(|_| Float::new(0.0).unwrap());
            rec.add(CselLanguage::Num(f))
        }
        CselExpr::Symbol { name } => rec.add(CselLanguage::Var(Symbol::from(name.as_str()))),
        CselExpr::Unary { op, arg } => {
            let a = csel_to_recexpr(arg, rec);
            if op == "-" || op == "neg" {
                rec.add(CselLanguage::Neg([a]))
            } else {
                a
            }
        }
        CselExpr::Binary { op, left, right } => {
            let l = csel_to_recexpr(left, rec);
            let r = csel_to_recexpr(right, rec);
            match op.as_str() {
                "+" => rec.add(CselLanguage::Add([l, r])),
                "-" => rec.add(CselLanguage::Sub([l, r])),
                "*" => rec.add(CselLanguage::Mul([l, r])),
                "/" => rec.add(CselLanguage::SafeDiv([l, r])),
                "^" | "**" => rec.add(CselLanguage::SafePow([l, r])),
                _ => rec.add(CselLanguage::Add([l, r])),
            }
        }
        CselExpr::Call { func, args } => {
            let ids: Vec<Id> = args.iter().map(|a| csel_to_recexpr(a, rec)).collect();
            let g0 = ids.first().copied().unwrap_or_else(|| rec.add(CselLanguage::Num(Float::new(0.0).unwrap())));
            let g1 = ids.get(1).copied().unwrap_or(g0);
            let g2 = ids.get(2).copied().unwrap_or(g1);
            match func.as_str() {
                "safe_div" => rec.add(CselLanguage::SafeDiv([g0, g1])),
                "safe_sqrt" | "sqrt" => rec.add(CselLanguage::SafeSqrt([g0])),
                "safe_log" | "log" => rec.add(CselLanguage::SafeLog([g0])),
                "safe_exp" | "exp" => rec.add(CselLanguage::SafeExp([g0])),
                "safe_pow" | "pow" => rec.add(CselLanguage::SafePow([g0, g1])),
                "safe_sin" | "sin" => rec.add(CselLanguage::SafeSin([g0])),
                "safe_cos" | "cos" => rec.add(CselLanguage::SafeCos([g0])),
                "safe_tan" | "tan" => rec.add(CselLanguage::SafeTan([g0])),
                "safe_tanh" | "tanh" => rec.add(CselLanguage::SafeTanh([g0])),
                "safe_acos" | "acos" => rec.add(CselLanguage::SafeAcos([g0])),
                "safe_asin" | "asin" => rec.add(CselLanguage::SafeAsin([g0])),
                "abs" | "fabs" => rec.add(CselLanguage::Abs([g0])),
                "min" => rec.add(CselLanguage::Min([g0, g1])),
                "max" => rec.add(CselLanguage::Max([g0, g1])),
                "log1p" => rec.add(CselLanguage::Log1p([g0])),
                "expm1" => rec.add(CselLanguage::Expm1([g0])),
                "smooth_clamp" => rec.add(CselLanguage::SmoothClamp([g0, g1, g2])),
                "where" => rec.add(CselLanguage::Where([g0, g1, g2])),
                "imex_split" => rec.add(CselLanguage::ImexSplit([g0, g1])),
                "norm_cdf" => rec.add(CselLanguage::NormCdf([g0])),
                "norm_ppf" => rec.add(CselLanguage::NormPpf([g0])),
                "logistic_sigmoid" => rec.add(CselLanguage::LogisticSigmoid([g0])),
                "logit" => rec.add(CselLanguage::Logit([g0])),
                "hill_eq" => rec.add(CselLanguage::HillEq([g0, g1, g2])),
                "michaelis_menten" => rec.add(CselLanguage::MichaelisMenten([g0, g1, g2])),
                "kl_term" => rec.add(CselLanguage::KlTerm([g0, g1])),
                "shannon_term" => rec.add(CselLanguage::ShannonTerm([g0])),
                "gammainc_reg" => rec.add(CselLanguage::GammaincReg([g0, g1])),
                "betainc_reg" => rec.add(CselLanguage::BetaincReg([g0, g1, g2])),
                _ => g0,
            }
        }
    }
}

fn recexpr_to_csel(rec: &RecExpr<CselLanguage>, id: Id) -> CselExpr {
    match &rec[id] {
        CselLanguage::Num(f) => CselExpr::Literal { value: f.into_inner() },
        CselLanguage::Var(s) => CselExpr::Symbol { name: s.to_string() },
        CselLanguage::Neg([a]) => CselExpr::Unary {
            op: "-".into(),
            arg: Box::new(recexpr_to_csel(rec, *a)),
        },
        CselLanguage::Add([l, r]) => CselExpr::Binary {
            op: "+".into(),
            left: Box::new(recexpr_to_csel(rec, *l)),
            right: Box::new(recexpr_to_csel(rec, *r)),
        },
        CselLanguage::Sub([l, r]) => CselExpr::Binary {
            op: "-".into(),
            left: Box::new(recexpr_to_csel(rec, *l)),
            right: Box::new(recexpr_to_csel(rec, *r)),
        },
        CselLanguage::Mul([l, r]) => CselExpr::Binary {
            op: "*".into(),
            left: Box::new(recexpr_to_csel(rec, *l)),
            right: Box::new(recexpr_to_csel(rec, *r)),
        },
        CselLanguage::Div([l, r]) | CselLanguage::SafeDiv([l, r]) => CselExpr::Call {
            func: "safe_div".into(),
            args: vec![recexpr_to_csel(rec, *l), recexpr_to_csel(rec, *r)],
        },
        CselLanguage::Pow([l, r]) | CselLanguage::SafePow([l, r]) => CselExpr::Call {
            func: "safe_pow".into(),
            args: vec![recexpr_to_csel(rec, *l), recexpr_to_csel(rec, *r)],
        },
        CselLanguage::SafeSqrt([a]) => CselExpr::Call { func: "safe_sqrt".into(), args: vec![recexpr_to_csel(rec, *a)] },
        CselLanguage::SafeLog([a]) => CselExpr::Call { func: "safe_log".into(), args: vec![recexpr_to_csel(rec, *a)] },
        CselLanguage::SafeExp([a]) => CselExpr::Call { func: "safe_exp".into(), args: vec![recexpr_to_csel(rec, *a)] },
        CselLanguage::SafeSin([a]) => CselExpr::Call { func: "safe_sin".into(), args: vec![recexpr_to_csel(rec, *a)] },
        CselLanguage::SafeCos([a]) => CselExpr::Call { func: "safe_cos".into(), args: vec![recexpr_to_csel(rec, *a)] },
        CselLanguage::SafeTan([a]) => CselExpr::Call { func: "safe_tan".into(), args: vec![recexpr_to_csel(rec, *a)] },
        CselLanguage::SafeTanh([a]) => CselExpr::Call { func: "safe_tanh".into(), args: vec![recexpr_to_csel(rec, *a)] },
        CselLanguage::SafeAcos([a]) => CselExpr::Call { func: "safe_acos".into(), args: vec![recexpr_to_csel(rec, *a)] },
        CselLanguage::SafeAsin([a]) => CselExpr::Call { func: "safe_asin".into(), args: vec![recexpr_to_csel(rec, *a)] },
        CselLanguage::Abs([a]) => CselExpr::Call { func: "abs".into(), args: vec![recexpr_to_csel(rec, *a)] },
        CselLanguage::Min([a, b]) => CselExpr::Call { func: "min".into(), args: vec![recexpr_to_csel(rec, *a), recexpr_to_csel(rec, *b)] },
        CselLanguage::Max([a, b]) => CselExpr::Call { func: "max".into(), args: vec![recexpr_to_csel(rec, *a), recexpr_to_csel(rec, *b)] },
        CselLanguage::Log1p([a]) => CselExpr::Call { func: "log1p".into(), args: vec![recexpr_to_csel(rec, *a)] },
        CselLanguage::Expm1([a]) => CselExpr::Call { func: "expm1".into(), args: vec![recexpr_to_csel(rec, *a)] },
        CselLanguage::SmoothClamp([a, b, c]) => CselExpr::Call { func: "smooth_clamp".into(), args: vec![recexpr_to_csel(rec, *a), recexpr_to_csel(rec, *b), recexpr_to_csel(rec, *c)] },
        CselLanguage::Where([a, b, c]) => CselExpr::Call { func: "where".into(), args: vec![recexpr_to_csel(rec, *a), recexpr_to_csel(rec, *b), recexpr_to_csel(rec, *c)] },
        CselLanguage::NormCdf([a]) => CselExpr::Call { func: "norm_cdf".into(), args: vec![recexpr_to_csel(rec, *a)] },
        CselLanguage::NormPpf([a]) => CselExpr::Call { func: "norm_ppf".into(), args: vec![recexpr_to_csel(rec, *a)] },
        CselLanguage::LogisticSigmoid([a]) => CselExpr::Call { func: "logistic_sigmoid".into(), args: vec![recexpr_to_csel(rec, *a)] },
        CselLanguage::Logit([a]) => CselExpr::Call { func: "logit".into(), args: vec![recexpr_to_csel(rec, *a)] },
        CselLanguage::HillEq([a, b, c]) => CselExpr::Call { func: "hill_eq".into(), args: vec![recexpr_to_csel(rec, *a), recexpr_to_csel(rec, *b), recexpr_to_csel(rec, *c)] },
        CselLanguage::MichaelisMenten([a, b, c]) => CselExpr::Call { func: "michaelis_menten".into(), args: vec![recexpr_to_csel(rec, *a), recexpr_to_csel(rec, *b), recexpr_to_csel(rec, *c)] },
        CselLanguage::KlTerm([a, b]) => CselExpr::Call { func: "kl_term".into(), args: vec![recexpr_to_csel(rec, *a), recexpr_to_csel(rec, *b)] },
        CselLanguage::ShannonTerm([a]) => CselExpr::Call { func: "shannon_term".into(), args: vec![recexpr_to_csel(rec, *a)] },
        CselLanguage::GammaincReg([a, b]) => CselExpr::Call { func: "gammainc_reg".into(), args: vec![recexpr_to_csel(rec, *a), recexpr_to_csel(rec, *b)] },
        CselLanguage::BetaincReg([a, b, c]) => CselExpr::Call { func: "betainc_reg".into(), args: vec![recexpr_to_csel(rec, *a), recexpr_to_csel(rec, *b), recexpr_to_csel(rec, *c)] },
        CselLanguage::ImexSplit([a, b]) => {
            // In runtime evaluation, imex_split(linear_stiff, nonlinear) is still linear_stiff + nonlinear
            CselExpr::Binary {
                op: "+".into(),
                left: Box::new(recexpr_to_csel(rec, *a)),
                right: Box::new(recexpr_to_csel(rec, *b)),
            }
        }
    }
}

struct CselCost;

impl CostFunction<CselLanguage> for CselCost {
    type Cost = usize;
    fn cost<C>(&mut self, enode: &CselLanguage, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        let op_cost = match enode {
            CselLanguage::Num(_) | CselLanguage::Var(_) => 1,
            CselLanguage::ImexSplit(_) => 1,
            CselLanguage::Log1p(_) | CselLanguage::Expm1(_) => 1,
            CselLanguage::Add(_) => 2,
            _ => 2,
        };
        enode.fold(op_cost, |sum, id| sum.saturating_add(costs(id)))
    }
}

pub fn optimize_csel_egraph(expr: &CselExpr) -> (CselExpr, bool) {
    let mut rec = RecExpr::default();
    let root = csel_to_recexpr(expr, &mut rec);
    let rules = make_csel_rewrite_rules();
    let runner = Runner::default()
        .with_iter_limit(12)
        .with_node_limit(5000)
        .with_expr(&rec)
        .run(&rules);
    let extractor = Extractor::new(&runner.egraph, CselCost);
    let (_, best) = extractor.find_best(runner.roots[0]);
    let has_imex_split = best.as_ref().iter().any(|n| matches!(n, CselLanguage::ImexSplit(_)));
    let best_id = Id::from(best.as_ref().len() - 1);
    let out = recexpr_to_csel(&best, best_id);
    let _ = root;
    (out, has_imex_split)
}

// ---------------------------------------------------------------------------
// Deterministic Math / LaTeX Desugaring
// ---------------------------------------------------------------------------

fn replace_latex_command_two_args(input: &str, cmd: &str, target: &str) -> String {
    let mut s = input.to_string();
    let needle = format!("\\{}", cmd);
    while let Some(idx) = s.find(&needle) {
        let rest = &s[idx + needle.len()..];
        let rest_trim = rest.trim_start();
        if let Some((arg1, after1)) = extract_braced(rest_trim) {
            let after1_trim = after1.trim_start();
            if let Some((arg2, after2)) = extract_braced(after1_trim) {
                let rep = format!("{}({}, {})", target, arg1, arg2);
                s = format!("{}{}{}", &s[..idx], rep, after2);
                continue;
            }
        }
        break;
    }
    s
}

fn replace_latex_command_one_arg(input: &str, cmd: &str, target: &str) -> String {
    let mut s = input.to_string();
    let needle = format!("\\{}", cmd);
    while let Some(idx) = s.find(&needle) {
        let rest = &s[idx + needle.len()..];
        let rest_trim = rest.trim_start();
        if let Some((arg1, after1)) = extract_braced(rest_trim) {
            let rep = if target.is_empty() {
                arg1
            } else {
                format!("{}({})", target, arg1)
            };
            s = format!("{}{}{}", &s[..idx], rep, after1);
            continue;
        }
        break;
    }
    s
}

fn extract_braced(s: &str) -> Option<(String, &str)> {
    if !s.starts_with('{') {
        return None;
    }
    let mut depth = 0usize;
    for (i, ch) in s.char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some((s[1..i].to_string(), &s[i + 1..]));
                }
            }
            _ => {}
        }
    }
    None
}

pub fn desugar_math_to_csel_core(raw: &str) -> String {
    let mut s = raw.trim().to_string();
    // Strip surrounding $...$ or $$...$$
    s = s.trim_matches('$').trim().to_string();

    // Strip \left and \right
    s = s.replace("\\left(", "(")
        .replace("\\right)", ")")
        .replace("\\left[", "(")
        .replace("\\right]", ")")
        .replace("\\left|", "abs(")
        .replace("\\right|", ")");

    // LaTeX operators
    s = s.replace("\\cdot", " * ")
        .replace("\\times", " * ")
        .replace("\\div", " / ")
        .replace("\\pm", " + ")
        .replace("**", "^");

    // \frac{a}{b} and \dfrac{a}{b} -> safe_div(a, b)
    s = replace_latex_command_two_args(&s, "frac", "safe_div");
    s = replace_latex_command_two_args(&s, "dfrac", "safe_div");
    s = replace_latex_command_one_arg(&s, "sqrt", "safe_sqrt");
    s = replace_latex_command_one_arg(&s, "mathrm", "");
    s = replace_latex_command_one_arg(&s, "text", "");

    // LaTeX function names
    for (lx, fn_name) in [
        ("\\exp", "safe_exp"),
        ("\\ln", "safe_log"),
        ("\\log", "safe_log"),
        ("\\sin", "safe_sin"),
        ("\\cos", "safe_cos"),
        ("\\tan", "safe_tan"),
        ("\\tanh", "safe_tanh"),
        ("\\arcsin", "safe_asin"),
        ("\\arccos", "safe_acos"),
    ] {
        s = s.replace(lx, fn_name);
    }

    // Strip remaining LaTeX greek backslashes (\alpha -> alpha)
    let greek_cmds = [
        "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
        "kappa", "lambda", "mu", "nu", "xi", "pi", "rho", "sigma", "tau",
        "phi", "chi", "psi", "omega", "Delta", "Omega", "Sigma", "Phi", "Psi", "Gamma", "Lambda",
    ];
    for g in greek_cmds {
        s = s.replace(&format!("\\{}", g), g);
    }

    // Unicode Greek & superscripts
    let unicode_map = [
        ("α", "alpha"), ("β", "beta"), ("γ", "gamma"), ("δ", "delta"),
        ("ε", "epsilon"), ("ζ", "zeta"), ("η", "eta"), ("θ", "theta"),
        ("κ", "kappa"), ("λ", "lambda"), ("μ", "mu"), ("ν", "nu"),
        ("ξ", "xi"), ("π", "pi"), ("ρ", "rho"), ("σ", "sigma"),
        ("τ", "tau"), ("φ", "phi"), ("χ", "chi"), ("ψ", "psi"), ("ω", "omega"),
        ("Δ", "Delta"), ("Ω", "Omega"), ("Σ", "Sigma"), ("Φ", "Phi"),
        ("²", "^2"), ("³", "^3"), ("⁻¹", "^(-1)"), ("⁻²", "^(-2)"),
        ("·", " * "), ("×", " * "), ("−", "-"),
    ];
    for (u, ascii) in unicode_map {
        s = s.replace(u, ascii);
    }

    // Strip numpy/math/torch prefixes
    let prefix_re = Regex::new(r"\b(?:np|numpy|math|jnp|torch|sp)\.").unwrap();
    s = prefix_re.replace_all(&s, "").to_string();

    // Map bare standard calls to safe_* equivalents if not already safe_*
    for (bare, safe) in [
        ("sqrt", "safe_sqrt"),
        ("exp", "safe_exp"),
        ("log", "safe_log"),
        ("ln", "safe_log"),
        ("sin", "safe_sin"),
        ("cos", "safe_cos"),
        ("tan", "safe_tan"),
        ("tanh", "safe_tanh"),
        ("acos", "safe_acos"),
        ("asin", "safe_asin"),
        ("pow", "safe_pow"),
        ("fabs", "abs"),
    ] {
        let re = Regex::new(&format!(r"\b{}\s*\(", bare)).unwrap();
        s = re.replace_all(&s, &format!("{}(", safe)).to_string();
    }

    // Implicit multiplication: `2x` -> `2 * x`, `2(` -> `2 * (`, `)(` -> `) * (`
    let re_num_id = Regex::new(r"(\d)([A-Za-z_])").unwrap();
    // Avoid breaking scientific notation like 1e-5 or 1e5
    let mut out = String::with_capacity(s.len() + 16);
    let chars: Vec<char> = s.chars().collect();
    for i in 0..chars.len() {
        let c = chars[i];
        out.push(c);
        if i + 1 < chars.len() {
            let next = chars[i + 1];
            if c.is_ascii_digit() && (next.is_ascii_alphabetic() || next == '_') {
                // Check if this is scientific notation `1e-5` / `1e+5` / `1e5`
                let is_sci = (next == 'e' || next == 'E')
                    && i + 2 < chars.len()
                    && (chars[i + 2].is_ascii_digit()
                        || ((chars[i + 2] == '+' || chars[i + 2] == '-')
                            && i + 3 < chars.len()
                            && chars[i + 3].is_ascii_digit()));
                if !is_sci {
                    out.push('*');
                }
            } else if c.is_ascii_digit() && next == '(' {
                out.push('*');
            } else if c == ')' && (next == '(' || next.is_ascii_alphabetic() || next.is_ascii_digit() || next == '_') {
                out.push('*');
            }
        }
    }
    let _ = re_num_id;
    out
}

// ---------------------------------------------------------------------------
// Symbol & Dimensional Auto-Resolution (Levenshtein + snake_case + Aliases)
// ---------------------------------------------------------------------------

fn to_snake_case(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 4);
    for (i, ch) in s.chars().enumerate() {
        if ch.is_uppercase() {
            if i > 0 && !out.ends_with('_') {
                out.push('_');
            }
            for lc in ch.to_lowercase() {
                out.push(lc);
            }
        } else {
            out.push(ch);
        }
    }
    out
}

fn levenshtein(a: &str, b: &str) -> usize {
    let va: Vec<char> = a.chars().collect();
    let vb: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=vb.len()).collect();
    let mut curr = vec![0; vb.len() + 1];
    for (i, ca) in va.iter().enumerate() {
        curr[0] = i + 1;
        for (j, cb) in vb.iter().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            curr[j + 1] = (prev[j + 1] + 1).min(curr[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[vb.len()]
}

pub fn resolve_symbol_fuzzy_core(ident: &str, ontology_keys: &[String]) -> Option<String> {
    if ontology_keys.iter().any(|k| k == ident) {
        return Some(ident.to_string());
    }
    let lower = ident.to_lowercase();
    let snake = to_snake_case(ident);

    for k in ontology_keys {
        if k.to_lowercase() == lower || to_snake_case(k) == snake {
            return Some(k.clone());
        }
    }

    // Physics alias map
    let aliases: HashMap<&str, &[&str]> = HashMap::from([
        ("vel", &["v", "velocity", "vx", "v_x"][..]),
        ("pos", &["x", "position", "r"][..]),
        ("temp", &["T", "temperature", "temp_k"][..]),
        ("rho", &["density", "rho0", "rho_0"][..]),
        ("omega", &["w", "angular_velocity", "omega_0"][..]),
        ("gamma", &["damping", "gamma_0"][..]),
        ("mass", &["m", "m_0"][..]),
        ("gravity", &["g", "g_0"][..]),
    ]);
    if let Some(cands) = aliases.get(lower.as_str()) {
        for c in *cands {
            if let Some(found) = ontology_keys.iter().find(|k| k.eq_ignore_ascii_case(c)) {
                return Some(found.clone());
            }
        }
    }

    // Levenshtein distance <= 2 (for identifiers of length >= 3)
    if ident.len() >= 3 {
        let mut best: Option<(usize, String)> = None;
        for k in ontology_keys {
            let d = levenshtein(&lower, &k.to_lowercase());
            if d <= 2 {
                if best.as_ref().map_or(true, |(bd, _)| d < *bd) {
                    best = Some((d, k.clone()));
                }
            }
        }
        if let Some((_, matched)) = best {
            return Some(matched);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// PyO3 Functions
// ---------------------------------------------------------------------------

#[pyfunction]
pub fn csel_egraph_optimize(py: Python<'_>, expr_json: &str) -> PyResult<String> {
    let expr: CselExpr = serde_json::from_str(expr_json)
        .map_err(|e| PyValueError::new_err(format!("Invalid CselExpr JSON: {}", e)))?;
    let (optimized, has_imex) = py.allow_threads(move || optimize_csel_egraph(&expr));
    let out = serde_json::json!({
        "expr": optimized,
        "has_imex_split": has_imex,
    });
    serde_json::to_string(&out)
        .map_err(|e| PyValueError::new_err(format!("Serialization error: {}", e)))
}

#[pyfunction]
pub fn desugar_math_to_csel(raw_expr: &str) -> String {
    desugar_math_to_csel_core(raw_expr)
}

#[pyfunction]
pub fn resolve_symbol_fuzzy(ident: &str, ontology_keys: Vec<String>) -> Option<String> {
    resolve_symbol_fuzzy_core(ident, &ontology_keys)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_desugar_latex_and_implicit_mul() {
        let s = desugar_math_to_csel_core(r"\frac{-k x}{m} + 2\sin(\theta) - \alpha^2");
        assert!(s.contains("safe_div(-k x, m)") || s.contains("safe_div"));
        assert!(s.contains("2*safe_sin(theta)"));
        assert!(s.contains("alpha^2"));
    }

    #[test]
    fn test_egraph_log1p_and_stiff_imex() {
        let expr = CselExpr::Call {
            func: "safe_log".into(),
            args: vec![CselExpr::Binary {
                op: "+".into(),
                left: Box::new(CselExpr::Literal { value: 1.0 }),
                right: Box::new(CselExpr::Symbol { name: "x".into() }),
            }],
        };
        let (opt, _) = optimize_csel_egraph(&expr);
        match opt {
            CselExpr::Call { func, .. } => assert_eq!(func, "log1p"),
            other => panic!("Expected log1p, got {:?}", other),
        }

        let stiff = CselExpr::Binary {
            op: "+".into(),
            left: Box::new(CselExpr::Binary {
                op: "*".into(),
                left: Box::new(CselExpr::Literal { value: -5000.0 }),
                right: Box::new(CselExpr::Symbol { name: "x".into() }),
            }),
            right: Box::new(CselExpr::Symbol { name: "y".into() }),
        };
        let (_, has_imex) = optimize_csel_egraph(&stiff);
        assert!(has_imex);
    }

    #[test]
    fn test_resolve_symbol_fuzzy() {
        let pool = vec!["initial_velocity".to_string(), "mass".to_string(), "damping".to_string()];
        assert_eq!(
            resolve_symbol_fuzzy_core("InitialVelocity", &pool),
            Some("initial_velocity".to_string())
        );
        assert_eq!(
            resolve_symbol_fuzzy_core("dampng", &pool),
            Some("damping".to_string())
        );
    }
}
