//! Exact SI dimensional reduction of equation residuals (each expression = 0).
//! Repeating quantities must be declared positive on the model domain; no
//! cancellation across a potentially zero dimensional reference is assumed.

use crate::csel_egraph::{
    canonicalize_dimensionless_system, AlphaCanonicalResult, CanonicalSymbol, CselLanguage,
    TaggedCselEq,
};
use crate::holographic_hin::CselExpr;
use egg::{Language, RecExpr};
use std::collections::BTreeMap;

type Result<T> = std::result::Result<T, String>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, serde::Serialize)]
struct Rational {
    n: i128,
    d: i128,
}

fn gcd(mut a: i128, mut b: i128) -> i128 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}
fn overflow() -> String {
    "exact Buckingham-Pi rational overflow".into()
}

impl Rational {
    const ZERO: Self = Self { n: 0, d: 1 };
    const ONE: Self = Self { n: 1, d: 1 };
    fn new(n: i128, d: i128) -> Result<Self> {
        if d == 0 {
            return Err("zero rational denominator".into());
        }
        let an = n.checked_abs().ok_or_else(overflow)?;
        let ad = d.checked_abs().ok_or_else(overflow)?;
        let g = gcd(an, ad);
        let n = if d < 0 {
            n.checked_neg().ok_or_else(overflow)?
        } else {
            n
        };
        Ok(Self {
            n: n / g,
            d: ad / g,
        })
    }
    fn neg(self) -> Result<Self> {
        Self::new(self.n.checked_neg().ok_or_else(overflow)?, self.d)
    }
    fn add(self, b: Self) -> Result<Self> {
        let g = gcd(self.d, b.d);
        Self::new(
            self.n
                .checked_mul(b.d / g)
                .and_then(|x| b.n.checked_mul(self.d / g).and_then(|y| x.checked_add(y)))
                .ok_or_else(overflow)?,
            self.d.checked_mul(b.d / g).ok_or_else(overflow)?,
        )
    }
    fn sub(self, b: Self) -> Result<Self> {
        self.add(b.neg()?)
    }
    fn mul(self, b: Self) -> Result<Self> {
        let a = gcd(self.n.abs(), b.d);
        let c = gcd(b.n.abs(), self.d);
        Self::new(
            (self.n / a).checked_mul(b.n / c).ok_or_else(overflow)?,
            (self.d / c).checked_mul(b.d / a).ok_or_else(overflow)?,
        )
    }
    fn div(self, b: Self) -> Result<Self> {
        self.mul(Self::new(b.d, b.n)?)
    }
    fn decimal(text: &str) -> Result<Self> {
        let (mantissa, exponent) = text
            .split_once(['e', 'E'])
            .map_or(Ok((text, 0_i32)), |(m, e)| {
                e.parse::<i32>().map(|e| (m, e)).map_err(|e| e.to_string())
            })?;
        let places = mantissa.split_once('.').map_or(0, |(_, f)| f.len() as i32);
        let digits = mantissa
            .replace('.', "")
            .parse::<i128>()
            .map_err(|e| e.to_string())?;
        let power = exponent - places;
        if power >= 0 {
            Self::new(
                digits
                    .checked_mul(10_i128.checked_pow(power as u32).ok_or_else(overflow)?)
                    .ok_or_else(overflow)?,
                1,
            )
        } else {
            Self::new(
                digits,
                10_i128
                    .checked_pow(power.unsigned_abs())
                    .ok_or_else(overflow)?,
            )
        }
    }
    fn text(self) -> String {
        if self.d == 1 {
            self.n.to_string()
        } else {
            format!("(/ {} {})", self.n, self.d)
        }
    }
}

/// Rows are SI M,L,T,Theta,I,N,J; columns follow an invariant symbol order.
fn rref(mut a: Vec<Vec<Rational>>, n: usize) -> Result<(Vec<Vec<Rational>>, Vec<usize>)> {
    let mut pivots = Vec::new();
    let mut row = 0;
    for col in 0..n {
        let Some(pivot) = (row..a.len()).find(|&i| a[i][col] != Rational::ZERO) else {
            continue;
        };
        a.swap(row, pivot);
        let scale = a[row][col];
        for j in 0..n {
            a[row][j] = a[row][j].div(scale)?;
        }
        for i in 0..a.len() {
            if i != row {
                let k = a[i][col];
                for j in 0..n {
                    a[i][j] = a[i][j].sub(k.mul(a[row][j])?)?;
                }
            }
        }
        pivots.push(col);
        row += 1;
        if row == a.len() {
            break;
        }
    }
    Ok((a, pivots))
}

fn integer_nullspace(
    a: &[Vec<Rational>],
    pivots: &[usize],
    n: usize,
) -> Result<Vec<Vec<Rational>>> {
    let mut basis = Vec::new();
    for free in (0..n).filter(|j| !pivots.contains(j)) {
        let mut v = vec![Rational::ZERO; n];
        v[free] = Rational::ONE;
        for (i, &p) in pivots.iter().enumerate() {
            v[p] = a[i][free].neg()?;
        }
        let mut lcm = 1_i128;
        for x in &v {
            lcm = (lcm / gcd(lcm, x.d))
                .checked_mul(x.d)
                .ok_or_else(overflow)?;
        }
        for x in &mut v {
            *x = x.mul(Rational::new(lcm, 1)?)?;
        }
        let divisor = v.iter().fold(0, |g, x| gcd(g, x.n.abs()));
        for x in &mut v {
            *x = x.div(Rational::new(divisor, 1)?)?;
        }
        basis.push(v);
    }
    Ok(basis)
}

type Monomial = BTreeMap<String, Rational>;
#[derive(Clone)]
struct Polynomial(BTreeMap<Monomial, Rational>);

impl Polynomial {
    fn constant(x: Rational) -> Self {
        Self(if x == Rational::ZERO {
            BTreeMap::new()
        } else {
            BTreeMap::from([(BTreeMap::new(), x)])
        })
    }
    fn atom(s: String) -> Self {
        Self(BTreeMap::from([(
            BTreeMap::from([(s, Rational::ONE)]),
            Rational::ONE,
        )]))
    }
    fn coefficient(&self) -> Option<Rational> {
        if self.0.len() == 1 {
            self.0.get(&BTreeMap::new()).copied()
        } else if self.0.is_empty() {
            Some(Rational::ZERO)
        } else {
            None
        }
    }
    fn content(&self) -> Result<Rational> {
        self.0.values().try_fold(Rational::ZERO, |sum, k| {
            sum.add(Rational::new(k.n.abs(), k.d)?)
        })
    }
    fn add(mut self, b: Self) -> Result<Self> {
        for (m, k) in b.0 {
            let k = self.0.get(&m).copied().unwrap_or(Rational::ZERO).add(k)?;
            if k == Rational::ZERO {
                self.0.remove(&m);
            } else {
                self.0.insert(m, k);
            }
        }
        if self.0.len() > 2048 {
            return Err("Buckingham-Pi polynomial expansion budget exhausted".into());
        }
        Ok(self)
    }
    fn scale(mut self, k: Rational) -> Result<Self> {
        for c in self.0.values_mut() {
            *c = c.mul(k)?;
        }
        if k == Rational::ZERO {
            self.0.clear();
        }
        Ok(self)
    }
    fn mul(self, b: Self) -> Result<Self> {
        if self.0.len().saturating_mul(b.0.len()) > 2048 {
            return Err("Buckingham-Pi polynomial expansion budget exhausted".into());
        }
        let mut out = Self::constant(Rational::ZERO);
        for (m, c) in &self.0 {
            for (p, d) in &b.0 {
                let mut key = m.clone();
                for (s, e) in p {
                    let exponent = key.get(s).copied().unwrap_or(Rational::ZERO).add(*e)?;
                    if exponent == Rational::ZERO {
                        key.remove(s);
                    } else {
                        key.insert(s.clone(), exponent);
                    }
                }
                out = out.add(Self(BTreeMap::from([(key, c.mul(*d)?)])))?;
            }
        }
        Ok(out)
    }
    fn div(self, b: Self) -> Result<Self> {
        if b.0.is_empty() {
            return Err("zero denominator in dimensional equation".into());
        }
        if b.0.len() == 1 {
            let (m, c) = b.0.into_iter().next().unwrap();
            let mut inverse = BTreeMap::new();
            for (s, e) in m {
                inverse.insert(s, e.neg()?);
            }
            self.mul(Self(BTreeMap::from([(inverse, Rational::ONE.div(c)?)])))
        } else {
            let c = *b.0.values().next().unwrap();
            let normalized = b.scale(Rational::ONE.div(c)?)?;
            self.scale(Rational::ONE.div(c)?)?
                .mul(Self::atom(format!("(/ 1 {})", normalized.text())))
        }
    }
    fn pow(self, e: Rational) -> Result<Self> {
        if e.d == 1 && (0..=16).contains(&e.n) {
            let mut out = Self::constant(Rational::ONE);
            for _ in 0..e.n {
                out = out.mul(self.clone())?;
            }
            Ok(out)
        } else if e.d == 1 && (-16..0).contains(&e.n) {
            Self::constant(Rational::ONE).div(self.pow(e.neg()?)?)
        } else {
            Ok(Self::atom(format!("(^ {} {})", self.text(), e.text())))
        }
    }
    fn residual(self) -> Result<Self> {
        let content = self.content()?;
        if content == Rational::ZERO {
            Ok(self)
        } else {
            self.scale(Rational::ONE.div(content)?)
        }
    }
    fn text(&self) -> String {
        fn fold(op: &str, items: Vec<String>, empty: &str) -> String {
            items
                .into_iter()
                .reduce(|a, b| format!("({op} {a} {b})"))
                .unwrap_or_else(|| empty.into())
        }
        let terms = self
            .0
            .iter()
            .map(|(m, c)| {
                let mut factors = Vec::new();
                if *c != Rational::ONE || m.is_empty() {
                    factors.push(c.text());
                }
                for (s, e) in m {
                    factors.push(if *e == Rational::ONE {
                        s.clone()
                    } else {
                        format!("(^ {s} {})", e.text())
                    });
                }
                fold("*", factors, "1")
            })
            .collect();
        fold("+", terms, "0")
    }
}

#[derive(Clone)]
enum Expr {
    Number(Rational),
    Symbol(String),
    Apply(String, Vec<Expr>),
}
impl Expr {
    fn colored_text(&self, labels: &BTreeMap<String, String>, distinguished: &str) -> String {
        match self {
            Self::Number(x) => x.text(),
            Self::Symbol(s) => {
                if s == distinguished {
                    "self".into()
                } else {
                    labels[s].clone()
                }
            }
            Self::Apply(op, args) => {
                fn flatten(e: &Expr, op: &str, out: &mut Vec<Expr>) {
                    match e {
                        Expr::Apply(other, args) if other == op => {
                            for a in args {
                                flatten(a, op, out)
                            }
                        }
                        _ => out.push(e.clone()),
                    }
                }
                let mut args = args.clone();
                if op == "+" || op == "*" {
                    let mut flat = Vec::new();
                    for a in &args {
                        flatten(a, op, &mut flat)
                    }
                    args = flat;
                }
                let mut rendered: Vec<_> = args
                    .iter()
                    .map(|a| a.colored_text(labels, distinguished))
                    .collect();
                if op == "+" || op == "*" {
                    rendered.sort();
                }
                format!("({op} {})", rendered.join(" "))
            }
        }
    }
    fn parse(s: &str) -> Result<Self> {
        if s.trim_start().starts_with('{') {
            fn convert(e: CselExpr) -> Result<Expr> {
                Ok(match e {
                    CselExpr::Literal { value } if value.is_finite() => {
                        Expr::Number(Rational::decimal(&value.to_string())?)
                    }
                    CselExpr::Symbol { name } => Expr::Symbol(name),
                    CselExpr::Unary { op, arg } => Expr::Apply(
                        if op == "-" { "neg".into() } else { op },
                        vec![convert(*arg)?],
                    ),
                    CselExpr::Binary { op, left, right } => {
                        Expr::Apply(op, vec![convert(*left)?, convert(*right)?])
                    }
                    CselExpr::Call { func, args } => {
                        Expr::Apply(func, args.into_iter().map(convert).collect::<Result<_>>()?)
                    }
                    _ => return Err("nonfinite dimensional literal".into()),
                })
            }
            return convert(serde_json::from_str(s).map_err(|e| e.to_string())?);
        }
        let aliases = regex::Regex::new(r"\((exp|log|sqrt|sin|cos|tan|tanh|acos|asin)\s").unwrap();
        let s = aliases.replace_all(s, "(safe_$1 ");
        let rec: RecExpr<CselLanguage> = s
            .parse::<RecExpr<CselLanguage>>()
            .map_err(|e| e.to_string())?;
        let mut out: Vec<Expr> = Vec::new();
        for node in rec.as_ref() {
            out.push(match node {
                CselLanguage::Num(x) => Self::Number(Rational::decimal(&x.to_string())?),
                CselLanguage::Var(s) => Self::Symbol(s.to_string()),
                _ => Self::Apply(
                    node.to_string(),
                    node.children()
                        .iter()
                        .map(|i| out[usize::from(*i)].clone())
                        .collect(),
                ),
            });
        }
        out.pop()
            .ok_or_else(|| "empty dimensional expression".into())
    }
}

#[derive(Clone)]
struct Value {
    form: Polynomial,
    dimension: Vec<Rational>,
    positive: bool,
    bounds: Option<(f64, f64)>,
}
fn symbol_bounds(t: &CanonicalSymbol) -> Option<(f64, f64)> {
    let [lo, hi] = t.domain?;
    let scale = (t.unit_scale[1] as f64) / (t.unit_scale[0] as f64);
    let (a, b) = (lo * scale, hi * scale);
    if a.is_finite() && b.is_finite() {
        Some((a.min(b), a.max(b)))
    } else {
        None
    }
}
fn iv_add(a: Option<(f64, f64)>, b: Option<(f64, f64)>) -> Option<(f64, f64)> {
    let ((a0, a1), (b0, b1)) = (a?, b?);
    let (lo, hi) = (a0 + b0, a1 + b1);
    if lo.is_finite() && hi.is_finite() { Some((lo, hi)) } else { None }
}
fn iv_sub(a: Option<(f64, f64)>, b: Option<(f64, f64)>) -> Option<(f64, f64)> {
    let ((a0, a1), (b0, b1)) = (a?, b?);
    let (lo, hi) = (a0 - b1, a1 - b0);
    if lo.is_finite() && hi.is_finite() { Some((lo, hi)) } else { None }
}
fn iv_mul(a: Option<(f64, f64)>, b: Option<(f64, f64)>) -> Option<(f64, f64)> {
    let ((a0, a1), (b0, b1)) = (a?, b?);
    let ps = [a0 * b0, a0 * b1, a1 * b0, a1 * b1];
    if ps.iter().all(|x| x.is_finite()) {
        Some((ps.into_iter().fold(f64::INFINITY, f64::min), ps.into_iter().fold(f64::NEG_INFINITY, f64::max)))
    } else {
        None
    }
}
fn iv_div(a: Option<(f64, f64)>, b: Option<(f64, f64)>) -> Option<(f64, f64)> {
    let ((a0, a1), (b0, b1)) = (a?, b?);
    if !(b0 > 0.0 || b1 < 0.0) {
        return None;
    }
    let qs = [a0 / b0, a0 / b1, a1 / b0, a1 / b1];
    if qs.iter().all(|x| x.is_finite()) {
        Some((qs.into_iter().fold(f64::INFINITY, f64::min), qs.into_iter().fold(f64::NEG_INFINITY, f64::max)))
    } else {
        None
    }
}
impl Value {
    fn scalar(form: Polynomial) -> Self {
        let positive = form.coefficient().map_or(false, |x| x.n > 0);
        let bounds = form.coefficient().and_then(|x| {
            let v = (x.n as f64) / (x.d as f64);
            if v.is_finite() { Some((v, v)) } else { None }
        });
        Self {
            form,
            dimension: vec![Rational::ZERO; 7],
            positive,
            bounds,
        }
    }
    fn known_nonzero(&self) -> bool {
        self.positive
            || self.form.coefficient().map_or(false, |x| x.n != 0)
            || self.bounds.map_or(false, |(lo, hi)| lo.is_finite() && hi.is_finite() && (lo > 0.0 || hi < 0.0))
    }
    fn lower(e: &Expr, variables: &BTreeMap<String, Value>) -> Result<Self> {
        let (op, args) = match e {
            Expr::Number(x) => return Ok(Self::scalar(Polynomial::constant(*x))),
            Expr::Symbol(s) => {
                return variables
                    .get(s)
                    .cloned()
                    .ok_or_else(|| format!("undeclared alpha-canonical symbol: {s}"))
            }
            Expr::Apply(op, args) => (
                op.as_str(),
                args.iter()
                    .map(|x| Self::lower(x, variables))
                    .collect::<Result<Vec<_>>>()?,
            ),
        };
        let need = |n| {
            if args.len() == n {
                Ok(())
            } else {
                Err(format!("invalid dimensional operator arity: {op}"))
            }
        };
        match op {
            "neg" | "-" if args.len() == 1 => Ok(Self {
                form: args[0].form.clone().scale(Rational::new(-1, 1)?)?,
                dimension: args[0].dimension.clone(),
                positive: false,
                bounds: args[0].bounds.map(|(lo, hi)| (-hi, -lo)),
            }),
            "+" | "-" => {
                need(2)?;
                if !args[0].form.0.is_empty()
                    && !args[1].form.0.is_empty()
                    && args[0].dimension != args[1].dimension
                {
                    return Err("inhomogeneous dimensional equation".into());
                }
                let b = if op == "-" {
                    args[1].form.clone().scale(Rational::new(-1, 1)?)?
                } else {
                    args[1].form.clone()
                };
                let bounds = if op == "+" {
                    iv_add(args[0].bounds, args[1].bounds)
                } else {
                    iv_sub(args[0].bounds, args[1].bounds)
                };
                Ok(Self {
                    form: args[0].form.clone().add(b)?,
                    dimension: if args[0].form.0.is_empty() {
                        args[1].dimension.clone()
                    } else {
                        args[0].dimension.clone()
                    },
                    positive: (op == "+" && args[0].positive && args[1].positive)
                        || bounds.map_or(false, |(lo, _)| lo > 0.0),
                    bounds,
                })
            }
            "*" | "/" => {
                need(2)?;
                let a = &args[0];
                let b = &args[1];
                if op == "/" && !b.known_nonzero() {
                    return Err(
                        "formal division requires a domain-proven nonzero denominator".into(),
                    );
                }
                let bounds = if op == "*" {
                    iv_mul(a.bounds, b.bounds)
                } else {
                    iv_div(a.bounds, b.bounds)
                };
                Ok(Self {
                    form: if op == "*" {
                        a.form.clone().mul(b.form.clone())?
                    } else {
                        a.form.clone().div(b.form.clone())?
                    },
                    dimension: a
                        .dimension
                        .iter()
                        .zip(&b.dimension)
                        .map(|(a, b)| if op == "*" { a.add(*b) } else { a.sub(*b) })
                        .collect::<Result<_>>()?,
                    positive: (a.positive && b.positive)
                        || bounds.map_or(false, |(lo, _)| lo > 0.0),
                    bounds,
                })
            }
            "^" | "**" | "pow" => {
                need(2)?;
                let e = args[1]
                    .form
                    .coefficient()
                    .ok_or("dimensional power requires an exact rational exponent")?;
                if args[1].dimension.iter().any(|x| *x != Rational::ZERO) {
                    return Err("dimensionful exponent".into());
                }
                if e.d != 1 && args[0].dimension.iter().any(|x| *x != Rational::ZERO) {
                    return Err(
                        "dimensionful rational powers require symbolic unit-root normalization"
                            .into(),
                    );
                }
                if e.n < 0 && !args[0].known_nonzero() {
                    return Err("negative powers require a domain-proven nonzero base".into());
                }
                Ok(Self {
                    form: args[0].form.clone().pow(e)?,
                    dimension: args[0]
                        .dimension
                        .iter()
                        .map(|x| x.mul(e))
                        .collect::<Result<_>>()?,
                    positive: args[0].positive || e.n == 0,
                    bounds: None,
                })
            }
            "sqrt" | "safe_sqrt" => {
                need(1)?;
                if args[0].dimension.iter().any(|x| *x != Rational::ZERO) {
                    return Err(
                        "dimensionful radicals require symbolic unit-root normalization".into(),
                    );
                }
                Ok(Self::scalar(Polynomial::atom(format!(
                    "(safe_sqrt {})",
                    args[0].form.text()
                ))))
            }
            "abs" | "fabs" => {
                need(1)?;
                let content = args[0].form.content()?;
                let form = if content == Rational::ZERO {
                    Polynomial::constant(Rational::ZERO)
                } else {
                    Polynomial::atom(format!("(abs {})", args[0].form.clone().residual()?.text()))
                        .scale(content)?
                };
                let bounds = args[0].bounds.map(|(lo, hi)| {
                    if lo <= 0.0 && hi >= 0.0 {
                        (0.0, (-lo).max(hi))
                    } else {
                        (lo.abs().min(hi.abs()), lo.abs().max(hi.abs()))
                    }
                });
                Ok(Self {
                    form,
                    dimension: args[0].dimension.clone(),
                    positive: args[0].known_nonzero(),
                    bounds,
                })
            }
            _ => {
                if args
                    .iter()
                    .any(|x| x.dimension.iter().any(|d| *d != Rational::ZERO))
                {
                    return Err(format!("{op} requires dimensionless arguments (totalized dimensional functions are not unit-covariant)"));
                }
                let op = match op {
                    "exp" => "safe_exp",
                    "log" => "safe_log",
                    "sin" => "safe_sin",
                    "cos" => "safe_cos",
                    "tan" => "safe_tan",
                    "tanh" => "safe_tanh",
                    x => x,
                };
                let mut value = Self::scalar(Polynomial::atom(format!(
                    "({op} {})",
                    args.iter()
                        .map(|a| a.form.text())
                        .collect::<Vec<_>>()
                        .join(" ")
                )));
                value.positive = matches!(op, "safe_exp" | "logistic_sigmoid");
                Ok(value)
            }
        }
    }
}

fn compress(labels: BTreeMap<String, String>) -> BTreeMap<String, String> {
    let mut values: Vec<_> = labels.values().cloned().collect();
    values.sort();
    values.dedup();
    labels
        .into_iter()
        .map(|(s, c)| (s, format!("c{}", values.binary_search(&c).unwrap())))
        .collect()
}

pub(crate) fn canonical_system(
    eqs: &[TaggedCselEq],
    max_branches: usize,
) -> Result<AlphaCanonicalResult> {
    let mut symbols = BTreeMap::<String, CanonicalSymbol>::new();
    let mut exprs = Vec::new();
    for eq in eqs {
        for (s, t) in &eq.symbols {
            let scale = Rational::new(t.unit_scale[0] as i128, t.unit_scale[1] as i128)?;
            if scale.n <= 0 {
                return Err("unit scales must be strictly positive".into());
            }
            if let Some(old) = symbols.insert(s.clone(), t.clone()) {
                if old != *t {
                    return Err("inconsistent symbol labels".into());
                }
            }
        }
        exprs.push(Expr::parse(&eq.expr)?);
    }
    if symbols.len() > 64 {
        return Err("Buckingham-Pi supports at most 64 symbols".into());
    }
    let initial = symbols
        .iter()
        .map(|(s, t)| {
            (
                s.clone(),
                format!(
                    "{}:{:?}:{:?}:{}",
                    !t.positive, t.role, t.si_dim, t.semantic_tag
                ),
            )
        })
        .collect();
    let mut colors = compress(initial);
    // SI-normalized contexts prevent a mere change of coordinate units from
    // splitting a color cell before individualization/refinement.
    for _ in 0..symbols.len() {
        let mut signatures = BTreeMap::new();
        for distinguished in symbols.keys() {
            let variables = symbols
                .iter()
                .map(|(s, t)| {
                    Ok((
                        s.clone(),
                        Value {
                            form: Polynomial::atom(s.clone()).scale(Rational::new(
                                t.unit_scale[1] as i128,
                                t.unit_scale[0] as i128,
                            )?)?,
                            dimension: t
                                .si_dim
                                .iter()
                                .map(|&x| Rational::new(x as i128, 1))
                                .collect::<Result<_>>()?,
                            positive: t.positive,
                            bounds: symbol_bounds(t),
                        },
                    ))
                })
                .collect::<Result<BTreeMap<_, _>>>()?;
            let mut contexts = exprs
                .iter()
                .map(|e| {
                    let v = Value::lower(e, &variables)?;
                    let expression = Expr::parse(&v.form.residual()?.text())?;
                    Ok((v.dimension, expression.colored_text(&colors, distinguished)))
                })
                .collect::<Result<Vec<_>>>()?;
            contexts.sort();
            signatures.insert(
                distinguished.clone(),
                format!("{}:{contexts:?}", colors[distinguished]),
            );
        }
        let next = compress(signatures);
        let stable = symbols.keys().all(|a| {
            symbols
                .keys()
                .all(|b| (colors[a] == colors[b]) == (next[a] == next[b]))
        });
        colors = next;
        if stable {
            break;
        }
    }
    let mut cells = BTreeMap::<String, Vec<String>>::new();
    for (s, c) in colors {
        cells.entry(c).or_default().push(s);
    }
    let limit = max_branches.min(64);
    let mut count = 1usize;
    for cell in cells.values() {
        for k in 2..=cell.len() {
            count = count.saturating_mul(k);
        }
    }
    if limit == 0 || count > limit {
        return Err(
            "alpha-canonical symmetry budget exhausted; no invariant signature emitted".into(),
        );
    }
    fn permutations(xs: &mut [String], i: usize, out: &mut Vec<Vec<String>>) {
        if i == xs.len() {
            out.push(xs.to_vec());
        } else {
            for j in i..xs.len() {
                xs.swap(i, j);
                permutations(xs, i + 1, out);
                xs.swap(i, j);
            }
        }
    }
    let mut orders = vec![Vec::new()];
    for cell in cells.into_values() {
        let mut p = Vec::new();
        permutations(&mut cell.clone(), 0, &mut p);
        orders = orders
            .into_iter()
            .flat_map(|v| {
                p.iter().map(move |p| {
                    let mut v = v.clone();
                    v.extend(p.clone());
                    v
                })
            })
            .collect();
    }
    let eval_order = |order: &[String]| -> Result<String> {
        let n = order.len();
        let dimensions = (0..7)
            .map(|i| {
                order
                    .iter()
                    .map(|s| Rational::new(symbols[s].si_dim[i] as i128, 1))
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        let (matrix, pivots) = rref(dimensions, n)?;
        if pivots.iter().any(|&p| !symbols[&order[p]].positive) {
            return Err(
                "dimensional reduction requires positive repeating quantities covering the SI rank"
                    .into(),
            );
        }
        let basis = integer_nullspace(&matrix, &pivots, n)?;
        let free: Vec<_> = (0..n).filter(|j| !pivots.contains(j)).collect();
        let mut variables = BTreeMap::new();
        let mut pi_symbols = BTreeMap::new();
        for (i, &f) in free.iter().enumerate() {
            let coeff_ok = basis[i][f] == Rational::ONE || basis[i][f] == Rational::new(-1, 1)?;
            if !coeff_ok && !symbols[&order[f]].positive {
                return Err(
                    "primitive rational-power Pi groups require a positive free quantity".into(),
                );
            }
            let mut tag = symbols[&order[f]].clone();
            tag.si_dim = [0; 7];
            tag.unit_scale = [1, 1];
            tag.semantic_tag = i as u8;
            pi_symbols.insert(format!("pi{i}"), tag);
        }
        for (j, s) in order.iter().enumerate() {
            let form = match free.iter().position(|&f| f == j) {
                Some(i) => Polynomial(BTreeMap::from([(
                    BTreeMap::from([(format!("pi{i}"), Rational::ONE.div(basis[i][j])?)]),
                    Rational::ONE,
                )])),
                None => Polynomial::constant(Rational::ONE),
            };
            variables.insert(
                s.clone(),
                Value {
                    form: form.scale(Rational::new(
                        symbols[s].unit_scale[1] as i128,
                        symbols[s].unit_scale[0] as i128,
                    )?)?,
                    dimension: symbols[s]
                        .si_dim
                        .iter()
                        .map(|&x| Rational::new(x as i128, 1))
                        .collect::<Result<_>>()?,
                    positive: symbols[s].positive,
                    bounds: symbol_bounds(&symbols[s]),
                },
            );
        }
        let mut reduced = Vec::new();
        let mut equation_dimensions = Vec::new();
        let mut exact = Vec::new();
        for e in &exprs {
            let v = Value::lower(e, &variables)?;
            let expression = v.form.residual()?.text();
            equation_dimensions.push(v.dimension);
            exact.push(expression.clone());
            reduced.push(TaggedCselEq {
                expr: expression,
                symbols: pi_symbols.clone(),
            });
        }
        let mut equations = Vec::new();
        for ((eq, dimension), exact) in reduced.iter().zip(equation_dimensions).zip(exact) {
            let canon = canonicalize_dimensionless_system(std::slice::from_ref(eq), 64)?;
            equations.push((dimension, exact, canon.canonical_system));
        }
        equations.sort();
        let tags: Vec<_> = order
            .iter()
            .map(|s| {
                let mut t = symbols[s].clone();
                t.unit_scale = [1, 1];
                t
            })
            .collect();
        serde_json::to_string(&(tags, basis, equations)).map_err(|e| e.to_string())
    };

    let mut best: Option<String> = None;
    let mut last_err: Option<String> = None;
    for mut order in orders {
        // Positive domain-proven references precede potentially zero states.
        order.sort_by_key(|s| !symbols[s].positive);
        match eval_order(&order) {
            Ok(candidate) => {
                if best.as_ref().map_or(true, |b| candidate < *b) {
                    best = Some(candidate);
                }
            }
            Err(e) => {
                last_err = Some(e);
            }
        }
    }
    // NF-08(b): If greedy pivot choice failed on signed states, enumerate subsets of positive symbols
    // as leading pivot candidates and pick the lexicographically smallest valid canonical form.
    if best.is_none() {
        let pos_syms: Vec<String> = symbols
            .iter()
            .filter(|(_, s)| s.positive)
            .map(|(k, _)| k.clone())
            .collect();
        let nonpos_syms: Vec<String> = symbols
            .iter()
            .filter(|(_, s)| !s.positive)
            .map(|(k, _)| k.clone())
            .collect();
        if !pos_syms.is_empty() && pos_syms.len() <= 10 {
            fn gen_subsets(
                items: &[String],
                k: usize,
                start: usize,
                cur: &mut Vec<String>,
                out: &mut Vec<Vec<String>>,
            ) {
                if cur.len() == k {
                    out.push(cur.clone());
                    return;
                }
                for idx in start..items.len() {
                    cur.push(items[idx].clone());
                    gen_subsets(items, k, idx + 1, cur, out);
                    cur.pop();
                }
            }
            let max_k = pos_syms.len().min(7);
            for k in 1..=max_k {
                let mut subsets = Vec::new();
                gen_subsets(&pos_syms, k, 0, &mut Vec::new(), &mut subsets);
                for sub in subsets {
                    let mut order = sub;
                    for p in &pos_syms {
                        if !order.contains(p) {
                            order.push(p.clone());
                        }
                    }
                    order.extend(nonpos_syms.iter().cloned());
                    if let Ok(candidate) = eval_order(&order) {
                        if best.as_ref().map_or(true, |b| candidate < *b) {
                            best = Some(candidate);
                        }
                    }
                }
            }
        }
    }
    let canonical_system = match best {
        Some(b) => b,
        None => return Err(last_err.unwrap_or_else(|| "dimensional reduction failed".into())),
    };
    Ok(AlphaCanonicalResult {
        canonical_system,
        saturated: true,
        branches_explored: count,
    })
}

/// Exact AC-ring multivariate polynomial normal form for `RecExpr<CselLanguage>`.
/// Used when binary AC e-graph rewriting exceeds the 2^k associativity node cap on
/// large sums/products (e.g. expanded Jacobian determinants). Fails closed on
/// unsupported operators, non-finite numbers, or polynomial expansion overflow.
pub(crate) fn canonicalize_polynomial_recexpr(rec: &RecExpr<CselLanguage>) -> Result<String> {
    let nodes = rec.as_ref();
    if nodes.is_empty() {
        return Err("empty expression".into());
    }
    let mut vals: Vec<Polynomial> = Vec::with_capacity(nodes.len());
    for node in nodes {
        let p = match node {
            CselLanguage::Num(x) => {
                let v = x.into_inner();
                if !v.is_finite() {
                    return Err("nonfinite constant".into());
                }
                Polynomial::constant(Rational::decimal(&v.to_string())?)
            }
            CselLanguage::Var(s) => Polynomial::atom(s.to_string()),
            CselLanguage::Neg([a]) => vals[usize::from(*a)].clone().scale(Rational::new(-1, 1)?)?,
            CselLanguage::Add([a, b]) => vals[usize::from(*a)].clone().add(vals[usize::from(*b)].clone())?,
            CselLanguage::Sub([a, b]) => {
                let nb = vals[usize::from(*b)].clone().scale(Rational::new(-1, 1)?)?;
                vals[usize::from(*a)].clone().add(nb)?
            }
            CselLanguage::Mul([a, b]) => vals[usize::from(*a)].clone().mul(vals[usize::from(*b)].clone())?,
            CselLanguage::SafeDiv([a, b]) => {
                let pa = vals[usize::from(*a)].clone();
                let pb = vals[usize::from(*b)].clone();
                if let Some(c) = pb.coefficient() {
                    if c == Rational::ZERO {
                        return Err("zero denominator".into());
                    }
                    pa.scale(Rational::ONE.div(c)?)?
                } else {
                    Polynomial::atom(format!("(safe_div {} {})", pa.text(), pb.text()))
                }
            }
            CselLanguage::SafePow([a, b]) => {
                let pa = vals[usize::from(*a)].clone();
                let pb = vals[usize::from(*b)].clone();
                if let Some(c) = pb.coefficient() {
                    if c.d == 1 && c.n >= 0 && c.n <= 16 {
                        let mut acc = Polynomial::constant(Rational::ONE);
                        for _ in 0..c.n {
                            acc = acc.mul(pa.clone())?;
                        }
                        acc
                    } else {
                        Polynomial::atom(format!("(safe_pow {} {})", pa.text(), pb.text()))
                    }
                } else {
                    Polynomial::atom(format!("(safe_pow {} {})", pa.text(), pb.text()))
                }
            }
            CselLanguage::SafeExp([a]) => Polynomial::atom(format!("(safe_exp {})", vals[usize::from(*a)].text())),
            CselLanguage::SafeLog([a]) => Polynomial::atom(format!("(safe_log {})", vals[usize::from(*a)].text())),
            CselLanguage::SafeSqrt([a]) => Polynomial::atom(format!("(safe_sqrt {})", vals[usize::from(*a)].text())),
            CselLanguage::SafeSin([a]) => Polynomial::atom(format!("(safe_sin {})", vals[usize::from(*a)].text())),
            CselLanguage::SafeCos([a]) => Polynomial::atom(format!("(safe_cos {})", vals[usize::from(*a)].text())),
            CselLanguage::Abs([a]) => Polynomial::atom(format!("(abs {})", vals[usize::from(*a)].text())),
            _ => return Err("unsupported operator in polynomial canonicalization".into()),
        };
        vals.push(p);
    }
    Ok(vals.last().unwrap().text())
}

/// Diagonal affine nondimensionalization scales for extreme SI conditioning (`1e-24..1e18`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DiagonalAffineScaling {
    /// Characteristic scale `S_i > 0` for each STATE slot (`y_hat_i = y_i / S_i`).
    pub state_scales: Vec<f64>,
    /// Characteristic time scale `tau_0 > 0` (`t_hat = (t - t_0) / tau_0`).
    pub tau_0: f64,
    /// Nondimensionalized initial state vector `y_hat(0)`.
    pub scaled_initial_states: Vec<f64>,
    /// Dimensionless Pi-group descriptors inferred from symbol scales.
    pub pi_groups: Vec<String>,
}

/// Compute diagonal affine scaling `(S, tau_0, pi_groups)` for a `SciIR` system.
/// States belonging to the same `ConservationPoolSpec` share the pool's characteristic
/// total scale so linear conservation laws stay exact in scaled coordinates.
pub fn compute_diagonal_affine_scales(ir: &crate::sci_ir::SciIR) -> DiagonalAffineScaling {
    let mut states: Vec<&crate::sci_ir::SymbolDecl> = ir
        .symbols
        .iter()
        .filter(|s| s.role == crate::sci_ir::SlotRole::State)
        .collect();
    states.sort_by_key(|s| s.slot_index);

    // Map pool_id -> characteristic pool scale (> 0)
    let mut pool_scale: BTreeMap<String, f64> = BTreeMap::new();
    for pool in &ir.conservation_pools {
        if pool.expected_total.is_finite() && pool.expected_total.abs() > 1e-30 {
            pool_scale.insert(pool.pool_id.clone(), pool.expected_total.abs());
        }
    }

    let mut state_scales = Vec::with_capacity(states.len());
    let mut scaled_initial_states = Vec::with_capacity(states.len());
    let mut pi_groups = Vec::new();

    for st in &states {
        let mut s = if let Some(ref pid) = st.pool_id {
            pool_scale.get(pid).copied().unwrap_or(0.0)
        } else {
            0.0
        };
        if s <= 1e-30 {
            if st.initial_or_default.is_finite() && st.initial_or_default.abs() > 1e-30 {
                s = st.initial_or_default.abs();
            } else if let Some(ref dom) = st.domain {
                let mag = dom.hi.abs().max(dom.lo.abs());
                s = if mag > 1e-30 { mag } else { 1.0 };
            } else {
                s = 1.0;
            }
        }
        // Normalize to nearest power-of-2 or exact positive characteristic magnitude
        if !s.is_finite() || s <= 1e-30 {
            s = 1.0;
        }
        state_scales.push(s);
        scaled_initial_states.push(st.initial_or_default / s);
        pi_groups.push(format!("pi_{} = {} / {:.6e}", st.name, st.name, s));
    }

    let tau_0 = match &ir.config {
        crate::sci_ir::ArchetypeConfig::OdePde { t_span, .. } => {
            let dt = (t_span.1 - t_span.0).abs();
            if dt.is_finite() && dt > 1e-30 {
                dt
            } else {
                1.0
            }
        }
        _ => 1.0,
    };

    DiagonalAffineScaling {
        state_scales,
        tau_0,
        scaled_initial_states,
        pi_groups,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uast::WireRole;

    #[test]
    fn exact_rref_has_primitive_integer_kernel_and_checked_errors() {
        let input = vec![
            vec![Rational::new(2, 1).unwrap(), Rational::ONE, Rational::ZERO],
            vec![Rational::ZERO, Rational::ONE, Rational::ONE],
        ];
        let (a, p) = rref(input.clone(), 3).unwrap();
        let basis = integer_nullspace(&a, &p, 3).unwrap();
        assert_eq!(p, vec![0, 1]);
        assert_eq!(basis.len(), 1);
        assert_eq!(
            basis[0],
            vec![
                Rational::new(1, 1).unwrap(),
                Rational::new(-2, 1).unwrap(),
                Rational::new(2, 1).unwrap()
            ]
        );
        for row in input {
            let dot = row
                .iter()
                .zip(&basis[0])
                .try_fold(Rational::ZERO, |s, (a, b)| s.add(a.mul(*b)?))
                .unwrap();
            assert_eq!(dot, Rational::ZERO);
        }
        assert!(Rational::new(1, 0).is_err());
        assert!(Rational::new(i128::MAX, 1)
            .unwrap()
            .mul(Rational::new(2, 1).unwrap())
            .is_err());
        assert_eq!(
            Rational::decimal("1.25e-2").unwrap(),
            Rational::new(1, 80).unwrap()
        );
        assert_eq!(rref(vec![vec![]; 7], 0).unwrap().1, Vec::<usize>::new());
    }

    fn allen_eggers(names: [&str; 6], km: bool, cubic: bool) -> TaggedCselEq {
        let [h, reference, k, v, z, a] = names;
        let tag = |role, si_dim, positive, length_power: i32| CanonicalSymbol {
            role,
            si_dim,
            semantic_tag: 0,
            positive,
            domain: None,
            unit_scale: if km {
                if length_power >= 0 {
                    [1000_i64.pow(length_power as u32), 1]
                } else {
                    [1, 1000_i64.pow((-length_power) as u32)]
                }
            } else {
                [1, 1]
            },
        };
        let symbols = BTreeMap::from([
            (
                h.into(),
                tag(WireRole::ControlParam, [0, 1, 0, 0, 0, 0, 0], true, 1),
            ),
            (
                reference.into(),
                tag(WireRole::ControlParam, [0, 1, -1, 0, 0, 0, 0], true, 1),
            ),
            (
                k.into(),
                tag(WireRole::ControlParam, [0, -1, 0, 0, 0, 0, 0], true, -1),
            ),
            (
                v.into(),
                tag(WireRole::DynamicState, [0, 1, -1, 0, 0, 0, 0], false, 1),
            ),
            (
                z.into(),
                tag(WireRole::DynamicState, [0, 1, 0, 0, 0, 0, 0], false, 1),
            ),
            (
                a.into(),
                tag(
                    WireRole::DerivedObservable,
                    [0, 1, -2, 0, 0, 0, 0],
                    false,
                    1,
                ),
            ),
        ]);
        let velocity = if cubic {
            format!("(/ (* {v} (* {v} {v})) {reference})")
        } else {
            format!("(* {v} {v})")
        };
        TaggedCselEq {
            expr: format!("(+ {a} (* {k} (* {velocity} (safe_exp (neg (/ {z} {h}))))))"),
            symbols,
        }
    }

    #[test]
    fn allen_eggers_renamed_kilometre_clone_matches_but_cubic_differs() {
        let original = allen_eggers(["H", "v_ref", "k", "v", "z", "a"], false, false);
        let clone = allen_eggers(
            [
                "scale_height",
                "speed_scale",
                "drag",
                "speed",
                "altitude",
                "decel",
            ],
            true,
            false,
        );
        let canonical = canonical_system(&[original.clone()], 64)
            .unwrap()
            .canonical_system;
        let renamed = canonical_system(&[clone.clone()], 64)
            .unwrap()
            .canonical_system;
        assert_eq!(canonical, renamed);
        assert_eq!(
            crate::his::compute_cds_digests(&canonical, "claim", "domain").model_hash,
            crate::his::compute_cds_digests(&renamed, "claim", "domain").model_hash
        );
        let cubic = allen_eggers(["H", "v_ref", "k", "v", "z", "a"], false, true);
        assert_ne!(
            canonical,
            canonical_system(&[cubic], 64).unwrap().canonical_system
        );
        let mut explicit_si = clone;
        explicit_si.expr="(+ (* 1000 decel) (* (* 0.001 drag) (* (* (* 1000 speed) (* 1000 speed)) (safe_exp (neg (/ (* 1000 altitude) (* 1000 scale_height)))))))".into();
        assert_eq!(
            canonical,
            canonical_system(&[explicit_si], 64)
                .unwrap()
                .canonical_system
        );
        let mut no_domain = original.clone();
        for t in no_domain.symbols.values_mut() {
            t.positive = false;
        }
        assert!(canonical_system(&[no_domain], 64)
            .unwrap_err()
            .contains("domain-proven nonzero denominator"));
        let mut bad_scale = original;
        bad_scale.symbols.get_mut("H").unwrap().unit_scale = [0, 1];
        assert!(canonical_system(&[bad_scale], 64).is_err());
    }

    #[test]
    fn dimensional_errors_fail_closed_and_si_order_includes_all_seven_bases() {
        let tag = CanonicalSymbol {
            role: WireRole::ControlParam,
            si_dim: [1, 1, 1, 1, 1, 1, 1],
            semantic_tag: 0,
            unit_scale: [1, 1],
            positive: true,
            domain: None,
        };
        let symbols = BTreeMap::from([("reference".into(), tag.clone()), ("x".into(), tag)]);
        let eq = TaggedCselEq {
            expr: "(- x reference)".into(),
            symbols,
        };
        assert!(canonical_system(&[eq.clone()], 64).is_ok());
        for expression in ["(safe_exp x)", "(/ x 0)", "(+ x 1)", "(sqrt x)", "missing"] {
            let bad = TaggedCselEq {
                expr: expression.into(),
                ..eq.clone()
            };
            assert!(canonical_system(&[bad], 64).is_err(), "{expression}");
        }
        assert!(canonical_system(&[], 64).is_ok());
        assert!(canonical_system(&[], 0).is_err());
        let unguarded = TaggedCselEq {
            expr: "(/ x x)".into(),
            symbols: BTreeMap::from([(
                "x".into(),
                CanonicalSymbol {
                    role: WireRole::DynamicState,
                    si_dim: [0; 7],
                    semantic_tag: 0,
                    unit_scale: [1, 1],
                    positive: false,
                    domain: None,
                },
            )]),
        };
        assert!(canonical_system(&[unguarded], 64)
            .unwrap_err()
            .contains("nonzero denominator"));
    }
}
