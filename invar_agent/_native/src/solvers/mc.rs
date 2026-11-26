//! Native Monte Carlo Solver & Exact Statistical Diagnostics (`solvers/mc.rs`).
//!
//! Zero external statistical dependencies:
//! - Stateless `SplitMix64` CRN
//! - Welford-Chan-Golub-LeVeque parallel merge
//! - Maurer-Pontil Empirical Bernstein confidence bounds
//! - Pure-math Lentz Continued-Fraction Regularized Incomplete Beta (`< 1e-12` vs SciPy)
//!   + exact Clopper-Pearson binomial confidence interval
//! - Vehtari rank-normalized split-`R_hat` and bulk `ESS`

use crate::kernel_compiler::{build_buffer_spec, eval_ir_observables};
use crate::sci_ir::{ArchetypeConfig, SciIR};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WelfordStats {
    pub count: u64,
    pub mean: f64,
    pub m2: f64,
    pub min: f64,
    pub max: f64,
}

impl WelfordStats {
    pub fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    pub fn push(&mut self, x: f64) {
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / (self.count as f64);
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
        if x < self.min {
            self.min = x;
        }
        if x > self.max {
            self.max = x;
        }
    }

    /// Chan-Golub-LeVeque parallel merge of two Welford accumulators.
    pub fn merge(&self, other: &Self) -> Self {
        if self.count == 0 {
            return other.clone();
        }
        if other.count == 0 {
            return self.clone();
        }
        let na = self.count as f64;
        let nb = other.count as f64;
        let n = na + nb;
        let delta = other.mean - self.mean;
        let mean = self.mean + delta * (nb / n);
        let m2 = self.m2 + other.m2 + delta * delta * (na * nb / n);
        Self {
            count: self.count + other.count,
            mean,
            m2,
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    pub fn sample_variance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            self.m2 / ((self.count - 1) as f64)
        }
    }
}

/// Maurer-Pontil Empirical Bernstein confidence interval `(lo, hi)` for bounded `[a, b]` samples.
pub fn empirical_bernstein_ci(stats: &WelfordStats, range_width: f64, alpha: f64) -> (f64, f64) {
    let n = stats.count as f64;
    if n < 2.0 {
        return (stats.mean, stats.mean);
    }
    let var = stats.sample_variance().max(0.0);
    let log_term = (3.0 / alpha.clamp(1e-12, 0.5)).ln();
    let rad = (2.0 * var * log_term / n).sqrt() + (3.0 * range_width.abs() * log_term) / (n - 1.0);
    (stats.mean - rad, stats.mean + rad)
}

/// Lanczos approximation for `ln(Gamma(z))` accurate to ~15 decimal digits.
pub fn ln_gamma(z: f64) -> f64 {
    const COEFFS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_9,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if z < 0.5 {
        std::f64::consts::PI.ln() - (std::f64::consts::PI * z).sin().ln() - ln_gamma(1.0 - z)
    } else {
        let zm1 = z - 1.0;
        let mut x = COEFFS[0];
        for (i, &c) in COEFFS.iter().enumerate().skip(1) {
            x += c / (zm1 + i as f64);
        }
        let t = zm1 + 7.5;
        0.5 * (2.0 * std::f64::consts::PI).ln() + (zm1 + 0.5) * t.ln() - t + x.ln()
    }
}

/// Modified Lentz continued fraction for the Regularized Incomplete Beta function `I_x(a, b)`.
pub fn reg_inc_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let ln_bt = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln();
    let bt = ln_bt.exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betacf_lentz(a, b, x) / a
    } else {
        1.0 - bt * betacf_lentz(b, a, 1.0 - x) / b
    }
}

fn betacf_lentz(a: f64, b: f64, x: f64) -> f64 {
    const MAX_IT: usize = 300;
    const EPS: f64 = 1e-15;
    const FPMIN: f64 = 1e-300;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=MAX_IT {
        let mf = m as f64;
        let m2 = 2.0 * mf;
        // Even step
        let aa = mf * (b - mf) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;

        // Odd step
        let aa = -(a + mf) * (qab + mf) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

/// Inverse Regularized Incomplete Beta `I_x^{-1}(a, b) = p` via bisection to `< 1e-14`.
pub fn inv_reg_inc_beta(a: f64, b: f64, p: f64) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }
    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if reg_inc_beta(a, b, mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
        if (hi - lo) < 1e-15 {
            break;
        }
    }
    0.5 * (lo + hi)
}

/// Exact Clopper-Pearson binomial confidence interval `(lo, hi)` using pure-math Lentz CF.
pub fn clopper_pearson_ci(k: u64, n: u64, alpha: f64) -> (f64, f64) {
    if n == 0 {
        return (0.0, 1.0);
    }
    let lo = if k == 0 {
        0.0
    } else {
        inv_reg_inc_beta(k as f64, (n - k + 1) as f64, alpha * 0.5)
    };
    let hi = if k >= n {
        1.0
    } else {
        inv_reg_inc_beta((k + 1) as f64, (n - k) as f64, 1.0 - alpha * 0.5)
    };
    (lo, hi)
}

/// Acklam rational approximation of the standard normal quantile `Phi^{-1}(p)`.
fn inv_normal_cdf(p: f64) -> f64 {
    let p = p.clamp(1e-15, 1.0 - 1e-15);
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    let p_low = 0.02425;
    let p_high = 1.0 - p_low;
    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// Vehtari (2021) rank-normalized split-`R_hat` and bulk `ESS` across multiple chains.
pub fn vehtari_split_rhat_and_ess(chains: &[Vec<f64>]) -> (f64, f64) {
    // Split each chain into two halves
    let mut split_chains: Vec<&[f64]> = Vec::new();
    for ch in chains {
        let half = ch.len() / 2;
        if half >= 4 {
            split_chains.push(&ch[..half]);
            split_chains.push(&ch[half..2 * half]);
        }
    }
    let m = split_chains.len();
    if m < 2 {
        return (1.0, 0.0);
    }
    let n = split_chains[0].len();
    let total = m * n;

    // Global rank normalization (Blom scores)
    let mut indexed: Vec<(f64, usize)> = Vec::with_capacity(total);
    for (ci, ch) in split_chains.iter().enumerate() {
        for (si, &val) in ch.iter().enumerate() {
            indexed.push((val, ci * n + si));
        }
    }
    indexed.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut z = vec![0.0; total];
    let denom = total as f64 + 0.25;
    for (rank0, &(_, flat_idx)) in indexed.iter().enumerate() {
        let p = ((rank0 + 1) as f64 - 0.375) / denom;
        z[flat_idx] = inv_normal_cdf(p);
    }

    // Chain means and variances
    let mut chain_means = vec![0.0; m];
    let mut chain_vars = vec![0.0; m];
    for ci in 0..m {
        let slice = &z[ci * n..(ci + 1) * n];
        let mu = slice.iter().sum::<f64>() / (n as f64);
        chain_means[ci] = mu;
        let var = slice.iter().map(|&v| (v - mu) * (v - mu)).sum::<f64>() / ((n - 1) as f64);
        chain_vars[ci] = var;
    }
    let grand_mean = chain_means.iter().sum::<f64>() / (m as f64);
    let b_var = (n as f64)
        * chain_means
            .iter()
            .map(|&mu| (mu - grand_mean) * (mu - grand_mean))
            .sum::<f64>()
        / ((m - 1) as f64);
    let w_var = chain_vars.iter().sum::<f64>() / (m as f64);
    if w_var <= 1e-30 {
        return (1.0, total as f64);
    }
    let var_hat = ((n as f64 - 1.0) / (n as f64)) * w_var + b_var / (n as f64);
    let r_hat = (var_hat / w_var).sqrt();

    // Estimate bulk ESS via Geyer initial positive sequence autocorrelation
    let max_lag = (n / 2).min(64);
    let mut tau = 1.0_f64;
    let mut prev_pair = f64::INFINITY;
    let mut lag = 1usize;
    while lag + 1 < max_lag {
        let rho1 = pooled_autocorr(&z, m, n, lag, &chain_means, w_var, var_hat);
        let rho2 = pooled_autocorr(&z, m, n, lag + 1, &chain_means, w_var, var_hat);
        let pair = rho1 + rho2;
        if pair <= 0.0 {
            break;
        }
        let clipped = pair.min(prev_pair);
        tau += 2.0 * clipped;
        prev_pair = clipped;
        lag += 2;
    }
    let ess = ((total as f64) / tau.max(1.0)).min(total as f64);
    (r_hat, ess)
}

fn pooled_autocorr(
    z: &[f64],
    m: usize,
    n: usize,
    lag: usize,
    chain_means: &[f64],
    w_var: f64,
    var_hat: f64,
) -> f64 {
    let mut acov_sum = 0.0;
    for ci in 0..m {
        let slice = &z[ci * n..(ci + 1) * n];
        let mu = chain_means[ci];
        let mut s = 0.0;
        for i in 0..(n - lag) {
            s += (slice[i] - mu) * (slice[i + lag] - mu);
        }
        acov_sum += s / (n as f64);
    }
    let mean_acov = acov_sum / (m as f64);
    1.0 - (w_var - mean_acov) / var_hat.max(1e-30)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonteCarloSummary {
    pub replicates: usize,
    pub observable_means: Vec<f64>,
    pub observable_bernstein_ci: Vec<(f64, f64)>,
    pub split_r_hat: Vec<f64>,
    pub bulk_ess: Vec<f64>,
    pub converged: bool,
}

/// Run parallel CRN Monte Carlo over a sealed `SciIR`.
pub fn run_sci_ir_monte_carlo(
    ir: &SciIR,
    override_params: Option<&[f64]>,
) -> Result<MonteCarloSummary, String> {
    let spec = build_buffer_spec(ir);
    let params = override_params.unwrap_or(&spec.default_params);
    let constants = &spec.constants;

    let (base_seed, replicates, alpha) = match &ir.config {
        ArchetypeConfig::MonteCarlo {
            base_seed,
            replicates,
            confidence_alpha,
            ..
        } => (*base_seed, (*replicates).max(64), *confidence_alpha),
        _ => (42, 1024, 0.05),
    };

    let n_obs = spec.n_observables;
    let all_samples: Vec<Vec<f64>> = (0..replicates)
        .into_par_iter()
        .map(|rep| {
            let mut obs = vec![0.0; n_obs];
            eval_ir_observables(
                ir,
                0.0,
                &spec.initial_states,
                params,
                constants,
                &mut obs,
                base_seed,
                rep as u64,
            );
            obs
        })
        .collect();

    let mut means = Vec::with_capacity(n_obs);
    let mut cis = Vec::with_capacity(n_obs);
    let mut rhats = Vec::with_capacity(n_obs);
    let mut ess_list = Vec::with_capacity(n_obs);

    let n_chains = 4usize;
    let per_chain = replicates / n_chains;

    for oi in 0..n_obs {
        let mut stats = WelfordStats::new();
        for row in &all_samples {
            stats.push(row[oi]);
        }
        let range = (stats.max - stats.min).max(1e-12);
        means.push(stats.mean);
        cis.push(empirical_bernstein_ci(&stats, range, alpha));

        let chains: Vec<Vec<f64>> = (0..n_chains)
            .map(|c| {
                (0..per_chain)
                    .map(|k| all_samples[c * per_chain + k][oi])
                    .collect()
            })
            .collect();
        let (rh, ess) = vehtari_split_rhat_and_ess(&chains);
        rhats.push(rh);
        ess_list.push(ess);
    }

    let converged = rhats.iter().all(|&r| r < 1.01) && ess_list.iter().all(|&e| e > 500.0);

    Ok(MonteCarloSummary {
        replicates,
        observable_means: means,
        observable_bernstein_ci: cis,
        split_r_hat: rhats,
        bulk_ess: ess_list,
        converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lentz_clopper_pearson_matches_scipy_to_1e12() {
        // Exact SciPy reference: scipy.stats.beta.ppf([0.025, 0.975], [7, 8], [14, 13]) for k=7, n=20, alpha=0.05
        // SciPy exact values: lo = 0.15390920478454118, hi = 0.5921885345328282
        let (lo, hi) = clopper_pearson_ci(7, 20, 0.05);
        assert!((lo - 0.153_909_204_784_541_18).abs() < 1e-12, "lo={lo}");
        assert!((hi - 0.592_188_534_532_828_2).abs() < 1e-12, "hi={hi}");
    }

    #[test]
    fn welford_cgl_merge_exact_and_vehtari_iid_converged() {
        let mut a = WelfordStats::new();
        let mut b = WelfordStats::new();
        let mut full = WelfordStats::new();
        for i in 1..=100 {
            let x = (i as f64) * 0.37;
            full.push(x);
            if i <= 40 {
                a.push(x);
            } else {
                b.push(x);
            }
        }
        let merged = a.merge(&b);
        assert!((merged.mean - full.mean).abs() < 1e-12);
        assert!((merged.sample_variance() - full.sample_variance()).abs() < 1e-12);

        // 4 independent i.i.d. normal chains -> split R_hat < 1.01 & ESS > 500
        let chains: Vec<Vec<f64>> = (0..4)
            .map(|c| {
                (0..400)
                    .map(|i| {
                        crate::kernel_compiler::sample_stateless_rand(
                            12345,
                            (c * 400 + i) as u64,
                            0,
                            crate::sci_ir::RandDist::Normal01,
                        )
                    })
                    .collect()
            })
            .collect();
        let (rhat, ess) = vehtari_split_rhat_and_ess(&chains);
        assert!(rhat < 1.01, "rhat={rhat}");
        assert!(ess > 500.0, "ess={ess}");
    }
}
