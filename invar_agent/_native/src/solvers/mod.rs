//! Native Rust numerical solver suite for Invar-Agent vNext.
//!
//! - `ode`: RK4, DOPRI5 (Hairer–Wanner PI + stiffness sensor), MPRK22(1.0)
//!   (strictly rejecting alpha != 1.0, disjoint pools, 2-cycle sensor, zero-start -> ROS2),
//!   PartitionedMprkRos2, ROS2(gamma_+), Rosenbrock23 (+h*d*f_t), SDIRK2, ImplicitMidpoint.
//! - `mc`: Stateless SplitMix64 CRN, Welford-CGL merge, Maurer-Pontil Empirical Bernstein,
//!   pure-math Lentz CF Clopper-Pearson (< 1e-12 vs SciPy), Vehtari rank-normalized split-R_hat & ESS.
//! - `fixed_point`: Deuflhard affine-invariant damped Newton, FD Jacobian cross-check,
//!   Krawczyk enclosure, 3-tier stability certifier (Routh-Hurwitz, Bauer-Fike, Lyapunov).
//! - `sweep`: Parallel Rayon control-parameter sweep, signed obligation slack, adaptive bisection,
//!   and SHBT open-boundary algebraic closure.

pub mod fixed_point;
pub mod mc;
pub mod ode;
pub mod sweep;
