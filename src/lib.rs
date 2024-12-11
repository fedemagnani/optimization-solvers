#[cfg(feature = "lbfgsb")]
use lbfgsb_sys::lbfgsb as ffi;
#[cfg(feature = "lbfgsb")]
use lbfgsb_sys::string as ffi_string;
use nalgebra::{DMatrix, DVector};
#[cfg(feature = "lbfgsb")]
use std::ffi::CStr;

use tracing::{debug, error, info, trace, warn};
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::{
    fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer, Registry,
};

// Deriving new solvers for STRONGLY CONVEX functions in UNCONSTRAINED OPTIMIZATION setting (which assume EXACT LINE SEARCH, so a bit more optimistic than the general case in which you use inexact line search):
// - Define a new line search method (or use existing one like backtracking)
// - Define a new direction update method (or use existing one like steepest descent)
// - Since the objective is strongly convex, there is a fixed range [m,M] in which the hessian of the function lives (at any point)
// - Considering the upper bound of the hessian, assume EXACT line search minimizing the rhs of the inequality f(x_k + t*d_k) <= f(x_k) + t*grad_k.dot(d_k) + t^2*M/m
//      - If instead you have bounds of the step length of a certain inexact line search (typically you infer them from the exit condition of the line search), substitute the step length with the min of the possible alternatives
//      - notice that this is the simple inequality f(y) >= f(x) + grad_f(x).dot(y-x) + 1/2 * (y-x).dot(H(x).(y-x)) for y=x+t*d substituting the hessian with its upper bound (so we revert the inequality sign)
// - Once that you have an optimal step length, inject it in the equation
// - Using some known equalities get rid of the gradient of the function
// - Put -p^* on both side of the equation (so that you can build the suboptimal error E_k = f(x_k) - p^*)
// - Describe the inequality as a Differential Inequality in discrete time (E_{k+1} <= f(E_k)) and compute the general solution
//      - The general solution draws the trajectory of the upper bound of the suboptimal error:
//      - first check you want to do is that it converges asymptotically to zero
//      - Define a tolerance level "q" and set the general solution <= q. Apply some transformations if needed to solve the inequality for the iteration number k: this gives you a lower bound of the number of iterations needed to achieve the desired tolerance. Typically this lower bound is increasing function of suboptimality of the initial point (f(x_0) - p^*) and the upper bound of the condition number of the hessian (M/m). Of course it should be decreasing of the tolerance level q (the more tolerance you have, the lower is the number of required iterations to achieve that convergence).
//      - From the inequality of the general solution, apply some transformations to verify what is the convergence rate of the algorithm (for example in the gradient descent if you apply the log you come up with a function which is linear in k, describing a linear convergence rate).

pub mod tracer;
pub use tracer::*;

pub mod ls_solver;
pub use ls_solver::*;

pub mod func_eval;
pub use func_eval::*;

pub mod line_search;
pub use line_search::*;

pub mod number;
pub use number::*;

pub mod quasi_newton {
    use super::*;
    pub mod bfgs;
    pub use bfgs::*;
    pub mod bfgs_b;
    pub use bfgs_b::*;
    pub mod broyden;
    pub use broyden::*;
    pub mod broyden_b;
    pub use broyden_b::*;
    pub mod dfp;
    pub use dfp::*;
    pub mod dfp_b;
    pub use dfp_b::*;
    pub mod sr1;
    pub use sr1::*;
    pub mod sr1_b;
    pub use sr1_b::*;

    #[cfg(feature = "lbfgsb")]
    pub mod lbfgsb;
    #[cfg(feature = "lbfgsb")]
    pub use lbfgsb::*;
}
pub use quasi_newton::*;

pub mod steepest_descent {
    use super::*;
    pub mod gradient_descent;
    pub use gradient_descent::*;

    pub mod coordinate_descent;
    pub use coordinate_descent::*;

    pub mod pnorm_descent;
    pub use pnorm_descent::*;

    pub mod spg;
    pub use spg::*;

    pub mod projected_gradient_descent;
    pub use projected_gradient_descent::*;
}

pub use steepest_descent::*;

pub mod newton;
pub use newton::*;

pub mod online {
    use super::*;
    pub mod osgmg;
    pub use osgmg::*;
}
pub use online::*;

pub use line_search::*;

pub mod plotter_3d;
pub use plotter_3d::*;
