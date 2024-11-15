use core::panic;

// Inexact line search described in chapter 9.2 of Boyd's convex optimization book
use super::*;
pub struct ProjectedBackTracking {
    alpha: Floating, // recommended: [0.01, 0.3]
    beta: Floating,  // recommended: [0.1, 0.8]
    lower_bound: DVector<Floating>,
    upper_bound: DVector<Floating>,
}
impl ProjectedBackTracking {
    pub fn new(
        alpha: Floating,
        beta: Floating,
        lower_bound: DVector<Floating>,
        upper_bound: DVector<Floating>,
    ) -> Self {
        ProjectedBackTracking {
            alpha,
            beta,
            lower_bound,
            upper_bound,
        }
    }

    // check if the change in the image has been lower than a proportion (alpha) of the directional derivative
    // known also as Armijo condition
    pub fn sufficient_decrease_condition(
        &self,
        f_k: &Floating,
        f_kp1: &Floating,
        x_k: &DVector<Floating>,
        x_kp1: &DVector<Floating>,
        step: &Floating,
    ) -> bool {
        let diff = x_kp1 - x_k;
        let norm_squared_diff = diff.dot(&diff);
        f_kp1 - f_k <= -self.alpha / step * norm_squared_diff
    }
}

impl LineSearch for ProjectedBackTracking {
    fn compute_step_len(
        &self,
        x_k: &DVector<Floating>,
        direction_k: &DVector<Floating>,
        oracle: &impl Fn(&DVector<Floating>) -> FuncEvalMultivariate,
        max_iter: usize,
    ) -> Floating {
        let mut t = 1.0;
        let mut i = 0;

        while max_iter > i {
            let eval = oracle(x_k);
            let x_kp1 = x_k + t * direction_k;

            //differently from the backtracking line search, we project the next iterate into the feasible domain
            let x_kp1 = x_kp1.box_projection(&self.lower_bound, &self.upper_bound);

            let eval_kp1 = oracle(&x_kp1);

            // we check if we are out of domain
            if eval_kp1.f().is_nan() || eval_kp1.f().is_infinite() {
                warn!(target: "backtracking line search", "Step size too big: next iterate is out of domain. Decreasing step by beta ({:?})", x_kp1);

                t *= self.beta;
                continue;
            }

            if self.sufficient_decrease_condition(eval.f(), eval_kp1.f(), x_k, &x_kp1, &t) {
                return t;
            }

            //if we are here, it means that the we still didn't meet the exit condition, so we decrease the step size accordingly
            t *= self.beta;
            i += 1;
        }
        warn!(target: "backtracking line search", "Max iter reached. Early stopping.");
        t
        // worst case scenario: t=0 (or t>0 but t<1 because of early stopping).
        // if t=0 we are not updating the iterate
        // if early stop triggered, we benefit from some image reduction but it is not enough to be considered satisfactory
    }
}

mod backtracking_tests {
    use super::*;
    use nalgebra::DVector;
    use tracing::info;

    #[test]
    pub fn test_backtracking() {
        std::env::set_var("RUST_LOG", "info");

        // in this example the objecive function has constant hessian, thus its condition number doesn't change on different points.
        // Recall that in gradient descent method, the upper bound of the log error is positive function of the upper bound of condition number of the hessian (ratio between max and min eigenvalue).
        // This causes poor performance when the hessian is ill conditioned
        let tracer = Tracer::default()
            .with_stdout_layer(Some(LogFormat::Normal))
            .build();
        let gamma = 90.0;
        let f_and_g = |x: &DVector<Floating>| -> FuncEvalMultivariate {
            let f = 0.5 * (x[0].powi(2) + gamma * x[1].powi(2));
            let g = DVector::from(vec![x[0], gamma * x[1]]);
            (f, g).into()
        };
        let max_iter = 1000000;
        //here we define a rough gradient descent method that uses backtracking line search
        let mut k = 1;
        let mut iterate = DVector::from(vec![180.0, 152.0]);
        let lower_bound: DVector<Floating> = vec![1., 1.].into();
        let upper_bound: DVector<Floating> = vec![Floating::INFINITY, Floating::INFINITY].into();
        let backtracking =
            ProjectedBackTracking::new(1e-4, 0.5, lower_bound.clone(), upper_bound.clone());
        let gradient_tol = 1e-12;

        while max_iter > k {
            debug!("Iterate: {:?}", iterate);
            let eval = f_and_g(&iterate);
            // we do a rough check on the squared norm of the gradient to verify convergence
            if eval.g().dot(eval.g()) < gradient_tol {
                warn!("Gradient norm is lower than tolerance. Convergence!.");
                break;
            }
            let direction = -eval.g();
            let t = <ProjectedBackTracking as LineSearch>::compute_step_len(
                &backtracking,
                &iterate,
                &direction,
                &f_and_g,
                max_iter,
            );
            // As exit condition, we take the infinity norm of the difference between the projected iterate and the next iterate
            //we perform the update
            let cached_iterate = iterate.clone();
            iterate += t * direction;
            iterate = iterate.box_projection(&lower_bound, &upper_bound);
            if (cached_iterate - &iterate)
                .iter()
                .fold(0.0f64, |acc, x| acc.max(x.abs()))
                < gradient_tol
            {
                warn!("Infinity norm of iterate difference is lower than tolerance. Convergence!.");
                break;
            }
            k += 1;
        }
        println!("Iterate: {:?}", iterate);
        println!("Function eval: {:?}", f_and_g(&iterate));
        info!("Test took {} iterations", k);
    }
}
