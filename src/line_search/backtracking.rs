// Inexact line search described in chapter 9.2 of Boyd's convex optimization book
use super::*;
pub struct BackTracking {
    c1: Floating,   // recommended: [0.01, 0.3]
    beta: Floating, // recommended: [0.1, 0.8]
    lower_bound: Option<DVector<Floating>>,
    upper_bound: Option<DVector<Floating>>,
}
impl BackTracking {
    pub fn new(c1: Floating, beta: Floating) -> Self {
        BackTracking {
            c1,
            beta,
            lower_bound: None,
            upper_bound: None,
        }
    }
    pub fn with_lower_bound(mut self, lower_bound: DVector<Floating>) -> Self {
        self.lower_bound = Some(lower_bound);
        self
    }
    pub fn with_upper_bound(mut self, upper_bound: DVector<Floating>) -> Self {
        self.upper_bound = Some(upper_bound);
        self
    }

    fn sufficient_decrease_with_bounds(
        &self,
        x0: &DVector<Floating>,
        x: &DVector<Floating>,
        f0: &Floating,
        f: &Floating,
        t: &Floating, //step len
    ) -> bool {
        let diff = x - x0;
        // debug!(target: "backtracking line search", "Modified Armijo rule: f0: {:?}, f: {:?}, t: {:?}, x0: {:?}, x: {:?}, diff: {:?}", f0, f, t, x0, x, diff);
        f - f0 <= (-&self.c1 / t) * diff.dot(&diff)
    }
}

impl SufficientDecreaseCondition for BackTracking {
    fn c1(&self) -> Floating {
        self.c1
    }
}

impl LineSearch for BackTracking {
    fn compute_step_len(
        &self,
        x_k: &DVector<Floating>,
        direction_k: &DVector<Floating>,
        oracle: &impl Fn(&DVector<Floating>) -> FuncEvalMultivariate,
        max_iter: usize,
    ) -> Floating {
        let mut t = 1.0;
        let mut i = 0;
        let eval = oracle(x_k);
        let has_bounds = self.lower_bound.is_some() || self.upper_bound.is_some();

        while max_iter > i {
            let mut x_kp1 = x_k + t * direction_k;
            // if it has bounds, we project the next iterate onto the feasible set
            if has_bounds {
                if let Some(lower_bound) = &self.lower_bound {
                    x_kp1 = x_kp1.sup(lower_bound);
                }
                if let Some(upper_bound) = &self.upper_bound {
                    x_kp1 = x_kp1.inf(upper_bound);
                }
            }
            let eval_kp1 = oracle(&x_kp1);

            // we check if we are out of domain
            if eval_kp1.f().is_nan() || eval_kp1.f().is_infinite() {
                warn!(target: "backtracking line search", "Step size too big: next iterate is out of domain. Decreasing step by beta ({:?})", x_kp1);
                t *= self.beta;
                continue;
            }

            if has_bounds {
                if self.sufficient_decrease_with_bounds(x_k, &x_kp1, eval.f(), eval_kp1.f(), &t) {
                    debug!(target: "backtracking line search", "Modified Armijo rule met. Exiting with step size: {:?} at iteration {:?}", t, i);
                    return t;
                }
            } else if self.sufficient_decrease_with_directional_derivative(
                eval.f(),
                eval_kp1.f(),
                eval.g(),
                direction_k,
            ) {
                debug!(target: "backtracking line search", "Sufficient decrease condition met. Exiting with step size: {:?}", t);
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
        let max_iter = 1000;
        //here we define a rough gradient descent method that uses backtracking line search
        let mut k = 1;
        let mut iterate = DVector::from(vec![180.0, 152.0]);
        let backtracking = BackTracking::new(1e-4, 0.5);
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
            let t = <BackTracking as LineSearch>::compute_step_len(
                &backtracking,
                &iterate,
                &direction,
                &f_and_g,
                max_iter,
            );
            //we perform the update
            iterate += t * direction;
            k += 1;
        }
        println!("Iterate: {:?}", iterate);
        println!("Function eval: {:?}", f_and_g(&iterate));
        assert!((iterate[0] - 0.0).abs() < 1e-6);
        info!("Test took {} iterations", k);
    }
}
