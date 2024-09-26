// Inexact line search described in chapter 9.2 of Boyd's convex optimization book
use super::*;
pub struct BackTracking {
    alpha: Floating, // recommended: [0.01, 0.3]
    beta: Floating,  // recommended: [0.1, 0.8]
}
impl BackTracking {
    pub fn new(alpha: Floating, beta: Floating) -> Self {
        BackTracking { alpha, beta }
    }

    // check if the change in the image has been lower than a proportion (alpha) of the directional derivative
    pub fn sufficient_decrease_condition(
        &self,
        f_k: Floating,
        f_kp1: Floating,
        grad_k: Array1<Floating>,
        direction_k: &Array1<Floating>,
    ) -> bool {
        f_kp1 - f_k <= self.alpha * grad_k.dot(direction_k)
    }
}

impl LineSearch for BackTracking {
    fn compute_step_len(
        &self,
        x_k: &Array1<Floating>,
        direction_k: &Array1<Floating>,
        f_and_g: impl Fn(&Array1<Floating>) -> (Floating, Array1<Floating>),
        max_iter: usize,
    ) -> Floating {
        let mut t = 1.0;
        let mut i = 0;
        while max_iter > i {
            let (f_k, grad_k) = f_and_g(x_k);
            let x_kp1 = x_k + t * direction_k;
            let (f_kp1, _) = f_and_g(&x_kp1);

            // we check if we are out of domain
            if f_kp1.is_nan() || f_kp1.is_infinite() {
                warn!(target: "backtracking line search", "Step size too big: next iterate is out of domain. Decreasing step by beta.");
                t *= self.beta;
                continue;
            }

            if self.sufficient_decrease_condition(f_k, f_kp1, grad_k, direction_k) {
                break;
            }

            //if we are here, it means that the we still didn't meet the exit condition, so we decrease the step size accordingly
            t *= self.beta;
            i += 1;
        }
        // worst case scenario: t=0 (or t>0 but t<1 because of early stopping).
        // if t=0 we are not updating the iterate
        // if early stop triggered, we benefit from some image reduction but it is not enough to be considered satisfactory
        t
    }
}

mod backtracking_tests {
    use super::*;
    use ndarray::arr1;
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
        let f_and_g = |x: &Array1<Floating>| -> (Floating, Array1<Floating>) {
            let f = 0.5 * (x[0].powi(2) + gamma * x[1].powi(2));
            let g = arr1(&[x[0], gamma * x[1]]);
            (f, g)
        };
        let max_iter = 1000;
        //here we define a rough gradient descent method that uses backtracking line search
        let mut k = 1;
        let mut iterate = arr1(&[180.0, 152.0]);
        let backtracking = BackTracking::new(1e-4, 0.5);
        let gradient_tol = 1e-12;

        while max_iter > k {
            debug!("Iterate: {:?}", iterate);
            let (_, grad_k) = f_and_g(&iterate);
            // we do a rough check on the squared norm of the gradient to verify convergence
            if grad_k.dot(&grad_k) < gradient_tol {
                warn!("Gradient norm is lower than tolerance. Convergence!.");
                break;
            }
            let direction = -&grad_k;
            let t = backtracking.compute_step_len(&iterate, &direction, f_and_g, max_iter);
            //we perform the update
            iterate = iterate + t * direction;
            k += 1;
        }

        assert!((iterate[0] - 0.0).abs() < 1e-6);
        info!("Test took {} iterations", k);
    }
}
