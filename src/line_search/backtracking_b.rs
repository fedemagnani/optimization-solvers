// Inexact line search described in chapter 9.2 of Boyd's convex optimization book, adapted for solvers handling constraints (such as Projected Gradient Descent)

use super::*;
pub struct BackTrackingB {
    c1: Floating,   // recommended: [0.01, 0.3]
    beta: Floating, // recommended: [0.1, 0.8]
    lower_bound: DVector<Floating>,
    upper_bound: DVector<Floating>,
}
impl BackTrackingB {
    pub fn new(
        c1: Floating,
        beta: Floating,
        lower_bound: DVector<Floating>,
        upper_bound: DVector<Floating>,
    ) -> Self {
        BackTrackingB {
            c1,
            beta,
            lower_bound,
            upper_bound,
        }
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
        f - f0 <= (-&self.c1 / t) * diff.dot(&diff)
    }
}

impl HasBounds for BackTrackingB {
    fn lower_bound(&self) -> &DVector<Floating> {
        &self.lower_bound
    }
    fn upper_bound(&self) -> &DVector<Floating> {
        &self.upper_bound
    }
    fn set_lower_bound(&mut self, lower_bound: DVector<Floating>) {
        self.lower_bound = lower_bound;
    }
    fn set_upper_bound(&mut self, upper_bound: DVector<Floating>) {
        self.upper_bound = upper_bound;
    }
}

impl LineSearch for BackTrackingB {
    fn compute_step_len(
        &mut self,
        x_k: &DVector<Floating>,
        eval_x_k: &FuncEvalMultivariate,
        direction_k: &DVector<Floating>,
        oracle: &impl Fn(&DVector<Floating>) -> FuncEvalMultivariate,
        max_iter: usize,
    ) -> Floating {
        let mut t = 1.0;
        let mut i = 0;

        while max_iter > i {
            let x_kp1 = x_k + t * direction_k;
            // we project the next iterate onto the feasible set
            let x_kp1 = x_kp1.box_projection(&self.lower_bound, &self.upper_bound);
            let eval_kp1 = oracle(&x_kp1);
            // we check if we are out of domain
            if eval_kp1.f().is_nan() || eval_kp1.f().is_infinite() {
                trace!(target: "backtracking_b line search", "Step size too big: next iterate is out of domain. Decreasing step by beta ({:?})", x_kp1);
                t *= self.beta;
                continue;
            }
            if self.sufficient_decrease_with_bounds(x_k, &x_kp1, eval_x_k.f(), eval_kp1.f(), &t) {
                trace!(target: "backtracking_b line search", "Modified Armijo rule met. Exiting with step size: {:?} at iteration {:?}", t, i);
                return t;
            }

            //if we are here, it means that the we still didn't meet the exit condition, so we decrease the step size accordingly
            t *= self.beta;
            i += 1;
        }
        trace!(target: "backtracking_b line search", "Max iter reached. Early stopping.");
        t
        // worst case scenario: t=0 (or t>0 but t<1 because of early stopping).
        // if t=0 we are not updating the iterate
        // if early stop triggered, we benefit from some image reduction but it is not enough to be considered satisfactory
    }
}
