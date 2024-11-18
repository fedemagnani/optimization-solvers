use super::*;
pub mod backtracking;
pub use backtracking::*;
pub mod morethuente;
pub use morethuente::*;
pub trait LineSearch {
    fn compute_step_len(
        &self,
        x_k: &DVector<Floating>,         // current iterate
        direction_k: &DVector<Floating>, // direction of the ray along which we are going to search
        oracle: &impl Fn(&DVector<Floating>) -> FuncEvalMultivariate, // oracle
        max_iter: usize, // maximum number of iterations during line search (if direction update is costly, set this high to perform more exact line search)
    ) -> Floating; //returns the scalar step size
}

pub trait SufficientDecreaseCondition {
    fn c1(&self) -> Floating; // Armijo senstivity
    fn sufficient_decrease_with_directional_derivative(
        &self,
        f_k: &Floating,
        f_kp1: &Floating,
        grad_k: &DVector<Floating>,
        direction_k: &DVector<Floating>,
    ) -> bool {
        f_kp1 - f_k <= self.c1() * grad_k.dot(direction_k)
    }
    fn sufficient_decrease_with_norm_of_diff(
        &self,
        f_k: &Floating,
        f_kp1: &Floating,
        x_k: &DVector<Floating>,
        x_kp1: &DVector<Floating>,
        step_len: &Floating,
    ) -> bool {
        let diff = x_kp1 - x_k;
        f_kp1 - f_k <= -self.c1() / step_len * diff.dot(&diff)
    }
}

pub trait CurvatureCondition {
    fn c2(&self) -> Floating; // curvature senstivity
    fn curvature_condition(
        &self,
        grad_k: &DVector<Floating>,
        grad_kp1: &DVector<Floating>,
        direction_k: &DVector<Floating>,
    ) -> bool {
        grad_kp1.dot(direction_k) >= self.c2() * grad_k.dot(direction_k)
    }
    fn strong_curvature_condition(
        &self,
        grad_k: &DVector<Floating>,
        grad_kp1: &DVector<Floating>,
        direction_k: &DVector<Floating>,
    ) -> bool {
        grad_kp1.dot(direction_k).abs() <= self.c2() * grad_k.dot(direction_k).abs()
    }
}

pub trait WolfeConditions: SufficientDecreaseCondition + CurvatureCondition {
    fn wolfe_conditions_with_directional_derivative(
        &self,
        f_k: &Floating,
        f_kp1: &Floating,
        grad_k: &DVector<Floating>,
        grad_kp1: &DVector<Floating>,
        direction_k: &DVector<Floating>,
    ) -> bool {
        self.sufficient_decrease_with_directional_derivative(f_k, f_kp1, grad_k, direction_k)
            && self.curvature_condition(grad_k, grad_kp1, direction_k)
    }
    fn strong_wolfe_conditions_with_directional_derivative(
        &self,
        f_k: &Floating,
        f_kp1: &Floating,
        grad_k: &DVector<Floating>,
        grad_kp1: &DVector<Floating>,
        direction_k: &DVector<Floating>,
    ) -> bool {
        self.sufficient_decrease_with_directional_derivative(f_k, f_kp1, grad_k, direction_k)
            && self.strong_curvature_condition(grad_k, grad_kp1, direction_k)
    }
}
// Blanket implementation for WolfeConditions
impl<T> WolfeConditions for T where T: SufficientDecreaseCondition + CurvatureCondition {}
