use super::*;
pub mod backtracking;
pub use backtracking::*;

pub trait LineSearch {
    fn compute_step_len(
        &self,
        x_k: &DVector<Floating>,                          // current iterate
        direction_k: &DVector<Floating>, // direction of the ray along which we are going to search
        oracle: &impl Fn(&DVector<Floating>) -> FuncEval, // oracle
        max_iter: usize, // maximum number of iterations during line search (if direction update is costly, set this high to perform more exact line search)
    ) -> Floating; //returns the scalar step size
}
