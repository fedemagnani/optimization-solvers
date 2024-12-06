use super::*;

pub struct NoSearch;
impl LineSearch for NoSearch {
    fn compute_step_len(
        &mut self,
        _: &DVector<Floating>,    // current iterate
        _: &FuncEvalMultivariate, // function evaluation at x_k
        _: &DVector<Floating>,    // direction of the ray along which we are going to search
        _: &impl Fn(&DVector<Floating>) -> FuncEvalMultivariate, // oracle
        _: usize, // maximum number of iterations during line search (if direction update is costly, set this high to perform more exact line search)
    ) -> Floating {
        1.0
    }
}
