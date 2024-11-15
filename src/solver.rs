use super::*;

pub trait ComputeDirection {
    fn compute_direction(&mut self, eval: &FuncEvalMultivariate) -> DVector<Floating>;
}

#[derive(thiserror::Error, Debug)]
pub enum SolverError {
    #[error("Max iter reached")]
    MaxIterReached,
    #[error("Out of domain")]
    OutOfDomain,
}

//Template pattern for solvers
pub trait Solver: ComputeDirection {
    type LS: LineSearch;
    fn line_search(&self) -> &Self::LS;
    fn line_search_mut(&mut self) -> &mut Self::LS;
    fn xk(&self) -> &DVector<Floating>;
    fn xk_mut(&mut self) -> &mut DVector<Floating>;
    fn k(&self) -> &usize;
    fn k_mut(&mut self) -> &mut usize;
    fn has_converged(&self, eval: &FuncEvalMultivariate) -> bool;
    fn evaluation_hook(&mut self, _eval: &FuncEvalMultivariate) {}
    fn direction_hook(&mut self, _eval: &FuncEvalMultivariate, _direction: &DVector<Floating>) {}
    fn step_hook(
        &mut self,
        _eval: &FuncEvalMultivariate,
        _direction: &DVector<Floating>,
        _step: &Floating,
        _next_iterate: &DVector<Floating>,
        _oracle: &impl Fn(&DVector<Floating>) -> FuncEvalMultivariate,
    ) {
    }

    fn minimize(
        &mut self,
        oracle: impl Fn(&DVector<Floating>) -> FuncEvalMultivariate,
        max_iter_solver: usize,
        max_iter_line_search: usize,
    ) -> Result<(), SolverError> {
        *self.k_mut() = 0;
        while &max_iter_solver > self.k() {
            let eval = oracle(self.xk());
            if eval.f().is_nan() || eval.f().is_infinite() {
                error!(target: "solver","Minimization completed: next iterate is out of domain");
                return Err(SolverError::OutOfDomain);
            }
            self.evaluation_hook(&eval);
            if self.has_converged(&eval) {
                info!(
                    target: "solver",
                    "Minimization completed: convergence in {} iterations",
                    self.k()
                );
                return Ok(());
            }
            let direction = self.compute_direction(&eval);
            self.direction_hook(&eval, &direction);
            let step = self.line_search().compute_step_len(
                self.xk(),
                &direction,
                &oracle,
                max_iter_line_search,
            );
            let next_iterate = self.xk() + step * &direction;
            self.step_hook(&eval, &direction, &step, &next_iterate, &oracle);
            *self.xk_mut() = next_iterate;
            *self.k_mut() += 1;
        }
        warn!(target: "solver","Minimization completed: max iter reached during");
        Err(SolverError::MaxIterReached)
    }
}
