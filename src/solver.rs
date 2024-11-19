use core::panic;

use super::*;

pub trait ComputeDirection {
    fn compute_direction(
        &mut self,
        eval: &FuncEvalMultivariate,
    ) -> Result<DVector<Floating>, SolverError>;
}

#[derive(thiserror::Error, Debug)]
pub enum SolverError {
    #[error("Max iter reached")]
    MaxIterReached,
    #[error("Out of domain")]
    OutOfDomain,
}

//Template pattern for solvers. Methods that are already implemented can be freely overriden.
pub trait Solver: ComputeDirection {
    type LS: LineSearch;
    fn line_search(&self) -> &Self::LS;
    fn line_search_mut(&mut self) -> &mut Self::LS;
    fn xk(&self) -> &DVector<Floating>;
    fn xk_mut(&mut self) -> &mut DVector<Floating>;
    fn k(&self) -> &usize;
    fn k_mut(&mut self) -> &mut usize;
    fn has_converged(&self, eval: &FuncEvalMultivariate) -> bool;

    fn setup(&mut self) {}

    fn evaluate_x_k(
        &mut self,
        oracle: &impl Fn(&DVector<Floating>) -> FuncEvalMultivariate,
    ) -> Result<FuncEvalMultivariate, SolverError> {
        let eval = oracle(self.xk());
        if eval.f().is_nan() || eval.f().is_infinite() {
            error!(target: "solver","Minimization completed: next iterate is out of domain");
            return Err(SolverError::OutOfDomain);
        }
        Ok(eval)
    }

    fn update_next_iterate(
        &mut self,
        _: &FuncEvalMultivariate, //eval: &FuncEvalMultivariate,
        oracle: &impl Fn(&DVector<Floating>) -> FuncEvalMultivariate,
        direction: &DVector<Floating>,
        max_iter_line_search: usize,
    ) -> Result<(), SolverError> {
        let step = self.line_search().compute_step_len(
            self.xk(),
            direction,
            &oracle,
            max_iter_line_search,
        );

        let next_iterate = self.xk() + step * direction;
        *self.xk_mut() = next_iterate;

        Ok(())
    }

    fn minimize(
        &mut self,
        oracle: impl Fn(&DVector<Floating>) -> FuncEvalMultivariate,
        max_iter_solver: usize,
        max_iter_line_search: usize,
    ) -> Result<(), SolverError> {
        *self.k_mut() = 0;

        self.setup();

        while &max_iter_solver > self.k() {
            let eval = self.evaluate_x_k(&oracle)?;

            if self.has_converged(&eval) {
                info!(
                    target: "solver",
                    "Minimization completed: convergence in {} iterations",
                    self.k()
                );
                return Ok(());
            }

            let direction = self.compute_direction(&eval)?;
            debug!(target: "solver","Gradient: {:?}, Direction: {:?}", eval.g(), direction);
            self.update_next_iterate(&eval, &oracle, &direction, max_iter_line_search)?;

            debug!(target: "solver","Iterate: {:?}", self.xk());
            debug!(target: "solver","Function eval: {:?}", eval);

            *self.k_mut() += 1;
        }
        debug!(target: "solver","Minimization completed: max iter reached during minimization");
        Err(SolverError::MaxIterReached)
    }
}
