use core::panic;

use super::*;

pub trait ComputeDirection {
    fn compute_direction(
        &mut self,
        eval_x_k: &FuncEvalMultivariate,
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
pub trait OptimizationSolver: ComputeDirection {
    fn xk(&self) -> &DVector<Floating>;
    fn xk_mut(&mut self) -> &mut DVector<Floating>;
    fn k(&self) -> &usize;
    fn k_mut(&mut self) -> &mut usize;
    fn has_converged(&self, eval_x_k: &FuncEvalMultivariate) -> bool;

    fn setup(&mut self) {}

    fn evaluate_x_k(
        &mut self,
        oracle: &impl Fn(&DVector<Floating>) -> FuncEvalMultivariate,
    ) -> Result<FuncEvalMultivariate, SolverError> {
        let eval_x_k = oracle(self.xk());
        if eval_x_k.f().is_nan() || eval_x_k.f().is_infinite() {
            error!(target: "solver","Minimization completed: next iterate is out of domain");
            return Err(SolverError::OutOfDomain);
        }
        Ok(eval_x_k)
    }

    fn update_next_iterate<LS: LineSearch>(
        &mut self,
        line_search: &mut LS,
        eval_x_k: &FuncEvalMultivariate, //eval_x_k: &FuncEvalMultivariate,
        oracle: &impl Fn(&DVector<Floating>) -> FuncEvalMultivariate,
        direction: &DVector<Floating>,
        max_iter_line_search: usize,
    ) -> Result<(), SolverError> {
        let step = line_search.compute_step_len(
            self.xk(),
            eval_x_k,
            direction,
            &oracle,
            max_iter_line_search,
        );

        let next_iterate = self.xk() + step * direction;
        *self.xk_mut() = next_iterate;

        Ok(())
    }

    fn minimize<LS: LineSearch>(
        &mut self,
        line_search: &mut LS,
        oracle: impl Fn(&DVector<Floating>) -> FuncEvalMultivariate,
        max_iter_solver: usize,
        max_iter_line_search: usize,
    ) -> Result<(), SolverError> {
        *self.k_mut() = 0;

        self.setup();

        while &max_iter_solver > self.k() {
            let eval_x_k = self.evaluate_x_k(&oracle)?;

            if self.has_converged(&eval_x_k) {
                info!(
                    target: "solver",
                    "Minimization completed: convergence in {} iterations",
                    self.k()
                );
                return Ok(());
            }

            let direction = self.compute_direction(&eval_x_k)?;
            debug!(target: "solver","Gradient: {:?}, Direction: {:?}", eval_x_k.g(), direction);
            self.update_next_iterate(
                line_search,
                &eval_x_k,
                &oracle,
                &direction,
                max_iter_line_search,
            )?;

            debug!(target: "solver","Iterate: {:?}", self.xk());
            debug!(target: "solver","Function eval: {:?}", eval_x_k);

            *self.k_mut() += 1;
        }
        warn!(target: "solver","Minimization completed: max iter reached during minimization");
        Err(SolverError::MaxIterReached)
    }
}

pub trait HasBounds {
    fn lower_bound(&self) -> &DVector<Floating>;
    fn upper_bound(&self) -> &DVector<Floating>;
    fn set_lower_bound(&mut self, lower_bound: DVector<Floating>);
    fn set_upper_bound(&mut self, upper_bound: DVector<Floating>);
}

pub trait HasProjectedGradient: OptimizationSolver + HasBounds {
    fn projected_gradient(&self, eval: &FuncEvalMultivariate) -> DVector<Floating> {
        let mut proj_grad = eval.g().clone();
        for (i, x) in self.xk().iter().enumerate() {
            if (x == &self.lower_bound()[i] && proj_grad[i] > 0.0)
                || (x == &self.upper_bound()[i] && proj_grad[i] < 0.0)
            {
                proj_grad[i] = 0.0;
            }
        }
        proj_grad
    }
}

//Blanket implementation for all optimization solvers that have bounds
impl<T> HasProjectedGradient for T where T: OptimizationSolver + HasBounds {}
