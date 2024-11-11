use super::*;

pub trait ComputeDirection {
    fn compute_direction(&mut self, eval: &FuncEval) -> DVector<Floating>;
}

//Template pattern for solvers
pub trait Solver: ComputeDirection {
    type LS: LineSearch;
    fn line_search(&self) -> &Self::LS;
    fn xk(&self) -> &DVector<Floating>;
    fn xk_mut(&mut self) -> &mut DVector<Floating>;
    fn k(&self) -> &usize;
    fn k_mut(&mut self) -> &mut usize;
    fn has_converged(&self, eval: &FuncEval) -> bool;
    fn evaluation_hook(&mut self, _eval: &FuncEval) {}
    fn direction_hook(&mut self, _eval: &FuncEval, _direction: &DVector<Floating>) {}
    fn step_hook(
        &mut self,
        _eval: &FuncEval,
        _direction: &DVector<Floating>,
        _step: &Floating,
        _next_iterate: &DVector<Floating>,
        _oracle: &impl Fn(&DVector<Floating>) -> FuncEval,
    ) {
    }

    fn minimize(&mut self, oracle: impl Fn(&DVector<Floating>) -> FuncEval, max_iter: usize) {
        *self.k_mut() = 0;
        while &max_iter > self.k() {
            let eval = oracle(self.xk());
            self.evaluation_hook(&eval);
            if self.has_converged(&eval) {
                info!(
                    "Minimization completed: convergence in {} iterations",
                    self.k()
                );
                return;
            }
            let direction = self.compute_direction(&eval);
            self.direction_hook(&eval, &direction);
            let step =
                self.line_search()
                    .compute_step_len(self.xk(), &direction, &oracle, max_iter);
            let next_iterate = self.xk() + step * &direction;
            self.step_hook(&eval, &direction, &step, &next_iterate, &oracle);
            *self.xk_mut() = next_iterate;
            *self.k_mut() += 1;
        }
        warn!("Minimization completed: max iter reached during");
    }
}
