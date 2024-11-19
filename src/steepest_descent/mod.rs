use super::*;
pub mod gradient_descent;
pub use gradient_descent::*;

pub mod coordinate_descent;
pub use coordinate_descent::*;

pub mod pnorm_descent;
pub use pnorm_descent::*;

pub mod spg;
pub use spg::*;

// All the algorithms in the family of steepest descent differ only in the way they compute the descent direction (i.e. they differ in the norm used so that the associated unit ball is the constraint set on which search the direction that minimizes the directional derivative at the current iterate. Typically this minimizer is a unit vector but any scaled version of the vector is good (the line search will adjust the direction later), so it's good supplying the rescaled version of the minimizer which has minimal computational cost).

// the family of steepest descent algorithms has (at most) linear convergence rate, and it's possible to see it by computing the trajectory of the upper bound of the log-suboptimality error ln(f(x_k)-p^*) where p^* is the optimal value of the problem. In particular, the convergence drops significantly if the upper bound of the condition number of the hessian matrix of the function is high (you can see it by solving the log-suboptimality error trajectory for the iteration number k). Recall that an upper bound on the condition number of the hessian can be derived by taking the ratio between the maximal and the minimal eigenvalue of the hessian matrix. This condition number can be also thought as the volume of the ellipsoid {x: x^T H x <= 1} where H is the hessian matrix of the function, which is always relatable to the volume of the euclidean unit ball gamma*sqrt{det (H^TH)} where gamma is the volume of the euclidean unit ball.The p-norm descent tries to tackle this issue by taking a Matrix P that proxies correctly the hessian matrix (i.e. its unit norm {x: x^T P x <= 1} is a good approximation of the sublevel sets of the function), and this adjustments decreases the condition number of P^{-0.5} H P^{-0.5} because it would resemble (more or less) the identity matrix. It's from this intuition that the newton and quasi-newton methods become more clear.

// The steepest descent version with bounds (paired with backtracking line search with bounds) is the implementation of algorithm 12.1 from [Neculai Andrei, 2022]

#[derive(derive_getters::Getters)]
pub struct SteepestDescent<T, S> {
    line_search: T,
    grad_tol: Floating,
    x: DVector<Floating>,
    k: usize,
    direction_strategy: S,
    lower_bound: Option<DVector<Floating>>,
    upper_bound: Option<DVector<Floating>>,
    pg: Option<DVector<Floating>>, // 12.14 from [Neculai Andrei, 2022]
}

impl<T, S> SteepestDescent<T, S> {
    pub fn with_upper_bound(mut self, upper_bound: DVector<Floating>) -> Self {
        self.upper_bound = Some(upper_bound);
        self
    }
    pub fn with_lower_bound(mut self, lower_bound: DVector<Floating>) -> Self {
        self.lower_bound = Some(lower_bound);
        self
    }
}

impl<T, S> ComputeDirection for SteepestDescent<T, S>
where
    S: ComputeDirection,
{
    fn compute_direction(
        &mut self,
        eval: &FuncEvalMultivariate,
    ) -> Result<DVector<Floating>, SolverError> {
        self.direction_strategy.compute_direction(eval)
    }
}

impl<T, S> Solver for SteepestDescent<T, S>
where
    T: LineSearch,
    S: ComputeDirection,
{
    type LS = T;
    fn line_search(&self) -> &Self::LS {
        &self.line_search
    }
    fn line_search_mut(&mut self) -> &mut Self::LS {
        &mut self.line_search
    }
    fn xk(&self) -> &DVector<Floating> {
        &self.x
    }
    fn xk_mut(&mut self) -> &mut DVector<Floating> {
        &mut self.x
    }
    fn k(&self) -> &usize {
        &self.k
    }
    fn k_mut(&mut self) -> &mut usize {
        &mut self.k
    }
    fn has_converged(&self, eval: &FuncEvalMultivariate) -> bool {
        // we verify that the norm of the gradient is below the tolerance. If the projected gradient is available, then it means that we are in a constrained optimization setting and we verify if it is zero since this is equivalent to first order conditions of optimality in the setting of optimization with simple bounds (Theorem 12.3 from [Neculai Andrei, 2022])
        let grad = match &self.pg {
            Some(pg) => pg,
            None => eval.g(),
        };
        // grad.dot(grad) < self.grad_tol
        // we compute the infinity norm of the gradient
        grad.iter()
            .fold(Floating::NEG_INFINITY, |acc, x| x.abs().max(acc))
            < self.grad_tol
    }

    fn update_next_iterate(
        &mut self,
        _: &FuncEvalMultivariate, //eval: &FuncEvalMultivariate,
        oracle: &impl Fn(&DVector<Floating>) -> FuncEvalMultivariate,
        direction: &DVector<Floating>,
        max_iter_line_search: usize,
    ) -> Result<(), SolverError> {
        let step =
            self.line_search()
                .compute_step_len(self.xk(), direction, oracle, max_iter_line_search);

        debug!(target: "steepest_descent", "ITERATE: {} + {} * {} = {}", self.xk(), step, direction, self.xk() + step * direction);

        let next_iterate = self.xk() + step * direction;

        // we project the next iterate and compute the projected gradient if bounds are present;
        let next_iterate = match (&self.lower_bound, &self.upper_bound) {
            (Some(lower), Some(upper)) => {
                let next_proj = next_iterate.box_projection(lower, upper);
                self.pg = Some(&next_proj - self.xk());
                next_proj
            }
            (Some(lower), None) => {
                let next_proj = next_iterate.sup(lower);
                self.pg = Some(&next_proj - self.xk());
                next_proj
            }
            (None, Some(upper)) => {
                let next_proj = next_iterate.inf(upper);
                self.pg = Some(&next_proj - self.xk());
                next_proj
            }
            (None, None) => next_iterate,
        };
        *self.xk_mut() = next_iterate;

        Ok(())
    }
}
