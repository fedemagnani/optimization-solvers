use super::*;

// The projected gradient method is a simple naturalization of the steepest descent method in the setting of optimization with simple bounds. In this context, I've implemented algorithm 12.1 from [Neculai Andrei, 2022]

#[derive(derive_getters::Getters)]
pub struct ProjectedGradientDescent<LS> {
    line_search: LS,
    grad_tol: Floating,
    x: DVector<Floating>,
    k: usize,
    lower_bound: DVector<Floating>,
    upper_bound: DVector<Floating>,
}

impl<LS> ProjectedGradientDescent<LS>
where
    LS: LineSearch + HasBounds,
{
    pub fn new(
        line_search: LS,
        grad_tol: Floating,
        x0: DVector<Floating>,
        lower_bound: DVector<Floating>,
        upper_bound: DVector<Floating>,
    ) -> Self {
        let x0 = x0.box_projection(&lower_bound, &upper_bound);
        // let
        // let pg = DVector::zeros(x0.len());
        Self {
            line_search,
            grad_tol,
            x: x0,
            k: 0,
            lower_bound,
            upper_bound,
            // pg,
        }
    }

    pub fn projected_gradient(&self, eval: &FuncEvalMultivariate) -> DVector<Floating> {
        let mut proj_grad = eval.g().clone();
        for (i, x) in self.xk().iter().enumerate() {
            if (x == &self.lower_bound[i] && proj_grad[i] > 0.0)
                || (x == &self.upper_bound[i] && proj_grad[i] < 0.0)
            {
                proj_grad[i] = 0.0;
            }
        }
        proj_grad
    }
}

impl<LS> HasBounds for ProjectedGradientDescent<LS> {
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

impl<LS> ComputeDirection for ProjectedGradientDescent<LS> {
    fn compute_direction(
        &mut self,
        eval: &FuncEvalMultivariate,
    ) -> Result<DVector<Floating>, SolverError> {
        Ok(-eval.g())
    }
}

impl<LS> OptimizationSolver for ProjectedGradientDescent<LS>
where
    LS: LineSearch + HasBounds,
{
    type LS = LS;
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

        let proj_grad = self.projected_gradient(eval);
        // warn!(target: "projected_gradient_descent", "Projected gradient: {:?}", proj_grad);
        // we compute the infinity norm of the projected gradient
        proj_grad
            .iter()
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

        debug!(target: "projected_gradient_descent", "ITERATE: {} + {} * {} = {}", self.xk(), step, direction, self.xk() + step * direction);

        let next_iterate = self.xk() + step * direction;

        // we project the next iterate;
        let next_iterate = next_iterate.box_projection(&self.lower_bound, &self.upper_bound);

        // // compute the projected gradient;
        // self.pg = &next_iterate - self.xk();

        *self.xk_mut() = next_iterate;

        Ok(())
    }
}

mod projected_gradient_test {
    use super::*;
    #[test]
    pub fn constrained_grad_desc_backtracking() {
        std::env::set_var("RUST_LOG", "info");

        let tracer = Tracer::default()
            .with_stdout_layer(Some(LogFormat::Normal))
            .build();
        let gamma = 999.0;
        let f_and_g = |x: &DVector<Floating>| -> FuncEvalMultivariate {
            let f = 0.5 * (x[0].powi(2) + gamma * x[1].powi(2));
            let g = DVector::from(vec![x[0], gamma * x[1]]);
            (f, g).into()
        };

        //bounds
        let lower_bounds = DVector::from_vec(vec![-f64::INFINITY, -f64::INFINITY]);
        let upper_oounds = DVector::from_vec(vec![f64::INFINITY, f64::INFINITY]);
        // Linesearch builder
        let alpha = 1e-4;
        let beta = 0.5;
        let ls = BackTrackingB::new(alpha, beta, lower_bounds.clone(), upper_oounds.clone());

        // Gradient descent builder
        let tol = 1e-12;
        let x_0 = DVector::from(vec![180.0, 152.0]);
        let mut gd = ProjectedGradientDescent::new(ls, tol, x_0, lower_bounds, upper_oounds);

        // Minimization
        let max_iter_solver = 10000;
        let max_iter_line_search = 1000;

        gd.minimize(f_and_g, max_iter_solver, max_iter_line_search);

        println!("Iterate: {:?}", gd.xk());

        let eval = f_and_g(gd.xk());
        println!("Function eval: {:?}", eval);
        println!("Gradient norm: {:?}", eval.g().norm());
        println!("tol: {:?}", tol);

        let convergence = gd.has_converged(&eval);
        println!("Convergence: {:?}", convergence);
    }
}
