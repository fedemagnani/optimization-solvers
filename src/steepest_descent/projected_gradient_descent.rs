use super::*;

// The projected gradient method is a simple naturalization of the steepest descent method in the setting of optimization with simple bounds. In this context, I've implemented algorithm 12.1 from [Neculai Andrei, 2022]

#[derive(derive_getters::Getters)]
pub struct ProjectedGradientDescent {
    grad_tol: Floating,
    x: DVector<Floating>,
    k: usize,
    lower_bound: DVector<Floating>,
    upper_bound: DVector<Floating>,
}

impl ProjectedGradientDescent {
    pub fn new(
        grad_tol: Floating,
        x0: DVector<Floating>,
        lower_bound: DVector<Floating>,
        upper_bound: DVector<Floating>,
    ) -> Self {
        let x0 = x0.box_projection(&lower_bound, &upper_bound);
        // let
        // let pg = DVector::zeros(x0.len());
        Self {
            grad_tol,
            x: x0,
            k: 0,
            lower_bound,
            upper_bound,
            // pg,
        }
    }
}

impl HasBounds for ProjectedGradientDescent {
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

impl ComputeDirection for ProjectedGradientDescent {
    fn compute_direction(
        &mut self,
        eval: &FuncEvalMultivariate,
    ) -> Result<DVector<Floating>, SolverError> {
        // Ok(-eval.g())
        let direction = &self.x - eval.g();
        let direction = direction.box_projection(&self.lower_bound, &self.upper_bound);
        let direction = direction - &self.x;
        Ok(direction)
    }
}

impl OptimizationSolver for ProjectedGradientDescent {
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
        proj_grad.infinity_norm() < self.grad_tol
    }

    fn update_next_iterate<LS: LineSearch>(
        &mut self,
        line_search: &mut LS,
        eval_x_k: &FuncEvalMultivariate, //eval: &FuncEvalMultivariate,
        oracle: &impl Fn(&DVector<Floating>) -> FuncEvalMultivariate,
        direction: &DVector<Floating>,
        max_iter_line_search: usize,
    ) -> Result<(), SolverError> {
        let step = line_search.compute_step_len(
            self.xk(),
            eval_x_k,
            direction,
            oracle,
            max_iter_line_search,
        );

        debug!(target: "projected_gradient_descent", "ITERATE: {} + {} * {} = {}", self.xk(), step, direction, self.xk() + step * direction);

        let next_iterate = self.xk() + step * direction;

        // // we project the next iterate;
        // let next_iterate = next_iterate.box_projection(&self.lower_bound, &self.upper_bound);

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

        //bounds for the problem p
        let lower_bounds = DVector::from_vec(vec![-f64::INFINITY, -f64::INFINITY]);
        let upper_oounds = DVector::from_vec(vec![f64::INFINITY, f64::INFINITY]);
        // Linesearch builder
        let alpha = 1e-4;
        let beta = 0.5;
        let mut ls = BackTrackingB::new(alpha, beta, lower_bounds.clone(), upper_oounds.clone());

        // Gradient descent builder
        let tol = 1e-6;
        let x_0 = DVector::from(vec![180.0, 152.0]);
        let mut gd = ProjectedGradientDescent::new(tol, x_0, lower_bounds, upper_oounds);

        // Minimization
        let max_iter_solver = 10000;
        let max_iter_line_search = 1000;

        gd.minimize(&mut ls, f_and_g, max_iter_solver, max_iter_line_search)
            .unwrap();

        println!("Iterate: {:?}", gd.xk());

        let eval = f_and_g(gd.xk());
        println!("Function eval: {:?}", eval);
        println!(
            "projected Gradient norm: {:?}",
            gd.projected_gradient(&eval).infinity_norm()
        );
        println!("tol: {:?}", tol);

        let convergence = gd.has_converged(&eval);
        println!("Convergence: {:?}", convergence);
    }
}
