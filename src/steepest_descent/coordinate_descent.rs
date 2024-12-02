use super::*;

use super::*;

// All the algorithms in the family of steepest descent differ only in the way they compute the descent direction (i.e. they differ in the norm used so that the associated unit ball is the constraint set on which search the direction that minimizes the directional derivative at the current iterate. Typically this minimizer is a unit vector but any scaled version of the vector is good (the line search will adjust the direction later), so it's good supplying the rescaled version of the minimizer which has minimal computational cost).

// the family of steepest descent algorithms has (at most) linear convergence rate, and it's possible to see it by computing the trajectory of the upper bound of the log-suboptimality error ln(f(x_k)-p^*) where p^* is the optimal value of the problem. In particular, the convergence drops significantly if the upper bound of the condition number of the hessian matrix of the function is high (you can see it by solving the log-suboptimality error trajectory for the iteration number k). Recall that an upper bound on the condition number of the hessian can be derived by taking the ratio between the maximal and the minimal eigenvalue of the hessian matrix. This condition number can be also thought as the volume of the ellipsoid {x: x^T H x <= 1} where H is the hessian matrix of the function, which is always relatable to the volume of the euclidean unit ball gamma*sqrt{det (H^TH)} where gamma is the volume of the euclidean unit ball.

#[derive(derive_getters::Getters)]
pub struct CoordinateDescent {
    pub grad_tol: Floating,
    pub x: DVector<Floating>,
    pub k: usize,
}

impl CoordinateDescent {
    pub fn new(grad_tol: Floating, x0: DVector<Floating>) -> Self {
        Self {
            grad_tol,
            x: x0,
            k: 0,
        }
    }
}

impl ComputeDirection for CoordinateDescent {
    fn compute_direction(
        &mut self,
        eval: &FuncEvalMultivariate,
    ) -> Result<DVector<Floating>, SolverError> {
        // Differently from the gradient descent, here we pick the highest absolute value of the gradient and we multiply it with the vector of the canonical basis associated with its entry
        let grad_k = eval.g();
        let (position, max_value) =
            grad_k
                .iter()
                .enumerate()
                .fold((0, 0.0), |(idx, max), (i, g)| {
                    if g.abs() > max {
                        (i, g.abs())
                    } else {
                        (idx, max)
                    }
                });
        let mut direction_k = DVector::zeros(grad_k.len());
        direction_k[position] = -max_value.signum();
        Ok(direction_k)
    }
}

impl OptimizationSolver for CoordinateDescent {
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
        // we verify that the norm of the gradient is below the tolerance.
        let grad = eval.g();
        // we compute the infinity norm of the gradient
        grad.iter()
            .fold(Floating::NEG_INFINITY, |acc, x| x.abs().max(acc))
            < self.grad_tol
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

        debug!(target: "coordinate_descent", "ITERATE: {} + {} * {} = {}", self.xk(), step, direction, self.xk() + step * direction);

        let next_iterate = self.xk() + step * direction;

        *self.xk_mut() = next_iterate;

        Ok(())
    }
}

mod steepest_descent_l1_test {
    use super::*;
    use nalgebra::DVector;

    #[test]
    pub fn coordinate_descent_morethuente() {
        std::env::set_var("RUST_LOG", "info");

        let tracer = Tracer::default()
            .with_stdout_layer(Some(LogFormat::Normal))
            .build();
        let gamma = 90.0;
        let f_and_g = |x: &DVector<Floating>| -> FuncEvalMultivariate {
            let f = 0.5 * (x[0].powi(2) + gamma * x[1].powi(2));
            let g = DVector::from(vec![x[0], gamma * x[1]]);
            (f, g).into()
        };
        // Linesearch builder

        let mut ls = MoreThuente::default();

        // Gradient descent builder
        let tol = 1e-12;

        let x_0 = DVector::from(vec![180.0, 152.0]);
        let mut sdl1 = CoordinateDescent::new(tol, x_0);

        // Minimization
        let max_iter_solver = 1000;
        let max_iter_line_search = 100;

        sdl1.minimize(
            &mut ls,
            f_and_g,
            max_iter_solver,
            max_iter_line_search,
            None,
        )
        .unwrap();

        println!("Iterate: {:?}", sdl1.xk());

        let eval = f_and_g(sdl1.xk());
        println!("Function eval: {:?}", eval);
        println!("Gradient norm: {:?}", eval.g().norm());
        println!("tol: {:?}", tol);

        let convergence = sdl1.has_converged(&eval);
        println!("Convergence: {:?}", convergence);

        assert!((eval.f() - 0.0).abs() < 1e-6);
    }

    #[test]
    pub fn coordinate_descent_backtracking() {
        std::env::set_var("RUST_LOG", "info");

        let tracer = Tracer::default()
            .with_stdout_layer(Some(LogFormat::Normal))
            .build();
        let gamma = 90.0;
        let f_and_g = |x: &DVector<Floating>| -> FuncEvalMultivariate {
            let f = 0.5 * (x[0].powi(2) + gamma * x[1].powi(2));
            let g = DVector::from(vec![x[0], gamma * x[1]]);
            (f, g).into()
        };
        // Linesearch builder
        let alpha = 1e-4;
        let beta = 0.5;
        let mut ls = BackTracking::new(alpha, beta);

        // Gradient descent builder
        let tol = 1e-12;

        let x_0 = DVector::from(vec![180.0, 152.0]);
        let mut sdl1 = CoordinateDescent::new(tol, x_0);

        // Minimization
        let max_iter_solver = 1000;
        let max_iter_line_search = 100;

        sdl1.minimize(
            &mut ls,
            f_and_g,
            max_iter_solver,
            max_iter_line_search,
            None,
        )
        .unwrap();

        println!("Iterate: {:?}", sdl1.xk());

        let eval = f_and_g(sdl1.xk());
        println!("Function eval: {:?}", eval);
        println!("Gradient norm: {:?}", eval.g().norm());
        println!("tol: {:?}", tol);

        let convergence = sdl1.has_converged(&eval);
        println!("Convergence: {:?}", convergence);

        assert!((eval.f() - 0.0).abs() < 1e-6);
    }
}
