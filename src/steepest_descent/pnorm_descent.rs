use super::*;

// All the algorithms in the family of steepest descent differ only in the way they compute the descent direction (i.e. they differ in the norm used so that the associated unit ball is the constraint set on which search the direction that minimizes the directional derivative at the current iterate. Typically this minimizer is a unit vector but any scaled version of the vector is good (the line search will adjust the direction later), so it's good supplying the rescaled version of the minimizer which has minimal computational cost).

// the family of steepest descent algorithms has (at most) linear convergence rate, and it's possible to see it by computing the trajectory of the upper bound of the log-suboptimality error ln(f(x_k)-p^*) where p^* is the optimal value of the problem. In particular, the convergence drops significantly if the upper bound of the condition number of the hessian matrix of the function is high (you can see it by solving the log-suboptimality error trajectory for the iteration number k). Recall that an upper bound on the condition number of the hessian can be derived by taking the ratio between the maximal and the minimal eigenvalue of the hessian matrix. This condition number can be also thought as the volume of the ellipsoid {x: x^T H x <= 1} where H is the hessian matrix of the function, which is always relatable to the volume of the euclidean unit ball gamma*sqrt{det (H^TH)} where gamma is the volume of the euclidean unit ball.The p-norm descent tries to tackle this issue by taking a Matrix P that proxies correctly the hessian matrix (i.e. its unit norm {x: x^T P x <= 1} is a good approximation of the sublevel sets of the function), and this adjustments decreases the condition number of P^{-0.5} H P^{-0.5} because it would resemble (more or less) the identity matrix. It's from this intuition that the newton and quasi-newton methods become more clear.

// Notice that the the pnorm descent is equivalent to the steepest descent with P= identity matrix
// This approach finds the direction of the steepest descent by minimizing the directional derivative (at current iterate) over the ellipsoid {d: d^T P d <= 1} (which could be thought as the unit ball of the P-norm ||P^(-1/2) d||_2)
// The best thing would be picking a matrix P (and then compute its inverse) such that the P is a good approximation of the hessian of the function. By doing this, the condition number of the hessian is in control and the convergence rate of the algorithm is improved. It's from this rationale that newton and quasi-newton methods are born.

#[derive(derive_getters::Getters)]
pub struct PnormDescent {
    pub grad_tol: Floating,
    pub x: DVector<Floating>,
    pub k: usize,
    pub inverse_p: DMatrix<Floating>,
}

impl PnormDescent {
    pub fn new(grad_tol: Floating, x0: DVector<Floating>, inverse_p: DMatrix<Floating>) -> Self {
        PnormDescent {
            grad_tol,
            x: x0,
            k: 0,
            inverse_p,
        }
    }
}

impl ComputeDirection for PnormDescent {
    fn compute_direction(
        &mut self,
        eval: &FuncEvalMultivariate,
    ) -> Result<DVector<Floating>, SolverError> {
        Ok(-&self.inverse_p * eval.g())
    }
}

impl LineSearchSolver for PnormDescent {
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

        debug!(target: "pnorm_descent", "ITERATE: {} + {} * {} = {}", self.xk(), step, direction, self.xk() + step * direction);

        let next_iterate = self.xk() + step * direction;

        *self.xk_mut() = next_iterate;

        Ok(())
    }
}

mod gpnorm_descent_test {
    use super::*;
    use nalgebra::{Matrix, Matrix2, Vector2};

    #[test]
    pub fn pnorm_morethuente() {
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
        // We compute the inverse hessian of the function. Notice that the hessian is constant since the objective function is quadratic
        let inv_hessian = DMatrix::from_iterator(2, 2, vec![1.0, 0.0, 0.0, 1.0 / gamma]);
        // let inv_hessian = DMatrix::identity(2, 2);

        // Linesearch builder
        let mut ls = MoreThuente::default();

        // pnorm descent builder
        let tol = 1e-12;
        let x_0 = DVector::from(vec![180.0, 152.0]);
        let mut gd = PnormDescent::new(tol, x_0, inv_hessian);

        // Minimization
        let max_iter_solver = 1000;
        let max_iter_line_search = 100;

        gd.minimize(
            &mut ls,
            f_and_g,
            max_iter_solver,
            max_iter_line_search,
            None,
        )
        .unwrap();

        println!("Iterate: {:?}", gd.xk());

        let eval = f_and_g(gd.xk());
        println!("Function eval: {:?}", eval);
        println!("Gradient norm: {:?}", eval.g().norm());
        println!("tol: {:?}", tol);

        let convergence = gd.has_converged(&eval);
        println!("Convergence: {:?}", convergence);

        assert!((eval.f() - 0.0).abs() < 1e-6);
    }

    #[test]
    pub fn pnorm_backtracking() {
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
        // We compute the inverse hessian of the function. Notice that the hessian is constant since the objective function is quadratic
        let inv_hessian = DMatrix::from_iterator(2, 2, vec![1.0, 0.0, 0.0, 1.0 / gamma]);
        // let inv_hessian = DMatrix::identity(2, 2);

        // Linesearch builder
        let alpha = 1e-4;
        let beta = 0.5;
        let mut ls = BackTracking::new(alpha, beta);

        // pnorm descent builder
        let tol = 1e-12;
        let x_0 = DVector::from(vec![180.0, 152.0]);
        let mut gd = PnormDescent::new(tol, x_0, inv_hessian);

        // Minimization
        let max_iter_solver = 1000;
        let max_iter_line_search = 100;

        gd.minimize(
            &mut ls,
            f_and_g,
            max_iter_solver,
            max_iter_line_search,
            None,
        )
        .unwrap();

        println!("Iterate: {:?}", gd.xk());

        let eval = f_and_g(gd.xk());
        println!("Function eval: {:?}", eval);
        println!("Gradient norm: {:?}", eval.g().norm());
        println!("tol: {:?}", tol);

        let convergence = gd.has_converged(&eval);
        println!("Convergence: {:?}", convergence);

        assert!((eval.f() - 0.0).abs() < 1e-6);
    }
}
