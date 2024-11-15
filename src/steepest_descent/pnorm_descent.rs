use super::*;
// Notice that the the pnorm descent is equivalent to the steepest descent with P= identity matrix
// This approach finds the direction of the steepest descent by minimizing the directional derivative (at current iterate) over the ellipsoid {d: d^T P d <= 1} (which could be thought as the unit ball of the P-norm ||P^(-1/2) d||_2)
// The best thing would be picking a matrix P (and then compute its inverse) such that the P is a good approximation of the hessian of the function. By doing this, the condition number of the hessian is in control and the convergence rate of the algorithm is improved. It's from this rationale that newton and quasi-newton methods are born.

#[derive(Default)]
pub struct PnormDescentStrategy {
    inverse_p: DMatrix<Floating>,
}
impl PnormDescentStrategy {
    pub fn new(inverse_p: DMatrix<Floating>) -> Self {
        PnormDescentStrategy { inverse_p }
    }
}

impl ComputeDirection for PnormDescentStrategy {
    fn compute_direction(&mut self, eval: &FuncEvalMultivariate) -> DVector<Floating> {
        // let grad_k = eval.g();
        // self.inverse_p.mul_to(grad_k, &mut self.direction);
        // self.direction.neg_mut();
        // self.direction.clone()
        -&self.inverse_p * eval.g()
    }
}
pub type PnormDescent = SteepestDescent<BackTracking, PnormDescentStrategy>;

impl PnormDescent {
    pub fn new(
        line_search: BackTracking,
        grad_tol: Floating,
        x0: DVector<Floating>,
        inverse_p: DMatrix<Floating>,
    ) -> Self {
        PnormDescent {
            line_search,
            grad_tol,
            x: x0,
            k: 0,
            direction_strategy: PnormDescentStrategy::new(inverse_p),
        }
    }
}

mod gpnorm_descent_test {
    use super::*;
    use nalgebra::{Matrix, Matrix2, Vector2};

    #[test]
    pub fn test_min() {
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
        let ls = BackTracking::new(alpha, beta);

        // pnorm descent builder
        let tol = 1e-12;
        let x_0 = DVector::from(vec![180.0, 152.0]);
        let mut gd = PnormDescent::new(ls, tol, x_0, inv_hessian);

        // Minimization
        let max_iter_solver = 1000;
        let max_iter_line_search = 100;

        gd.minimize(f_and_g, max_iter_solver, max_iter_line_search)
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
