use super::*;

#[derive(Default)]
pub struct GradientDescentStrategy;
impl ComputeDirection for GradientDescentStrategy {
    fn compute_direction(&mut self, eval: &FuncEval) -> DVector<Floating> {
        -eval.g()
    }
}
pub type GradientDescent = SteepestDescent<BackTracking, GradientDescentStrategy>;

impl GradientDescent {
    pub fn new(line_search: BackTracking, grad_tol: Floating, x0: DVector<Floating>) -> Self {
        Self {
            line_search,
            grad_tol,
            x: x0,
            k: 0,
            direction_strategy: GradientDescentStrategy,
        }
    }
}

mod gradient_descent_test {
    use super::*;

    #[test]
    pub fn test_min() {
        std::env::set_var("RUST_LOG", "info");

        let tracer = Tracer::default()
            .with_stdout_layer(Some(LogFormat::Normal))
            .build();
        let gamma = 90.0;
        let f_and_g = |x: &DVector<Floating>| -> FuncEval {
            let f = 0.5 * (x[0].powi(2) + gamma * x[1].powi(2));
            let g = DVector::from(vec![x[0], gamma * x[1]]);
            (f, g).into()
        };
        // Linesearch builder
        let alpha = 1e-4;
        let beta = 0.5;
        let ls = BackTracking::new(alpha, beta);

        // Gradient descent builder
        let tol = 1e-12;
        let x_0 = DVector::from(vec![180.0, 152.0]);
        let mut gd = GradientDescent::new(ls, tol, x_0);

        // Minimization
        let max_iter = 1000;

        gd.minimize(f_and_g, max_iter);

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
