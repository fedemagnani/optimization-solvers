use tracing::info;

use super::*;

#[derive(Default)]
pub struct CoordinateDescentStrategy;

impl ComputeDirection for CoordinateDescentStrategy {
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
pub type CoordinateDescent<LS> = SteepestDescent<LS, CoordinateDescentStrategy>;

impl<LS> CoordinateDescent<LS> {
    pub fn new(line_search: LS, grad_tol: Floating, x0: DVector<Floating>) -> Self {
        Self {
            line_search,
            grad_tol,
            x: x0,
            k: 0,
            direction_strategy: CoordinateDescentStrategy,
            lower_bound: None,
            upper_bound: None,
            pg: None,
        }
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

        let ls = MoreThuente::default();

        // Gradient descent builder
        let tol = 1e-12;

        let x_0 = DVector::from(vec![180.0, 152.0]);
        let mut sdl1 = CoordinateDescent::new(ls, tol, x_0);

        // Minimization
        let max_iter_solver = 1000;
        let max_iter_line_search = 100;

        sdl1.minimize(f_and_g, max_iter_solver, max_iter_line_search)
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
        let ls = BackTracking::new(alpha, beta);

        // Gradient descent builder
        let tol = 1e-12;

        let x_0 = DVector::from(vec![180.0, 152.0]);
        let mut sdl1 = CoordinateDescent::new(ls, tol, x_0);

        // Minimization
        let max_iter_solver = 1000;
        let max_iter_line_search = 100;

        sdl1.minimize(f_and_g, max_iter_solver, max_iter_line_search)
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
