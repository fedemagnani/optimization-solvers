use nalgebra::{DMatrix, DVector};
use optimization_solvers::{FuncEvalMultivariate, LineSearchSolver, MoreThuente, Newton, Tracer};

fn main() {
    // Setting up logging
    std::env::set_var("RUST_LOG", "info");
    let _ = Tracer::default().with_normal_stdout_layer().build();

    // Convex function: f(x,y) = x^2 + y^2 + exp(x^2 + y^2)
    // This function is convex and has a unique minimum at (0, 0)
    let f_and_g = |x: &DVector<f64>| -> FuncEvalMultivariate {
        let x1 = x[0];
        let x2 = x[1];

        // Function value
        let f = x1.powi(2) + x2.powi(2) + (x1.powi(2) + x2.powi(2)).exp();

        // Gradient: ∇f = [2x + 2x*exp(x^2+y^2), 2y + 2y*exp(x^2+y^2)]
        let exp_term = (x1.powi(2) + x2.powi(2)).exp();
        let g1 = 2.0 * x1 * (1.0 + exp_term);
        let g2 = 2.0 * x2 * (1.0 + exp_term);
        let g = DVector::from_vec(vec![g1, g2]);

        // Hessian: ∇²f = [[2(1+exp) + 4x^2*exp, 4xy*exp], [4xy*exp, 2(1+exp) + 4y^2*exp]]
        let h11 = 2.0 * (1.0 + exp_term) + 4.0 * x1.powi(2) * exp_term;
        let h12 = 4.0 * x1 * x2 * exp_term;
        let h21 = h12;
        let h22 = 2.0 * (1.0 + exp_term) + 4.0 * x2.powi(2) * exp_term;
        let hessian = DMatrix::from_vec(2, 2, vec![h11, h21, h12, h22]);

        FuncEvalMultivariate::new(f, g).with_hessian(hessian)
    };

    // Setting up the line search (More-Thuente line search)
    let mut ls = MoreThuente::default();

    // Setting up the solver
    let tol = 1e-6;
    let x0 = DVector::from_vec(vec![1.0, 1.0]); // Starting point
    let mut solver = Newton::new(tol, x0.clone());

    // Running the solver
    let max_iter_solver = 20;
    let max_iter_line_search = 20;

    println!("=== Newton's Method Example ===");
    println!("Objective: f(x,y) = x^2 + y^2 + exp(x^2 + y^2) (convex)");
    println!("Global minimum: (0, 0) with f(0,0) = 1");
    println!("Starting point: {:?}", x0);
    println!("Tolerance: {}", tol);
    println!();

    match solver.minimize(
        &mut ls,
        f_and_g,
        max_iter_solver,
        max_iter_line_search,
        None,
    ) {
        Ok(()) => {
            let x = solver.x();
            let eval = f_and_g(x);
            println!("✅ Optimization completed successfully!");
            println!("Final iterate: {:?}", x);
            println!("Function value: {:.6}", eval.f());
            println!("Gradient norm: {:.6}", eval.g().norm());
            println!("Iterations: {}", solver.k());

            // Show Newton decrement
            if let Some(decrement_squared) = solver.decrement_squared() {
                println!("Newton decrement squared: {:.6}", decrement_squared);
                println!("Newton decrement: {:.6}", decrement_squared.sqrt());
            }

            // Check if we're close to the known minimum
            let true_min = DVector::from_vec(vec![0.0, 0.0]);
            let distance_to_min = (x - true_min).norm();
            println!("Distance to true minimum: {:.6}", distance_to_min);
            println!("Expected function value: 1.0");
        }
        Err(e) => {
            println!("❌ Optimization failed: {:?}", e);
        }
    }
}
