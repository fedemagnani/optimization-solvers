use nalgebra::DVector;
use optimization_solvers::{
    BackTracking, FuncEvalMultivariate, GradientDescent, LineSearchSolver, Tracer,
};

fn main() {
    // Setting up logging
    std::env::set_var("RUST_LOG", "info");
    let _ = Tracer::default().with_normal_stdout_layer().build();

    // Convex quadratic function: f(x,y) = x^2 + 2y^2
    // Global minimum at (0, 0) with f(0,0) = 0
    let f_and_g = |x: &DVector<f64>| -> FuncEvalMultivariate {
        let x1 = x[0];
        let x2 = x[1];

        // Function value
        let f = x1.powi(2) + 2.0 * x2.powi(2);

        // Gradient
        let g1 = 2.0 * x1;
        let g2 = 4.0 * x2;
        let g = DVector::from_vec(vec![g1, g2]);

        FuncEvalMultivariate::new(f, g)
    };

    // Setting up the line search (backtracking with Armijo condition)
    let armijo_factor = 1e-4;
    let beta = 0.5;
    let mut ls = BackTracking::new(armijo_factor, beta);

    // Setting up the solver
    let tol = 1e-6;
    let x0 = DVector::from_vec(vec![2.0, 1.0]); // Starting point
    let mut solver = GradientDescent::new(tol, x0.clone());

    // Running the solver
    let max_iter_solver = 100;
    let max_iter_line_search = 20;

    println!("=== Gradient Descent Example ===");
    println!("Objective: f(x,y) = x^2 + 2y^2 (convex quadratic)");
    println!("Global minimum: (0, 0) with f(0,0) = 0");
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

            // Check if we're close to the known minimum
            let true_min = DVector::from_vec(vec![0.0, 0.0]);
            let distance_to_min = (x - true_min).norm();
            println!("Distance to true minimum: {:.6}", distance_to_min);
            println!("Expected function value: 0.0");
        }
        Err(e) => {
            println!("❌ Optimization failed: {:?}", e);
        }
    }
}
