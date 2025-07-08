use nalgebra::DVector;
use optimization_solvers::{
    BackTracking, CoordinateDescent, FuncEvalMultivariate, LineSearchSolver, Tracer,
};

fn main() {
    // Setting up logging
    std::env::set_var("RUST_LOG", "info");
    let _ = Tracer::default().with_normal_stdout_layer().build();

    // Separable convex function: f(x,y,z) = x^2 + 2y^2 + 3z^2
    // This function is separable and has a minimum at (0, 0, 0)
    let f_and_g = |x: &DVector<f64>| -> FuncEvalMultivariate {
        let x1 = x[0];
        let x2 = x[1];
        let x3 = x[2];

        // Function value
        let f = x1.powi(2) + 2.0 * x2.powi(2) + 3.0 * x3.powi(2);

        // Gradient
        let g1 = 2.0 * x1;
        let g2 = 4.0 * x2;
        let g3 = 6.0 * x3;
        let g = DVector::from_vec(vec![g1, g2, g3]);

        FuncEvalMultivariate::new(f, g)
    };

    // Setting up the line search (backtracking)
    let armijo_factor = 1e-4;
    let beta = 0.5;
    let mut ls = BackTracking::new(armijo_factor, beta);

    // Setting up the solver
    let tol = 1e-6;
    let x0 = DVector::from_vec(vec![1.0, 1.0, 1.0]); // Starting point
    let mut solver = CoordinateDescent::new(tol, x0.clone());

    // Running the solver
    let max_iter_solver = 100;
    let max_iter_line_search = 10;

    println!("=== Coordinate Descent Example ===");
    println!("Objective: f(x,y,z) = x^2 + 2y^2 + 3z^2 (separable convex)");
    println!("Global minimum: (0, 0, 0) with f(0,0,0) = 0");
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
            let true_min = DVector::from_vec(vec![0.0, 0.0, 0.0]);
            let distance_to_min = (x - true_min).norm();
            println!("Distance to true minimum: {:.6}", distance_to_min);
            println!("Expected function value: 0.0");

            // Verify optimality conditions
            let gradient_at_solution = eval.g();
            println!("Gradient at solution: {:?}", gradient_at_solution);
            println!(
                "Gradient norm should be close to 0: {}",
                gradient_at_solution.norm()
            );
        }
        Err(e) => {
            println!("❌ Optimization failed: {:?}", e);
        }
    }
}
