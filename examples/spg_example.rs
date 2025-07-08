use nalgebra::DVector;
use optimization_solvers::{
    BackTracking, FuncEvalMultivariate, LineSearchSolver, SpectralProjectedGradient, Tracer,
};

fn main() {
    // Setting up logging
    std::env::set_var("RUST_LOG", "info");
    let _ = Tracer::default().with_normal_stdout_layer().build();

    // Convex function: f(x,y) = x^2 + y^2 + exp(x^2 + y^2)
    // This function is convex and has a minimum at (0, 0)
    let f_and_g = |x: &DVector<f64>| -> FuncEvalMultivariate {
        let x1 = x[0];
        let x2 = x[1];

        // Function value
        let f = x1.powi(2) + x2.powi(2) + (x1.powi(2) + x2.powi(2)).exp();

        // Gradient
        let exp_term = (x1.powi(2) + x2.powi(2)).exp();
        let g1 = 2.0 * x1 * (1.0 + exp_term);
        let g2 = 2.0 * x2 * (1.0 + exp_term);
        let g = DVector::from_vec(vec![g1, g2]);

        FuncEvalMultivariate::new(f, g)
    };

    // Setting up the line search (backtracking)
    let armijo_factor = 1e-4;
    let beta = 0.5;
    let mut ls = BackTracking::new(armijo_factor, beta);

    // Setting up the solver with box constraints
    let tol = 1e-6;
    let x0 = DVector::from_vec(vec![0.5, 0.5]); // Starting point
    let lower_bound = DVector::from_vec(vec![-1.0, -1.0]); // -1 <= x <= 1, -1 <= y <= 1
    let upper_bound = DVector::from_vec(vec![1.0, 1.0]);

    // Create a mutable oracle for SPG initialization
    let mut oracle_for_init = f_and_g;
    let mut solver = SpectralProjectedGradient::new(
        tol,
        x0.clone(),
        &mut oracle_for_init,
        lower_bound.clone(),
        upper_bound.clone(),
    );

    // Running the solver
    let max_iter_solver = 100;
    let max_iter_line_search = 20;

    println!("=== Spectral Projected Gradient (SPG) Example ===");
    println!("Objective: f(x,y) = x^2 + y^2 + exp(x^2 + y^2) (convex)");
    println!("Global minimum: (0, 0) with f(0,0) = 1");
    println!("Constraints: -1 <= x <= 1, -1 <= y <= 1");
    println!("Starting point: {:?}", x0);
    println!("Lower bounds: {:?}", lower_bound);
    println!("Upper bounds: {:?}", upper_bound);
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

            // Check constraint satisfaction
            println!("Constraint satisfaction:");
            for i in 0..x.len() {
                println!(
                    "  x[{}] = {:.6} (bounds: [{:.1}, {:.1}])",
                    i, x[i], lower_bound[i], upper_bound[i]
                );
            }

            // Check if we're close to the known minimum
            let true_min = DVector::from_vec(vec![0.0, 0.0]);
            let distance_to_min = (x - true_min).norm();
            println!("Distance to true minimum: {:.6}", distance_to_min);
            println!("Expected function value: 1.0");

            // Show some properties of SPG
            println!("SPG properties:");
            println!("  - Uses spectral step length estimation");
            println!("  - Handles box constraints efficiently");
            println!("  - Often faster than standard projected gradient");
        }
        Err(e) => {
            println!("❌ Optimization failed: {:?}", e);
        }
    }
}
