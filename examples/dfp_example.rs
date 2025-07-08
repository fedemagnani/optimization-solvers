use nalgebra::DVector;
use optimization_solvers::{FuncEvalMultivariate, LineSearchSolver, MoreThuente, Tracer, DFP};

fn main() {
    // Setting up logging
    std::env::set_var("RUST_LOG", "info");
    let _ = Tracer::default().with_normal_stdout_layer().build();

    // Convex function: f(x,y) = x^2 + 5y^2 + xy
    // This function is convex and has a unique minimum
    let f_and_g = |x: &DVector<f64>| -> FuncEvalMultivariate {
        let x1 = x[0];
        let x2 = x[1];

        // Function value
        let f = x1.powi(2) + 5.0 * x2.powi(2) + x1 * x2;

        // Gradient
        let g1 = 2.0 * x1 + x2;
        let g2 = 10.0 * x2 + x1;
        let g = DVector::from_vec(vec![g1, g2]);

        FuncEvalMultivariate::new(f, g)
    };

    // Setting up the line search (More-Thuente line search)
    let mut ls = MoreThuente::default();

    // Setting up the solver
    let tol = 1e-6;
    let x0 = DVector::from_vec(vec![2.0, 1.0]); // Starting point
    let mut solver = DFP::new(tol, x0.clone());

    // Running the solver
    let max_iter_solver = 100;
    let max_iter_line_search = 20;

    println!("=== DFP (Davidon-Fletcher-Powell) Quasi-Newton Example ===");
    println!("Objective: f(x,y) = x^2 + 5y^2 + xy (convex quadratic)");
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

            // Verify optimality conditions
            let gradient_at_solution = eval.g();
            println!("Gradient at solution: {:?}", gradient_at_solution);
            println!(
                "Gradient norm should be close to 0: {}",
                gradient_at_solution.norm()
            );

            // For this convex quadratic function, the minimum should be at the solution of the linear system
            // ∇f(x) = 0, which gives us: 2x + y = 0, x + 10y = 0
            // Solving: x = 0, y = 0
            let expected_min = DVector::from_vec(vec![0.0, 0.0]);
            let distance_to_expected = (x - expected_min).norm();
            println!(
                "Distance to expected minimum (0,0): {:.6}",
                distance_to_expected
            );
            println!("Expected function value at (0,0): 0.0");
        }
        Err(e) => {
            println!("❌ Optimization failed: {:?}", e);
        }
    }
}
