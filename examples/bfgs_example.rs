use nalgebra::{DMatrix, DVector};
use optimization_solvers::{FuncEvalMultivariate, LineSearchSolver, MoreThuente, Tracer, BFGS};

fn main() {
    // Setting up logging
    std::env::set_var("RUST_LOG", "info");
    let _ = Tracer::default().with_normal_stdout_layer().build();

    // Convex quadratic function: f(x,y,z) = x^2 + 2y^2 + 3z^2 + xy + yz
    // This function has a unique minimum that we can verify
    let f_and_g = |x: &DVector<f64>| -> FuncEvalMultivariate {
        let x1 = x[0];
        let x2 = x[1];
        let x3 = x[2];

        // Function value
        let f = x1.powi(2) + 2.0 * x2.powi(2) + 3.0 * x3.powi(2) + x1 * x2 + x2 * x3;

        // Gradient
        let g1 = 2.0 * x1 + x2;
        let g2 = 4.0 * x2 + x1 + x3;
        let g3 = 6.0 * x3 + x2;
        let g = DVector::from_vec(vec![g1, g2, g3]);

        FuncEvalMultivariate::new(f, g)
    };

    // Setting up the line search (More-Thuente line search)
    let mut ls = MoreThuente::default();

    // Setting up the solver
    let tol = 1e-8;
    let x0 = DVector::from_vec(vec![1.0, 1.0, 1.0]); // Starting point
    let mut solver = BFGS::new(tol, x0.clone());

    // Running the solver
    let max_iter_solver = 50;
    let max_iter_line_search = 20;

    println!("=== BFGS Quasi-Newton Example ===");
    println!("Objective: f(x,y,z) = x^2 + 2y^2 + 3z^2 + xy + yz (convex quadratic)");
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
            println!("Function value: {:.8}", eval.f());
            println!("Gradient norm: {:.8}", eval.g().norm());
            println!("Iterations: {}", solver.k());

            // Verify optimality conditions
            let gradient_at_solution = eval.g();
            println!("Gradient at solution: {:?}", gradient_at_solution);
            println!(
                "Gradient norm should be close to 0: {}",
                gradient_at_solution.norm()
            );

            // For this convex quadratic function, the minimum should be at the solution of the linear system
            // ∇f(x) = 0, which gives us a system of linear equations
            println!("Expected minimum: solution of ∇f(x) = 0");
        }
        Err(e) => {
            println!("❌ Optimization failed: {:?}", e);
        }
    }
}
