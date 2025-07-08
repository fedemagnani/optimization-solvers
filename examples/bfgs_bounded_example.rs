use nalgebra::DVector;
use optimization_solvers::{FuncEvalMultivariate, LineSearchSolver, MoreThuente, Tracer, BFGSB};

fn main() {
    // Setting up logging
    std::env::set_var("RUST_LOG", "info");
    let _ = Tracer::default().with_normal_stdout_layer().build();

    // Convex function: f(x,y) = x^2 + 2y^2 + xy
    // This function is convex and has a unique minimum
    let f_and_g = |x: &DVector<f64>| -> FuncEvalMultivariate {
        let x1 = x[0];
        let x2 = x[1];

        // Function value
        let f = x1.powi(2) + 2.0 * x2.powi(2) + x1 * x2;

        // Gradient
        let g1 = 2.0 * x1 + x2;
        let g2 = 4.0 * x2 + x1;
        let g = DVector::from_vec(vec![g1, g2]);

        FuncEvalMultivariate::new(f, g)
    };

    // Setting up the line search (More-Thuente line search)
    let mut ls = MoreThuente::default();

    // Setting up the solver with box constraints
    let tol = 1e-6;
    let x0 = DVector::from_vec(vec![0.5, 0.5]); // Starting point
    let lower_bound = DVector::from_vec(vec![0.0, 0.0]); // x >= 0, y >= 0
    let upper_bound = DVector::from_vec(vec![2.0, 2.0]); // x <= 2, y <= 2
    let mut solver = BFGSB::new(tol, x0.clone(), lower_bound.clone(), upper_bound.clone());

    // Running the solver
    let max_iter_solver = 100;
    let max_iter_line_search = 20;

    println!("=== BFGSB (Bounded BFGS) Example ===");
    println!("Objective: f(x,y) = x^2 + 2y^2 + xy (convex quadratic)");
    println!("Unconstrained minimum: solution of ∇f(x) = 0");
    println!("Constraints: 0 <= x <= 2, 0 <= y <= 2");
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

            // Verify optimality conditions
            let gradient_at_solution = eval.g();
            println!("Gradient at solution: {:?}", gradient_at_solution);
            println!(
                "Gradient norm should be close to 0: {}",
                gradient_at_solution.norm()
            );

            // Show some properties of BFGSB
            println!("BFGSB properties:");
            println!("  - BFGS with box constraints");
            println!("  - Maintains BFGS Hessian approximation");
            println!("  - Handles bounds through projection");
        }
        Err(e) => {
            println!("❌ Optimization failed: {:?}", e);
        }
    }
}
