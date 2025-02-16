use nalgebra::{DMatrix, DVector};
use optimization_solvers::{
    BackTracking, FuncEvalMultivariate, GradientDescent, LineSearchSolver, Plotter3d, Tracer,
};

fn main() {
    // Setting up log verbosity and _.
    std::env::set_var("RUST_LOG", "debug");
    let _ = Tracer::default().with_normal_stdout_layer().build();
    // Setting up the oracle
    let matrix = DMatrix::from_vec(2, 2, vec![100., 0., 0., 100.]);
    let mut f_and_g = |x: &DVector<f64>| -> FuncEvalMultivariate {
        let f = x.dot(&(&matrix * x));
        let g = 2. * &matrix * x;
        FuncEvalMultivariate::new(f, g)
    };
    // Setting up the line search
    let armijo_factr = 1e-4;
    let beta = 0.5; // (beta in (0, 1), ntice that beta = 0.5 corresponds to bisection)
    let mut ls = BackTracking::new(armijo_factr, beta);
    // Setting up the main solver, with its parameters and the initial guess
    let tol = 1e-6;
    let x0 = DVector::from_vec(vec![10., 10.]);
    let mut solver = GradientDescent::new(tol, x0);
    // We define a callback to store iterates and function evaluations
    let mut iterates = vec![];
    let mut solver_callback = |s: &GradientDescent| {
        iterates.push(s.x().clone());
    };
    // Running the solver
    let max_iter_solver = 100;
    let max_iter_line_search = 10;

    solver
        .minimize(
            &mut ls,
            f_and_g,
            max_iter_solver,
            max_iter_line_search,
            Some(&mut solver_callback),
        )
        .unwrap();
    // Printing the result
    let x = solver.x();
    let eval = f_and_g(x);
    println!("x: {:?}", x);
    println!("f(x): {}", eval.f());
    println!("g(x): {:?}", eval.g());

    // Plotting the iterates
    let n = 50;
    let start = -5.0;
    let end = 5.0;
    let plotter = Plotter3d::new(start, end, start, end, n)
        .append_plot(&mut f_and_g, "Objective function", 0.5)
        .append_scatter_points(&mut f_and_g, &iterates, "Iterates")
        .set_layout_size(1600, 1000);
    plotter.build("quadratic.html");
}
