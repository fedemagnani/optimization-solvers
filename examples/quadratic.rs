use nalgebra::{DMatrix, DVector};
use optimization_solvers::{FuncEvalMultivariate, MoreThuente, OptimizationSolver, Tracer, BFGS};

fn main() {
    // Setting up log verbosity and tracer
    std::env::set_var("RUST_LOG", "debug");
    let tracer = Tracer::default().with_normal_stdout_layer().build();
    // Setting up the oracle
    let matrix = DMatrix::from_vec(2, 2, vec![1., 0., 0., 1.]);
    let f_and_g = |x: &DVector<f64>| -> FuncEvalMultivariate {
        let f = x.dot(&(&matrix * x));
        let g = 2. * &matrix * x;
        FuncEvalMultivariate::new(f, g)
    };
    // Setting up the line search
    let mut ls = MoreThuente::default();
    // Setting up the main solver, with its parameters and the initial guess
    let tol = 1e-6;
    let x0 = DVector::from_vec(vec![1., 1.]);
    let mut solver = BFGS::new(tol, x0);
    // Running the solver
    let max_iter_solver = 100;
    let max_iter_line_search = 10;
    let callback = None;
    solver
        .minimize(
            &mut ls,
            f_and_g,
            max_iter_solver,
            max_iter_line_search,
            callback,
        )
        .unwrap();
    // Printing the result
    let x = solver.x();
    let eval = f_and_g(x);
    println!("x: {:?}", x);
    println!("f(x): {}", eval.f());
    println!("g(x): {:?}", eval.g());
    assert_eq!(eval.f(), &0.0);
}
