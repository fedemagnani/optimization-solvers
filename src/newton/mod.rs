use super::*;

pub mod projected_newton;
pub use projected_newton::*;
pub mod spn;
pub use spn::*;
#[derive(derive_getters::Getters)]
pub struct Newton {
    tol: Floating,
    decrement_squared: Option<Floating>,
    x: DVector<Floating>,
    k: usize,
}

impl Newton {
    pub fn new(tol: Floating, x0: DVector<Floating>) -> Self {
        Newton {
            tol,
            decrement_squared: None,
            x: x0,
            k: 0,
        }
    }
}

impl ComputeDirection for Newton {
    fn compute_direction(
        &mut self,
        eval: &FuncEvalMultivariate,
    ) -> Result<DVector<Floating>, SolverError> {
        let hessian = eval
            .hessian()
            .clone()
            .expect("Hessian not available in the oracle");
        //[TODO]: Boyd recommends several alternatives to the solution of Newton system which take advantage of prior information about sparsity/banded bandwidth of the hessian.
        match hessian.try_inverse() {
            Some(hessian_inv) => {
                let direction = -&hessian_inv * eval.g();
                // we compute also the squared newton decrement
                self.decrement_squared = Some((hessian_inv * &direction).dot(&direction));
                Ok(direction)
            }
            None => {
                warn!(target:"newton","Hessian is singular. Using gradient descent direction.");
                Ok(-eval.g())
            }
        }
    }
}

impl LineSearchSolver for Newton {
    fn xk(&self) -> &DVector<Floating> {
        &self.x
    }
    fn k(&self) -> &usize {
        &self.k
    }
    fn xk_mut(&mut self) -> &mut DVector<Floating> {
        &mut self.x
    }
    fn k_mut(&mut self) -> &mut usize {
        &mut self.k
    }
    fn has_converged(&self, _: &FuncEvalMultivariate) -> bool {
        match self.decrement_squared {
            Some(decrement_squared) => decrement_squared * 0.5 < self.tol,
            None => false,
        }
    }
}

#[cfg(test)]
mod newton_test {
    use super::*;

    #[test]
    pub fn newton_morethuente() {
        std::env::set_var("RUST_LOG", "info");

        let _ = Tracer::default()
            .with_stdout_layer(Some(LogFormat::Normal))
            .build();
        let gamma = 1222.0;
        let oracle = |x: &DVector<Floating>| -> FuncEvalMultivariate {
            let f: f64 = 0.5 * (x[0].powi(2) + gamma * x[1].powi(2));
            let g = DVector::from(vec![x[0], gamma * x[1]]);
            let hessian = DMatrix::from_iterator(2, 2, vec![1.0, 0.0, 0.0, gamma]);
            FuncEvalMultivariate::new(f, g).with_hessian(hessian)
        };

        // Linesearch builder

        let mut ls = MoreThuente::default();

        // newton builder
        let tol = 1e-8;
        let x_0 = DVector::from(vec![1.0, 1.0]);
        let mut nt = Newton::new(tol, x_0);

        // Minimization
        let max_iter_solver = 1000;
        let max_iter_line_search = 100;

        nt.minimize(&mut ls, oracle, max_iter_solver, max_iter_line_search, None)
            .unwrap();

        println!("Iterate: {:?}", nt.xk());

        let eval = oracle(nt.xk());
        println!("Function eval: {:?}", eval);
        println!("Gradient norm: {:?}", eval.g().norm());
        println!("tol: {:?}", tol);

        let convergence = nt.has_converged(&eval);
        println!("Convergence: {:?}", convergence);

        assert!((eval.f() - 0.0).abs() < 1e-6);
    }

    #[test]
    pub fn newton_backtracking() {
        std::env::set_var("RUST_LOG", "info");

        let _ = Tracer::default()
            .with_stdout_layer(Some(LogFormat::Normal))
            .build();
        let gamma = 1222.0;
        let oracle = |x: &DVector<Floating>| -> FuncEvalMultivariate {
            let f: f64 = 0.5 * (x[0].powi(2) + gamma * x[1].powi(2));
            let g = DVector::from(vec![x[0], gamma * x[1]]);
            let hessian = DMatrix::from_iterator(2, 2, vec![1.0, 0.0, 0.0, gamma]);
            FuncEvalMultivariate::new(f, g).with_hessian(hessian)
        };

        // Linesearch builder
        let alpha = 1e-4;
        let beta = 0.5;
        let mut ls = BackTracking::new(alpha, beta);

        // newton builder
        let tol = 1e-8;
        let x_0 = DVector::from(vec![1.0, 1.0]);
        let mut nt = Newton::new(tol, x_0);

        // Minimization
        let max_iter_solver = 1000;
        let max_iter_line_search = 100;

        nt.minimize(&mut ls, oracle, max_iter_solver, max_iter_line_search, None)
            .unwrap();

        println!("Iterate: {:?}", nt.xk());

        let eval = oracle(nt.xk());
        println!("Function eval: {:?}", eval);
        println!("Gradient norm: {:?}", eval.g().norm());
        println!("tol: {:?}", tol);

        let convergence = nt.has_converged(&eval);
        println!("Convergence: {:?}", convergence);

        assert!((eval.f() - 0.0).abs() < 1e-6);
    }
}
