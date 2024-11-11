use super::*;

#[derive(derive_getters::Getters)]
pub struct BFGS<T> {
    line_search: T,
    approx_inv_hessian: DMatrix<Floating>,
    x: DVector<Floating>,
    k: usize,
    tol: Floating,
    s_norm: Option<Floating>,
    y_norm: Option<Floating>,
    identity: DMatrix<Floating>,
}

impl<T> BFGS<T> {
    pub fn next_iterate_too_close(&self) -> bool {
        match self.s_norm() {
            Some(s) => s < &self.tol,
            None => false,
        }
    }
    pub fn gradient_next_iterate_too_close(&self) -> bool {
        match self.y_norm() {
            Some(y) => y < &self.tol,
            None => false,
        }
    }
    pub fn new(line_search: T, tol: Floating, x0: DVector<Floating>) -> Self {
        let n = x0.len();
        let identity = DMatrix::identity(n, n);
        BFGS {
            line_search,
            approx_inv_hessian: identity.clone(),
            x: x0,
            k: 0,
            tol,
            s_norm: None,
            y_norm: None,
            identity,
        }
    }
}

impl<T> ComputeDirection for BFGS<T> {
    fn compute_direction(&mut self, eval: &FuncEval) -> DVector<Floating> {
        -&self.approx_inv_hessian * eval.g()
    }
}

impl<T> Solver for BFGS<T>
where
    T: LineSearch,
{
    type LS = T;
    fn k(&self) -> &usize {
        &self.k
    }
    fn xk(&self) -> &DVector<Floating> {
        &self.x
    }
    fn xk_mut(&mut self) -> &mut DVector<Floating> {
        &mut self.x
    }
    fn k_mut(&mut self) -> &mut usize {
        &mut self.k
    }
    fn line_search(&self) -> &Self::LS {
        &self.line_search
    }
    fn has_converged(&self, eval: &FuncEval) -> bool {
        // either the gradient is small or the difference between the iterates is small
        // eval.g().norm() < self.tol || self.next_iterate_too_close()
        if self.next_iterate_too_close() {
            warn!("Minimization completed: next iterate too close");
            true
        } else if self.gradient_next_iterate_too_close() {
            warn!("Minimization completed: gradient next iterate too close");
            true
        } else {
            eval.g().norm() < self.tol
        }
    }
    fn step_hook(
        &mut self,
        _eval: &FuncEval,
        _direction: &DVector<Floating>,
        _step: &Floating,
        _next_iterate: &DVector<Floating>,
        _oracle: &impl Fn(&DVector<Floating>) -> FuncEval,
    ) {
        // We update the inverse hessian in the corrections in this hook which is triggered just after the calculation of the next iterate
        let s = _next_iterate - &self.x;
        self.s_norm = Some(s.norm());
        if self.next_iterate_too_close() {
            return;
        }
        let y = _oracle(_next_iterate).g() - _eval.g();
        self.y_norm = Some(y.norm());
        if self.gradient_next_iterate_too_close() {
            return;
        }
        let ys = &y.dot(&s);
        let rho = 1.0 / ys;
        let w_a = &s * &y.transpose();
        let w_b = &y * &s.transpose();
        let innovation = &s * &s.transpose();
        let left_term = self.identity() - (w_a * rho);
        let right_term = self.identity() - (w_b * rho);
        self.approx_inv_hessian =
            (left_term * &self.approx_inv_hessian * right_term) + innovation * rho;
    }
}

mod test_bfgs {
    use super::*;
    #[test]
    fn test_outer() {
        let a = DVector::from_vec(vec![1.0, 2.0]);
        let b = DVector::from_vec(vec![3.0, 4.0]);
        let c = a * b.transpose();
        println!("{:?}", c);
    }

    #[test]
    pub fn test_min() {
        std::env::set_var("RUST_LOG", "info");

        let tracer = Tracer::default()
            .with_stdout_layer(Some(LogFormat::Normal))
            .build();
        let gamma = 90.0;
        let f_and_g = |x: &DVector<Floating>| -> FuncEval {
            let f = 0.5 * (x[0].powi(2) + gamma * x[1].powi(2));
            let g = DVector::from(vec![x[0], gamma * x[1]]);
            (f, g).into()
        };

        // Linesearch builder
        let alpha = 1e-4;
        let beta = 0.5;
        let ls = BackTracking::new(alpha, beta);

        // pnorm descent builder
        let tol = 1e-12;
        let x_0 = DVector::from(vec![180.0, 152.0]);
        let mut gd = BFGS::new(ls, tol, x_0);

        // Minimization
        let max_iter = 1000;

        gd.minimize(f_and_g, max_iter);

        println!("Iterate: {:?}", gd.xk());

        let eval = f_and_g(gd.xk());
        println!("Function eval: {:?}", eval);
        println!("Gradient norm: {:?}", eval.g().norm());
        println!("tol: {:?}", tol);

        let convergence = gd.has_converged(&eval);
        println!("Convergence: {:?}", convergence);

        assert!((eval.f() - 0.0).abs() < 1e-6);
    }
}
