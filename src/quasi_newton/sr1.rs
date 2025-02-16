use super::*;

#[derive(derive_getters::Getters)]
pub struct SR1 {
    approx_inv_hessian: DMatrix<Floating>,
    x: DVector<Floating>,
    k: usize,
    tol: Floating,
    s_norm: Option<Floating>,
    y_norm: Option<Floating>,
    identity: DMatrix<Floating>,
}

impl SR1 {
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
    pub fn new(tol: Floating, x0: DVector<Floating>) -> Self {
        let n = x0.len();
        let identity = DMatrix::identity(n, n);
        SR1 {
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

impl ComputeDirection for SR1 {
    fn compute_direction(
        &mut self,
        eval: &FuncEvalMultivariate,
    ) -> Result<DVector<Floating>, SolverError> {
        Ok(-&self.approx_inv_hessian * eval.g())
    }
}

impl LineSearchSolver for SR1 {
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
    fn has_converged(&self, eval: &FuncEvalMultivariate) -> bool {
        // either the gradient is small or the difference between the iterates is small
        // eval.g().norm() < self.tol || self.next_iterate_too_close()
        if self.next_iterate_too_close() {
            warn!(target: "SR1","Minimization completed: next iterate too close");
            true
        } else if self.gradient_next_iterate_too_close() {
            warn!(target: "SR1","Minimization completed: gradient next iterate too close");
            true
        } else {
            eval.g().norm() < self.tol
        }
    }

    fn update_next_iterate<LS: LineSearch>(
        &mut self,
        line_search: &mut LS,
        eval_x_k: &FuncEvalMultivariate,
        oracle: &mut impl FnMut(&DVector<Floating>) -> FuncEvalMultivariate,
        direction: &DVector<Floating>,
        max_iter_line_search: usize,
    ) -> Result<(), SolverError> {
        let step = line_search.compute_step_len(
            self.xk(),
            eval_x_k,
            direction,
            oracle,
            max_iter_line_search,
        );

        let next_iterate = self.xk() + step * direction;

        let s = &next_iterate - &self.x;
        self.s_norm = Some(s.norm());
        let y = oracle(&next_iterate).g() - eval_x_k.g();
        self.y_norm = Some(y.norm());

        //updating iterate here, and then we will update the inverse hessian (if corrections are not too small)
        *self.xk_mut() = next_iterate;

        // We update the inverse hessian and the corrections in this hook which is triggered just after the calculation of the next iterate

        if self.next_iterate_too_close() {
            return Ok(());
        }

        if self.gradient_next_iterate_too_close() {
            return Ok(());
        }

        // SR1 update
        let hy = &self.approx_inv_hessian * &y;
        let shy = s - hy;
        self.approx_inv_hessian += &shy * shy.transpose() / shy.dot(&y);
        Ok(())
    }
}

#[cfg(test)]
mod test_sr1 {
    use super::*;
    #[test]
    fn test_outer() {
        let a = DVector::from_vec(vec![1.0, 2.0]);
        let b = DVector::from_vec(vec![3.0, 4.0]);
        let c = a * b.transpose();
        println!("{:?}", c);
    }

    #[test]
    pub fn sr1_morethuente() {
        std::env::set_var("RUST_LOG", "info");

        let _ = Tracer::default()
            .with_stdout_layer(Some(LogFormat::Normal))
            .build();
        let gamma = 1.;
        let f_and_g = |x: &DVector<Floating>| -> FuncEvalMultivariate {
            let f = 0.5 * ((x[0] + 1.).powi(2) + gamma * (x[1] - 1.).powi(2));
            let g = DVector::from(vec![x[0] + 1., gamma * (x[1] - 1.)]);
            (f, g).into()
        };

        // Linesearch builder

        let mut ls = MoreThuente::default();

        // pnorm descent builder
        let tol = 1e-12;
        let x_0 = DVector::from(vec![180.0, 152.0]);
        let mut gd = SR1::new(tol, x_0);

        // Minimization
        let max_iter_solver = 1000;
        let max_iter_line_search = 100000;

        gd.minimize(
            &mut ls,
            f_and_g,
            max_iter_solver,
            max_iter_line_search,
            None,
        )
        .unwrap();

        println!("Iterate: {:?}", gd.xk());

        let eval = f_and_g(gd.xk());
        println!("Function eval: {:?}", eval);
        println!("Gradient norm: {:?}", eval.g().norm());
        println!("tol: {:?}", tol);

        let convergence = gd.has_converged(&eval);
        println!("Convergence: {:?}", convergence);

        assert!((eval.f() - 0.0).abs() < 1e-6);
    }

    #[test]
    pub fn sr1_backtracking() {
        std::env::set_var("RUST_LOG", "info");

        let _ = Tracer::default()
            .with_stdout_layer(Some(LogFormat::Normal))
            .build();
        let gamma = 1.;
        let f_and_g = |x: &DVector<Floating>| -> FuncEvalMultivariate {
            let f = 0.5 * ((x[0] + 1.).powi(2) + gamma * (x[1] - 1.).powi(2));
            let g = DVector::from(vec![x[0] + 1., gamma * (x[1] - 1.)]);
            (f, g).into()
        };

        // Linesearch builder
        let alpha = 1e-4;
        let beta = 0.5; //0.5 is like backtracking line search
                        // let ls = BackTracking::new(alpha, beta);
        let mut ls = BackTracking::new(alpha, beta);

        // pnorm descent builder
        let tol = 1e-12;
        let x_0 = DVector::from(vec![180.0, 152.0]);
        let mut gd = SR1::new(tol, x_0);

        // Minimization
        let max_iter_solver = 1000;
        let max_iter_line_search = 100000;

        gd.minimize(
            &mut ls,
            f_and_g,
            max_iter_solver,
            max_iter_line_search,
            None,
        )
        .unwrap();

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
