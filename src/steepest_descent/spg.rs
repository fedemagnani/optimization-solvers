//implementation of Spectral Projected Gradient method from Birgin, Ernesto & Martínez, José Mario & Raydan, Marcos. (2014). Spectral Projected Gradient Methods: Review and Perspectives. Journal of Statistical Software. 60. 1-21. 10.18637/jss.v060.i03.

// The spectral projected gradient follows the same approach of the projected gradient method, but:
// - It rescales the descent direction via a safeguarded Barzila-Borwein scalar, that is between the min and max eigenvalues of the average hessian between x_k and x_k + ts_k (hence the ``spectral'' denomination)

// - The algorithm is typically paired with a non-monotone line search (quadratic or cubic) as that one descibed in [Grippo, Lampariello, Lucidi, 1986] because sometimes enforcing the sufficient decrease condition, which is typical in armijo line search, can be too restrictive. Notice that this kind of line-search embeds the monotone line-search by simply setting to 1 the look-back parameter when evaluating the armijo condition.

use super::*;

#[derive(derive_getters::Getters)]
pub struct SpectralProjectedGradient {
    grad_tol: Floating,
    x: DVector<Floating>,
    k: usize,
    lower_bound: DVector<Floating>,
    upper_bound: DVector<Floating>,
    lambda: Floating,
    lambda_min: Floating,
    lambda_max: Floating,
}

impl SpectralProjectedGradient {
    pub fn with_lambdas(mut self, lambda_min: Floating, lambda_max: Floating) -> Self {
        self.lambda_min = lambda_min;
        self.lambda_max = lambda_max;
        self
    }
    pub fn new(
        grad_tol: Floating,
        x0: DVector<Floating>,
        oracle: &impl Fn(&DVector<Floating>) -> FuncEvalMultivariate,
        lower_bound: DVector<Floating>,
        upper_bound: DVector<Floating>,
    ) -> Self {
        let x0 = x0.box_projection(&lower_bound, &upper_bound);
        let lambda_min = 1e-3;
        let lambda_max = 1e3;

        // we initialize lambda0 as equation 8 from [Birgin, Martínez, Raydan, 2014]
        let eval0 = oracle(&x0);
        let direction0 = &x0 - eval0.g();
        let direction0 = direction0.box_projection(&lower_bound, &upper_bound);
        let direction0 = direction0 - &x0;
        let lambda = (1. / direction0.infinity_norm())
            .min(lambda_max)
            .max(lambda_min);

        Self {
            grad_tol,
            x: x0,
            k: 0,
            lower_bound,
            upper_bound,
            lambda,
            lambda_min,
            lambda_max,
        }
    }
}

impl HasBounds for SpectralProjectedGradient {
    fn lower_bound(&self) -> &DVector<Floating> {
        &self.lower_bound
    }
    fn set_lower_bound(&mut self, lower_bound: DVector<Floating>) {
        self.lower_bound = lower_bound;
    }
    fn set_upper_bound(&mut self, upper_bound: DVector<Floating>) {
        self.upper_bound = upper_bound;
    }
    fn upper_bound(&self) -> &DVector<Floating> {
        &self.upper_bound
    }
}

impl ComputeDirection for SpectralProjectedGradient {
    fn compute_direction(
        &mut self,
        eval: &FuncEvalMultivariate,
    ) -> Result<DVector<Floating>, SolverError> {
        let direction = &self.x - self.lambda * eval.g();
        let direction = direction.box_projection(&self.lower_bound, &self.upper_bound);
        let direction = direction - &self.x;
        Ok(direction)
    }
}

impl OptimizationSolver for SpectralProjectedGradient {
    fn has_converged(&self, eval: &FuncEvalMultivariate) -> bool {
        let projected_gradient = self.projected_gradient(eval);
        projected_gradient.infinity_norm() < self.grad_tol
    }
    fn xk(&self) -> &DVector<Floating> {
        &self.x
    }
    fn xk_mut(&mut self) -> &mut DVector<Floating> {
        &mut self.x
    }
    fn k(&self) -> &usize {
        &self.k
    }
    fn k_mut(&mut self) -> &mut usize {
        &mut self.k
    }
    fn update_next_iterate<LS: LineSearch>(
        &mut self,
        line_search: &mut LS,
        eval_x_k: &FuncEvalMultivariate, //eval: &FuncEvalMultivariate,
        oracle: &impl Fn(&DVector<Floating>) -> FuncEvalMultivariate,
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

        let xk = self.xk(); //immutable borrow

        debug!(target: "spectral_projected_gradient", "ITERATE: {} + {} * {} = {}", xk, step, direction, xk + step * direction);

        let next_iterate = xk + step * direction;

        // we compute the correction terms:
        let s_k = &next_iterate - xk;
        let y_k = oracle(&next_iterate).g() - eval_x_k.g();

        *self.xk_mut() = next_iterate;

        // we update the Barzilai-Borwein scalar to be used in the next iteration for computing the descent direction
        let skyk = s_k.dot(&y_k);
        if skyk <= 0. {
            debug!(target: "spectral_projected_gradient", "skyk = {} <= 0. Resetting lambda to lambda_max", skyk);
            self.lambda = self.lambda_max;
            return Ok(());
        }
        let sksk = s_k.dot(&s_k);
        self.lambda = (sksk / skyk).min(self.lambda_max).max(self.lambda_min);
        Ok(())
    }
}

mod spg_test {
    use super::*;
    #[test]
    pub fn constrained_spg_backtracking() {
        std::env::set_var("RUST_LOG", "info");

        let tracer = Tracer::default()
            .with_stdout_layer(Some(LogFormat::Normal))
            .build();
        let gamma = 1e9;
        let f_and_g = |x: &DVector<Floating>| -> FuncEvalMultivariate {
            let f = 0.5 * (x[0].powi(2) + gamma * x[1].powi(2));
            let g = DVector::from(vec![x[0], gamma * x[1]]);
            (f, g).into()
        };

        //bounds
        let lower_bounds = DVector::from_vec(vec![-1., 47.]);
        let upper_oounds = DVector::from_vec(vec![f64::INFINITY, f64::INFINITY]);
        // Linesearch builder
        let c1 = 1e-4;
        let m = 10;
        // let ls = BackTrackingB::new(alpha, beta, lower_bounds.clone(), upper_oounds.clone());
        let mut ls = GLLQuadratic::new(c1, m);

        // Gradient descent builder
        let tol = 1e-12;
        let x_0 = DVector::from(vec![180.0, 152.0]);
        let mut gd = SpectralProjectedGradient::new(tol, x_0, &f_and_g, lower_bounds, upper_oounds);

        // Minimization
        let max_iter_solver = 10000;
        let max_iter_line_search = 1000;

        gd.minimize(&mut ls, f_and_g, max_iter_solver, max_iter_line_search)
            .unwrap();

        println!("Iterate: {:?}", gd.xk());

        let eval = f_and_g(gd.xk());
        println!("Function eval: {:?}", eval);
        println!(
            "Projected Gradient norm: {:?}",
            gd.projected_gradient(&eval).norm()
        );
        println!("tol: {:?}", tol);

        let convergence = gd.has_converged(&eval);
        println!("Convergence: {:?}", convergence);
    }
}
