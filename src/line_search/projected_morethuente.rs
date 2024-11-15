use super::*;

// Implementation from https://www.ii.uib.no/~lennart/drgrad/More1994.pdf (More, Thuente 1994) and https://bayanbox.ir/view/1460469776013846613/Sun-Yuan-Optimization-theory.pdf (Sun, Yuan 2006)

#[derive(Debug, Clone, derive_getters::Getters)]
pub struct ProjectedMoreThuente {
    c1: Floating, //mu (armijo sensitivity)
    c2: Floating, //eta (curvature sensitivity)
    t_min: Floating,
    t_max: Floating,
    delta_min: Floating,
    delta: Floating,
    delta_max: Floating,
    lower_bound: DVector<Floating>,
    upper_bound: DVector<Floating>,
}

impl ProjectedMoreThuente {
    pub fn new(lower_bound: DVector<Floating>, upper_bound: DVector<Floating>) -> Self {
        ProjectedMoreThuente {
            c1: 1e-4,
            c2: 0.9,
            t_min: 0.0,
            t_max: Floating::INFINITY,
            delta_min: 0.58333333,
            delta: 0.66,
            delta_max: 1.1,
            lower_bound,
            upper_bound,
        }
    }
    pub fn with_deltas(
        mut self,
        delta_min: Floating,
        delta: Floating,
        delta_max: Floating,
    ) -> Self {
        self.delta_min = delta_min;
        self.delta = delta;
        self.delta_max = delta_max;
        self
    }
    pub fn with_t_min(mut self, t_min: Floating) -> Self {
        self.t_min = t_min;
        self
    }
    pub fn with_t_max(mut self, t_max: Floating) -> Self {
        self.t_max = t_max;
        self
    }
    pub fn with_c1(mut self, c1: Floating) -> Self {
        assert!(c1 > 0.0, "c1 must be positive");
        assert!(c1 < self.c2, "c1 must be less than c2");
        self.c1 = c1;
        self
    }
    pub fn with_c2(mut self, c2: Floating) -> Self {
        assert!(c2 > 0.0, "c2 must be positive");
        assert!(c2 < 1.0, "c2 must be less than 1");
        assert!(c2 > self.c1, "c2 must be greater than c1");
        self.c2 = c2;
        self
    }

    pub fn update_interval(
        f_tl: &Floating,
        f_t: &Floating,
        g_t: &Floating,
        tl: &mut Floating,
        t: Floating,
        tu: &mut Floating,
    ) -> bool {
        // case U1 in Update Algorithm and Case a in Modified Update Algorithm
        if f_t > f_tl {
            *tu = t;
            false
        }
        // case U2 in Update Algorithm and Case b in Modified Update Algorithm
        else if g_t * (*tl - t) > 0. {
            *tl = t;
            false
        }
        // case U3 in Update Algorithm and Case c in Modified Update Algorithm
        else if g_t * (*tl - t) < 0. {
            *tu = *tl;
            *tl = t;
            false
        } else {
            //interval converged to a point
            true
        }
    }

    pub fn cubic_minimizer(
        ta: &Floating,
        tb: &Floating,
        f_ta: &Floating,
        f_tb: &Floating,
        g_ta: &Floating,
        g_tb: &Floating,
    ) -> Floating {
        // Equation 2.4.51 [Sun, Yuan 2006]

        let s = 3. * (f_tb - f_ta) / (tb - ta);
        let z = s - g_ta - g_tb;
        let w = (z.powi(2) - g_ta * g_tb).sqrt();
        // Equation 2.4.56 [Sun, Yuan 2006]
        ta + ((tb - ta) * ((w - g_ta - z) / (g_tb - g_ta + 2. * w)))
    }

    pub fn quadratic_minimzer_1(
        ta: &Floating,
        tb: &Floating,
        f_ta: &Floating,
        f_tb: &Floating,
        g_ta: &Floating,
    ) -> Floating {
        // Equation 2.4.2 [Sun, Yuan 2006]
        let lin_int = (f_ta - f_tb) / (ta - tb);

        ta - 0.5 * ((ta - tb) * g_ta / (g_ta - lin_int))
    }

    pub fn quadratic_minimizer_2(
        ta: &Floating,
        tb: &Floating,
        g_ta: &Floating,
        g_tb: &Floating,
    ) -> Floating {
        // Equation 2.4.5 [Sun, Yuan 2006]

        ta - g_ta * ((ta - tb) / (g_ta - g_tb))
    }

    pub fn phi(eval: &FuncEvalMultivariate, direction_k: &DVector<Floating>) -> FuncEvalUnivariate {
        // recall that phi(t) = f(x + t * direction). Thus via chain rule nabla phi(t) = <nabla f(x + t * direction), direction> (i.e. the directional derivative of f at x + t * direction in the direction of direction)
        let image = eval.f();
        let derivative = eval.g().dot(direction_k);
        FuncEvalUnivariate::new(*image, derivative)
    }
    pub fn psi(
        &self,
        phi_0: &FuncEvalUnivariate,
        phi_t: &FuncEvalUnivariate,
        t: &Floating,
    ) -> FuncEvalUnivariate {
        let image = phi_t.f() - phi_0.f() - self.c1 * t * phi_0.g();
        let derivative = phi_t.g() - self.c1 * phi_0.g();
        FuncEvalUnivariate::new(image, derivative)
    }
}

impl SufficientDecreaseCondition for ProjectedMoreThuente {
    fn c1(&self) -> Floating {
        self.c1
    }
}

impl CurvatureCondition for ProjectedMoreThuente {
    fn c2(&self) -> Floating {
        self.c2
    }
}

impl LineSearch for ProjectedMoreThuente {
    fn compute_step_len(
        &self,
        x_k: &DVector<Floating>,         // current iterate
        direction_k: &DVector<Floating>, // direction of the ray along which we are going to search
        oracle: &impl Fn(&DVector<Floating>) -> FuncEvalMultivariate, // oracle
        max_iter: usize, // maximum number of iterations during line search (if direction update is costly, set this high to perform more exact line search)
    ) -> Floating {
        let mut use_modified_updating = false;
        let mut interval_converged = false;

        let mut t = 1.0f64.max(self.t_min).min(self.t_max);
        let mut tl = self.t_min;
        let mut tu = self.t_max;

        for i in 0..max_iter {
            let eval_0 = oracle(x_k);
            let proj_xt =
                (x_k + t * direction_k).box_projection(&self.lower_bound, &self.upper_bound);
            let eval_t = oracle(&proj_xt);
            // Check for convergence
            if self.strong_wolfe_conditions(
                eval_0.f(),
                eval_t.f(),
                eval_0.g(),
                eval_t.g(),
                direction_k,
            ) {
                info!("Strong Wolfe conditions satisfied at iteration {}", i);

                return t;
            } else if interval_converged {
                info!("Interval converged at iteration {}", i);

                return t;
            } else if t == self.t_min {
                info!("t is at the minimum value at iteration {}", i);
                return t;
            } else if t == self.t_max {
                info!("t is at the maximum value at iteration {}", i);
                return t;
            }

            let phi_t = Self::phi(&eval_t, direction_k);
            let phi_0 = Self::phi(&eval_0, direction_k);

            let psi_t = self.psi(&phi_0, &phi_t, &t);

            if !use_modified_updating && psi_t.f() <= &0. && phi_t.g() > &0. {
                //paper suggests that when the conidition is verified, you start using the modified updating and never go back
                use_modified_updating = true;
            }

            let proj_xtl =
                (x_k + tl * direction_k).box_projection(&self.lower_bound, &self.upper_bound);
            let eval_tl = oracle(&proj_xtl);
            let phi_tl = Self::phi(&eval_tl, direction_k);

            // using auxiliary or modified evaluation according to the flag
            let (f_tl, g_tl, f_t, g_t) = if use_modified_updating {
                (*phi_tl.f(), *phi_tl.g(), phi_t.f(), phi_t.g())
            } else {
                let psi_tl = self.psi(&phi_0, &phi_tl, &tl);
                (*psi_tl.f(), *psi_tl.g(), psi_t.f(), psi_t.g())
            };

            //Trial value selection (section 4 of the paper)
            //case 1
            if f_t > &f_tl {
                let tc = Self::cubic_minimizer(&tl, &t, &f_tl, f_t, &g_tl, g_t);
                let tq = Self::quadratic_minimzer_1(&tl, &t, &f_tl, f_t, &g_tl);

                if (tc - tl).abs() < (tq - tl).abs() {
                    t = tc;
                } else {
                    t = 0.5 * (tq + tc); //midpoint
                }
            }
            //case 2 (here f_t <= &f_tl)
            else if g_t * g_tl < 0. {
                let tc = Self::cubic_minimizer(&tl, &t, &f_tl, f_t, &g_tl, g_t);
                let ts = Self::quadratic_minimizer_2(&tl, &t, &g_tl, g_t);

                if (tc - t).abs() >= (ts - t).abs() {
                    t = tc;
                } else {
                    t = ts;
                }
            }
            //case 3 (here f_t <= &f_tl, g_t * g_tl >= 0.)
            else if g_t.abs() <= g_tl.abs() {
                let tc = Self::cubic_minimizer(&tl, &t, &f_tl, f_t, &g_tl, g_t);
                let ts = Self::quadratic_minimizer_2(&tl, &t, &g_tl, g_t);

                let t_plus = if (tc - t).abs() < (ts - t).abs() {
                    tc
                } else {
                    ts
                };
                if t > tl {
                    t = t_plus.min(t + self.delta * (tu - t));
                } else {
                    t = t_plus.max(t + self.delta * (tu - t));
                }
            }
            // case 4 (here f_t <= &f_tl, g_t * g_tl >= 0., g_t.abs() > g_tl.abs())
            else {
                let (f_tu, g_tu) = {
                    let eval_tu = oracle(&(x_k + tu * direction_k));
                    let phi_tu = Self::phi(&eval_tu, direction_k);
                    if use_modified_updating {
                        (*phi_tu.f(), *phi_tu.g())
                    } else {
                        let psi_tu = self.psi(&phi_0, &phi_tu, &tu);
                        (*psi_tu.f(), *psi_tu.g())
                    }
                };
                t = Self::cubic_minimizer(&tu, &t, f_t, &f_tu, g_t, &g_tu);
            }

            //clamping t to the max and min values
            t = t.max(self.t_min).min(self.t_max);

            //Updating algorithm (section 2 and 3 of the paper)
            interval_converged = Self::update_interval(&f_tl, f_t, g_t, &mut tl, t, &mut tu)
        }
        warn!("Line search did not converge in {} iterations", max_iter);
        t
    }
}

mod morethuente_test {
    use super::*;
    #[test]
    pub fn test_phi() {
        std::env::set_var("RUST_LOG", "info");

        // in this example the objecive function has constant hessian, thus its condition number doesn't change on different points.
        // Recall that in gradient descent method, the upper bound of the log error is positive function of the upper bound of condition number of the hessian (ratio between max and min eigenvalue).
        // This causes poor performance when the hessian is ill conditioned
        let tracer = Tracer::default()
            .with_stdout_layer(Some(LogFormat::Normal))
            .build();
        let gamma = 90.0;
        let f_and_g = |x: &DVector<Floating>| -> FuncEvalMultivariate {
            let f = 0.5 * (x[0].powi(2) + gamma * x[1].powi(2));
            let g = DVector::from(vec![x[0], gamma * x[1]]);
            (f, g).into()
        };
        let max_iter = 1000;
        //here we define a rough gradient descent method that uses ls line search
        let mut k = 1;
        let mut iterate = DVector::from(vec![180.0, 152.0]);
        let lower_bound: DVector<Floating> = vec![1e-5, 1.].into();
        let upper_bound: DVector<Floating> = vec![Floating::INFINITY, Floating::INFINITY].into();
        let ls = ProjectedMoreThuente::new(lower_bound.clone(), upper_bound.clone());
        // let ls = BackTracking::new(1e-4, 0.5);
        let gradient_tol = 1e-12;

        while max_iter > k {
            debug!("Iterate: {:?}", iterate);
            let eval = f_and_g(&iterate);
            // we do a rough check on the squared norm of the gradient to verify convergence
            if eval.g().dot(eval.g()) < gradient_tol {
                warn!("Gradient norm is lower than tolerance. Convergence!.");
                break;
            }

            let direction = -eval.g();
            let t = <ProjectedMoreThuente as LineSearch>::compute_step_len(
                &ls, &iterate, &direction, &f_and_g, max_iter,
            );

            // As exit condition, we take the infinity norm of the difference between the projected iterate and the next iterate
            //we perform the update
            let cached_iterate = iterate.clone();
            iterate += t * direction;
            iterate = iterate.box_projection(&lower_bound, &upper_bound);
            if (cached_iterate - &iterate)
                .iter()
                .fold(0.0f64, |acc, x| acc.max(x.abs()))
                < gradient_tol
            {
                warn!("Infinity norm of iterate difference is lower than tolerance. Convergence!.");
                break;
            }
            k += 1;
        }
        println!("Iterate: {:?}", iterate);
        println!("Function eval: {:?}", f_and_g(&iterate));
        // assert!((iterate[0] - 0.0).abs() < 1e-6);
        info!("Test took {} iterations", k);
    }
}
