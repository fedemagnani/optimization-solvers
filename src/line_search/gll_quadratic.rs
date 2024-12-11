use super::*;

pub struct GLLQuadratic {
    c1: Floating,
    // setting m equal to one is equivalent to the monotone line search vith armijo condition
    m: usize,
    f_previous: Vec<Floating>,
    sigma1: Floating,
    sigma2: Floating,
}

impl GLLQuadratic {
    pub fn new(c1: Floating, m: usize) -> Self {
        let sigma1 = 0.1;
        let sigma2 = 0.9;
        Self {
            c1,
            m,
            f_previous: vec![],
            sigma1,
            sigma2,
        }
    }
    pub fn with_sigmas(mut self, sigma1: Floating, sigma2: Floating) -> Self {
        self.sigma1 = sigma1;
        self.sigma2 = sigma2;
        self
    }

    fn append_new_f(&mut self, f: Floating) {
        if self.f_previous.len() == self.m {
            self.f_previous.remove(0);
        }
        self.f_previous.push(f);
    }

    fn f_max(&self) -> Floating {
        let max = self
            .f_previous
            .iter()
            .fold(Floating::NEG_INFINITY, |acc, x| x.max(acc));
        max
    }
}

impl SufficientDecreaseCondition for GLLQuadratic {
    fn c1(&self) -> Floating {
        self.c1
    }
}

impl LineSearch for GLLQuadratic {
    fn compute_step_len(
        &mut self,
        x_k: &DVector<Floating>,         // current iterate
        eval_x_k: &FuncEvalMultivariate, // function evaluation at x_k
        direction_k: &DVector<Floating>, // direction of the ray along which we are going to search
        oracle: & mut impl FnMut(&DVector<Floating>) -> FuncEvalMultivariate, // oracle
        max_iter: usize, // maximum number of iterations during line search (if direction update is costly, set this high to perform more exact line search)
    ) -> Floating {
        // we append the function eval to the previous function evals
        self.append_new_f(*eval_x_k.f());
        let mut t = 1.0;
        let f_max = self.f_max();
        let mut i = 0;

        while max_iter > i {
            let x_kp1 = x_k + t * direction_k;

            let eval_kp1 = oracle(&x_kp1);

            // armijo condition
            if self.sufficient_decrease(&f_max, eval_kp1.f(), eval_x_k.g(), &t, direction_k) {
                trace!(target: "gll quadratic line search", "Sufficient decrease condition met. Exiting with step size: {:?}", t);
                return t;
            }

            if t <= 0.1 {
                trace!(target: "gll quadratic line search", "Step size too small: {}; Bissecting.", t);
                t *= 0.5;
            } else {
                // here step size is sufficiently large to perform a quadratic interpolation
                let t_tmp = -0.5 * t * t * eval_x_k.g().dot(direction_k)
                    / (eval_kp1.f() - eval_x_k.f() - t * eval_x_k.g().dot(direction_k));
                if t_tmp > self.sigma1 && t_tmp < self.sigma2 * t {
                    trace!(target: "gll quadratic line search", "Safeguarded step size: {}", t_tmp);
                    t = t_tmp;
                } else {
                    // if step is not safeguarded, we take a conservative bissected step
                    trace!(target: "gll quadratic line search", "t_tmp = {} not in [{}, {}]. Bissecting t_tmp.", t_tmp, self.sigma1, self.sigma2 * t);
                    t = t_tmp * 0.5;
                }
            }

            i += 1;
        }
        trace!(target: "gll quadratic line search", "Max iter reached. Early stopping.");
        t
    }
}
