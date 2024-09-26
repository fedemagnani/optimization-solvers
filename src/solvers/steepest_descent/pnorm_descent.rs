use super::*;

// Notice that the the pnorm descent is equivalent to the steepest descent with P= identity matrix
// This approach finds the direction of the steepest descent by minimizing the directional derivative (at current iterate) over the ellipsoid {d: d^T P d <= 1} (which could be thought as the unit ball of the P-norm ||P^(-1/2) d||_2)
// The best thing would be picking a matrix P (and then compute its inverse) such that the P is a good approximation of the hessian of the function. By doing this, the condition number of the hessian is in control and the convergence rate of the algorithm is improved. It's from this rationale that newton and quasi-newton methods are born.

pub struct PnormDescent<T> {
    line_search: T, // line search algorithm to compute step length after finding a direction
    grad_tol: Floating, // tolerance for the gradient as exit condition
    inverse_p: Array2<Floating>, // inverse of the matrix P
}

impl<T> PnormDescent<T>
where
    T: LineSearch,
{
    pub fn new(line_search: T, grad_tol: Floating, inverse_p: Array2<Floating>) -> Self {
        PnormDescent {
            line_search,
            grad_tol,
            inverse_p,
        }
    }
}

impl<T> SteepestDescent for PnormDescent<T>
where
    T: LineSearch,
{
    type LineSearch = T;

    fn compute_direction(&self, grad_k: &Array1<Floating>) -> Array1<Floating> {
        -self.inverse_p.dot(grad_k)
    }

    fn grad_tol(&self) -> Floating {
        self.grad_tol
    }

    fn line_search(&self) -> &Self::LineSearch {
        &self.line_search
    }
}

mod gpnorm_descent_test {
    use super::*;
    use ndarray::arr1;

    #[test]
    pub fn test_min() {
        std::env::set_var("RUST_LOG", "info");

        let tracer = Tracer::default()
            .with_stdout_layer(Some(LogFormat::Normal))
            .build();
        let gamma = 90.0;
        let f_and_g = |x: &Array1<Floating>| -> (Floating, Array1<Floating>) {
            let f = 0.5 * (x[0].powi(2) + gamma * x[1].powi(2));
            let g = arr1(&[x[0], gamma * x[1]]);
            (f, g)
        };
        // Linesearch builder
        let alpha = 1e-4;
        let beta = 0.5;
        let ls = BackTracking::new(alpha, beta);

        // pnorm descent builder
        let tol = 1e-12;
        let gd = PnormDescent::new(ls, tol, Array2::eye(2));

        // Minimization
        let max_iter = 1000;
        let x_0 = arr1(&[180.0, 152.0]);
        let x_min = gd.minimize(x_0, f_and_g, max_iter);

        assert!((x_min[0] - 0.0).abs() < 1e-6);
    }
}
