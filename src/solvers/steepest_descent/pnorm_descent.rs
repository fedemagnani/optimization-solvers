use super::*;
// Notice that the the pnorm descent is equivalent to the steepest descent with P= identity matrix
// This approach finds the direction of the steepest descent by minimizing the directional derivative (at current iterate) over the ellipsoid {d: d^T P d <= 1} (which could be thought as the unit ball of the P-norm ||P^(-1/2) d||_2)
// The best thing would be picking a matrix P (and then compute its inverse) such that the P is a good approximation of the hessian of the function. By doing this, the condition number of the hessian is in control and the convergence rate of the algorithm is improved. It's from this rationale that newton and quasi-newton methods are born.

pub struct PnormDescent<T> {
    line_search: T, // line search algorithm to compute step length after finding a direction
    grad_tol: Floating, // tolerance for the gradient as exit condition
    inverse_p: DMatrix<Floating>, // inverse of the matrix P
    direction: DVector<Floating>, // buffer to store the result of the matrix-vector product
}

impl<T> PnormDescent<T>
where
    T: LineSearch,
{
    pub fn new(line_search: T, grad_tol: Floating, inverse_p: DMatrix<Floating>) -> Self {
        PnormDescent {
            line_search,
            grad_tol,
            direction: DVector::zeros(inverse_p.nrows()),
            inverse_p,
        }
    }
}

impl<T> SteepestDescent for PnormDescent<T>
where
    T: LineSearch,
{
    type LineSearch = T;

    fn compute_direction(&mut self, grad_k: &DVector<Floating>) {
        self.inverse_p.mul_to(grad_k, &mut self.direction);
        // We make the direction negative
        self.direction.neg_mut();
    }

    fn direction(&self) -> &DVector<Floating> {
        &self.direction
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
    use nalgebra::{Matrix, Matrix2, Vector2};

    #[test]
    pub fn test_min() {
        std::env::set_var("RUST_LOG", "info");

        let tracer = Tracer::default()
            .with_stdout_layer(Some(LogFormat::Normal))
            .build();
        let gamma = 90.0;
        let f_and_g = |x: &DVector<Floating>| -> (Floating, DVector<Floating>) {
            let f = 0.5 * (x[0].powi(2) + gamma * x[1].powi(2));
            let g = DVector::from(vec![x[0], gamma * x[1]]);
            (f, g)
        };
        // We compute the inverse hessian of the function. Notice that the hessian is constant since the objective function is quadratic
        let inv_hessian = DMatrix::from_iterator(2, 2, vec![1.0, 0.0, 0.0, 1.0 / gamma]);
        // Linesearch builder
        let alpha = 1e-4;
        let beta = 0.5;
        let ls = BackTracking::new(alpha, beta);

        // pnorm descent builder
        let tol = 1e-12;
        let mut gd = PnormDescent::new(ls, tol, inv_hessian);

        // Minimization
        let max_iter = 1000;
        let x_0 = DVector::from(vec![180.0, 152.0]);
        let x_min = gd.minimize(x_0, f_and_g, max_iter);

        assert!((x_min[0] - 0.0).abs() < 1e-6);
    }
}
