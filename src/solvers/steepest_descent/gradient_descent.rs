use super::*;

pub struct GradientDescent<T> {
    line_search: T, // line search algorithm to compute step length after finding a direction
    grad_tol: Floating, // tolerance for the gradient as exit condition
    direction: DVector<Floating>, // buffer to store the result of the matrix-vector product
}

impl<T> GradientDescent<T>
where
    T: LineSearch,
{
    pub fn new(line_search: T, grad_tol: Floating, n: usize) -> Self {
        GradientDescent {
            line_search,
            grad_tol,
            direction: DVector::zeros(n),
        }
    }
}

impl<T> SteepestDescent for GradientDescent<T>
where
    T: LineSearch,
{
    type LineSearch = T;

    fn compute_direction(&mut self, grad_k: &DVector<Floating>) {
        self.direction = -grad_k;
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

mod gradient_descent_test {
    use super::*;

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
        // Linesearch builder
        let alpha = 1e-4;
        let beta = 0.5;
        let ls = BackTracking::new(alpha, beta);

        // Gradient descent builder
        let tol = 1e-12;
        let n = 2;
        let mut gd = GradientDescent::new(ls, tol, n);

        // Minimization
        let max_iter = 1000;
        let x_0 = DVector::from(vec![180.0, 152.0]);
        let x_min = gd.minimize(x_0, f_and_g, max_iter);

        assert!((x_min[0] - 0.0).abs() < 1e-6);
    }
}
