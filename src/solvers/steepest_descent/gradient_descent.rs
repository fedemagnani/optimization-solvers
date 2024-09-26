use super::*;

pub struct GradientDescent<T> {
    line_search: T, // line search algorithm to compute step length after finding a direction
    grad_tol: Floating, // tolerance for the gradient as exit condition
}

impl<T> Default for GradientDescent<T>
where
    T: Default,
{
    fn default() -> Self {
        GradientDescent {
            line_search: T::default(),
            grad_tol: 1e-6,
        }
    }
}

impl<T> GradientDescent<T>
where
    T: LineSearch,
{
    pub fn new(line_search: T, grad_tol: Floating) -> Self {
        GradientDescent {
            line_search,
            grad_tol,
        }
    }
}

impl<T> SteepestDescent for GradientDescent<T>
where
    T: LineSearch,
{
    type LineSearch = T;

    fn compute_direction(&self, grad_k: &Array1<Floating>) -> Array1<Floating> {
        -grad_k
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

        // Gradient descent builder
        let tol = 1e-12;
        let gd = GradientDescent::new(ls, tol);

        // Minimization
        let max_iter = 1000;
        let x_0 = arr1(&[180.0, 152.0]);
        let x_min = gd.minimize(x_0, f_and_g, max_iter);

        assert!((x_min[0] - 0.0).abs() < 1e-6);
    }
}
