use tracing::info;

use super::*;

pub struct CoordinateDescent<T> {
    line_search: T, // line search algorithm to compute step length after finding a direction
    grad_tol: Floating, // tolerance for the gradient as exit condition
    direction: DVector<Floating>, // buffer to store the result of the matrix-vector product
}

impl<T> CoordinateDescent<T>
where
    T: LineSearch,
{
    pub fn new(line_search: T, grad_tol: Floating, n: usize) -> Self {
        CoordinateDescent {
            line_search,
            grad_tol,
            direction: DVector::zeros(n),
        }
    }
}

impl<T> SteepestDescent for CoordinateDescent<T>
where
    T: LineSearch,
{
    type LineSearch = T;

    fn compute_direction(&mut self, grad_k: &DVector<Floating>) {
        // Differently from the gradient descent, here we pick the highest absolute value of the gradient and we multiply it with the vector of the canonical basis associated with its entry
        let (position, max_value) =
            grad_k
                .into_iter()
                .enumerate()
                .fold((0, 0.0), |(idx, max), (i, g)| {
                    if g.abs() > max {
                        (i, g.abs())
                    } else {
                        (idx, max)
                    }
                });
        let mut direction_k = DVector::zeros(grad_k.len());
        direction_k[position] = -max_value.signum();
        self.direction.copy_from(&direction_k);
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

mod steepest_descent_l1_test {
    use super::*;
    use nalgebra::DVector;

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
        let mut sdl1 = CoordinateDescent::new(ls, tol, n);

        // Minimization
        let max_iter = 1000;
        let x_0 = DVector::from(vec![180.0, 152.0]);
        let x_min = sdl1.minimize(x_0, f_and_g, max_iter);

        assert!((x_min[0] - 0.0).abs() < 1e-6);
    }
}
