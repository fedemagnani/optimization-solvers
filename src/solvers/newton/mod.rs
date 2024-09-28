use super::*;

pub struct Newton<T> {
    pub line_search: T,
    pub tol: Floating,
}

impl<T> Newton<T>
where
    T: LineSearch,
{
    pub fn new(line_search: T, tol: Floating) -> Self {
        Newton { line_search, tol }
    }

    // compute the descent direction
    fn compute_direction_and_decrement_squared(
        &self,
        grad_k: &DVector<Floating>,
        hessian_k: &DMatrix<Floating>,
    ) -> (DVector<Floating>, Floating) {
        // //[TODO]: Boyd recommends several alternatives to the solution of Newton system which take advantage of prior information about sparsity/banded bandwidth of the hessian.
        // let cholesky = hessian_k.clone().cholesky().unwrap();
        // let w = cholesky.solve(&-grad_k);
        // (cholesky.solve(&w), w.dot(&w))

        // we verify that the matrix is not null
        match hessian_k.clone().try_inverse() {
            Some(hessian_inv) => {
                let direction = -&hessian_inv * grad_k;
                let decrement_squared = (hessian_inv.clone().mul(&direction)).dot(&direction);
                (direction, decrement_squared)
            }
            None => {
                warn!(target:"newton","Hessian is singular. Using gradient descent direction.");
                (-grad_k.clone(), grad_k.dot(grad_k))
            }
        }
    }
    fn minimize(
        &mut self,
        x_0: DVector<Floating>, // initial iterate
        f_and_g: impl Fn(&DVector<Floating>) -> (Floating, DVector<Floating>), // oracle that returns the value of the function and its gradient
        hessian: impl Fn(&DVector<Floating>) -> DMatrix<Floating>, // oracle that returns the hessian of the function
        max_iter: usize,                                           // maximum number of iterations
    ) -> DVector<Floating> {
        let mut x_k = x_0;
        let mut i = 0;
        let mut convergence = false;
        while max_iter > i {
            let (_, grad_k) = f_and_g(&x_k);

            let hessian_k = hessian(&x_k);
            let (direction, decrement_squared) =
                self.compute_direction_and_decrement_squared(&grad_k, &hessian_k);

            if decrement_squared * 0.5 < self.tol {
                // exit condition: half of the squared newton decrement is lower than tolerance
                convergence = true;
                break;
            }

            let step = self
                .line_search
                .compute_step_len(&x_k, &direction, &f_and_g, max_iter);
            x_k += step * &direction;
            i += 1;
        }
        if convergence {
            warn!(target:"steepest descent","Convergence in {} iterations. Optimal point: {:?}", i, x_k);
        } else {
            warn!(target:"steepest descent","Maximum number of iterations reached. No convergence.");
        }
        x_k
    }
}

mod newton_test {
    use super::*;
    use nalgebra::{Matrix, Matrix2, Vector2};

    #[test]
    pub fn test_min() {
        std::env::set_var("RUST_LOG", "info");

        let tracer = Tracer::default()
            .with_stdout_layer(Some(LogFormat::Normal))
            .build();
        let gamma = 1222.0;
        let f_and_g = |x: &DVector<Floating>| -> (Floating, DVector<Floating>) {
            let f = 0.5 * (x[0].powi(2) + gamma * x[1].powi(2));
            let g = DVector::from(vec![x[0], gamma * x[1]]);
            (f, g)
        };
        let hessian = |_: &DVector<Floating>| -> DMatrix<Floating> {
            DMatrix::from_iterator(2, 2, vec![1.0, 0.0, 0.0, gamma])
        };

        // Linesearch builder
        let alpha = 1e-4;
        let beta = 0.5;
        let ls = BackTracking::new(alpha, beta);

        // newton builder
        let tol = 1e-8;
        let mut nt = Newton::new(ls, tol);

        // Minimization
        let max_iter = 1000;
        let x_0 = DVector::from(vec![1.0, 1.0]);
        let x_min = nt.minimize(x_0, f_and_g, hessian, max_iter);
        assert!((x_min[0] - 0.0).abs() < 1e-4);
    }
}
