use super::*;
pub mod gradient_descent;
pub use gradient_descent::*;

pub mod coordinate_descent;
pub use coordinate_descent::*;

pub mod pnorm_descent;
pub use pnorm_descent::*;

// All the algorithms in the family of steepest descent differ only in the way they compute the descent direction (i.e. they differ in the norm used so that the associated unit ball is the constraint set on which search the direction that minimizes the directional derivative at the current iterate. Typically this minimizer is a unit vector but any scaled version of the vector is good (the line search will adjust the direction later), so it's good supplying the rescaled version of the minimizer which has minimal computational cost).

// the family of steepest descent algorithms has (at most) linear convergence rate, and it's possible to see it by computing the trajectory of the upper bound of the log-suboptimality error ln(f(x_k)-p^*) where p^* is the optimal value of the problem. In particular, the convergence drops significantly if the upper bound of the condition number of the hessian matrix of the function is high (you can see it by solving the log-suboptimality error trajectory for the iteration number k). Recall that an upper bound on the condition number of the hessian can be derived by taking the ratio between the maximal and the minimal eigenvalue of the hessian matrix. This condition number can be also thought as the volume of the ellipsoid {x: x^T H x <= 1} where H is the hessian matrix of the function, which is always relatable to the volume of the euclidean unit ball gamma*sqrt{det (H^TH)} where gamma is the volume of the euclidean unit ball.The p-norm descent tries to tackle this issue by taking a Matrix P that proxies correctly the hessian matrix (i.e. its unit norm {x: x^T P x <= 1} is a good approximation of the sublevel sets of the function), and this adjustments decreases the condition number of P^{-0.5} H P^{-0.5} because it would resemble (more or less) the identity matrix. It's from this intuition that the newton and quasi-newton methods become more clear.

pub trait SteepestDescent {
    type LineSearch: LineSearch;
    // compute the descent direction
    fn compute_direction(&mut self, grad_k: &DVector<Floating>);
    fn direction(&self) -> &DVector<Floating>;
    fn grad_tol(&self) -> Floating;
    fn line_search(&self) -> &Self::LineSearch;
    fn minimize(
        &mut self,
        x_0: DVector<Floating>, // initial iterate
        f_and_g: impl Fn(&DVector<Floating>) -> (Floating, DVector<Floating>), // oracle that returns the value of the function and its gradient
        max_iter: usize, // maximum number of iterations
    ) -> DVector<Floating> {
        let mut x_k = x_0;
        let mut i = 0;
        let mut convergence = false;
        while max_iter > i {
            let (_, grad_k) = f_and_g(&x_k);
            if grad_k.dot(&grad_k) < self.grad_tol() {
                // exit condition: squared norm of gradient is lower than tolerance
                convergence = true;
                break;
            }
            self.compute_direction(&grad_k);
            let direction_k = self.direction();
            let step = self
                .line_search()
                .compute_step_len(&x_k, &direction_k, &f_and_g, max_iter);
            x_k = x_k + step * direction_k;
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
