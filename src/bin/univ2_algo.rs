use nalgebra::{DMatrix, DVector};
use optimization_solvers::Floating;

fn hessian_portfolio(v: &DVector<f64>) -> DMatrix<f64> {
    let v1 = v[0];
    let v2 = v[1];
    let mut hessian = DMatrix::zeros(2, 2);
    hessian[(0, 0)] = -0.5 * v2.sqrt() / (v1 * v1.sqrt());
    let sym_term = 0.5 / ((v1 * v2).sqrt());
    hessian[(0, 1)] = sym_term;
    hessian[(1, 0)] = sym_term;
    hessian[(1, 1)] = -0.5 * v1.sqrt() / (v2 * v2.sqrt());
    hessian
}

fn gradient_portfolio(v: &DVector<f64>) -> DVector<f64> {
    let v1 = v[0];
    let v2 = v[1];
    let mut gradient = DVector::zeros(2);
    gradient[0] = (v2 / v1).sqrt();
    gradient[1] = (v1 / v2).sqrt();
    gradient
}

fn liquidity(x: &DVector<f64>) -> f64 {
    let x1 = x[0];
    let x2 = x[1];
    (x1 * x2).sqrt()
}

fn main() {
    let mut v = DVector::from_vec(vec![1., 1.]);
    let ra = DVector::from_vec(vec![1e6, 1e6]);
    let rb = DVector::from_vec(vec![1e3, 2e3]);
    let mut direction = DVector::zeros(2);
    let max = 1;

    let reg_term = (1001413.2135623731 + 2.) * DMatrix::identity(2, 2);
    for _ in 0..max {
        println!("x_k {:?}", v);
        println!("direction {:?}", direction);
        v += &direction;
        println!("x_k+1 {:?}", v);
        let gradient = gradient_portfolio(&v);
        let hessian = hessian_portfolio(&v);
        let liquidity_a = liquidity(&ra);
        let liquidity_b = liquidity(&rb);

        let hessian_a = &hessian * liquidity_a;
        let hessian_b = &hessian * liquidity_b;
        let a = hessian_a + hessian_b;
        let a: DMatrix<Floating> = a + &reg_term;
        println!("invertible hessian: {:?}", a.is_invertible());
        let eigenvalues = a.eigenvalues().unwrap();
        println!("eigenvalues: {:?}", eigenvalues);
        let b = (&ra - liquidity_a * &gradient) + (&rb - liquidity_b * &gradient);
        println!("a: {:?}", a);
        direction = (a).cholesky().unwrap().solve(&(b)); //apprently direction lives in the same line emanated from the origin by v
    }
}

#[test]
fn solve_system() {
    let a = DMatrix::from_vec(2, 2, vec![3., 2., 2., 5.]);
    let b = DVector::from_vec(vec![1., 2.]);
    let x = a.cholesky().unwrap().solve(&b);
    println!("{:?}", x);
}
