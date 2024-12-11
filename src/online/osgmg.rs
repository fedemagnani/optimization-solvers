use super::*;
// Online scaled gradient method with ratio surrogate and lower bound from [Gao, Chu, Ye, Udell]

pub enum SparsityPattern {
    Diagonal(DVector<Floating>),
    Normal(DMatrix<Floating>),
}
impl SparsityPattern {
    fn default_diagonal(n: usize) -> Self {
        SparsityPattern::Diagonal(DVector::from_element(n, 1.0))
    }
    fn default_normal(n: usize) -> Self {
        SparsityPattern::Normal(DMatrix::from_element(n, n, 1.0))
    }
    fn new_diagonal(p: DVector<Floating>) -> Self {
        SparsityPattern::Diagonal(p)
    }
    fn new_normal(p: DMatrix<Floating>) -> Self {
        SparsityPattern::Normal(p)
    }
}

pub struct OSGMG {}
impl OSGMG {
    pub fn minimize(
        x0: DVector<Floating>,
        mut oracle: impl FnMut(&DVector<Floating>) -> FuncEvalMultivariate,
        max_iter: usize,
        s_pattern: SparsityPattern,
        adagrad_alpha: Floating,
        grad_tol: Floating,
    ) -> DVector<Floating> {
        let mut x = x0;
        let n = x.len();
        let mut ngradevl = 0;
        match s_pattern {
            SparsityPattern::Diagonal(mut p) => {
                //here pv is the elementwise product of p and v
                let pv = |p: &DVector<Floating>, g: &DVector<Floating>| {
                    let mut res = DVector::zeros(n);
                    for i in 0..n {
                        res[i] = p[i] * g[i];
                    }
                    res
                };
                let mut cap_g = DVector::zeros(n);
                for i in 0..max_iter {
                    info!("x: {:?}", x);
                    let eval = oracle(&x);
                    let g = eval.g();
                    let nrmg = g.norm();
                    let xtmp = &x - &pv(&p, &g);
                    let eval_tmp = oracle(&xtmp);
                    let gtmp = eval_tmp.g();
                    let nrmgtmp = gtmp.norm();

                    let hesstmp = eval_tmp.hessian().clone().expect("Hessian not provided");

                    let gr = (hesstmp * gtmp)
                        .iter()
                        .enumerate()
                        .map(|(i, x)| x * g[i])
                        .collect::<Vec<_>>();
                    let gr = -DVector::from_vec(gr) / (nrmg * nrmgtmp);
                    cap_g += DVector::from_vec(gr.iter().map(|x| x * x).collect::<Vec<Floating>>());
                    p += -adagrad_alpha
                        * DVector::from_vec(
                            gr.iter()
                                .enumerate()
                                .map(|(i, x)| x * (cap_g[i] + 1e-20).sqrt())
                                .collect::<Vec<Floating>>(),
                        );
                    info!("xtmp: {:?}", xtmp);
                    // Monotone oracle
                    if nrmgtmp < nrmg {
                        x = xtmp;
                    }
                    if nrmg < grad_tol {
                        break;
                    }
                }
                x
            }
            SparsityPattern::Normal(mut p) => {
                //here pv is the elementwise product of p and v
                let mut pv = |p: &DMatrix<Floating>, g: &DVector<Floating>| p * g;
                let mut cap_g = DMatrix::zeros(n, n);
                for i in 0..max_iter {
                    let eval = oracle(&x);
                    let g = eval.g();
                    let f = eval.f();
                    let nrmg = g.norm();
                    let xtmp = &x - &pv(&p, &g);
                    let eval_tmp = oracle(&xtmp);
                    let gtmp = eval_tmp.g();
                    let hesstmp = eval_tmp.hessian().clone().expect("Hessian not provided");
                    let nrmgtmp = gtmp.norm();
                    let gr = (hesstmp * gtmp) * g.transpose();

                    let gr = -gr / (nrmg * nrmgtmp);
                    cap_g += DMatrix::from_vec(n, n, gr.iter().map(|x| x * x).collect::<Vec<_>>());
                    p += -adagrad_alpha
                        * DMatrix::from_vec(
                            n,
                            n,
                            gr.iter()
                                .map(|x| (x + 1e-20).sqrt())
                                .collect::<Vec<Floating>>(),
                        );

                    // Monotone oracle
                    if nrmgtmp < nrmg {
                        x = xtmp;
                    }

                    if nrmg < grad_tol {
                        break;
                    }
                }
                x
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::func_eval::FuncEvalMultivariate;
    use crate::line_search::MoreThuente;
    use crate::number::Floating;
    use crate::steepest_descent::gradient_descent::GradientDescent;
    use crate::tracer::LogFormat;
    use crate::tracer::Tracer;
    use nalgebra::{DMatrix, DVector};

    // #[test]
    pub fn osgmg() {
        std::env::set_var("RUST_LOG", "info");

        let tracer = Tracer::default()
            .with_stdout_layer(Some(LogFormat::Normal))
            .build();
        let gamma = 2.;
        let matrix = DMatrix::from_vec(2, 2, vec![100., 0., 0., 100.]);
        let f_and_g = |x: &DVector<f64>| -> FuncEvalMultivariate {
            let f = x.dot(&(&matrix * x));
            let g = 2. * &matrix * x;
            let hessian = 2. * &matrix;
            FuncEvalMultivariate::new(f, g).with_hessian(hessian)
        };
        // Linesearch builder
        let mut ls = MoreThuente::default();

        // Gradient descent builder
        let tol = 1e-12;
        let x_0 = DVector::from(vec![4.0, 300.0]);
        let x = OSGMG::minimize(
            x_0,
            f_and_g,
            10000000,
            SparsityPattern::default_diagonal(2),
            1.,
            tol,
        );

        let eval = f_and_g(&x);
        println!("Iterate: {:?}", x);
        println!("Function eval: {:?}", eval);
        println!("Gradient norm: {:?}", eval.g().norm());
        println!("tol: {:?}", tol);

        let convergence = eval.g().norm() < tol;
        println!("Convergence: {:?}", convergence);

        assert!((eval.f() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn outeer() {
        let x1 = DVector::from_vec(vec![1.0, 1.0]);
        let x2 = DVector::from_vec(vec![1.0, 1.0]);
        let r = x1 * x2.transpose();
        println!("{:?}", r);
        let x3 = DVector::from_vec(vec![2.0, 2.0]);
        let r = r * x3;
        println!("{:?}", r);
    }
}
