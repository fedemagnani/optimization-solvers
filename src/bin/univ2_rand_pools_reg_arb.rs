use core::f64;

use nalgebra::{DMatrix, DVector, Matrix2, Vector2};

use optimization_solvers::*;
use rand::{Rng, SeedableRng};
use rayon::iter::{IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator};
use tracing::{debug, error, info, warn};

#[derive(Debug, Clone, derive_getters::Getters)]
struct Univ2 {
    r0: f64,
    r1: f64,
    asset0: usize,
    asset1: usize,
    gamma: f64,
    liquidity: f64,
    liquidity_grad: Vector2<f64>,
    portfolio_grad: Vector2<f64>,
    portoflio_hessian: Matrix2<f64>,
}

impl Univ2 {
    pub fn new(r0: f64, r1: f64, asset0: usize, asset1: usize, gamma: f64) -> Self {
        Univ2 {
            r0,
            r1,
            asset0,
            asset1,
            gamma,
            liquidity: (r0 * r1).sqrt(),
            liquidity_grad: Vector2::new(0.5 * (r1 / r0).sqrt(), 0.5 * (r0 / r1).sqrt()),
            portfolio_grad: Vector2::new(0.0, 0.0).into(),
            portoflio_hessian: Matrix2::new(0.0, 0.0, 0.0, 0.0),
        }
    }
    pub fn update_portfolio_grad(&mut self, p: &DVector<f64>) {
        self.portfolio_grad[0] = (p[1] / p[0]).sqrt();
        self.portfolio_grad[1] = (p[0] / p[1]).sqrt();
    }
    pub fn update_portoflio_hessian(&mut self, p: &DVector<f64>) {
        self.portoflio_hessian[(0, 0)] = -0.5 / p[0] * (p[1] / p[0]).sqrt();
        self.portoflio_hessian[(0, 1)] = 0.5 / (p[0] * p[1]).sqrt();
        self.portoflio_hessian[(1, 0)] = 0.5 / (p[0] * p[1]).sqrt();
        self.portoflio_hessian[(1, 1)] = -0.5 / p[1] * (p[0] / p[1]).sqrt();
    }

    //gradient returned has dimension assets_n
    pub fn find_arb(&mut self, v: &DVector<f64>) -> FuncEvalMultivariate {
        if self.asset0 >= v.len() || self.asset1 >= v.len() {
            println!(
                "v, asset0, asset1: {:?}, {:?}, {:?}",
                v.len(),
                self.asset0,
                self.asset1
            );
        }
        let assets_n = v.len();
        let v0 = v[self.asset0];
        let v1 = v[self.asset1];

        let v = [v0, v1];

        let g_liq = self.liquidity_grad();
        // we find the rescaling factor as the inifnity norm of diag(g_liq)^{-1} * v
        let rescaling_factor = v
            .iter()
            .zip(g_liq.iter())
            .fold(0.0f64, |acc, (v, g)| acc.max(v / g));
        let rescaled_g = rescaling_factor * g_liq;

        // we compute p clamping rescaling_factor * g_liq in [v,v/gamma]
        let p = rescaled_g
            .iter()
            .enumerate()
            .map(|(i, rescaled_g)| rescaled_g.max(v[i]).min(v[i] / self.gamma))
            .collect::<Vec<f64>>()
            .into();
        // debug!("v: {:?}", v);
        // debug!("p: {:?}", p);
        // debug!("rescaled_g: {:?}", rescaled_g);
        // we compute the inifnity norm between p and rescaled g: if it is infinitesimally small we are in the no-arb zone
        let diff: Vector2<f64> = &p - rescaled_g;
        let diff_norm = diff.iter().fold(0.0f64, |acc, x| acc.max(x.abs()));
        if diff_norm < f64::EPSILON {
            // debug!("No-arb zone");
            return FuncEvalMultivariate::new(0.0, DVector::zeros(assets_n))
                .with_hessian(DMatrix::zeros(assets_n, assets_n));
        }
        self.update_portfolio_grad(&p);
        let w = self.portfolio_grad();

        //we should explicitly set the no-arb zone, without computing the portfolio grad
        let mut p_with_fees = p.clone();

        let mut swap0 = self.r0 - self.liquidity() * w[0];
        let mut swap1 = self.r1 - self.liquidity() * w[1];

        if swap0 < 0.0 {
            swap0 /= self.gamma;
            p_with_fees[0] /= self.gamma;
        }
        if swap1 < 0.0 {
            swap1 /= self.gamma;
            p_with_fees[1] /= self.gamma;
        }
        self.update_portoflio_hessian(&p_with_fees);
        let h = self.portoflio_hessian();
        // let gradient = DVector::from_vec(vec![swap0, swap1]);
        let image = v[0] * swap0 + v[1] * swap1;
        let mut gradient = DVector::zeros(assets_n);
        gradient[self.asset0] = swap0;
        gradient[self.asset1] = swap1;
        let hessian_low_dim = -self.liquidity() * h;
        let mut hessian = DMatrix::zeros(assets_n, assets_n);
        hessian[(self.asset0, self.asset0)] = hessian_low_dim[(0, 0)];
        hessian[(self.asset0, self.asset1)] = hessian_low_dim[(0, 1)];
        hessian[(self.asset1, self.asset0)] = hessian_low_dim[(1, 0)];
        hessian[(self.asset1, self.asset1)] = hessian_low_dim[(1, 1)];

        FuncEvalMultivariate::new(image, gradient).with_hessian(hessian)
    }
}

#[test]
pub fn test_univ2_analytical() {
    let g = 0.997;
    let pool_1_asset0 = 0;
    let pool_1_asset1 = 1;
    let mut pool_1 = Univ2::new(1e6, 1e6, pool_1_asset0, pool_1_asset1, g);
    let pool_2_asset0 = 0;
    let pool_2_asset1 = 1;
    let mut pool_2 = Univ2::new(1e3, 2e3, pool_2_asset0, pool_2_asset1, g);

    let v1 = 1.;
    let term = (pool_1.r0 + pool_2.r0) / (pool_1.liquidity() + pool_2.liquidity());
    let term = term.powi(2);
    println!("term: {:?}", term);
    let v2 = v1 * term;
    let v1 = v1 / v2;
    let v2 = v2 / v2;
    println!("v1: {:?}", v1);
    println!("v2: {:?}", v2);

    let eval_1 = pool_1.find_arb(&DVector::from_vec(vec![v1, v2]));
    let eval_2 = pool_2.find_arb(&DVector::from_vec(vec![v1, v2]));
    println!("eval_1: {:?}", eval_1);
    println!("eval_2: {:?}", eval_2);
    let y = eval_1.g() + eval_2.g();
    println!("y: {:?}", y);

    // we now try to derive the fee-aware prices
    let fee_factor = 0.997;
    let v = vec![v1, v2];
    let g_liq = pool_1.liquidity_grad();
    // we find the rescaling factor as the inifnity norm of diag(g_liq)^{-1} * v
    let rescaling_factor = v
        .iter()
        .zip(g_liq.iter())
        .fold(0.0f64, |acc, (v, g)| acc.max(v / g));
    let rescaled_g = rescaling_factor * g_liq;
    let p = rescaled_g
        .iter()
        .enumerate()
        .map(|(i, rescaled_g)| rescaled_g.max(v[i]).min(v[i] / fee_factor))
        .collect::<Vec<f64>>();
    println!("p1: {:?}", p);
    let g_liq = pool_2.liquidity_grad();
    // we find the rescaling factor as the inifnity norm of diag(g_liq)^{-1} * v
    let rescaling_factor = v
        .iter()
        .zip(g_liq.iter())
        .fold(0.0f64, |acc, (v, g)| acc.max(v / g));
    let rescaled_g = rescaling_factor * g_liq;
    let p = rescaled_g
        .iter()
        .enumerate()
        .map(|(i, rescaled_g)| rescaled_g.max(v[i]).min(v[i] / fee_factor))
        .collect::<Vec<f64>>();
    println!("p2: {:?}", p);

    let arb1 = pool_1.find_arb(&DVector::from_vec(p.to_vec()));
    let arb2 = pool_2.find_arb(&DVector::from_vec(p.to_vec()));
    let y = arb1.g() + arb2.g();
    println!("y with fees: {:?}", y);
    println!("Pnl: {:?}", y.iter().fold(0.0f64, |acc, x| acc + x));
}

fn main() {
    // we set seet, tracer and log info
    std::env::set_var("RUST_LOG", "debug");
    let mut rng = rand::rngs::StdRng::seed_from_u64(12);
    let tracer = Tracer::default()
        .with_stdout_layer(Some(LogFormat::Normal))
        .build();
    info!("Start binaries");
    // we mock the pools of the optimization
    let fee_factor = 0.997;
    let assets_n = 2;
    let pools_m = 1000;
    let prices = vec![1.; assets_n];
    let mut univ2_pools = vec![];
    // for _ in 0..pools_m {
    //     let r0 = 1e6 * rng.gen::<f64>() + 1e12 * rng.gen::<f64>();
    //     let r1 = 1e6 * rng.gen::<f64>() + 1e12 * rng.gen::<f64>();
    //     let asset0 = rng.gen::<usize>() % assets_n;
    //     let mut asset1 = rng.gen::<usize>() % assets_n;
    //     if asset1 == asset0 {
    //         asset1 = (asset1 + 1) % assets_n;
    //     }

    //     let gamma = fee_factor;
    //     let pool = Univ2::new(r0, r1, asset0, asset1, gamma);
    //     univ2_pools.push(pool);
    // }

    let pool_1_asset0 = 0;
    let pool_1_asset1 = 1;
    let pool_1 = Univ2::new(1e6, 1e6, pool_1_asset0, pool_1_asset1, fee_factor);
    let pool_2_asset0 = 0;
    let pool_2_asset1 = 1;
    let pool_2 = Univ2::new(1e3, 2e3, pool_2_asset0, pool_2_asset1, fee_factor);
    univ2_pools.push(pool_1.clone());
    univ2_pools.push(pool_2.clone());

    // // Below a L-ininfity regularization on reserves vector (divide by the max infinity norm of reserves)
    // let reg_fac = univ2_pools
    //     .iter()
    //     .fold(0.0f64, |acc, pool| acc.max(pool.r0.max(pool.r1)));

    // // Below a L-2 regularization on reserves vector (divide by the max euclidean norm of reserves)
    // let reg_fac = univ2_pools.iter().fold(0.0f64, |acc, pool| {
    //     acc.max((pool.r0.powi(2) + pool.r1.powi(2)).sqrt())
    // });

    // // Below a L-infinity regularization on liquidity vector (divide by the infinity norm of the vector of liquidities)
    // let reg_fac = univ2_pools
    //     .iter()
    //     .fold(0.0f64, |acc, pool| acc.max(*pool.liquidity()));

    // // //Below a L-2 regularization on liquidity vector (divide by the euclidean norm of the vector of liquidities)
    // let reg_fac = univ2_pools
    //     .iter()
    //     .fold(0.0f64, |acc, pool| acc + pool.liquidity().powi(2))
    //     .sqrt();

    // // Below a regularization factor based on the mean of the liquidity vector
    // let reg_fac = univ2_pools
    //     .iter()
    //     .fold(0.0f64, |acc, pool| acc + pool.liquidity())
    //     / univ2_pools.len() as f64;

    // // Below a regularization factor based on the mean of the reserves vector
    // let reg_fac = univ2_pools
    //     .iter()
    //     .fold(0.0f64, |acc, pool| acc + pool.r0 + pool.r1)
    //     / (2. * univ2_pools.len() as f64);

    let reg_fac = 1.;

    let reinf = 1.;

    // Setting up the oracle
    let min_amount_out = f64::EPSILON;
    let mut f_and_g = |v: &DVector<Floating>| -> FuncEvalMultivariate {
        // we define as utility function sum -ln(vi -ci) for i=1,2,...
        let (mut image, mut gradient, mut hessian) = univ2_pools
            .par_iter_mut()
            .fold(
                || {
                    (
                        0.0,
                        DVector::zeros(assets_n),
                        DMatrix::zeros(assets_n, assets_n),
                    )
                },
                |(mut acc, mut g, mut hes), pool| {
                    // for p in univ2_pools.iter_mut() {
                    //     p.r0 /= reg_fac;
                    //     p.r1 /= reg_fac;
                    // }

                    // //Regularize pool reserves
                    // let mut pool = pool.clone();
                    // pool.r0 /= reg_fac;
                    // pool.r1 /= reg_fac;

                    let mut eval = pool.find_arb(v);
                    let image = eval.f();
                    let gradient = eval.g();
                    acc += image;
                    g += gradient;
                    hes += eval.take_hessian();

                    (acc, g, hes)
                },
            )
            .reduce(
                || {
                    (
                        0.0,
                        DVector::zeros(assets_n),
                        DMatrix::zeros(assets_n, assets_n),
                    )
                },
                |(acc1, g1, h1), (acc2, g2, h2)| (acc1 + acc2, g1 + g2, h1 + h2),
            );
        // Instead of regularizing reserves (costly) we leverage on the positive homogenous of the find arbitrage
        image /= reg_fac;
        gradient /= reg_fac;
        hessian /= reg_fac;

        let mut image_sum = 0.0;
        let mut gradient_sum = DVector::zeros(assets_n);
        let mut hessian_sum = DMatrix::zeros(assets_n, assets_n);
        for (i, (c, v)) in prices.iter().zip(v.iter()).enumerate() {
            if c >= v {
                image_sum += 0.5 * (c - v).powi(2) - v.ln();
                gradient_sum[i] += v - c - 1.0 / v;
                hessian_sum[(i, i)] += 1.0 + 1.0 / v.powi(2);
            }
        }
        image += reinf * image_sum;
        gradient += reinf * gradient_sum;
        hessian += reinf * hessian_sum;
        // let eigenvalues = hessian.clone().symmetric_eigenvalues();
        // let condition_number = eigenvalues.iter().fold(0.0f64, |acc, x| acc.max(x.abs()))
        //     / eigenvalues
        //         .iter()
        //         .fold(f64::INFINITY, |acc, x| acc.min(x.abs()));
        // info!("Condition number: {:?}", condition_number);

        FuncEvalMultivariate::new(image, gradient).with_hessian(hessian)
    };

    // Setting up the parameters for main solver
    let max_iter = 10000;
    let max_iter_line_search: usize = 1000;
    let grad_tol = 1e-15;
    let mut x0 = DVector::from_vec(vec![0.; assets_n]);
    // x0[0] = 48.249371895149;
    // x0[1] = 58.972880935049645;

    // let upper_bound = DVector::from_vec(vec![f64::INFINITY; assets_n]);
    let lower_bound = DVector::from_vec(vec![f64::EPSILON; assets_n]);
    let upper_bound = DVector::from_vec(prices.clone());

    // Setting up the line search
    // let mut ls = BackTrackingB::new(1e-4, 0.5, lower_bound.clone(), upper_bound.clone());
    let mut ls = GLLQuadratic::new(1e-4, 8).with_sigmas(0.1, 0.9);
    // let mut ls = MoreThuenteB::new(assets_n)
    //     .with_lower_bound(lower_bound.clone())
    //     .with_upper_bound(upper_bound.clone());

    // Setting up the solver
    // let mut gd: SpectralProjectedGradient = SpectralProjectedGradient::new(
    //     grad_tol,
    //     x0.clone(),
    //     &mut f_and_g,
    //     lower_bound.clone(),
    //     upper_bound.clone(),
    // );
    // let mut gd: ProjectedGradientDescent =
    //     ProjectedGradientDescent::new(grad_tol, x0, lower_bound, upper_bound);
    let mut gd: BFGSB = BFGSB::new(grad_tol, x0, lower_bound.clone(), upper_bound.clone());
    // let mut gd: SR1B = SR1B::new(grad_tol, x0, lower_bound.clone(), upper_bound.clone());
    // let mut gd: BroydenB = BroydenB::new(grad_tol, x0, lower_bound.clone(), upper_bound.clone());
    // let mut gd: DFPB = DFPB::new(grad_tol, x0, lower_bound.clone(), upper_bound.clone());

    // let mut gd: ProjectedNewton =
    //     ProjectedNewton::new(grad_tol, x0, lower_bound.clone(), upper_bound.clone());
    // let mut gd: SpectralProjectedNewton = SpectralProjectedNewton::new(
    //     grad_tol,
    //     x0,
    //     &mut f_and_g,
    //     lower_bound.clone(),
    //     upper_bound.clone(),
    // );

    // We define a callback to store iterates and function evaluations
    // let mut iterates = vec![];
    let mut condition_numbers = vec![];
    let mut store_iterates = |s: &BFGSB| {
        // let eig = s.approx_inv_hessian().clone().eigenvalues();
        // let Some(eig) = eig else {
        //     return;
        // };
        // let (max_eig, min_eig) = eig.iter().fold((0.0f64, f64::INFINITY), |acc, x| {
        //     (acc.0.max(x.abs()), acc.1.min(x.abs()))
        // });
        // let condition_number = max_eig / min_eig;
        // condition_numbers.push(condition_number);
    };

    info!("Starting optimization");
    // Running the solver
    let res = gd.minimize(
        &mut ls,
        &mut f_and_g,
        max_iter,
        max_iter_line_search,
        Some(&mut store_iterates),
    );

    // info!("Function eval: {:?}", f_and_g(gd.xk()));
    let iterate = gd.xk();
    info!("Iterate: {:?}", iterate);
    info!("reg_fac: {:?}", reg_fac);
    // let iterate = iterate * reg_fac;

    info!(
        "Infinity norm projected gradient: {:?}",
        gd.projected_gradient(&mut f_and_g(&iterate))
            .infinity_norm()
    );
    info!(
        "Max condition number: {:?}",
        condition_numbers.iter().fold(0.0f64, |acc, x| acc.max(*x))
    );
    info!(
        "Min condition number: {:?}",
        condition_numbers
            .iter()
            .fold(f64::INFINITY, |acc, x| acc.min(*x))
    );
    info!(
        "Mean condition number: {:?}",
        condition_numbers.iter().fold(0.0f64, |acc, x| acc + *x) / condition_numbers.len() as f64
    );

    // // Plotting the iterates
    // let n = 1000;

    // let x_min = f64::EPSILON.sqrt();
    // let x_max = 10.;
    // let y_min = f64::EPSILON.sqrt();
    // let y_max = 10.;
    // let plotter = Plotter3d::new(x_min, x_max, y_min, y_max, n)
    //     .append_plot(&mut f_and_g, "Objective function", 0.5)
    //     .append_scatter_points(&mut f_and_g, &iterates, "Iterates")
    //     .set_layout_size(1600, 1000);
    // plotter.build("t.html");

    let y = univ2_pools
        .par_iter_mut()
        .fold(
            || DVector::zeros(assets_n),
            |mut acc, pool| {
                let arb = pool.find_arb(&iterate);
                acc += arb.g();
                acc
            },
        )
        .reduce(|| DVector::zeros(assets_n), |acc1, acc2| acc1 + acc2);
    info!("Net trade vector: {:?}", y);
    let neg = y.iter().filter(|x| x < &&0.).count();

    info!("NEGATIVE NET AMOUNTS: {:?}", neg);

    // // L-BFGS-B
    // let mut lbfgsb = Lbfgsb::new(assets_n);
    // for (i, (l, u)) in lower_bound.iter().zip(upper_bound.iter()).enumerate() {
    //     lbfgsb.set_lower_bound(i, *l);
    //     // lbfgsb.set_upper_bound(i, *u);
    // }
    // lbfgsb.set_pgtol(1e-18);
    // lbfgsb.set_factr(1e-18);
    // lbfgsb.max_iteration(1e9 as u32);
    // lbfgsb.set_verbosity(0);
    // lbfgsb.set_m(15);

    // let res = lbfgsb.minimize(f_and_g, &mut x0);
    // // println!("Res: {:?}", res);
    // let y = univ2_pools
    //     .par_iter_mut()
    //     .fold(
    //         || DVector::zeros(assets_n),
    //         |mut acc, pool| {
    //             let arb = pool.find_arb(&x0);
    //             acc += arb.g();
    //             acc
    //         },
    //     )
    //     .reduce(|| DVector::zeros(assets_n), |acc1, acc2| acc1 + acc2);

    // // we count how many negaive net trades we have
    // let neg = y.iter().filter(|x| x < &&0.).count();
    // println!("NEGATIVE NET AMOUNTS: {:?}", neg);
    // // println!("optimal point: {:?}", x0);

    // // // println!("optimal point: {:?}", x0);
    // println!("Net trade vector: {:?}", y);
    // // // println!("Prices: {:?}", prices);
    // // // println!("f_and_g: {:?}", f_and_g(&x0));
}
