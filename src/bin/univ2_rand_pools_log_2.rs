use core::f64;

use nalgebra::{DMatrix, DVector};

use optimization_solvers::*;
use rand::{Rng, SeedableRng};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use tracing::{debug, error, info, warn};

#[derive(Debug, Clone)]
struct Univ2 {
    r0: f64,
    r1: f64,
    asset0: usize,
    asset1: usize,
    gamma: f64,
}

impl Univ2 {
    pub fn new(r0: f64, r1: f64, asset0: usize, asset1: usize, gamma: f64) -> Self {
        Univ2 {
            r0,
            r1,
            asset0,
            asset1,
            gamma,
        }
    }
    pub fn liquidity(&self) -> f64 {
        (self.r0 * self.r1).sqrt()
    }
    pub fn liquidity_grad(&self) -> DVector<f64> {
        let g = vec![
            0.5 * (self.r1 / self.r0).sqrt(),
            0.5 * (self.r0 / self.r1).sqrt(),
        ];
        g.into()
    }
    pub fn portfolio_grad(p: &DVector<f64>) -> DVector<f64> {
        let g = vec![(p[1] / p[0]).sqrt(), (p[0] / p[1]).sqrt()];
        g.into()
    }

    //gradient returned has dimension assets_n
    pub fn find_arb(&self, v: &DVector<f64>) -> FuncEvalMultivariate {
        let assets_n = v.len();
        let v0 = v[self.asset0];
        let v1 = v[self.asset1];
        let v = [v0, v1];
        let liquidity = self.liquidity();
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
        let diff: DVector<Floating> = &p - rescaled_g;
        let diff_norm = diff.iter().fold(0.0f64, |acc, x| acc.max(x.abs()));
        if diff_norm < f64::EPSILON {
            // debug!("No-arb zone");
            return FuncEvalMultivariate::new(0.0, DVector::zeros(assets_n))
                .with_hessian(DMatrix::identity(assets_n, assets_n));
        }

        let w = Self::portfolio_grad(&p);
        //we should explicitly set the no-arb zone, without computing the portfolio grad

        let mut swap0 = self.r0 - liquidity * w[0];
        let mut swap1 = self.r1 - liquidity * w[1];

        if swap0 < 0.0 {
            swap0 /= self.gamma;
        }
        if swap1 < 0.0 {
            swap1 /= self.gamma;
        }
        // let gradient = DVector::from_vec(vec![swap0, swap1]);
        let image = v[0] * swap0 + v[1] * swap1;
        let mut gradient = DVector::zeros(assets_n);
        gradient[self.asset0] = swap0;
        gradient[self.asset1] = swap1;

        FuncEvalMultivariate::new(image, gradient)
    }
}

#[test]
fn prec() {
    let v1: f64 = 3636372848294849.;
    let v2: f64 = 4959590052485959003.;
    let s1 = (v2 / v1).sqrt();
    let s2 = (v1 / v2).sqrt();
    let f = s1 * v1 + s2 * v2;
    let g = 2. * (v1 * v2).sqrt();
    println!("f: {:?}", f);
    println!("g: {:?}", g);
    assert!((f - g).abs() < f64::EPSILON);
}

#[test]
pub fn single_pool_univ2() {
    let g = 0.5;
    let pool_2_asset0 = 0;
    let pool_2_asset1 = 1;
    let pool_2 = Univ2::new(1e3, 2e3, pool_2_asset0, pool_2_asset1, g);

    let v = DVector::from_vec(vec![1.0038392886213285, 1.0]);

    let eval_2 = pool_2.find_arb(&v);

    println!("eval_2: {:?}", eval_2);
}

#[test]
pub fn test_univ2_analytical() {
    let g = 1.;
    let pool_1_asset0 = 0;
    let pool_1_asset1 = 1;
    let pool_1 = Univ2::new(1e6, 1e6, pool_1_asset0, pool_1_asset1, g);
    let pool_2_asset0 = 0;
    let pool_2_asset1 = 1;
    let pool_2 = Univ2::new(1e3, 2e3, pool_2_asset0, pool_2_asset1, g);

    let v1 = 1.;
    let term = (pool_1.r0 + pool_2.r0) / (pool_1.liquidity() + pool_2.liquidity());
    let term = term.powi(2);
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

    //we know try to derive the fee-aware prices
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
}

#[test]
pub fn text_univ2() {
    let g = 0.997;
    let pool_1_asset0 = 0;
    let pool_1_asset1 = 1;
    let pool_1 = Univ2::new(1e6, 1e6, pool_1_asset0, pool_1_asset1, g);
    let pool_2_asset0 = 0;
    let pool_2_asset1 = 1;
    let pool_2 = Univ2::new(1e3, 2e3, pool_2_asset0, pool_2_asset1, g);

    // let v = DVector::from_vec(vec![1.0038332579839788216702345380326732993, 1.]);
    // let v = DVector::from_vec(vec![1.0008277707554645, 1.000000000000000]);
    let v = DVector::from_vec(vec![1.0038392886213285, 1.0]);
    let eval_1 = pool_1.find_arb(&v);
    let eval_2 = pool_2.find_arb(&v);
    println!("eval_1: {:?}", eval_1);
    println!("eval_2: {:?}", eval_2);
    let y = eval_1
        .g()
        .iter()
        .zip(eval_2.g().iter())
        .map(|(g1, g2)| g1 + g2)
        .collect::<Vec<f64>>();
    println!("y: {:?}", y);
    println!("f: {:?}", eval_1.f() + eval_2.f());
}

#[test]
fn validate() {
    let l1 = (526557.4090027738_f64 * 542725.2099031439).sqrt();
    let l2 = (34342.8179549562_f64 * 414956.8461853601).sqrt();
    let v = vec![1.0548_f64, 1.2953999999999999];
    let image1 =
        v[1] * 526557.4090027738 + v[0] * 542725.2099031439 - 2. * l1 * (v[0] * v[1]).sqrt();
    let image2 =
        v[1] * 34342.8179549562 + v[0] * 414956.8461853601 - 2. * l2 * (v[0] * v[1]).sqrt();
    println!("image1: {:?}", image1);
    println!("image2: {:?}", image2);
    println!("image: {:?}", image1 + image2);
}

fn main() {
    // we set seet, tracer and log info
    std::env::set_var("RUST_LOG", "info");
    let mut rng = rand::rngs::StdRng::seed_from_u64(12);
    let tracer = Tracer::default()
        .with_stdout_layer(Some(LogFormat::Normal))
        .build();
    // we mock the pools of the optimization
    let fee_factor = 0.997;
    let assets_n = 2;
    let pools_m = 200;
    let prices = vec![1.; assets_n];
    let mut univ2_pools = vec![];
    // for _ in 0..pools_m {
    //     let r0 = 1e6 * rng.gen::<f64>() + 1e8 * rng.gen::<f64>();
    //     let r1 = 1e6 * rng.gen::<f64>() + 1e8 * rng.gen::<f64>();
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

    // Below a L-infinity regularization on liquidity vector (divide by the infinity norm of the vector of liquidities)
    let reg_fac = univ2_pools
        .iter()
        .fold(0.0f64, |acc, pool| acc.max(pool.liquidity()));

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

    // let reg_fac = 1.;

    // Setting up the oracle
    let min_amount_out = 0.;
    let f_and_g = |v: &DVector<Floating>| -> FuncEvalMultivariate {
        // we define as utility function sum -ln(vi -ci) for i=1,2,...

        let (mut image, mut gradient) = univ2_pools
            .par_iter()
            .fold(
                || (0.0, DVector::zeros(assets_n)),
                |(mut acc, mut g), pool| {
                    let eval = pool.find_arb(v);
                    let image = eval.f();
                    let gradient = eval.g();
                    acc += image;
                    g += gradient;
                    (acc, g)
                },
            )
            .reduce(
                || (0.0, DVector::zeros(assets_n)),
                |(acc1, g1), (acc2, g2)| (acc1 + acc2, g1 + g2),
            );

        let mut image_sum = 0.0;
        let mut gradient_sum = DVector::zeros(assets_n);
        for (i, (c, v)) in prices.iter().zip(v.iter()).enumerate() {
            let c: f64 = *c;
            let v: f64 = *v;
            image_sum += (v - c) * (c + min_amount_out) - c * (c.ln() - (v - c).ln());
            gradient_sum[i] = (c + min_amount_out) + c / (v - c);
        }
        image += image_sum;
        gradient += gradient_sum;

        // // Instead of regularizing reserves (costly) we leverage on the positive homogenous of the find arbitrage
        // image /= reg_fac;
        // gradient /= reg_fac;
        // // hessian /= reg_fac;

        FuncEvalMultivariate::new(image, gradient)
    };

    // Setting up the parameters for main solver
    let max_iter = 10000;
    let max_iter_line_search = 1000;
    let grad_tol = 1e-12;
    let mut x0 = DVector::from_vec(vec![3.; assets_n]);
    // let upper_bound = DVector::from_vec(
    //     prices
    //         .clone()
    //         .into_iter()
    //         .map(|x| x - f64::EPSILON)
    //         .collect(),
    // );
    let upper_bound = DVector::from_vec(vec![f64::INFINITY; assets_n]);
    let lower_bound = DVector::from_vec(prices.iter().map(|x| x + f64::EPSILON.sqrt()).collect());

    // Setting up the line search
    // let mut ls = BackTrackingB::new(1e-4, 0.1, lower_bound.clone(), upper_bound.clone());
    let mut ls = GLLQuadratic::new(1e-4, 5).with_sigmas(0.1, 0.9);
    // let mut ls = MoreThuente::default();

    // Setting up the solver
    // let mut gd = SpectralProjectedGradient::new(
    //     grad_tol,
    //     x0.clone(),
    //     &f_and_g,
    //     lower_bound.clone(),
    //     upper_bound.clone(),
    // );
    // let mut gd = ProjectedGradientDescent::new(grad_tol, x0, lower_bound, upper_bound);
    let mut gd = BFGSB::new(grad_tol, x0, lower_bound.clone(), upper_bound.clone());

    // We define a callback to store iterates and function evaluations
    let mut iterates = vec![];
    let mut store_iterates = |s: &BFGSB| {
        iterates.push(s.xk().clone());
    };

    // Running the solver
    let res = gd.minimize(
        &mut ls,
        f_and_g,
        max_iter,
        max_iter_line_search,
        Some(&mut store_iterates),
    );

    info!("Function eval: {:?}", f_and_g(gd.xk()));
    let iterate = gd.xk();

    info!("Iterate: {:?}", iterate);

    info!(
        "Infinity norm projected gradient: {:?}",
        gd.projected_gradient(&f_and_g(iterate)).infinity_norm()
    );

    let y = univ2_pools
        .par_iter()
        .fold(
            || DVector::zeros(assets_n),
            |mut acc, pool| {
                let arb = pool.find_arb(iterate);
                acc += arb.g();
                acc
            },
        )
        .reduce(|| DVector::zeros(assets_n), |acc1, acc2| acc1 + acc2);
    info!("Net trade vector: {:?}", y);
    let neg = y.iter().filter(|x| x < &&0.).count();
    info!("NEGATIVE NET AMOUNTS: {:?}", neg);

    // // Plotting the iterates
    // let n = 1000;

    // let x_min = 0.;
    // let x_max = 1.;
    // let y_min = 0.;
    // let y_max = 1.;
    // let plotter = Plotter3d::new(x_min, x_max, y_min, y_max, n)
    //     .append_plot(&f_and_g, "Objective function", 0.5)
    //     .append_scatter_points(&f_and_g, &iterates, "Iterates")
    //     .set_layout_size(1600, 1000);
    // plotter.build("t.html");

    // // L-BFGS-B
    // let mut lbfgsb = Lbfgsb::new(assets_n);
    // for (i, (l, u)) in lower_bound.iter().zip(upper_bound.iter()).enumerate() {
    //     lbfgsb.set_lower_bound(i, *l);
    //     // lbfgsb.set_upper_bound(i, *u);
    // }
    // lbfgsb.set_pgtol(1e-18);
    // lbfgsb.set_factr(1e-18);
    // lbfgsb.max_iteration(1e9 as u32);
    // lbfgsb.set_verbosity(1);
    // lbfgsb.set_m(25);

    // let res = lbfgsb.minimize(f_and_g, &mut x0);
    // // println!("Res: {:?}", res);
    // let y = univ2_pools
    //     .par_iter()
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
    // println!("optimal point: {:?}", x0);

    // println!("optimal point: {:?}", x0);
    // println!("Net trade vector: {:?}", y);
    // println!("Prices: {:?}", prices);
    // println!("f_and_g: {:?}", f_and_g(&x0));
}
