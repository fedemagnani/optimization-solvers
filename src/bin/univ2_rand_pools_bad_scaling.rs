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
            portfolio_grad: Vector2::new(0.0, 0.0),
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

        let g_liq = self.liquidity_grad();

        let (p0, p1) = {
            let v1_v0 = v1 / v0;
            let g1_gammag0 = g_liq[1] / (self.gamma * g_liq[0]);
            let gammag1_g0 = self.gamma * g_liq[1] / g_liq[0];
            if (gammag1_g0 < v1_v0) && (v1_v0 < g1_gammag0) {
                return FuncEvalMultivariate::new(0.0, DVector::zeros(assets_n))
                    .with_hessian(DMatrix::zeros(assets_n, assets_n));
            }
            if g1_gammag0 < v1_v0 {
                (v0 / self.gamma, v1)
            } else {
                (v0, v1 / self.gamma)
            }
        };

        let p = DVector::from_vec(vec![p0, p1]);
        self.update_portfolio_grad(&p);
        let w = self.portfolio_grad();

        let mut swap0 = self.r0 - self.liquidity() * w[0];
        let mut swap1 = self.r1 - self.liquidity() * w[1];

        if swap0 < 0.0 {
            swap0 /= self.gamma;
        }
        if swap1 < 0.0 {
            swap1 /= self.gamma;
        }
        self.update_portoflio_hessian(&p);
        let h = self.portoflio_hessian();
        // let gradient = DVector::from_vec(vec![swap0, swap1]);
        let image = v0 * swap0 + v1 * swap1;
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
fn system_solve() {
    // the kernel of the liquidity hessian might define the aritrage free price
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
}

fn main() {
    // we set seet, tracer and log info
    std::env::set_var("RUST_LOG", "info");
    let mut rng = rand::rngs::StdRng::seed_from_u64(12);
    let tracer = Tracer::default()
        .with_stdout_layer(Some(LogFormat::Normal))
        .build();
    info!("Start binaries");
    // we mock the pools of the optimization
    let fee_factor = 0.997;
    let assets_n = 2;
    let pools_m = 1000;

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
    // println!("pool1: {:?}", univ2_pools[0]);

    let pool_1_asset0 = 0;
    let pool_1_asset1 = 1;
    let pool_1 = Univ2::new(1e0, 1e0, pool_1_asset0, pool_1_asset1, fee_factor);
    let pool_2_asset0 = 0;
    let pool_2_asset1 = 1;
    let pool_2 = Univ2::new(1e0, 1e7, pool_2_asset0, pool_2_asset1, fee_factor);
    univ2_pools.push(pool_1.clone());
    univ2_pools.push(pool_2.clone());

    // // Below a L-ininfity regularization on reserves vector (divide by the max infinity norm of reserves)
    // let reg_fac = univ2_pools
    //     .iter()
    //     .fold(0.0f64, |acc, pool| acc.max(pool.r0.max(pool.r1)));

    // let reg_fac = 1.;

    // for the optimization, we downscale all the reserves so that they are in (0,1]. NOtice that the arbitrage function is positive homogenous both w.r.t. prices and reserves. Thus from a mathematical standpoint, dividing all the reserves by the same regularization factor is equivalent to dividing the arbitrage function by the same factor. However, from a computational perspective, working with numbers in (0,1] is more stable, so it is preferable to downscale the reserves. The lower is the magnitude of the reserve, the higher is the variable value
    // let mut down_univ2_pools = univ2_pools
    //     .iter()
    //     .cloned()
    //     .map(|x| Univ2::new(x.r0 / reg_fac, x.r1 / reg_fac, x.asset0, x.asset1, x.gamma))
    //     .collect::<Vec<Univ2>>();

    // Setting up the oracle
    let f_and_g = |v: &DVector<Floating>| -> FuncEvalMultivariate {
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

        let mut image_sum = 0.0;
        let mut gradient_sum = DVector::zeros(assets_n);
        let mut hessian_sum = DMatrix::zeros(assets_n, assets_n);
        for (i, v) in v.iter().enumerate() {
            let v: f64 = *v;
            image_sum += -v.ln();
            gradient_sum[i] = -1.0 / v;
            hessian_sum[(i, i)] = 1.0 / v.powi(2);
        }
        image += image_sum;
        gradient += gradient_sum;
        hessian += hessian_sum;

        FuncEvalMultivariate::new(image, gradient).with_hessian(hessian)
    };

    // Setting up the parameters for main solver

    let mut x0 = DVector::from_vec(vec![1.; assets_n]);

    let upper_bound = DVector::from_vec(vec![f64::INFINITY; assets_n]);
    let lower_bound = DVector::from_vec(vec![f64::EPSILON; assets_n]);

    // L-BFGS-B
    let mut lbfgsb = Lbfgsb::new(assets_n);
    for (i, (l, u)) in lower_bound.iter().zip(upper_bound.iter()).enumerate() {
        lbfgsb.set_lower_bound(i, *l);
        lbfgsb.set_upper_bound(i, *u);
    }
    lbfgsb.set_pgtol(1e-18);
    lbfgsb.set_factr(1e-18);
    lbfgsb.max_iteration(1e9 as u32);
    lbfgsb.set_verbosity(0);
    lbfgsb.set_m(15);

    let _ = lbfgsb.minimize(f_and_g, &mut x0);
    let y = univ2_pools
        .par_iter_mut()
        .fold(
            || DVector::zeros(assets_n),
            |mut acc, pool| {
                let arb = pool.find_arb(&x0);
                acc += arb.g();
                acc
            },
        )
        .reduce(|| DVector::zeros(assets_n), |acc1, acc2| acc1 + acc2);

    // we count how many negaive net trades we have
    let neg = y.iter().filter(|x| x < &&0.).count();
    println!("NEGATIVE NET AMOUNTS: {:?}", neg);
    println!("Net trade vector: {:?}", y);

    let mean_price = x0.iter().fold(0.0, |acc, x| acc + x) / assets_n as f64;
    let variance_price =
        x0.iter().fold(0.0, |acc, x| acc + (x - mean_price).powi(2)) / (assets_n - 1) as f64;
    println!("Mean price: {:?}", mean_price);
    println!("Variance price: {:?}", variance_price);
    println!("Price vector: {:?}", x0);
}
