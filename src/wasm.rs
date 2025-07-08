use crate::{BackTracking, FuncEvalMultivariate, LineSearchSolver, MoreThuente};
use crate::{GradientDescent, Newton, BFGS};
use nalgebra::{DMatrix, DVector};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct OptimizationResult {
    x: Vec<f64>,
    f_value: f64,
    gradient_norm: f64,
    iterations: usize,
    success: bool,
    error_message: String,
}

#[wasm_bindgen]
impl OptimizationResult {
    pub fn new() -> Self {
        Self {
            x: Vec::new(),
            f_value: 0.0,
            gradient_norm: 0.0,
            iterations: 0,
            success: false,
            error_message: String::new(),
        }
    }

    pub fn get_x(&self) -> js_sys::Array {
        let array = js_sys::Array::new();
        for &value in &self.x {
            array.push(&JsValue::from_f64(value));
        }
        array
    }

    pub fn get_f_value(&self) -> f64 {
        self.f_value
    }

    pub fn get_gradient_norm(&self) -> f64 {
        self.gradient_norm
    }

    pub fn get_iterations(&self) -> usize {
        self.iterations
    }

    pub fn get_success(&self) -> bool {
        self.success
    }

    pub fn get_error_message(&self) -> String {
        self.error_message.clone()
    }
}

#[wasm_bindgen]
pub struct OptimizationSolver {
    tolerance: f64,
    max_iterations: usize,
}

#[wasm_bindgen]
impl OptimizationSolver {
    pub fn new(tolerance: f64, max_iterations: usize) -> Self {
        Self {
            tolerance,
            max_iterations,
        }
    }

    pub fn solve_gradient_descent(
        &self,
        x0: &[f64],
        f_and_g_fn: js_sys::Function,
    ) -> OptimizationResult {
        let mut result = OptimizationResult::new();

        // Convert initial point
        let x0_vec = DVector::from_vec(x0.to_vec());

        // Create objective function closure
        let objective = |x: &DVector<f64>| -> FuncEvalMultivariate {
            // Call JavaScript function
            let this = JsValue::NULL;
            let args = js_sys::Array::new();
            args.push(&JsValue::from_f64(x[0]));
            args.push(&JsValue::from_f64(x[1]));

            let js_result = f_and_g_fn.call1(&this, &args).unwrap();
            let js_array = js_sys::Array::from(&js_result);

            let f = js_array.get(0).as_f64().unwrap();
            let g1 = js_array.get(1).as_f64().unwrap();
            let g2 = js_array.get(2).as_f64().unwrap();

            let g = DVector::from_vec(vec![g1, g2]);
            FuncEvalMultivariate::new(f, g)
        };

        // Setup solver
        let mut solver = GradientDescent::new(self.tolerance, x0_vec);
        let mut ls = BackTracking::new(1e-4, 0.5);

        // Run optimization
        match solver.minimize(&mut ls, objective, self.max_iterations, 20, None) {
            Ok(()) => {
                let x = solver.x();
                let eval = objective(x);

                result.x = x.as_slice().to_vec();
                result.f_value = *eval.f();
                result.gradient_norm = eval.g().norm();
                result.iterations = *solver.k();
                result.success = true;
            }
            Err(e) => {
                result.error_message = format!("Optimization failed: {:?}", e);
                result.success = false;
            }
        }

        result
    }

    pub fn solve_bfgs(&self, x0: &[f64], f_and_g_fn: js_sys::Function) -> OptimizationResult {
        let mut result = OptimizationResult::new();

        // Convert initial point
        let x0_vec = DVector::from_vec(x0.to_vec());

        // Create objective function closure
        let objective = |x: &DVector<f64>| -> FuncEvalMultivariate {
            // Call JavaScript function
            let this = JsValue::NULL;
            let args = js_sys::Array::new();
            args.push(&JsValue::from_f64(x[0]));
            args.push(&JsValue::from_f64(x[1]));

            let js_result = f_and_g_fn.call1(&this, &args).unwrap();
            let js_array = js_sys::Array::from(&js_result);

            let f = js_array.get(0).as_f64().unwrap();
            let g1 = js_array.get(1).as_f64().unwrap();
            let g2 = js_array.get(2).as_f64().unwrap();

            let g = DVector::from_vec(vec![g1, g2]);
            FuncEvalMultivariate::new(f, g)
        };

        // Setup solver
        let mut solver = BFGS::new(self.tolerance, x0_vec);
        let mut ls = MoreThuente::default();

        // Run optimization
        match solver.minimize(&mut ls, objective, self.max_iterations, 20, None) {
            Ok(()) => {
                let x = solver.x();
                let eval = objective(x);

                result.x = x.as_slice().to_vec();
                result.f_value = *eval.f();
                result.gradient_norm = eval.g().norm();
                result.iterations = *solver.k();
                result.success = true;
            }
            Err(e) => {
                result.error_message = format!("Optimization failed: {:?}", e);
                result.success = false;
            }
        }

        result
    }

    pub fn solve_newton(
        &self,
        x0: &[f64],
        f_and_g_and_h_fn: js_sys::Function,
    ) -> OptimizationResult {
        let mut result = OptimizationResult::new();

        // Convert initial point
        let x0_vec = DVector::from_vec(x0.to_vec());

        // Create objective function closure with Hessian
        let objective = |x: &DVector<f64>| -> FuncEvalMultivariate {
            // Call JavaScript function
            let this = JsValue::NULL;
            let args = js_sys::Array::new();
            args.push(&JsValue::from_f64(x[0]));
            args.push(&JsValue::from_f64(x[1]));

            let js_result = f_and_g_and_h_fn.call1(&this, &args).unwrap();
            let js_array = js_sys::Array::from(&js_result);

            let f = js_array.get(0).as_f64().unwrap();
            let g1 = js_array.get(1).as_f64().unwrap();
            let g2 = js_array.get(2).as_f64().unwrap();

            let g = DVector::from_vec(vec![g1, g2]);

            // Extract Hessian (2x2 matrix)
            let h11 = js_array.get(3).as_f64().unwrap();
            let h12 = js_array.get(4).as_f64().unwrap();
            let h21 = js_array.get(5).as_f64().unwrap();
            let h22 = js_array.get(6).as_f64().unwrap();

            let hessian = DMatrix::from_vec(2, 2, vec![h11, h21, h12, h22]);

            FuncEvalMultivariate::new(f, g).with_hessian(hessian)
        };

        // Setup solver
        let mut solver = Newton::new(self.tolerance, x0_vec);
        let mut ls = MoreThuente::default();

        // Run optimization
        match solver.minimize(&mut ls, objective, self.max_iterations, 20, None) {
            Ok(()) => {
                let x = solver.x();
                let eval = objective(x);

                result.x = x.as_slice().to_vec();
                result.f_value = *eval.f();
                result.gradient_norm = eval.g().norm();
                result.iterations = *solver.k();
                result.success = true;
            }
            Err(e) => {
                result.error_message = format!("Optimization failed: {:?}", e);
                result.success = false;
            }
        }

        result
    }
}

// Utility functions for JavaScript
#[wasm_bindgen]
pub fn create_vector(data: &[f64]) -> JsValue {
    let array = js_sys::Array::new();
    for &value in data {
        array.push(&JsValue::from_f64(value));
    }
    JsValue::from(array)
}

#[wasm_bindgen]
pub fn log(message: &str) {
    web_sys::console::log_1(&JsValue::from_str(message));
}
