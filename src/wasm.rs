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
            // Add all vector components to the args array
            for &value in x.as_slice() {
                args.push(&JsValue::from_f64(value));
            }

            let js_result = f_and_g_fn.call1(&this, &args).unwrap();
            let js_array = js_sys::Array::from(&js_result);

            let f = js_array.get(0).as_f64().unwrap();
            // Extract gradient components dynamically
            let mut g_values = Vec::new();
            for i in 1..js_array.length() {
                if let Some(g_val) = js_array.get(i).as_f64() {
                    g_values.push(g_val);
                }
            }
            let g = DVector::from_vec(g_values);
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
            // Add all vector components to the args array
            for &value in x.as_slice() {
                args.push(&JsValue::from_f64(value));
            }

            let js_result = f_and_g_fn.call1(&this, &args).unwrap();
            let js_array = js_sys::Array::from(&js_result);

            let f = js_array.get(0).as_f64().unwrap();
            // Extract gradient components dynamically
            let mut g_values = Vec::new();
            for i in 1..js_array.length() {
                if let Some(g_val) = js_array.get(i).as_f64() {
                    g_values.push(g_val);
                }
            }
            let g = DVector::from_vec(g_values);
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
        let n = x0_vec.len();

        // Create objective function closure with Hessian
        let objective = |x: &DVector<f64>| -> FuncEvalMultivariate {
            // Call JavaScript function
            let this = JsValue::NULL;
            let args = js_sys::Array::new();
            // Add all vector components to the args array
            for &value in x.as_slice() {
                args.push(&JsValue::from_f64(value));
            }

            let js_result = f_and_g_and_h_fn.call1(&this, &args).unwrap();
            let js_array = js_sys::Array::from(&js_result);

            let f = js_array.get(0).as_f64().unwrap();

            // Extract gradient components
            let mut g_values = Vec::new();
            for i in 1..=n {
                if let Some(g_val) = js_array.get(i as u32).as_f64() {
                    g_values.push(g_val);
                } else {
                    panic!("Expected gradient component at index {}", i);
                }
            }
            let g = DVector::from_vec(g_values);

            // Extract Hessian components (n×n matrix)
            let mut hessian_values = Vec::new();
            let hessian_start = n + 1;
            let expected_hessian_size = n * n;

            for i in 0..expected_hessian_size {
                let idx = hessian_start + i;
                if let Some(h_val) = js_array.get(idx as u32).as_f64() {
                    hessian_values.push(h_val);
                } else {
                    panic!("Expected Hessian component at index {}", idx);
                }
            }

            let hessian = DMatrix::from_vec(n, n, hessian_values);

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
