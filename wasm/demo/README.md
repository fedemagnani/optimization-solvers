# Interactive Optimization Solvers Demo

This demo provides an interactive web interface for testing optimization algorithms with custom objective functions. Users can input their own functions, choose from different solvers, and compare results.

## Features

### 🎯 **Interactive Function Input**
- Write custom objective functions in JavaScript
- Support for both gradient-only and gradient+Hessian methods
- Real-time function validation and testing

### 🔧 **Multiple Solvers**
- **Gradient Descent**: First-order method, requires function value and gradient
- **BFGS Quasi-Newton**: Quasi-Newton method, requires function value and gradient
- **Newton's Method**: Second-order method, requires function value, gradient, and Hessian

### 📚 **Function Templates**
- **Quadratic**: Simple quadratic function f(x,y) = x² + 2y²
- **Rosenbrock**: Classic optimization test function
- **Ackley**: Multi-modal function with many local minima
- **Sphere**: Simple sphere function f(x,y) = x² + y²

### 📊 **Results & History**
- Detailed optimization results with convergence information
- History of previous runs for easy comparison
- Click on history items to reload previous configurations

## How to Use

### 1. Choose a Solver
Select from the dropdown menu:
- **Gradient Descent**: Good for simple problems, when Hessian computation is expensive
- **BFGS**: Good for medium-scale problems with good convergence properties
- **Newton's Method**: Best for small problems when Hessian is available

### 2. Set Initial Point
Enter the starting point as comma-separated values:
- For 2D: `2,1`
- For 3D: `1,2,3`
- The interface will automatically detect the number of dimensions

### 3. Configure Parameters
- **Tolerance**: Convergence criterion (default: 1e-6)
- **Max Iterations**: Maximum number of iterations (default: 100)

### 4. Write Your Objective Function

#### For Gradient-Only Methods (Gradient Descent, BFGS)
```javascript
function objective(x) {
    const x1 = x[0];
    const x2 = x[1];
    
    // Your function: f(x,y) = x² + 2y²
    const f = x1 * x1 + 2 * x2 * x2;
    
    // Gradient: [∂f/∂x, ∂f/∂y] = [2x, 4y]
    const g1 = 2 * x1;
    const g2 = 4 * x2;
    
    return [f, g1, g2];
}
```

#### For Newton's Method (with Hessian)
```javascript
function objective(x) {
    const x1 = x[0];
    const x2 = x[1];
    
    // Your function: f(x,y) = x² + 2y²
    const f = x1 * x1 + 2 * x2 * x2;
    
    // Gradient: [∂f/∂x, ∂f/∂y] = [2x, 4y]
    const g1 = 2 * x1;
    const g2 = 4 * x2;
    
    // Hessian: [[∂²f/∂x², ∂²f/∂x∂y], [∂²f/∂y∂x, ∂²f/∂y²]] = [[2, 0], [0, 4]]
    const h11 = 2;  // ∂²f/∂x²
    const h12 = 0;  // ∂²f/∂x∂y
    const h21 = 0;  // ∂²f/∂y∂x
    const h22 = 4;  // ∂²f/∂y²
    
    return [f, g1, g2, h11, h12, h21, h22];
}
```

### 5. Use Templates
Click on template buttons to load pre-defined functions:
- **Quadratic**: Simple test function
- **Rosenbrock**: Classic optimization benchmark
- **Ackley**: Multi-modal function
- **Sphere**: Basic sphere function

### 6. Run Optimization
Click "🚀 Run Optimization" to start the solver. Results will show:
- Final iterate (optimal point)
- Function value at the optimum
- Gradient norm (convergence measure)
- Number of iterations
- Success/failure status

## Function Requirements

### Input Format
- Function receives an array `x` where `x[0]`, `x[1]`, etc. are the variables
- Must return an array with the required derivatives

### Return Format
- **Gradient methods**: `[f, g1, g2, ...]` where `f` is function value and `g1,g2,...` are gradient components
- **Newton method**: `[f, g1, g2, h11, h12, h21, h22, ...]` where `h11,h12,h21,h22,...` are Hessian components

### Dimension Handling
- The interface automatically detects the number of dimensions from your initial point
- Make sure your function handles the correct number of variables
- For n-dimensional problems, access variables as `x[0]`, `x[1]`, ..., `x[n-1]`

## Example Functions

### 1. Simple Quadratic (2D)
```javascript
function objective(x) {
    const x1 = x[0], x2 = x[1];
    const f = x1 * x1 + 2 * x2 * x2;
    const g1 = 2 * x1;
    const g2 = 4 * x2;
    return [f, g1, g2];
}
```

### 2. Rosenbrock Function (2D)
```javascript
function objective(x) {
    const x1 = x[0], x2 = x[1];
    const f = Math.pow(1 - x1, 2) + 100 * Math.pow(x2 - x1 * x1, 2);
    const g1 = -2 * (1 - x1) - 400 * x1 * (x2 - x1 * x1);
    const g2 = 200 * (x2 - x1 * x1);
    return [f, g1, g2];
}
```

### 3. 3D Function Example
```javascript
function objective(x) {
    const x1 = x[0], x2 = x[1], x3 = x[2];
    const f = x1 * x1 + x2 * x2 + x3 * x3;
    const g1 = 2 * x1;
    const g2 = 2 * x2;
    const g3 = 2 * x3;
    return [f, g1, g2, g3];
}
```

## Tips for Best Results

1. **Start with templates**: Use the provided templates to understand the format
2. **Check dimensions**: Ensure your function handles the correct number of variables
3. **Test your function**: The interface validates your function before running
4. **Choose appropriate solver**: 
   - Use Gradient Descent for simple problems
   - Use BFGS for medium-scale problems
   - Use Newton's method only if you can compute the Hessian
5. **Adjust parameters**: Lower tolerance for more precise results, higher max iterations for complex functions

## Troubleshooting

### Common Errors
- **"Function must return an array"**: Make sure your function returns the correct format
- **"WASM module not loaded"**: Wait for the page to fully load
- **"Please enter a valid initial point"**: Use comma-separated numbers like `1,2,3`

### Performance Tips
- Start with simple functions to test the interface
- Use appropriate initial points (avoid points where gradient is undefined)
- For Newton's method, ensure your Hessian is positive definite near the optimum

## Building and Running

1. Build the WASM module:
   ```bash
   cd wasm
   ./build-wasm.sh
   ```

2. Serve the demo:
   ```bash
   cd demo
   python -m http.server 8000
   ```

3. Open `http://localhost:8000` in your browser

## Technical Details

- Built with WebAssembly for high-performance optimization
- Uses Rust optimization algorithms compiled to WASM
- JavaScript interface for easy function input and result display
- Real-time validation and error handling
- Responsive design for desktop and mobile use 