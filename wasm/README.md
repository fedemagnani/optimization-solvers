# WebAssembly (WASM) Support for Optimization Solvers

This guide explains how to compile and use the optimization solvers in WebAssembly for browser-based applications.

## Prerequisites

1. **Rust and Cargo** - Install from [rustup.rs](https://rustup.rs/)
2. **wasm-pack** - Install with: `cargo install wasm-pack`
3. **Node.js** (optional, for development server)

## Quick Start

### 1. Build for WASM

```bash
# Run the build script
cd wasm && ./build-wasm.sh

# Or manually:
wasm-pack build --target web --out-dir wasm/pkg
```

### 2. Test the Demo

```bash
# Start a local server (if you have Python)
python3 -m http.server 8000

# Or with Node.js
npx serve .

# Then open http://localhost:8000/wasm/demo/index.html
```

## Project Structure

```
optimization-solvers/
├── src/
│   ├── lib.rs          # Main library with conditional compilation
│   ├── wasm.rs         # WASM-specific interface
│   └── ...             # Core solver implementations
└── wasm/               # WASM-specific files
    ├── demo/
    │   └── index.html  # Interactive demo page
    ├── pkg/            # Generated WASM files (after build)
    ├── build-wasm.sh   # Build script
    └── README.md       # This file
```

## WASM Interface

### Available Solvers

The WASM interface provides three main optimization algorithms:

1. **Gradient Descent** - First-order method for unconstrained optimization
2. **BFGS** - Quasi-Newton method for faster convergence
3. **Newton's Method** - Second-order method (requires Hessian)

### JavaScript API

```javascript
import init, { OptimizationSolver } from './pkg/optimization_solvers.js';

// Initialize WASM module
await init();

// Create solver
const solver = OptimizationSolver.new('gradient_descent', 1e-6, 100);

// Define objective function (returns [f, g1, g2, ...])
function objective(x) {
    const x1 = x[0], x2 = x[1];
    const f = x1 * x1 + 2 * x2 * x2;  // f(x,y) = x² + 2y²
    const g1 = 2 * x1;                // ∂f/∂x
    const g2 = 4 * x2;                // ∂f/∂y
    return [f, g1, g2];
}

// Run optimization
const result = solver.solve_gradient_descent([2, 1], objective);

// Check results
if (result.success) {
    console.log('Solution:', result.x);
    console.log('Function value:', result.f_value);
    console.log('Iterations:', result.iterations);
} else {
    console.error('Error:', result.error_message);
}
```

### Result Object

```javascript
{
    x: [number, number, ...],     // Final iterate
    f_value: number,              // Function value at solution
    gradient_norm: number,        // Gradient norm at solution
    iterations: number,           // Number of iterations performed
    success: boolean,             // Whether optimization succeeded
    error_message: string         // Error message if failed
}
```

## Objective Function Format

### For Gradient Descent and BFGS

Return an array with function value and gradient components:

```javascript
function objective(x) {
    // x is an array [x1, x2, ...]
    const x1 = x[0], x2 = x[1];
    
    // Compute function value
    const f = /* your function */;
    
    // Compute gradient components
    const g1 = /* ∂f/∂x1 */;
    const g2 = /* ∂f/∂x2 */;
    
    return [f, g1, g2, ...];  // [f, ∂f/∂x1, ∂f/∂x2, ...]
}
```

### For Newton's Method

Return an array with function value, gradient, and Hessian components:

```javascript
function objective(x) {
    const x1 = x[0], x2 = x[1];
    
    // Function value
    const f = /* your function */;
    
    // Gradient
    const g1 = /* ∂f/∂x1 */;
    const g2 = /* ∂f/∂x2 */;
    
    // Hessian (2x2 matrix flattened)
    const h11 = /* ∂²f/∂x1² */;
    const h12 = /* ∂²f/∂x1∂x2 */;
    const h21 = /* ∂²f/∂x2∂x1 */;
    const h22 = /* ∂²f/∂x2² */;
    
    return [f, g1, g2, h11, h12, h21, h22];
}
```

## Example Functions

### 1. Quadratic Function
```javascript
// f(x,y) = x² + 2y²
function quadratic(x) {
    const x1 = x[0], x2 = x[1];
    const f = x1 * x1 + 2 * x2 * x2;
    const g1 = 2 * x1;
    const g2 = 4 * x2;
    return [f, g1, g2];
}
```

### 2. Rosenbrock Function
```javascript
// f(x,y) = (1-x)² + 100(y-x²)²
function rosenbrock(x) {
    const x1 = x[0], x2 = x[1];
    const f = Math.pow(1 - x1, 2) + 100 * Math.pow(x2 - x1 * x1, 2);
    const g1 = -2 * (1 - x1) - 400 * x1 * (x2 - x1 * x1);
    const g2 = 200 * (x2 - x1 * x1);
    return [f, g1, g2];
}
```

### 3. Exponential Function (for Newton)
```javascript
// f(x,y) = x² + y² + exp(x² + y²)
function exponential(x) {
    const x1 = x[0], x2 = x[1];
    const exp_term = Math.exp(x1 * x1 + x2 * x2);
    const f = x1 * x1 + x2 * x2 + exp_term;
    
    // Gradient
    const g1 = 2 * x1 * (1 + exp_term);
    const g2 = 2 * x2 * (1 + exp_term);
    
    // Hessian
    const h11 = 2 * (1 + exp_term) + 4 * x1 * x1 * exp_term;
    const h12 = 4 * x1 * x2 * exp_term;
    const h21 = h12;
    const h22 = 2 * (1 + exp_term) + 4 * x2 * x2 * exp_term;
    
    return [f, g1, g2, h11, h12, h21, h22];
}
```

## Integration with Web Frameworks

### React Example

```jsx
import { useState, useEffect } from 'react';
import init, { OptimizationSolver } from './pkg/optimization_solvers.js';

function OptimizationDemo() {
    const [solver, setSolver] = useState(null);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        init().then(() => {
            setSolver(new OptimizationSolver('gradient_descent', 1e-6, 100));
        });
    }, []);

    const runOptimization = () => {
        if (!solver) return;
        
        setLoading(true);
        const result = solver.solve_gradient_descent([2, 1], objective);
        setResult(result);
        setLoading(false);
    };

    return (
        <div>
            <button onClick={runOptimization} disabled={loading}>
                {loading ? 'Running...' : 'Run Optimization'}
            </button>
            {result && (
                <div>
                    <h3>Result: {result.success ? 'Success' : 'Failed'}</h3>
                    {result.success && (
                        <p>Solution: [{result.x.join(', ')}]</p>
                    )}
                </div>
            )}
        </div>
    );
}
```

### Vue.js Example

```vue
<template>
  <div>
    <button @click="runOptimization" :disabled="loading">
      {{ loading ? 'Running...' : 'Run Optimization' }}
    </button>
    <div v-if="result">
      <h3>Result: {{ result.success ? 'Success' : 'Failed' }}</h3>
      <p v-if="result.success">Solution: [{{ result.x.join(', ') }}]</p>
    </div>
  </div>
</template>

<script>
import init, { OptimizationSolver } from './pkg/optimization_solvers.js';

export default {
  data() {
    return {
      solver: null,
      result: null,
      loading: false
    };
  },
  async mounted() {
    await init();
    this.solver = new OptimizationSolver('gradient_descent', 1e-6, 100);
  },
  methods: {
    runOptimization() {
      if (!this.solver) return;
      
      this.loading = true;
      this.result = this.solver.solve_gradient_descent([2, 1], objective);
      this.loading = false;
    }
  }
};
</script>
```

## Performance Considerations

1. **WASM vs JavaScript**: The core optimization algorithms run in WASM for better performance
2. **Function Calls**: Each objective function evaluation requires a JavaScript → WASM call
3. **Memory**: WASM has its own memory space; large problems may require careful memory management
4. **Initialization**: The WASM module needs to be loaded once before use

## Troubleshooting

### Common Issues

1. **"WASM module not loaded"**
   - Ensure you call `await init()` before using any solver functions
   - Check that the WASM files are accessible from your web server

2. **"Function not found"**
   - Verify that your objective function returns the correct number of values
   - For Newton's method, ensure you provide Hessian components

3. **"Optimization failed"**
   - Check that your objective function is well-defined
   - Try different initial points or tolerance values
   - Ensure the function is convex for guaranteed convergence

### Debug Tips

```javascript
// Enable console logging
import { log } from './pkg/optimization_solvers.js';
log('Debug message');

// Check WASM module status
console.log('WASM module loaded:', !!wasmModule);
```

## Browser Compatibility

- **Modern browsers**: Chrome 57+, Firefox 52+, Safari 11+, Edge 79+
- **Mobile browsers**: iOS Safari 11+, Chrome Mobile 57+
- **Node.js**: Requires `--experimental-wasm-modules` flag

## License

The WASM interface follows the same license as the main optimization-solvers crate (MIT OR Apache-2.0). 