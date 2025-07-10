# Optimization Solvers

A comprehensive Rust library implementing composable state-of-the-art numerical optimization algorithms with WebAssembly support for browser-based optimization.

- **[Live WASM demo](https://fedemagnani.github.io/optimization-solvers-demo/)** - Run some optimization problems from yout browser

## Quick Start

```rust
use optimization_solvers::{GradientDescent, BFGS, Newton};

// Run optimization with any solver
let result = solver.minimize(objective_function, max_iterations);
```

## 📚 Documentation & Resources

- **[Documentation](https://deepwiki.com/fedemagnani/optimization-solvers)** - Full Rust API documentation
- **[Examples](./examples/)** - Complete examples for all optimization algorithms
- **[WASM Documentation](./wasm/README.md)** - Browser-based optimization with WebAssembly
- **[Academic resources](./resources.md)** - Misc of papers/books about numerical optimization

## 🧮 Solver Categories

### First-Order Methods
- **[Gradient Descent](./src/steepest_descent/gradient_descent.rs)** - Classic steepest descent
- **[Projected Gradient](./src/steepest_descent/projected_gradient_descent.rs)** - Constrained optimization
- **[Coordinate Descent](./src/steepest_descent/coordinate_descent.rs)** - Coordinate-wise optimization
- **[SPG](./src/steepest_descent/spg.rs)** - Spectral Projected Gradient
- **[P-Norm Descent](./src/steepest_descent/pnorm_descent.rs)** - Lp-norm based descent

### Quasi-Newton Methods
- **[BFGS](./src/quasi_newton/bfgs.rs)** - Broyden-Fletcher-Goldfarb-Shanno
- **[DFP](./src/quasi_newton/dfp.rs)** - Davidon-Fletcher-Powell
- **[Broyden](./src/quasi_newton/broyden.rs)** - Broyden's method
- **[L-BFGS-B](./src/quasi_newton/lbfgsb.rs)** - Limited-memory BFGS with bounds (enable `lbfgsb` feature flag)

### Second-Order Methods
- **[Newton's Method](./src/newton/mod.rs)** - Classical Newton optimization
- **[Projected Newton](./src/newton/projected_newton.rs)** - Constrained Newton
- **[SPN](./src/newton/spn.rs)** - Spectral Projected Newton

## 🚀 Getting Started

```bash
# Add to Cargo.toml
cargo add optimization-solvers

# Run examples
cargo run --example gradient_descent_example
cargo run --example bfgs_example

# Build for WebAssembly
cd wasm && ./build-wasm.sh
```

## 📦 Features

- **Multiple Algorithms**: 15+ optimization algorithms
- **WebAssembly Support**: Run in browsers with full performance
- **Line Search Methods**: Backtracking, More-Thuente, and more
- **Bounded Optimization**: Support for box constraints
- **Comprehensive Examples**: Ready-to-run examples for all solvers





