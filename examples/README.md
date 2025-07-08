# Optimization Solver Examples


To run any example, use:

```bash
cargo run --example <example_name>
```

For example:
```bash
cargo run --example gradient_descent_example
cargo run --example bfgs_example
cargo run --example newton_example
```

## Available Examples

### Unconstrained Optimization Solvers

- **`gradient_descent_example.rs`** - Gradient Descent
   - **Function**: f(x,y) = x² + 2y² (convex quadratic)
   - **Minimum**: (0, 0) with f(0,0) = 0
   - **Line Search**: Backtracking with Armijo condition

- **`bfgs_example.rs`** - BFGS Quasi-Newton Method
   - **Function**: f(x,y,z) = x² + 2y² + 3z² + xy + yz (convex quadratic)
   - **Minimum**: Solution of ∇f(x) = 0
   - **Line Search**: More-Thuente line search

- **`newton_example.rs`** - Newton's Method
   - **Function**: f(x,y) = x² + y² + exp(x² + y²) (convex)
   - **Minimum**: (0, 0) with f(0,0) = 1
   - **Line Search**: More-Thuente line search
   - **Requires**: Hessian information

- **`coordinate_descent_example.rs`** - Coordinate Descent
   - **Function**: f(x,y,z) = x² + 2y² + 3z² (separable convex)
   - **Minimum**: (0, 0, 0) with f(0,0,0) = 0
   - **Line Search**: Backtracking


- **`dfp_example.rs`** - DFP (Davidon-Fletcher-Powell) Quasi-Newton
   - **Function**: f(x,y) = x² + 5y² + xy (convex quadratic)
   - **Minimum**: (0, 0) with f(0,0) = 0
   - **Line Search**: More-Thuente line search

- **`broyden_example.rs`** - Broyden Quasi-Newton
   - **Function**: f(x,y) = x² + 3y² + 2xy (convex quadratic)
   - **Minimum**: (0, 0) with f(0,0) = 0
   - **Line Search**: More-Thuente line search

- **`pnorm_descent_example.rs`** - P-Norm Descent
   - **Function**: f(x,y) = x² + 4y² (convex quadratic)
   - **Minimum**: (0, 0) with f(0,0) = 0
   - **Preconditioner**: P = diag(1, 1/4)
   - **Line Search**: Backtracking


### Constrained Optimization Solvers

- **`projected_gradient_example.rs`** - Projected Gradient Descent
    - **Function**: f(x,y) = (x-2)² + (y-3)² (convex quadratic)
    - **Unconstrained minimum**: (2, 3) with f(2,3) = 0
    - **Constraints**: 0 ≤ x ≤ 1, 0 ≤ y ≤ 1
    - **Constrained minimum**: (1, 1) with f(1,1) = 5
    - **Line Search**: More-Thuente line search

- **`spg_example.rs`** - Spectral Projected Gradient (SPG)
    - **Function**: f(x,y) = x² + y² + exp(x² + y²) (convex)
    - **Minimum**: (0, 0) with f(0,0) = 1
    - **Constraints**: -1 ≤ x ≤ 1, -1 ≤ y ≤ 1
    - **Line Search**: Backtracking

- **`bfgs_bounded_example.rs`** - BFGSB (Bounded BFGS)
    - **Function**: f(x,y) = x² + 2y² + xy (convex quadratic)
    - **Constraints**: 0 ≤ x ≤ 2, 0 ≤ y ≤ 2
    - **Line Search**: More-Thuente line search

- **`sr1_bounded_example.rs`** - SR1B (Bounded SR1)
    - **Function**: f(x,y) = x² + 3y² + xy (convex quadratic)
    - **Constraints**: -1 ≤ x ≤ 1, -1 ≤ y ≤ 1
    - **Line Search**: More-Thuente line search

- **`dfp_bounded_example.rs`** - DFPB (Bounded DFP)
    - **Function**: f(x,y) = x² + 4y² + xy (convex quadratic)
    - **Constraints**: 0 ≤ x ≤ 1.5, 0 ≤ y ≤ 1.5
    - **Line Search**: More-Thuente line search

- **`broyden_bounded_example.rs`** - BroydenB (Bounded Broyden)
    - **Function**: f(x,y) = x² + 2y² + xy (convex quadratic)
    - **Constraints**: 0 ≤ x ≤ 1, 0 ≤ y ≤ 1
    - **Line Search**: More-Thuente line search



