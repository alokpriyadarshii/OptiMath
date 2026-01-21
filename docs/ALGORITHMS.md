# Algorithms (high-level)

This project is intentionally **dependency-free** so you can inspect and extend the math.
For serious production, use these components as reference implementations and integrate
with a dedicated solver backend.

## LP: Two-phase primal simplex

- Supports constraints of the form **a^T x <= b**, **a^T x >= b**, and **a^T x = b**.
- Automatically normalizes constraints so RHS b is non-negative.
- Converts the model into standard form by adding:
  - slack vars for <=
  - surplus + artificial vars for >=
  - artificial vars for =
- Phase 1 minimizes the sum of artificials (implemented as maximizing the negative sum).
- Phase 2 optimizes the original objective with artificials disallowed from entering.
- Uses **Bland's rule** for entering/leaving variable tie-breaking to reduce cycling risk.

## MILP: Branch-and-bound (educational)

- Solves LP relaxations with simplex.
- Branches on a fractional integer variable using two subproblems.
- Prunes nodes by bound, infeasibility, or node/iteration limits.

## NLP: BFGS quasi-Newton

- Unconstrained minimization with BFGS inverse-Hessian updates.
- Backtracking Armijo line search.
- Uses analytic gradients when provided; otherwise finite differences.

## Constrained NLP: Penalty wrapper

- Transforms constrained problems into unconstrained ones via quadratic penalties:
  - inequalities g(x) <= 0 penalize max(0, g(x))^2
  - equalities h(x) == 0 penalize h(x)^2
- Increases penalty weight outer-loop until constraints are satisfied within tolerance.
