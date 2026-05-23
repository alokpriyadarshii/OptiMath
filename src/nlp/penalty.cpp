#include "optimath/nlp/penalty.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "optimath/core/timer.hpp"
#include "optimath/nlp/bfgs.hpp"

namespace optimath::nlp {
namespace {

bool has_grad(const std::function<optimath::linalg::Vector(const optimath::linalg::Vector&)>& g) {
    return static_cast<bool>(g);
}

optimath::linalg::Vector finite_difference_grad_scalar(
    const std::function<double(const optimath::linalg::Vector&)>& fn,
    const optimath::linalg::Vector& x,
    double eps = 1e-8) {

    const std::size_t n = x.size();
    optimath::linalg::Vector g(n, 0.0);
    optimath::linalg::Vector xx = x;

    for (std::size_t i = 0; i < n; ++i) {
        const double xi = x.raw()[i];
        const double h = eps * (std::abs(xi) + 1.0);

        xx.raw()[i] = xi + h;
        const double fp = fn(xx);

        xx.raw()[i] = xi - h;
        const double fm = fn(xx);

        xx.raw()[i] = xi;
        g.raw()[i] = (fp - fm) / (2.0 * h);
    }

    return g;
}

// Compute max constraint violation.
double max_violation(
    const optimath::linalg::Vector& x,
    const std::vector<InequalityConstraint>& ineq,
    const std::vector<EqualityConstraint>& eq) {

    double v = 0.0;
    for (const auto& c : ineq) {
        const double gx = c.value(x);
        v = std::max(v, std::max(0.0, gx));
    }
    for (const auto& c : eq) {
        const double hx = c.value(x);
        v = std::max(v, std::abs(hx));
    }
    return v;
}

class PenaltyObjective final : public Objective {
public:
    PenaltyObjective(
        const Objective& base,
        const std::vector<InequalityConstraint>& ineq,
        const std::vector<EqualityConstraint>& eq,
        double mu)
        : base_(base), ineq_(ineq), eq_(eq), mu_(mu) {}

    double value(const optimath::linalg::Vector& x) const override {
        double v = base_.value(x);
        double pen = 0.0;
        for (const auto& c : ineq_) {
            const double gx = c.value(x);
            if (gx > 0.0) pen += gx * gx;
        }
        for (const auto& c : eq_) {
            const double hx = c.value(x);
            pen += hx * hx;
        }
        return v + mu_ * pen;
    }

    std::optional<optimath::linalg::Vector> gradient(const optimath::linalg::Vector& x) const override {
        // Base gradient (analytic if available, else finite difference).
        optimath::linalg::Vector g;
        if (auto gb = base_.gradient(x)) {
            g = *gb;
        } else {
            g = finite_difference_gradient(base_, x);
        }

        if (mu_ <= 0.0) return g;

        // Add penalty gradients.
        for (const auto& c : ineq_) {
            const double gx = c.value(x);
            if (gx <= 0.0) continue;
            optimath::linalg::Vector gg;
            if (has_grad(c.grad)) {
                gg = c.grad(x);
            } else {
                gg = finite_difference_grad_scalar(c.value, x);
            }
            for (std::size_t i = 0; i < g.size(); ++i) {
                g.raw()[i] += mu_ * 2.0 * gx * gg.raw()[i];
            }
        }

        for (const auto& c : eq_) {
            const double hx = c.value(x);
            optimath::linalg::Vector hg;
            if (has_grad(c.grad)) {
                hg = c.grad(x);
            } else {
                hg = finite_difference_grad_scalar(c.value, x);
            }
            for (std::size_t i = 0; i < g.size(); ++i) {
                g.raw()[i] += mu_ * 2.0 * hx * hg.raw()[i];
            }
        }

        return g;
    }

private:
    const Objective& base_;
    const std::vector<InequalityConstraint>& ineq_;
    const std::vector<EqualityConstraint>& eq_;
    double mu_{1.0};
};

} // namespace

core::SolveResult<PenaltyResult> minimize_with_penalty(
    const Objective& f,
    const optimath::linalg::Vector& x0,
    const std::vector<InequalityConstraint>& ineq,
    const std::vector<EqualityConstraint>& eq,
    const core::SolverOptions& options) {

    core::WallTimer timer;

    // Outer iterations are typically small; use a conservative cap.
    const std::size_t max_outer = std::max<std::size_t>(5, std::min<std::size_t>(30, options.limits.max_iterations / 50));

    double mu = 10.0;
    optimath::linalg::Vector x = x0;

    PenaltyStats stats;
    double prev_violation = std::numeric_limits<double>::infinity();

    for (std::size_t outer = 0; outer < max_outer; ++outer) {
        stats.outer_iters = outer + 1;

        const double viol = max_violation(x, ineq, eq);
        if (viol <= std::max(1e-8, 10.0 * options.tol.primal_feas)) {
            PenaltyResult out;
            out.solution = {x, f.value(x)};
            out.stats = stats;

            core::SolveResult<PenaltyResult> res;
            res.status = core::Status::Ok();
            res.solution = out;
            res.solve_time_seconds = timer.elapsed_seconds();
            res.iterations = stats.inner_iters;
            return res;
        }

        // Solve penalized subproblem.
        PenaltyObjective pobj(f, ineq, eq, mu);

        core::SolverOptions inner = options;
        inner.limits.max_iterations = std::max<std::size_t>(200, options.limits.max_iterations / 2);

        auto inner_res = minimize_bfgs(pobj, x, inner);
        stats.inner_iters += inner_res.iterations;

        // Move to best found (even if inner hit max-it, it may have improved).
        x = inner_res.solution.solution.x;

        const double new_viol = max_violation(x, ineq, eq);

        // Update mu if feasibility is not improving sufficiently.
        if (new_viol > 0.5 * prev_violation) {
            mu *= 10.0;
        } else {
            mu *= 2.0;
        }
        prev_violation = new_viol;

        if (options.limits.time_limit_seconds > 0.0 && timer.elapsed_seconds() > options.limits.time_limit_seconds) {
            break;
        }
    }

    // Final check.
    const double viol = max_violation(x, ineq, eq);
    PenaltyResult out;
    out.solution = {x, f.value(x)};
    out.stats = stats;

    core::SolveResult<PenaltyResult> res;
    res.solution = out;
    res.solve_time_seconds = timer.elapsed_seconds();
    res.iterations = stats.inner_iters;

    if (viol <= std::max(1e-8, 10.0 * options.tol.primal_feas)) {
        res.status = core::Status::Ok();
    } else {
        res.status = core::Status::MaxIterations("Penalty: constraints not satisfied within outer iteration budget");
    }
    return res;
}

} // namespace optimath::nlp
