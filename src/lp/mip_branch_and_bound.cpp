#include "optimath/lp/mip_branch_and_bound.hpp"

#include <cmath>
#include <limits>
#include <optional>

#include "optimath/core/timer.hpp"
#include "optimath/lp/simplex.hpp"

namespace optimath::lp {
namespace {

struct Best {
    bool has{false};
    double obj{-std::numeric_limits<double>::infinity()};
    LPSolution sol{};
};

struct SearchState {
    bool stopped_on_limit{false};
    std::optional<optimath::core::Status> terminal_status{};
};

std::optional<std::size_t> find_fractional_var(const std::vector<double>& x,
                                               const std::vector<std::size_t>& integer_vars,
                                               double tol,
                                               double* out_val) {
    double best_frac = 0.0;
    std::optional<std::size_t> best_idx;
    double best_val = 0.0;
    for (auto idx : integer_vars) {
        if (idx >= x.size()) continue;
        const double v = x[idx];
        const double frac = std::abs(v - std::round(v));
        if (frac > tol && frac > best_frac) {
            best_frac = frac;
            best_idx = idx;
            best_val = v;
        }
    }
    if (best_idx && out_val) *out_val = best_val;
    return best_idx;
}

void bnb(LinearProgram lp,
         const std::vector<std::size_t>& integer_vars,
         const optimath::core::SolverOptions& options,
         optimath::core::WallTimer& timer,
         MIPStats& stats,
         Best& best,
         SearchState& state) {
    if (options.limits.time_limit_seconds > 0.0 && timer.elapsed_seconds() >= options.limits.time_limit_seconds) {
        state.stopped_on_limit = true;
        return;
    }
    if (stats.nodes_explored >= options.limits.max_nodes) {
        state.stopped_on_limit = true;
        return;
    }

    ++stats.nodes_explored;

    auto lp_res = solve_simplex(lp, options);
    ++stats.lp_solves;
    if (!lp_res.status.ok()) {
        if (lp_res.status.code != optimath::core::StatusCode::kInfeasible) {
            state.terminal_status = lp_res.status;
        }
        return;
    }

    const auto& sol = lp_res.solution.solution;

    // Bound.
    if (best.has && sol.objective_value <= best.obj + options.tol.objective) return;

    double branch_val = 0.0;
    auto branch_var = find_fractional_var(sol.x, integer_vars, 1e-8, &branch_val);
    if (!branch_var) {
        // Integral.
        best.has = true;
        best.obj = sol.objective_value;
        best.sol = sol;
        return;
    }

    const std::size_t j = *branch_var;
    const double lo = std::floor(branch_val);
    const double hi = std::ceil(branch_val);

    // Left: x_j <= lo
    {
        auto left = lp;
        std::vector<double> a(left.num_vars(), 0.0);
        a[j] = 1.0;
        left.add_constraint(a, lo, ConstraintSense::kLessEqual, "bnb_ub");
        bnb(std::move(left), integer_vars, options, timer, stats, best, state);
    }

    // Right: x_j >= hi
    {
        auto right = lp;
        std::vector<double> a(right.num_vars(), 0.0);
        a[j] = 1.0;
        right.add_constraint(a, hi, ConstraintSense::kGreaterEqual, "bnb_lb");
        bnb(std::move(right), integer_vars, options, timer, stats, best, state);
    }
}

} // namespace

optimath::core::SolveResult<MIPResult> solve_branch_and_bound(const MIPModel& mip, const optimath::core::SolverOptions& options) {
    optimath::core::WallTimer timer;

    MIPStats stats;
    Best best;
    SearchState state;

    bnb(mip.relaxation, mip.integer_vars, options, timer, stats, best, state);

    MIPResult out;
    out.stats = stats;
    out.has_feasible_solution = best.has;
    out.solution = best.sol;

    optimath::core::SolveResult<MIPResult> res;
    res.solve_time_seconds = timer.elapsed_seconds();
    res.iterations = stats.nodes_explored;
    res.solution = out;

    if (state.stopped_on_limit) {
        res.status = optimath::core::Status::MaxIterations("Branch-and-bound stopped before proving optimality");
    } else if (!best.has && state.terminal_status.has_value()) {
        res.status = *state.terminal_status;
    } else if (!best.has) {
        res.status = optimath::core::Status::Infeasible("No integer-feasible solution found within limits");
    } else {
        res.status = optimath::core::Status::Ok();
    }

    return res;
}

} // namespace optimath::lp
