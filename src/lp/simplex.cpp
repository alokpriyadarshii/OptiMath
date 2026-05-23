#include "optimath/lp/simplex.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "optimath/core/timer.hpp"

namespace optimath::lp {
namespace {

struct Tableau {
    // m constraint rows + 1 objective row
    std::vector<std::vector<double>> a; // rows x (nvars+1). Last col is RHS.
    std::vector<std::size_t> basis;     // size m: var index basic in each row
    std::vector<bool> is_artificial;    // size nvars
};

inline bool is_zero(double v, double tol) { return std::abs(v) <= tol; }

ConstraintSense flip_sense(ConstraintSense s) {
    switch (s) {
        case ConstraintSense::kLessEqual: return ConstraintSense::kGreaterEqual;
        case ConstraintSense::kGreaterEqual: return ConstraintSense::kLessEqual;
        case ConstraintSense::kEqual: return ConstraintSense::kEqual;
    }
    return ConstraintSense::kEqual;
}

void pivot(Tableau& T, std::size_t row, std::size_t col, double tol_pivot) {
    const std::size_t m = T.basis.size();
    const std::size_t ncols = T.a[0].size();
    const double p = T.a[row][col];
    if (std::abs(p) < tol_pivot) return;

    // Normalize pivot row.
    for (std::size_t j = 0; j < ncols; ++j) T.a[row][j] /= p;

    // Eliminate pivot col in other rows.
    for (std::size_t i = 0; i < m + 1; ++i) {
        if (i == row) continue;
        const double f = T.a[i][col];
        if (is_zero(f, 0.0)) continue;
        if (std::abs(f) < 1e-18) continue;
        for (std::size_t j = 0; j < ncols; ++j) {
            if (j == col) {
                T.a[i][j] = 0.0;
            } else {
                T.a[i][j] -= f * T.a[row][j];
            }
        }
    }

    T.basis[row] = col;
}

void build_objective_row(Tableau& T, const std::vector<double>& cost) {
    const std::size_t m = T.basis.size();
    const std::size_t nvars = cost.size();
    const std::size_t rhs = nvars;

    // objective row = -c
    std::fill(T.a[m].begin(), T.a[m].end(), 0.0);
    for (std::size_t j = 0; j < nvars; ++j) T.a[m][j] = -cost[j];
    T.a[m][rhs] = 0.0;

    // For each basic variable with cost cb, add cb * row to make basic column zero.
    for (std::size_t i = 0; i < m; ++i) {
        const std::size_t b = T.basis[i];
        const double cb = cost[b];
        if (is_zero(cb, 0.0)) continue;
        for (std::size_t j = 0; j <= nvars; ++j) {
            T.a[m][j] += cb * T.a[i][j];
        }
    }
}

struct IterationOutcome {
    core::Status status;
    std::size_t pivots{0};
};

IterationOutcome simplex_iterations(
    Tableau& T,
    const core::SolverOptions& options,
    const std::vector<bool>& allowed_enter) {

    const std::size_t m = T.basis.size();
    const std::size_t nvars = T.a[0].size() - 1;
    const std::size_t rhs = nvars;

    std::size_t pivots = 0;
    core::WallTimer timer;

    for (std::size_t it = 0; it < options.limits.max_iterations; ++it) {
        if (options.limits.time_limit_seconds > 0.0 && timer.elapsed_seconds() > options.limits.time_limit_seconds) {
            return {core::Status::MaxIterations("time limit exceeded"), pivots};
        }

        // Bland's rule: smallest index with negative coefficient in objective row (since objective row is -reduced costs)
        std::optional<std::size_t> enter;
        for (std::size_t j = 0; j < nvars; ++j) {
            if (!allowed_enter[j]) continue;
            if (T.a[m][j] < -options.tol.dual_feas) { enter = j; break; }
        }
        if (!enter.has_value()) {
            return {core::Status::Ok(), pivots};
        }
        const std::size_t e = *enter;

        // Ratio test
        double best_ratio = std::numeric_limits<double>::infinity();
        std::optional<std::size_t> leave;
        for (std::size_t i = 0; i < m; ++i) {
            const double aie = T.a[i][e];
            if (aie > options.tol.pivot) {
                const double r = T.a[i][rhs] / aie;
                if (r < best_ratio - 1e-15) {
                    best_ratio = r;
                    leave = i;
                } else if (std::abs(r - best_ratio) <= 1e-15 && leave.has_value()) {
                    // Tie-break: Bland by basis var index
                    if (T.basis[i] < T.basis[*leave]) leave = i;
                }
            }
        }
        if (!leave.has_value()) {
            return {core::Status::Unbounded("LP relaxation appears unbounded"), pivots};
        }

        pivot(T, *leave, e, options.tol.pivot);
        ++pivots;
    }

    return {core::Status::MaxIterations("max iterations reached"), pivots};
}

Tableau build_tableau(const LinearProgram& lp) {
    const std::size_t n = lp.num_vars();
    const auto& cons_in = lp.constraints();

    // Normalize constraints so RHS is non-negative.
    struct NCon { std::vector<double> a; double b; ConstraintSense s; };
    std::vector<NCon> cons;
    cons.reserve(cons_in.size());
    for (const auto& c : cons_in) {
        NCon nc{c.a, c.b, c.sense};
        if (nc.b < 0.0) {
            for (double& v : nc.a) v *= -1.0;
            nc.b *= -1.0;
            nc.s = flip_sense(nc.s);
        }
        cons.push_back(std::move(nc));
    }

    std::size_t n_slack = 0, n_surplus = 0, n_art = 0;
    for (const auto& c : cons) {
        if (c.s == ConstraintSense::kLessEqual) ++n_slack;
        else if (c.s == ConstraintSense::kGreaterEqual) { ++n_surplus; ++n_art; }
        else { ++n_art; }
    }

    const std::size_t nvars = n + n_slack + n_surplus + n_art;
    const std::size_t m = cons.size();

    Tableau T;
    T.a.assign(m + 1, std::vector<double>(nvars + 1, 0.0));
    T.basis.assign(m, 0);
    T.is_artificial.assign(nvars, false);

    std::size_t slack_base = n;
    std::size_t surplus_base = n + n_slack;
    std::size_t art_base = n + n_slack + n_surplus;
    std::size_t slack_i = 0, surplus_i = 0, art_i = 0;

    for (std::size_t i = 0; i < m; ++i) {
        const auto& c = cons[i];
        // original vars
        for (std::size_t j = 0; j < n; ++j) T.a[i][j] = c.a[j];
        // RHS
        T.a[i][nvars] = c.b;

        if (c.s == ConstraintSense::kLessEqual) {
            const std::size_t sidx = slack_base + slack_i++;
            T.a[i][sidx] = 1.0;
            T.basis[i] = sidx;
        } else if (c.s == ConstraintSense::kGreaterEqual) {
            const std::size_t suidx = surplus_base + surplus_i++;
            const std::size_t aidx = art_base + art_i++;
            T.a[i][suidx] = -1.0;
            T.a[i][aidx] = 1.0;
            T.is_artificial[aidx] = true;
            T.basis[i] = aidx;
        } else { // equal
            const std::size_t aidx = art_base + art_i++;
            T.a[i][aidx] = 1.0;
            T.is_artificial[aidx] = true;
            T.basis[i] = aidx;
        }
    }

    return T;
}

bool pivot_out_artificials(Tableau& T, const core::SolverOptions& options) {
    const std::size_t m = T.basis.size();
    const std::size_t nvars = T.a[0].size() - 1;

    for (std::size_t i = 0; i < m; ++i) {
        const std::size_t b = T.basis[i];
        if (!T.is_artificial[b]) continue;

        // Try to pivot in a non-artificial variable.
        std::optional<std::size_t> col;
        for (std::size_t j = 0; j < nvars; ++j) {
            if (T.is_artificial[j]) continue;
            if (std::abs(T.a[i][j]) > options.tol.pivot) { col = j; break; }
        }
        if (col.has_value()) {
            pivot(T, i, *col, options.tol.pivot);
        } else {
            // If RHS is ~0 and row has no non-art columns, it's redundant; keep as-is.
            // If RHS not ~0, something is wrong; but phase-1 objective should have caught infeasibility.
            if (std::abs(T.a[i][nvars]) > options.tol.primal_feas) return false;
        }
    }
    return true;
}

} // namespace

core::SolveResult<SimplexResult> solve_simplex(const LinearProgram& lp, const core::SolverOptions& options) {
    core::SolveResult<SimplexResult> out;
    core::WallTimer wall;

    if (lp.num_vars() == 0) {
        out.status = core::Status::Invalid("LP has zero variables");
        return out;
    }

    Tableau T = build_tableau(lp);
    const std::size_t m = T.basis.size();
    const std::size_t nvars = T.a[0].size() - 1;

    // Phase 1: maximize -sum artificial
    std::vector<double> c1(nvars, 0.0);
    for (std::size_t j = 0; j < nvars; ++j) {
        if (T.is_artificial[j]) c1[j] = -1.0;
    }
    build_objective_row(T, c1);

    std::vector<bool> allow_all(nvars, true);
    auto phase1 = simplex_iterations(T, options, allow_all);
    out.iterations += phase1.pivots;

    if (!phase1.status.ok()) {
        out.status = phase1.status;
        out.solve_time_seconds = wall.elapsed_seconds();
        return out;
    }

    const double phase1_obj = T.a[m][nvars];
    if (std::abs(phase1_obj) > options.tol.objective) {
        out.status = core::Status::Infeasible("LP infeasible (phase-1 objective > 0)");
        out.solve_time_seconds = wall.elapsed_seconds();
        return out;
    }

    if (!pivot_out_artificials(T, options)) {
        out.status = core::Status::Numerical("failed to remove artificial variables cleanly");
        out.solve_time_seconds = wall.elapsed_seconds();
        return out;
    }

    // Phase 2: original objective
    std::vector<double> c2(nvars, 0.0);
    const auto& c = lp.objective();
    for (std::size_t j = 0; j < lp.num_vars(); ++j) c2[j] = c[j];
    build_objective_row(T, c2);

    // Disallow artificial vars entering in phase2.
    std::vector<bool> allow_phase2(nvars, true);
    for (std::size_t j = 0; j < nvars; ++j) {
        if (T.is_artificial[j]) allow_phase2[j] = false;
    }

    auto phase2 = simplex_iterations(T, options, allow_phase2);
    out.iterations += phase2.pivots;

    if (!phase2.status.ok()) {
        out.status = phase2.status;
        out.solve_time_seconds = wall.elapsed_seconds();
        return out;
    }

    // Extract solution
    LPSolution sol;
    sol.x.assign(lp.num_vars(), 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        const std::size_t b = T.basis[i];
        if (b < lp.num_vars()) sol.x[b] = T.a[i][nvars];
    }
    double obj = 0.0;
    for (std::size_t j = 0; j < lp.num_vars(); ++j) obj += c[j] * sol.x[j];
    sol.objective_value = obj;

    out.solution.solution = std::move(sol);
    out.solution.stats.pivots = phase1.pivots + phase2.pivots;
    out.status = core::Status::Ok();
    out.solve_time_seconds = wall.elapsed_seconds();
    return out;
}

} // namespace optimath::lp
