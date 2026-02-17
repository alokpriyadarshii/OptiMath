#include "optimath/nlp/bfgs.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "optimath/core/timer.hpp"
#include "optimath/linalg/matrix.hpp"

namespace optimath::nlp {
namespace {

optimath::linalg::Matrix identity(std::size_t n) {
    optimath::linalg::Matrix I(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i) I(i, i) = 1.0;
    return I;
}

optimath::linalg::Vector matvec_local(const optimath::linalg::Matrix& A, const optimath::linalg::Vector& x) {
    return optimath::linalg::matvec(A, x);
}

optimath::linalg::Vector negate(const optimath::linalg::Vector& v) {
    return (-1.0) * v;
}

} // namespace

core::SolveResult<BFGSResult> minimize_bfgs(
    const Objective& f,
    const optimath::linalg::Vector& x0,
    const core::SolverOptions& options) {

    optimath::core::WallTimer timer;

    const double grad_tol = std::max(1e-12, options.tol.primal_feas);
    const std::size_t max_it = options.limits.max_iterations;

    const std::size_t n = x0.size();
    optimath::linalg::Vector x = x0;

    auto eval_grad = [&](const optimath::linalg::Vector& xx) {
        auto gopt = f.gradient(xxyy);
        if (gopt) return *gopt;
        return finite_difference_gradient(f, xx);
    };

    double fx = f.value(x);
    optimath::linalg::Vector gx = eval_grad(x);

    BFGSStats stats;
    stats.function_evals++;
    stats.gradient_evals++;

    optimath::linalg::Matrix H = identity(n);

    const double c1 = 1e-4;

    for (std::size_t it = 0; it < max_it; ++it) {
        stats.iterations = it;

        if (gx.norm2() <= grad_tol) {
            BFGSResult out;
            out.solution = {x, fx};
            out.stats = stats;

            core::SolveResult<BFGSResult> res;
            res.status = core::Status::Ok();
            res.solution = out;
            res.solve_time_seconds = timer.elapsed_seconds();
            res.iterations = it;
            return res;
        }

        // Search direction p = -H g
        optimath::linalg::Vector p = negate(matvec_local(H, gx));

        // Ensure descent; if not, reset.
        double gtp = gx.dot(p);
        if (gtp >= -1e-16) {
            H = identity(n);
            p = negate(gx);
            gtp = gx.dot(p);
        }

        // Backtracking Armijo line search.
        double alpha = 1.0;
        double fnew = fx;
        bool accepted = false;
        optimath::linalg::Vector xnew = x;
        for (int ls = 0; ls < 60; ++ls) {
            xnew = x + alpha * p;
            fnew = f.value(xnew);
            stats.function_evals++;
            if (fnew <= fx + c1 * alpha * gtp) {
                fx = fnew;
                accepted = true;
                break;
            }
            alpha *= 0.5;
            if (alpha < 1e-16) break;
        }

        if (!accepted) {
            BFGSResult out;
            out.solution = {x, fx};
            out.stats = stats;

            core::SolveResult<BFGSResult> res;
            res.status = core::Status::Numerical("BFGS: line search failed");
            res.solution = out;
            res.solve_time_seconds = timer.elapsed_seconds();
            res.iterations = it;
            return res;
        }

        optimath::linalg::Vector gnew = eval_grad(xnew);
        stats.gradient_evals++;

        optimath::linalg::Vector s = xnew - x;
        optimath::linalg::Vector y = gnew - gx;

        const double ys = y.dot(s);
        if (std::abs(ys) < 1e-18) {
            H = identity(n);
        } else {
            const double rho = 1.0 / ys;

            // BFGS update: H = (I - rho s y^T) H (I - rho y s^T) + rho s s^T
            // Implement with explicit loops (small/medium n).
            optimath::linalg::Matrix I = identity(n);
            optimath::linalg::Matrix A(n, n, 0.0);
            optimath::linalg::Matrix B(n, n, 0.0);
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t j = 0; j < n; ++j) {
                    A(i, j) = I(i, j) - rho * s.raw()[i] * y.raw()[j];
                    B(i, j) = I(i, j) - rho * y.raw()[i] * s.raw()[j];
                }
            }

            // tmp = A * H
            optimath::linalg::Matrix tmp(n, n, 0.0);
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t j = 0; j < n; ++j) {
                    double sum = 0.0;
                    for (std::size_t k = 0; k < n; ++k) sum += A(i, k) * H(k, j);
                    tmp(i, j) = sum;
                }
            }

            // Hnew = tmp * B
            optimath::linalg::Matrix Hnew(n, n, 0.0);
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t j = 0; j < n; ++j) {
                    double sum = 0.0;
                    for (std::size_t k = 0; k < n; ++k) sum += tmp(i, k) * B(k, j);
                    Hnew(i, j) = sum;
                }
            }

            // + rho s s^T
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t j = 0; j < n; ++j) {
                    Hnew(i, j) += rho * s.raw()[i] * s.raw()[j];
                }
            }
            H = std::move(Hnew);
        }

        x = xnew;
        gx = gnew;
    }

    BFGSResult out;
    out.solution = {x, fx};
    out.stats = stats;

    core::SolveResult<BFGSResult> res;
    res.status = core::Status::MaxIterations("BFGS: reached max iterations");
    res.solution = out;
    res.solve_time_seconds = timer.elapsed_seconds();
    res.iterations = max_it;
    return res;
}

} // namespace optimath::nlp

