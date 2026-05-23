#include "test_common.hpp"

#include <cmath>

#include "optimath/core/options.hpp"
#include "optimath/linalg/vector.hpp"
#include "optimath/nlp/bfgs.hpp"
#include "optimath/nlp/objective.hpp"
#include "optimath/nlp/penalty.hpp"

void test_nlp_bfgs() {
    using optimath::linalg::Vector;
    using optimath::nlp::LambdaObjective;

    // Rosenbrock: global min at (1,1)
    const auto f = LambdaObjective(
        [](const Vector& x) {
            const double a = 1.0 - x.raw()[0];
            const double b = x.raw()[1] - x.raw()[0] * x.raw()[0];
            return a * a + 100.0 * b * b;
        },
        LambdaObjective::GradFn([](const Vector& x) {
            Vector g(2, 0.0);
            const double x0 = x.raw()[0];
            const double x1 = x.raw()[1];
            g.raw()[0] = -2.0 * (1.0 - x0) - 400.0 * x0 * (x1 - x0 * x0);
            g.raw()[1] = 200.0 * (x1 - x0 * x0);
            return g;
        })
    );

    optimath::core::SolverOptions opt;
    opt.limits.max_iterations = 20000;
    opt.tol.primal_feas = 1e-10;

    auto res = optimath::nlp::minimize_bfgs(f, Vector({-1.2, 1.0}), opt);
    EXPECT_TRUE(res.status.ok());
    EXPECT_NEAR(res.solution.solution.objective_value, 0.0, 1e-6);
    EXPECT_NEAR(res.solution.solution.x.raw()[0], 1.0, 1e-4);
    EXPECT_NEAR(res.solution.solution.x.raw()[1], 1.0, 1e-4);

    // Constrained quadratic via penalty:
    // minimize (x-1)^2 + (y-2)^2 subject to x + y = 3 and x >= 0.
    const auto q = LambdaObjective(
        [](const Vector& x) {
            const double dx = x.raw()[0] - 1.0;
            const double dy = x.raw()[1] - 2.0;
            return dx * dx + dy * dy;
        },
        LambdaObjective::GradFn([](const Vector& x) {
            Vector g(2, 0.0);
            g.raw()[0] = 2.0 * (x.raw()[0] - 1.0);
            g.raw()[1] = 2.0 * (x.raw()[1] - 2.0);
            return g;
        })
    );

    std::vector<optimath::nlp::InequalityConstraint> ineq;
    ineq.push_back({
        .value = [](const Vector& x) { return -x.raw()[0]; }, // -x <= 0 => x >= 0
        .grad  = [](const Vector&) { return Vector({-1.0, 0.0}); }
    });

    std::vector<optimath::nlp::EqualityConstraint> eq;
    eq.push_back({
        .value = [](const Vector& x) { return x.raw()[0] + x.raw()[1] - 3.0; },
        .grad  = [](const Vector&) { return Vector({1.0, 1.0}); }
    });

    auto res2 = optimath::nlp::minimize_with_penalty(q, Vector({0.5, 0.5}), ineq, eq, opt);
    EXPECT_TRUE(res2.status.ok());
    EXPECT_NEAR(res2.solution.solution.x.raw()[0], 1.0, 1e-3);
    EXPECT_NEAR(res2.solution.solution.x.raw()[1], 2.0, 1e-3);
}
