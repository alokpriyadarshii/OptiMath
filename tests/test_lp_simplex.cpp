#include "test_common.hpp"

#include <vector>

#include "optimath/lp/linear_program.hpp"
#include "optimath/lp/simplex.hpp"

using optimath::lp::ConstraintSense;

void test_lp_simplex() {
    // Maximize: 3x + 2y
    // s.t.
    //   x + y <= 4
    //   x <= 2
    //   y <= 3
    //   x,y >= 0
    // Optimal: x=2, y=2, objective=10.

    optimath::lp::LinearProgram lp(2);
    lp.set_objective({3.0, 2.0});
    lp.add_constraint({1.0, 1.0}, 4.0, ConstraintSense::kLessEqual, "c1");
    lp.add_constraint({1.0, 0.0}, 2.0, ConstraintSense::kLessEqual, "c2");
    lp.add_constraint({0.0, 1.0}, 3.0, ConstraintSense::kLessEqual, "c3");

    optimath::core::SolverOptions opt;
    opt.limits.max_iterations = 10000;

    auto res = optimath::lp::solve_simplex(lp, opt);
    EXPECT_TRUE(res.status.ok());
    EXPECT_NEAR(res.solution.solution.objective_value, 10.0, 1e-7);
    EXPECT_NEAR(res.solution.solution.x[0], 2.0, 1e-7);
    EXPECT_NEAR(res.solution.solution.x[1], 2.0, 1e-7);

    // Exercise >= and = constraints to ensure Phase-I works.
    // Maximize x subject to x >= 1 and x <= 2.
    optimath::lp::LinearProgram lp2(1);
    lp2.set_objective({1.0});
    lp2.add_constraint({1.0}, 1.0, ConstraintSense::kGreaterEqual, "lb");
    lp2.add_constraint({1.0}, 2.0, ConstraintSense::kLessEqual, "ub");

    auto res2 = optimath::lp::solve_simplex(lp2, opt);
    EXPECT_TRUE(res2.status.ok());
    EXPECT_NEAR(res2.solution.solution.x[0], 2.0, 1e-7);
    EXPECT_NEAR(res2.solution.solution.objective_value, 2.0, 1e-7);
}
