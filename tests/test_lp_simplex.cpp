#include "test_common.hpp"

#include <vector>

#include "optimath/core/status.hpp"
#include "optimath/lp/linear_program.hpp"
#include "optimath/lp/mip_branch_and_bound.hpp"
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

    // 0/1 knapsack must enforce both integrality and binary upper bounds.
    // Without x_i <= 1 bounds, integer branching can choose multiple copies
    // of the same item and return an invalid "0/1" solution.
    const std::vector<double> value = {6.0, 10.0, 12.0, 7.0};
    const std::vector<double> weight = {2.0, 4.0, 6.0, 3.0};
    optimath::lp::LinearProgram knapsack(value.size());
    knapsack.set_objective(value);
    knapsack.add_constraint(weight, 9.0, ConstraintSense::kLessEqual, "capacity");
    for (std::size_t i = 0; i < value.size(); ++i) {
        std::vector<double> upper_bound(value.size(), 0.0);
        upper_bound[i] = 1.0;
        knapsack.add_constraint(upper_bound, 1.0, ConstraintSense::kLessEqual, "binary_ub");
    }

    optimath::lp::MIPModel mip;
    mip.relaxation = knapsack;
    mip.integer_vars = {0, 1, 2, 3};

    auto mip_res = optimath::lp::solve_branch_and_bound(mip, opt);
    EXPECT_TRUE(mip_res.status.ok());
    EXPECT_TRUE(mip_res.solution.has_feasible_solution);
    EXPECT_NEAR(mip_res.solution.solution.objective_value, 23.0, 1e-7);
    EXPECT_NEAR(mip_res.solution.solution.x[0], 1.0, 1e-7);
    EXPECT_NEAR(mip_res.solution.solution.x[1], 1.0, 1e-7);
    EXPECT_NEAR(mip_res.solution.solution.x[2], 0.0, 1e-7);
    EXPECT_NEAR(mip_res.solution.solution.x[3], 1.0, 1e-7);

    // If branch-and-bound stops because of a node limit, it should report
    // a limit status rather than incorrectly claiming infeasibility.
    optimath::lp::LinearProgram limited_lp(1);
    limited_lp.set_objective({1.0});
    limited_lp.add_constraint({2.0}, 1.0, ConstraintSense::kLessEqual, "fractional_bound");

    optimath::lp::MIPModel limited_mip;
    limited_mip.relaxation = limited_lp;
    limited_mip.integer_vars = {0};

    optimath::core::SolverOptions limited_opt = opt;
    limited_opt.limits.max_nodes = 1;

    auto limited_res = optimath::lp::solve_branch_and_bound(limited_mip, limited_opt);
    EXPECT_TRUE(limited_res.status.code == optimath::core::StatusCode::kMaxIterations);

    optimath::lp::MIPModel invalid_mip;
    invalid_mip.relaxation = limited_lp;
    invalid_mip.integer_vars = {1};

    auto invalid_res = optimath::lp::solve_branch_and_bound(invalid_mip, opt);
    EXPECT_TRUE(invalid_res.status.code == optimath::core::StatusCode::kInvalidArgument);
}
