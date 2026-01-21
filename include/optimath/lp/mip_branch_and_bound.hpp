#pragma once

#include <cstddef>
#include <vector>

#include "optimath/core/options.hpp"
#include "optimath/core/result.hpp"
#include "optimath/lp/linear_program.hpp"

namespace optimath::lp {

struct MIPModel {
    LinearProgram relaxation;
    std::vector<std::size_t> integer_vars; // indices of vars that must be integral
};

struct MIPStats {
    std::size_t nodes_explored{0};
    std::size_t lp_solves{0};
};

struct MIPResult {
    LPSolution solution{};
    MIPStats stats{};
    bool has_feasible_solution{false};
};

core::SolveResult<MIPResult> solve_branch_and_bound(const MIPModel& mip, const core::SolverOptions& options = {});

} // namespace optimath::lp
