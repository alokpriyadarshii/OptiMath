#pragma once

#include <cstddef>

#include "optimath/core/options.hpp"
#include "optimath/core/result.hpp"
#include "optimath/lp/linear_program.hpp"

namespace optimath::lp {

struct SimplexStats {
    std::size_t pivots{0};
};

struct SimplexResult {
    LPSolution solution{};
    SimplexStats stats{};
};

core::SolveResult<SimplexResult> solve_simplex(const LinearProgram& lp, const core::SolverOptions& options = {});

} // namespace optimath::lp
