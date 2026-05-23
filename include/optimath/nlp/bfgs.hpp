#pragma once

#include <cstddef>

#include "optimath/core/options.hpp"
#include "optimath/core/result.hpp"
#include "optimath/nlp/objective.hpp"

namespace optimath::nlp {

struct BFGSStats {
    std::size_t iterations{0};
    std::size_t function_evals{0};
    std::size_t gradient_evals{0};
};

struct BFGSResult {
    NLPSolution solution{};
    BFGSStats stats{};
};

core::SolveResult<BFGSResult> minimize_bfgs(
    const Objective& f,
    const optimath::linalg::Vector& x0,
    const core::SolverOptions& options = {});

} // namespace optimath::nlp
