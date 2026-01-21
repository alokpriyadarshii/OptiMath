#pragma once

#include <functional>
#include <vector>

#include "optimath/core/options.hpp"
#include "optimath/core/result.hpp"
#include "optimath/nlp/objective.hpp"

namespace optimath::nlp {

// Constraint g(x) <= 0.
struct InequalityConstraint {
    std::function<double(const optimath::linalg::Vector&)> value;
    std::function<optimath::linalg::Vector(const optimath::linalg::Vector&)> grad; // optional; empty => finite diff
};

// Constraint h(x) == 0.
struct EqualityConstraint {
    std::function<double(const optimath::linalg::Vector&)> value;
    std::function<optimath::linalg::Vector(const optimath::linalg::Vector&)> grad; // optional
};

struct PenaltyStats {
    std::size_t outer_iters{0};
    std::size_t inner_iters{0};
};

struct PenaltyResult {
    NLPSolution solution{};
    PenaltyStats stats{};
};

core::SolveResult<PenaltyResult> minimize_with_penalty(
    const Objective& f,
    const optimath::linalg::Vector& x0,
    const std::vector<InequalityConstraint>& ineq = {},
    const std::vector<EqualityConstraint>& eq = {},
    const core::SolverOptions& options = {});

} // namespace optimath::nlp
