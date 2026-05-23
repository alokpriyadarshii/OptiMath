#pragma once

#include <cstddef>

namespace optimath::core {

struct SolverLimits {
    std::size_t max_iterations{100000};
    std::size_t max_nodes{10000}; // for branch-and-bound
    double time_limit_seconds{0.0}; // 0 => no limit
};

struct Tolerances {
    double primal_feas{1e-9};
    double dual_feas{1e-9};
    double pivot{1e-12};
    double objective{1e-9};
};

struct SolverOptions {
    SolverLimits limits{};
    Tolerances tol{};
    bool verbose{false};
};

} // namespace optimath::core
