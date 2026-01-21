#pragma once

#include <cstddef>

#include "optimath/core/status.hpp"

namespace optimath::core {

template <typename TSolution>
struct SolveResult {
    Status status{};
    TSolution solution{};
    double solve_time_seconds{0.0};
    std::size_t iterations{0};
};

} // namespace optimath::core
