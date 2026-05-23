#pragma once

#include "optimath/linalg/matrix.hpp"

namespace optimath::linalg {

// Solves A x = b with Gaussian elimination + partial pivoting.
// Throws std::invalid_argument on size mismatch.
// Throws std::runtime_error on singular matrices.
Vector solve_linear_system(Matrix A, Vector b);

} // namespace optimath::linalg
