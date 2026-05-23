#include "optimath/linalg/solve.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace optimath::linalg {

Vector solve_linear_system(Matrix A, Vector b) {
    const std::size_t n = A.rows();
    if (A.cols() != n) throw std::invalid_argument("solve_linear_system: A must be square");
    if (b.size() != n) throw std::invalid_argument("solve_linear_system: b size mismatch");

    // Forward elimination with partial pivoting.
    for (std::size_t k = 0; k < n; ++k) {
        // Pivot row
        std::size_t piv = k;
        double best = std::abs(A(k, k));
        for (std::size_t i = k + 1; i < n; ++i) {
            double v = std::abs(A(i, k));
            if (v > best) { best = v; piv = i; }
        }
        if (best < 1e-15) throw std::runtime_error("solve_linear_system: singular matrix");
        if (piv != k) {
            for (std::size_t j = k; j < n; ++j) std::swap(A(k, j), A(piv, j));
            std::swap(b.raw()[k], b.raw()[piv]);
        }

        const double diag = A(k, k);
        for (std::size_t i = k + 1; i < n; ++i) {
            const double factor = A(i, k) / diag;
            A(i, k) = 0.0;
            for (std::size_t j = k + 1; j < n; ++j) {
                A(i, j) -= factor * A(k, j);
            }
            b.raw()[i] -= factor * b.raw()[k];
        }
    }

    // Back substitution.
    Vector x(n, 0.0);
    for (std::size_t i = n; i-- > 0;) {
        double s = b.raw()[i];
        for (std::size_t j = i + 1; j < n; ++j) {
            s -= A(i, j) * x.raw()[j];
        }
        const double diag = A(i, i);
        if (std::abs(diag) < 1e-15) throw std::runtime_error("solve_linear_system: singular matrix");
        x.raw()[i] = s / diag;
    }

    return x;
}

} // namespace optimath::linalg
