#include "optimath/linalg/matrix.hpp"

#include <stdexcept>

namespace optimath::linalg {

Vector matvec(const Matrix& A, const Vector& x) {
    if (A.cols() != x.size()) throw std::invalid_argument("matvec size mismatch");
    Vector y(A.rows(), 0.0);
    for (std::size_t i = 0; i < A.rows(); ++i) {
        double s = 0.0;
        for (std::size_t j = 0; j < A.cols(); ++j) {
            s += A(i, j) * x.raw()[j];
        }
        y.raw()[i] = s;
    }
    return y;
}

Matrix transpose(const Matrix& A) {
    Matrix B(A.cols(), A.rows(), 0.0);
    for (std::size_t i = 0; i < A.rows(); ++i)
        for (std::size_t j = 0; j < A.cols(); ++j)
            B(j, i) = A(i, j);
    return B;
}

} // namespace optimath::linalg
