#pragma once

#include <stdexcept>
#include <vector>

#include "optimath/linalg/vector.hpp"

namespace optimath::linalg {

class Matrix {
public:
    Matrix() = default;
    Matrix(std::size_t r, std::size_t c, double v = 0.0) : r_(r), c_(c), data_(r * c, v) {}

    std::size_t rows() const noexcept { return r_; }
    std::size_t cols() const noexcept { return c_; }

    double& operator()(std::size_t r, std::size_t c) {
        return data_.at(r * c_ + c);
    }
    const double& operator()(std::size_t r, std::size_t c) const {
        return data_.at(r * c_ + c);
    }

    Vector row(std::size_t r) const {
        Vector v(c_);
        for (std::size_t j = 0; j < c_; ++j) v.raw()[j] = (*this)(r, j);
        return v;
    }

private:
    std::size_t r_{0};
    std::size_t c_{0};
    std::vector<double> data_;
};

Vector matvec(const Matrix& A, const Vector& x);
Matrix transpose(const Matrix& A);

} // namespace optimath::linalg
