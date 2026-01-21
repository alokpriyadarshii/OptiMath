#pragma once

#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <vector>

namespace optimath::linalg {

class Vector {
public:
    Vector() = default;
    explicit Vector(std::size_t n, double v = 0.0) : data_(n, v) {}
    Vector(std::initializer_list<double> il) : data_(il) {}

    std::size_t size() const noexcept { return data_.size(); }

    double& operator[](std::size_t i) { return data_.at(i); }
    const double& operator[](std::size_t i) const { return data_.at(i); }

    double* data() noexcept { return data_.data(); }
    const double* data() const noexcept { return data_.data(); }

    std::vector<double>& raw() noexcept { return data_; }
    const std::vector<double>& raw() const noexcept { return data_; }

    double dot(const Vector& other) const {
        if (size() != other.size()) throw std::invalid_argument("Vector::dot size mismatch");
        double s = 0.0;
        for (std::size_t i = 0; i < size(); ++i) s += data_[i] * other.data_[i];
        return s;
    }

    double norm2() const { return std::sqrt(dot(*this)); }

private:
    std::vector<double> data_;
};

Vector operator+(const Vector& a, const Vector& b);
Vector operator-(const Vector& a, const Vector& b);
Vector operator*(double s, const Vector& a);
Vector operator*(const Vector& a, double s);

} // namespace optimath::linalg
