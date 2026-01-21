#include "optimath/linalg/vector.hpp"

#include <stdexcept>

namespace optimath::linalg {

Vector operator+(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) throw std::invalid_argument("Vector + size mismatch");
    Vector r(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) r.raw()[i] = a.raw()[i] + b.raw()[i];
    return r;
}

Vector operator-(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) throw std::invalid_argument("Vector - size mismatch");
    Vector r(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) r.raw()[i] = a.raw()[i] - b.raw()[i];
    return r;
}

Vector operator*(double s, const Vector& a) {
    Vector r(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) r.raw()[i] = s * a.raw()[i];
    return r;
}

Vector operator*(const Vector& a, double s) { return s * a; }

} // namespace optimath::linalg
