#include "optimath/nlp/objective.hpp"

#include <cmath>
#include <stdexcept>

namespace optimath::nlp {

std::optional<optimath::linalg::Vector> Objective::gradient(const optimath::linalg::Vector&) const {
    return std::nullopt;
}

optimath::linalg::Vector finite_difference_gradient(
    const Objective& f,
    const optimath::linalg::Vector& x,
    double eps) {
    const std::size_t n = x.size();
    optimath::linalg::Vector g(n, 0.0);
    optimath::linalg::Vector xpert = x;

    const double fx = f.value(x);
    for (std::size_t i = 0; i < n; ++i) {
        const double old = xpert.raw()[i];
        const double step = eps * (1.0 + std::abs(old));
        xpert.raw()[i] = old + step;
        const double f1 = f.value(xpert);
        g.raw()[i] = (f1 - fx) / step;
        xpert.raw()[i] = old;
    }
    return g;
}

LambdaObjective::LambdaObjective(ValueFn v, std::optional<GradFn> g)
    : v_(std::move(v)), g_(std::move(g)) {
    if (!v_) throw std::invalid_argument("LambdaObjective: value function is empty");
}

double LambdaObjective::value(const optimath::linalg::Vector& x) const { return v_(x); }

std::optional<optimath::linalg::Vector> LambdaObjective::gradient(const optimath::linalg::Vector& x) const {
    if (g_) return (*g_)(x);
    return std::nullopt;
}

} // namespace optimath::nlp
