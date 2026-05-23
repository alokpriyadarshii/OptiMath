#pragma once

#include <functional>
#include <optional>

#include "optimath/linalg/vector.hpp"

namespace optimath::nlp {

// Minimization objective f(x).
class Objective {
public:
    virtual ~Objective() = default;
    virtual double value(const optimath::linalg::Vector& x) const = 0;

    // Optional analytic gradient; default uses finite differences.
    virtual std::optional<optimath::linalg::Vector> gradient(const optimath::linalg::Vector& x) const;
};

optimath::linalg::Vector finite_difference_gradient(
    const Objective& f,
    const optimath::linalg::Vector& x,
    double eps = 1e-8);

// Convenience wrapper for lambdas.
class LambdaObjective final : public Objective {
public:
    using ValueFn = std::function<double(const optimath::linalg::Vector&)>;
    using GradFn = std::function<optimath::linalg::Vector(const optimath::linalg::Vector&)>;

    explicit LambdaObjective(ValueFn v, std::optional<GradFn> g = std::nullopt);

    double value(const optimath::linalg::Vector& x) const override;
    std::optional<optimath::linalg::Vector> gradient(const optimath::linalg::Vector& x) const override;

private:
    ValueFn v_;
    std::optional<GradFn> g_;
};

struct NLPSolution {
    optimath::linalg::Vector x;
    double objective_value{0.0};
};

} // namespace optimath::nlp
