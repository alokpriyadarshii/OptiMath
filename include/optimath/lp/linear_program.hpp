#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace optimath::lp {

enum class ConstraintSense {
    kLessEqual,
    kGreaterEqual,
    kEqual
};

struct LinearConstraint {
    std::vector<double> a;  // coefficients
    double b{0.0};          // rhs
    ConstraintSense sense{ConstraintSense::kLessEqual};
    std::string name{};
};

class LinearProgram {
public:
    explicit LinearProgram(std::size_t num_vars = 0);

    std::size_t num_vars() const noexcept { return n_; }
    std::size_t num_constraints() const noexcept { return constraints_.size(); }

    // Maximize c^T x.
    void set_objective(const std::vector<double>& c);

    // Adds a constraint a^T x (<=,>=,=) b.
    void add_constraint(const std::vector<double>& a, double b, ConstraintSense s = ConstraintSense::kLessEqual, std::string name = {});

    const std::vector<double>& objective() const { return c_; }
    const std::vector<LinearConstraint>& constraints() const { return constraints_; }

private:
    std::size_t n_{0};
    std::vector<double> c_;
    std::vector<LinearConstraint> constraints_;
};

struct LPSolution {
    std::vector<double> x;
    double objective_value{0.0};
};

} // namespace optimath::lp
