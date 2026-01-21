#include "optimath/lp/linear_program.hpp"

#include <stdexcept>

namespace optimath::lp {

LinearProgram::LinearProgram(std::size_t num_vars) : n_(num_vars), c_(num_vars, 0.0) {}

void LinearProgram::set_objective(const std::vector<double>& c) {
    if (c.size() != n_) throw std::invalid_argument("objective size mismatch");
    c_ = c;
}

void LinearProgram::add_constraint(const std::vector<double>& a, double b, ConstraintSense s, std::string name) {
    if (a.size() != n_) throw std::invalid_argument("constraint coefficient size mismatch");
    constraints_.push_back(LinearConstraint{a, b, s, std::move(name)});
}

} // namespace optimath::lp
