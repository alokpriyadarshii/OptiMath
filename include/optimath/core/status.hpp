#pragma once

#include <string>

namespace optimath::core {

enum class StatusCode {
    kOk = 0,
    kInvalidArgument,
    kInfeasible,
    kUnbounded,
    kMaxIterations,
    kNumericalIssue,
    kInternalError
};

struct Status {
    StatusCode code{StatusCode::kOk};
    std::string message{};

    constexpr bool ok() const noexcept { return code == StatusCode::kOk; }

    static Status Ok() { return {}; }
    static Status Invalid(std::string msg) { return {StatusCode::kInvalidArgument, std::move(msg)}; }
    static Status Infeasible(std::string msg) { return {StatusCode::kInfeasible, std::move(msg)}; }
    static Status Unbounded(std::string msg) { return {StatusCode::kUnbounded, std::move(msg)}; }
    static Status MaxIterations(std::string msg) { return {StatusCode::kMaxIterations, std::move(msg)}; }
    static Status Numerical(std::string msg) { return {StatusCode::kNumericalIssue, std::move(msg)}; }
    static Status Internal(std::string msg) { return {StatusCode::kInternalError, std::move(msg)}; }
};

} // namespace optimath::core
