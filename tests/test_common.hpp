#pragma once

#include <cmath>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>

namespace optimath::tests {

inline void fail(const std::string& msg, const char* file, int line) {
    std::ostringstream oss;
    oss << file << ":" << line << " - " << msg;
    throw std::runtime_error(oss.str());
}

inline void expect_true(bool cond, const std::string& expr, const char* file, int line) {
    if (!cond) fail("EXPECT_TRUE failed: " + expr, file, line);
}

inline void expect_near(double a, double b, double tol, const std::string& expr, const char* file, int line) {
    if (std::isnan(a) || std::isnan(b) || std::fabs(a - b) > tol) {
        std::ostringstream oss;
        oss << "EXPECT_NEAR failed: " << expr << " (" << a << " vs " << b << ", tol=" << tol << ")";
        fail(oss.str(), file, line);
    }
}

} // namespace optimath::tests

#define EXPECT_TRUE(expr) ::optimath::tests::expect_true((expr), #expr, __FILE__, __LINE__)
#define EXPECT_NEAR(a, b, tol) ::optimath::tests::expect_near((a), (b), (tol), #a ", " #b, __FILE__, __LINE__)
