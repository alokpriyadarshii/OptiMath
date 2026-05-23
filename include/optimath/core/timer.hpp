#pragma once

#include <chrono>

namespace optimath::core {

class WallTimer {
public:
    using clock = std::chrono::steady_clock;

    WallTimer() : start_(clock::now()) {}

    void reset() { start_ = clock::now(); }

    double elapsed_seconds() const {
        const auto d = std::chrono::duration_cast<std::chrono::duration<double>>(clock::now() - start_);
        return d.count();
    }

private:
    clock::time_point start_;
};

} // namespace optimath::core
