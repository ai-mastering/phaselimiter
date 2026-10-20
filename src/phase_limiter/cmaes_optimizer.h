#pragma once

#include <cstdint>
#include <functional>
#include <vector>

namespace phase_limiter {

struct CmaesResult {
    std::vector<double> params;
    double value;
};

// Minimize inside the supplied bounds using active CMA-ES. The initial value
// is already evaluated; max_evaluations counts only new objective calls.
// Evaluations are sequential. The returned result is never worse than initial.
// Independent calls own their random state and may run concurrently or nested.
CmaesResult OptimizeCmaes(
    const std::function<double(const std::vector<double>&)> &objective,
    const std::vector<double> &initial, double initial_value,
    const std::vector<double> &lower, const std::vector<double> &upper,
    int max_evaluations, std::uint64_t seed);

}
