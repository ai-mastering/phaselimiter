#include "phase_limiter/cmaes_optimizer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <libcmaes/cmaes.h>

namespace phase_limiter {

CmaesResult OptimizeCmaes(
    const std::function<double(const std::vector<double>&)> &objective,
    const std::vector<double> &initial, double initial_value,
    const std::vector<double> &lower, const std::vector<double> &upper,
    int max_evaluations, std::uint64_t seed) {
    if (initial.empty() || initial.size() != lower.size() ||
        initial.size() != upper.size() || max_evaluations < 0 ||
        !std::isfinite(initial_value)) {
        throw std::invalid_argument("Invalid CMA-ES dimensions, budget, or initial value");
    }
    const int dimension = static_cast<int>(initial.size());
    std::vector<double> normalized(dimension), lb(dimension, 0), ub(dimension, 1);
    for (int i = 0; i < dimension; ++i) {
        if (!std::isfinite(lower[i]) || !std::isfinite(upper[i]) ||
            !std::isfinite(initial[i]) || !(lower[i] < upper[i]) ||
            initial[i] < lower[i] || initial[i] > upper[i]) {
            throw std::invalid_argument("Invalid CMA-ES bounds or initial point");
        }
        normalized[i] = (initial[i] - lower[i]) / (upper[i] - lower[i]);
    }
    CmaesResult best{initial, initial_value};
    if (max_evaluations == 0) return best;

    using Transform = libcmaes::GenoPheno<libcmaes::pwqBoundStrategy>;
    Transform transform(lb.data(), ub.data(), dimension);
    libcmaes::CMAParameters<Transform> parameters(
        normalized, 0.2, -1, seed == 0 ? 1 : seed, transform);
    // libcmaes treats a constructor seed of zero as time-based. Set it after
    // construction so that the CLI's zero seed is reproducible too.
    if (seed == 0) parameters.set_seed(0);
    parameters.set_algo(aCMAES);
    parameters.set_quiet(true);
    parameters.set_mt_feval(false);
    parameters.set_max_fevals(max_evaluations);

    struct BudgetExhausted {};
    int evaluations = 0;
    libcmaes::FitFunc evaluate = [&](const double *x, const int n) {
        // libcmaes checks its budget between generations; enforce the exact
        // objective-call limit even when the final generation is incomplete.
        if (evaluations == max_evaluations) throw BudgetExhausted{};
        ++evaluations;
        std::vector<double> params(n);
        for (int i = 0; i < n; ++i) {
            params[i] = lower[i] + std::max(0.0, std::min(1.0, x[i])) *
                (upper[i] - lower[i]);
        }
        const double value = objective(params);
        if (std::isfinite(value) && value < best.value) best = {params, value};
        return std::isfinite(value) ? value : std::numeric_limits<double>::max();
    };
    try {
        libcmaes::cmaes<Transform>(evaluate, parameters);
    } catch (const BudgetExhausted &) {
        // The best evaluated point remains valid, including a partial generation.
    }
    return best;
}

}
