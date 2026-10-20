#include "phase_limiter/cmaes_optimizer.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>
#include <libcmaes/cmaes.h>
#include <omp.h>

namespace phase_limiter {
namespace {
// OpenMP settings belong to the calling task. Restore them on normal return,
// budget exhaustion, and exceptions, including nested independent searches.
struct OpenMpSettings {
    int threads = omp_get_max_threads();
    int dynamic = omp_get_dynamic();
    explicit OpenMpSettings(int workers) {
        omp_set_dynamic(0);
        omp_set_num_threads(workers);
    }
    ~OpenMpSettings() {
        omp_set_num_threads(threads);
        omp_set_dynamic(dynamic);
    }
};
}

CmaesResult OptimizeCmaes(
    const std::function<double(const std::vector<double>&)> &objective,
    const std::vector<double> &initial, double initial_value,
    const std::vector<double> &lower, const std::vector<double> &upper,
    int max_evaluations, std::uint64_t seed, int worker_count) {
    if (initial.empty() || initial.size() != lower.size() ||
        initial.size() != upper.size() || max_evaluations < 0 ||
        !std::isfinite(initial_value) || worker_count < 1) {
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
    parameters.set_mt_feval(worker_count > 1);
    parameters.set_max_fevals(max_evaluations);
    OpenMpSettings openmp(std::min(worker_count, parameters.lambda()));

    struct BudgetExhausted {};
    int evaluations = 0;
    const double *batch_data = nullptr;
    std::vector<std::vector<double>> evaluated;
    std::vector<std::exception_ptr> errors;
    libcmaes::FitFunc evaluate = [&](const double *x, const int n) {
        // The pinned ESOStrategy::eval passes each phenotype column directly.
        // Each OpenMP iteration exclusively owns its candidate's result slot.
        const auto index = (x - batch_data) / n;
        try {
            auto &params = evaluated[index];
            params.resize(n);
            for (int i = 0; i < n; ++i) {
                params[i] = lower[i] + std::max(0.0, std::min(1.0, x[i])) *
                    (upper[i] - lower[i]);
            }
            const double value = objective(params);
            return std::isfinite(value) ? value : std::numeric_limits<double>::max();
        } catch (...) {
            // Exceptions must be caught on the same thread inside the OpenMP
            // region. Rethrow after all workers have joined.
            errors[index] = std::current_exception();
            return std::numeric_limits<double>::max();
        }
    };
    libcmaes::CMAStrategy<libcmaes::ACovarianceUpdate, Transform> strategy(evaluate, parameters);
    const auto evaluate_batch = [&](const dMat &candidates, const dMat &phenotypes) {
        const int count = std::min<int>(candidates.cols(), max_evaluations - evaluations);
        if (count == 0) throw BudgetExhausted{};
        // Select the prefix before starting workers. A partial final generation
        // must evaluate the same candidates for every thread count.
        const dMat selected = candidates.leftCols(count);
        const dMat selected_phenotypes = phenotypes.leftCols(count);
        batch_data = selected_phenotypes.data();
        evaluated.assign(count, {});
        errors.assign(count, nullptr);
        strategy.eval(selected, selected_phenotypes); // libcmaes' native OpenMP loop
        for (int i = 0; i < count; ++i) {
            if (errors[i]) std::rethrow_exception(errors[i]);
            const double value = strategy.get_solutions().get_candidate(i).get_fvalue();
            if (value < best.value) best = {evaluated[i], value};
        }
        evaluations += count;
        // Never update covariance with an incomplete generation. Retain its
        // best evaluated point and exit outside the OpenMP region.
        if (count < candidates.cols()) throw BudgetExhausted{};
    };
    try {
        strategy.optimize(evaluate_batch, [&] { return strategy.ask(); },
                          [&] { strategy.tell(); });
    } catch (const BudgetExhausted &) {
        // The best evaluated point remains valid, including a partial generation.
    }
    return best;
}

}
