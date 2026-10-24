#include <oneapi/tbb/global_control.h>
#include <oneapi/tbb/info.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <vector>
#include <gtest/gtest.h>
#include <optim.hpp>

namespace {
struct SearchResult {
    std::vector<std::vector<double>> evaluations;
    std::vector<double> solution;
    double value;
};

SearchResult Search(int workers, std::uint64_t seed, bool bounded = false) {
    tbb::global_control tbb_limit(tbb::global_control::max_allowed_parallelism, workers);
    SearchResult result;
    std::mutex mutex;
    arma::vec x(4, arma::fill::zeros);
    optim::algo_settings_t settings;
    settings.de_rng_seed = seed;
    // Exercise all three population reductions (64 -> 32 -> 16 -> 8).
    settings.de_n_pop = 64;
    settings.de_max_fn_eval = 2048;
    settings.de_initial_lb = arma::vec(4, arma::fill::ones) * -2;
    settings.de_initial_ub = arma::vec(4, arma::fill::ones) * 2;
    settings.vals_bound = bounded;
    settings.lower_bounds = settings.de_initial_lb;
    settings.upper_bounds = settings.de_initial_ub;
    const auto objective = [&](const arma::vec& v, arma::vec*, void*) {
        double value = 0;
        for (size_t k = 0; k < v.n_elem; ++k) {
            const double d = v(k) - 0.25;
            value += d * d + 0.05 * (1 - std::cos(8 * d));
        }
        std::lock_guard<std::mutex> lock(mutex);
        result.evaluations.emplace_back(v.begin(), v.end());
        return value;
    };
    EXPECT_TRUE(optim::de_prmm(x, objective, nullptr, settings));
    result.solution.assign(x.begin(), x.end());
    result.value = objective(x, nullptr, nullptr);
    // Callback arrival order is allowed to vary; the candidates must not.
    std::sort(result.evaluations.begin(), result.evaluations.end());
    return result;
}
}

TEST(OptimRng, SameSearchAcrossWorkerCountsAndRepeatedCalls) {
    for (std::uint64_t seed : {1ULL, 0x100000001ULL}) {
        const auto reference = Search(1, seed);
        ASSERT_GT(reference.evaluations.size(), 1000u);
        EXPECT_LT(reference.value, 0.1);
        for (int workers : {2, 8, 32, 1}) {
            const auto actual = Search(workers, seed);
            EXPECT_EQ(reference.evaluations, actual.evaluations);
            EXPECT_EQ(reference.solution, actual.solution);
            EXPECT_EQ(reference.value, actual.value);
        }
    }
}

TEST(OptimRng, SeedChangesSearchAndBoundedSearchIsReproducible) {
    const auto first = Search(1, 1, true);
    const auto second = Search(8, 1, true);
    EXPECT_EQ(first.evaluations, second.evaluations);
    EXPECT_EQ(first.solution, second.solution);
    EXPECT_NE(first.evaluations, Search(8, 0x100000001ULL, true).evaluations);
    for (const auto& candidate : first.evaluations) {
        for (double v : candidate) {
            EXPECT_TRUE(std::isfinite(v));
            EXPECT_GE(v, -2.0);
            EXPECT_LE(v, 2.0);
        }
    }
}
