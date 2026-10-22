#include <cmath>
#include <algorithm>
#include <condition_variable>
#include <future>
#include <limits>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <omp.h>
#include <gtest/gtest.h>
#include "phase_limiter/cmaes_optimizer.h"

namespace {

using phase_limiter::OptimizeCmaes;

using SearchTrace = std::vector<std::vector<double>>;

SearchTrace TraceSearch(std::uint64_t seed, const std::function<void()> &on_first = {}) {
    SearchTrace points;
    OptimizeCmaes([&](const std::vector<double> &x) {
        if (points.empty() && on_first) on_first();
        points.push_back(x);
        return x[0] * x[0] + x[1] * x[1];
    }, {0, 0}, 0, {-1, -1}, {1, 1}, 101, seed);
    return points;
}

TEST(CmaesOptimizer, NestedSearchDoesNotChangeOuterRandomSequence) {
    const auto expected = TraceSearch(1);
    const auto actual = TraceSearch(1, [] { TraceSearch(42); });
    EXPECT_EQ(expected, actual);
}

TEST(CmaesOptimizer, ConcurrentSearchesMatchIndependentRuns) {
    const auto expected_zero = TraceSearch(0);
    const auto expected_other = TraceSearch(42);
    std::mutex mutex;
    std::condition_variable ready;
    int arrived = 0;
    const auto overlap = [&] {
        std::unique_lock<std::mutex> lock(mutex);
        ++arrived;
        ready.notify_all();
        ready.wait(lock, [&] { return arrived == 2; });
    };
    auto zero = std::async(std::launch::async, [&] { return TraceSearch(0, overlap); });
    auto other = std::async(std::launch::async, [&] { return TraceSearch(42, overlap); });
    EXPECT_EQ(expected_zero, zero.get());
    EXPECT_EQ(expected_other, other.get());
}

TEST(CmaesOptimizer, SameCandidatesAndResultAcrossWorkerCounts) {
    struct Run {
        SearchTrace points;
        phase_limiter::CmaesResult result;
    };
    const auto search = [](int workers, int budget, std::uint64_t seed, bool ties) {
        Run run;
        std::mutex mutex;
        run.result = OptimizeCmaes([&](const std::vector<double> &x) {
            {
                std::lock_guard<std::mutex> lock(mutex);
                run.points.push_back(x);
            }
            return ties ? -1.0 : std::pow(x[0] - 0.3, 2) + std::pow(x[1] + 0.2, 2);
        }, {0, 0}, 0.13, {-1, -1}, {1, 1}, budget, seed, workers);
        // Only callback arrival order may differ.
        std::sort(run.points.begin(), run.points.end());
        return run;
    };
    for (bool ties : {false, true}) {
        for (auto seed : {0ULL, 1ULL, 0x100000001ULL}) {
            for (int budget : {0, 1, 9, 17, 101}) {
                const auto reference = search(1, budget, seed, ties);
                EXPECT_LE(reference.points.size(), static_cast<size_t>(budget));
                for (int workers : {2, 4, 32}) {
                    const auto actual = search(workers, budget, seed, ties);
                    EXPECT_EQ(reference.points, actual.points);
                    EXPECT_EQ(reference.result.params, actual.result.params);
                    EXPECT_EQ(reference.result.value, actual.result.value);
                }
            }
        }
    }
}

TEST(CmaesOptimizer, UsesOpenMpWorkersAndRestoresCallerSettings) {
    const int original_threads = omp_get_max_threads();
    const int original_dynamic = omp_get_dynamic();
    std::mutex mutex;
    std::set<std::thread::id> threads;
    OptimizeCmaes([&](const std::vector<double> &) {
        std::lock_guard<std::mutex> lock(mutex);
        threads.insert(std::this_thread::get_id());
        return 1.0;
    }, {0, 0}, 0, {-1, -1}, {1, 1}, 17, 1, 4);
    EXPECT_GT(threads.size(), 1u);
    EXPECT_LE(threads.size(), 4u);
    EXPECT_EQ(omp_get_max_threads(), original_threads);
    EXPECT_EQ(omp_get_dynamic(), original_dynamic);

    EXPECT_THROW(OptimizeCmaes([](const std::vector<double> &) -> double {
        throw std::runtime_error("parallel objective failed");
    }, {0, 0}, 0, {-1, -1}, {1, 1}, 17, 1, 4), std::runtime_error);
    EXPECT_EQ(omp_get_max_threads(), original_threads);
    EXPECT_EQ(omp_get_dynamic(), original_dynamic);
}

TEST(CmaesOptimizer, ImprovesWithinUnequalBounds) {
    int calls = 0;
    const auto objective = [&](const std::vector<double> &x) {
        ++calls;
        EXPECT_GE(x[0], -100.0);
        EXPECT_LE(x[0], 100.0);
        EXPECT_GE(x[1], -0.01);
        EXPECT_LE(x[1], 0.01);
        return std::pow((x[0] - 30) / 100, 2) +
            std::pow((x[1] - 0.003) / 0.01, 2);
    };
    // Start at the boundary, as several mastering5 parameters do in practice.
    const auto result = OptimizeCmaes(objective, {-100, -0.01}, 3.38,
                                     {-100, -0.01}, {100, 0.01}, 1000, 42);
    EXPECT_LE(calls, 1000);
    EXPECT_LT(result.value, 1e-6);
    EXPECT_NEAR(result.params[0], 30, 0.1);
    EXPECT_NEAR(result.params[1], 0.003, 1e-5);
}

TEST(CmaesOptimizer, StrictSmallBudgetsAndBaselineFallback) {
    for (int budget : {0, 1, 2, 9, 17}) {
        int calls = 0;
        const auto result = OptimizeCmaes(
            [&](const std::vector<double> &) { ++calls; return 1.0; },
            {0, 0}, 0, {-1, -1}, {1, 1}, budget, 1);
        EXPECT_LE(calls, budget);
        if (budget > 0) { EXPECT_GT(calls, 0); }
        EXPECT_EQ(result.params, (std::vector<double>{0, 0}));
        EXPECT_EQ(result.value, 0);
    }
}

TEST(CmaesOptimizer, ReproducibleSeedsIncludingZero) {
    const auto trace = [](std::uint64_t seed) {
        std::vector<std::vector<double>> points;
        OptimizeCmaes([&](const std::vector<double> &x) {
            points.push_back(x);
            return x[0] * x[0] + x[1] * x[1];
        }, {0, 0}, 0, {-1, -1}, {1, 1}, 17, seed);
        EXPECT_EQ(points.size(), 17u);
        return points;
    };
    EXPECT_EQ(trace(1), trace(1));
    EXPECT_EQ(trace(0), trace(0));
    EXPECT_NE(trace(1), trace(2));
}

TEST(CmaesOptimizer, RetainsImprovementFromPartialGeneration) {
    int calls = 0;
    std::vector<double> evaluated;
    const auto result = OptimizeCmaes([&](const std::vector<double> &x) {
        ++calls;
        evaluated = x;
        return -1.0;
    }, {0, 0}, 0, {-1, -1}, {1, 1}, 1, 1);
    EXPECT_EQ(calls, 1);
    EXPECT_EQ(result.value, -1);
    EXPECT_EQ(result.params, evaluated);
}

TEST(CmaesOptimizer, NonfiniteCandidatesCannotReplaceBaseline) {
    const auto result = OptimizeCmaes(
        [](const std::vector<double> &) { return std::numeric_limits<double>::quiet_NaN(); },
        {0, 0}, 0, {-1, -1}, {1, 1}, 10, 1);
    EXPECT_EQ(result.value, 0);
    EXPECT_EQ(result.params, (std::vector<double>{0, 0}));
}

TEST(CmaesOptimizer, RejectsInvalidArgumentsAndPropagatesObjectiveErrors) {
    const auto objective = [](const std::vector<double> &) { return 0.0; };
    EXPECT_THROW(OptimizeCmaes(objective, {0}, 0, {}, {1}, 1, 1), std::invalid_argument);
    EXPECT_THROW(OptimizeCmaes(objective, {0}, 0, {1}, {1}, 1, 1), std::invalid_argument);
    EXPECT_THROW(OptimizeCmaes(objective, {2}, 0, {-1}, {1}, 1, 1), std::invalid_argument);
    EXPECT_THROW(OptimizeCmaes(objective, {0}, 0, {-1}, {1}, -1, 1), std::invalid_argument);
    EXPECT_THROW(OptimizeCmaes([](const std::vector<double> &) -> double {
        throw std::runtime_error("objective failed");
    }, {0, 0}, 0, {-1, -1}, {1, 1}, 10, 1), std::runtime_error);
}

}
