#include <cmath>
#include <condition_variable>
#include <future>
#include <limits>
#include <mutex>
#include <stdexcept>
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
        if (budget > 0) EXPECT_GT(calls, 0);
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
