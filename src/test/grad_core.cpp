#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <type_traits>
#include <vector>
#include "phase_limiter/GradCore.h"
#include <oneapi/tbb/global_control.h>

namespace {
template<class Simd> void CheckDirections() {
    using T = typename Simd::element_type;
    using Core = phase_limiter::GradCore<Simd>;
    const bool single = std::is_same<T, float>::value;
    // Differentiate in double to avoid cancellation from subtracting float
    // objectives. Check three step sizes, requiring agreement at two of them.
    const double tolerance = single ? 1e-3 : 1e-5;
    const double steps[3] = {.001, .0003, .0001};
    for (int n : {256, 1024, 4096, 16384}) {
        for (bool cache : {false, true}) {
            phase_limiter::GradCoreSettings::GetInstance().set_src_cache(cache);
            for (int cutoff : {22050, 8000}) {
                // A source cutoff is rounded to the SIMD width, which differs
                // across precisions. Check arbitrary cutoffs with double's own
                // objective; compare float to double only at full bandwidth.
                if (single && cutoff != 22050) continue;
                phase_limiter::GradOptions options;
                options.len = n;
                options.sample_rate = 44100;
                options.max_available_freq = cutoff;
                for (double noise : {.1, .0001}) {
                    // Random pair, DC/Nyquist against silence, random against
                    // silence, and a near-silent pair, respectively.
                    for (int pattern = 0; pattern < 4; ++pattern) {
                        SCOPED_TRACE(::testing::Message() << "n=" << n << " cache=" << cache
                            << " cutoff=" << cutoff << " noise=" << noise << " pattern=" << pattern);
                        std::mt19937 rng(1729 + n);
                        std::uniform_real_distribution<double> random(-.4, .4);
                        bakuage::AlignedPodVector<T> x(n), src(n), grad(n);
                        for (int i = 0; i < n; ++i) {
                            x[i] = T(random(rng));
                            src[i] = T(random(rng));
                            if (pattern == 1) { x[i] = T(.3 + (i % 2 ? -.2 : .2)); src[i] = 0; }
                            if (pattern == 2) src[i] = 0;
                            if (pattern == 3) { x[i] *= T(1e-4); src[i] *= T(1e-4); }
                        }
                        bakuage::AlignedPodVector<double> reference_wave(n), reference_src(n);
                        for (int i = 0; i < n; ++i) reference_src[i] = double(src[i]);
                        phase_limiter::GradContext<simdpp::float64x4> reference_context;
                        phase_limiter::GradContext<Simd> context;
                        const auto eval = [&](const T *wave) {
                            return double(Core::calcEval23(options, wave, src.data(), T(noise), &context));
                        };
                        const double value = Core::calcEvalGrad23(
                            options, x.data(), src.data(), grad.data(), T(noise), &context);
                        const double plain = eval(x.data());
                        ASSERT_TRUE(std::isfinite(value));
                        EXPECT_NEAR(value, plain, (single ? 1e-6 : 1e-13) * std::max(1., std::abs(value)));
                        double norm = 0;
                        for (int i = 0; i < n; ++i) {
                            ASSERT_TRUE(std::isfinite(double(grad[i])));
                            norm += double(grad[i]) * double(grad[i]);
                        }
                        norm = std::sqrt(norm);
                        // Two random directions, DC, Nyquist, and three sample
                        // coordinates, including the window's zero endpoint.
                        for (int direction = 0; direction < 7; ++direction) {
                            std::vector<double> d(n, 0);
                            if (direction < 2) {
                                for (auto &v : d) v = random(rng);
                            } else if (direction == 2) {
                                std::fill(d.begin(), d.end(), 1.);
                            } else if (direction == 3) {
                                for (int i = 0; i < n; ++i) d[i] = i % 2 ? -1. : 1.;
                            } else {
                                d[direction == 4 ? 0 : direction == 5 ? n / 2 : n - 1] = 1;
                            }
                            double dnorm = 0;
                            for (double v : d) dnorm += v * v;
                            dnorm = std::sqrt(dnorm);
                            double analytic = 0;
                            for (int i = 0; i < n; ++i) {
                                d[i] /= dnorm;
                                analytic += double(grad[i]) * d[i];
                            }
                            int agreeing_steps = 0;
                            double best_error = 1e300, best_numeric = 0;
                            for (double base_step : steps) {
                                // Scale the displacement with the near-silent signal.
                                const double h = base_step * (pattern == 3 ? 1e-4 : 1);
                                // Differentiate the double-precision objective even when checking a
                                // float gradient; subtracting float objectives loses low bits.
                                for (int i = 0; i < n; ++i) reference_wave[i] = double(x[i]) + h * d[i];
                                const double plus = phase_limiter::GradCore<simdpp::float64x4>::calcEval23(
                                    options, reference_wave.data(), reference_src.data(), double(T(noise)), &reference_context);
                                for (int i = 0; i < n; ++i) reference_wave[i] = double(x[i]) - h * d[i];
                                const double minus = phase_limiter::GradCore<simdpp::float64x4>::calcEval23(
                                    options, reference_wave.data(), reference_src.data(), double(T(noise)), &reference_context);
                                const double numeric = (plus - minus) / (2 * h);
                                const double denominator = std::max(std::abs(analytic), std::abs(numeric))
                                    + 1e-3 * norm + 1e-30;
                                const double error = std::abs(analytic - numeric) / denominator;
                                if (std::isfinite(error) && error <= tolerance) ++agreeing_steps;
                                if (error < best_error) { best_error = error; best_numeric = numeric; }
                            }
                            if (agreeing_steps < 2) {
                                ADD_FAILURE() << "direction=" << direction << " analytic=" << analytic
                                    << " best_numeric=" << best_numeric << " best_scaled_error=" << best_error
                                    << " agreeing_steps=" << agreeing_steps << "/3 tolerance=" << tolerance;
                            }
                        }
                    }
                }
            }
        }
    }
}

// With zero source and flat frequency weighting, Parseval reduces the objective
// to sum(Hann[i]^2 * x[i]^2) / (noise + floor)^2. This oracle uses no FFT.
template<class Simd> void CheckClosedFormQuadratic() {
    if (phase_limiter::GradCoreSettings::GetInstance().erb_eval_func_weighting()) {
        GTEST_SKIP() << "Flat-weighting closed form does not apply to ERB weighting";
    }
    using T = typename Simd::element_type;
    const bool single = std::is_same<T, float>::value;
    for (int n : {256, 1024, 4096, 16384}) {
        SCOPED_TRACE(n);
        bakuage::AlignedPodVector<T> x(n), src(n), grad(n);
        for (int i = 0; i < n; ++i) { x[i] = T(.3 + (i % 2 ? -.2 : .2)); src[i] = 0; }
        phase_limiter::GradOptions options;
        options.len = n; options.sample_rate = 44100; options.max_available_freq = 22050;
        phase_limiter::GradCoreSettings::GetInstance().set_src_cache(false);
        phase_limiter::GradContext<Simd> context;
        const T noise = T(.1);
        const double value = phase_limiter::GradCore<Simd>::calcEvalGrad23(
            options, x.data(), src.data(), grad.data(), noise, &context);
        ASSERT_TRUE(std::isfinite(value));
        const double denominator = std::pow(double(noise) + double(
            phase_limiter::GradCoreSettings::GetInstance().absolute_min_noise()), 2);
        double expected_value = 0, max_error = 0, max_expected = 0;
        for (int i = 0; i < n; ++i) {
            ASSERT_TRUE(std::isfinite(double(grad[i])));
            const double window = .5 - .5 * std::cos(2 * std::acos(-1.) * i / n);
            expected_value += window * window * double(x[i]) * double(x[i]) / denominator;
            const double expected = 2 * window * window * double(x[i]) / denominator;
            max_error = std::max(max_error, std::abs(double(grad[i]) - expected));
            max_expected = std::max(max_expected, std::abs(expected));
        }
        const double tolerance = single ? 5e-6 : 1e-11;
        EXPECT_NEAR(value, expected_value, tolerance * expected_value);
        EXPECT_LE(max_error / max_expected, tolerance);
    }
}
TEST(GradCore, ClosedFormQuadraticDouble) { CheckClosedFormQuadratic<simdpp::float64x4>(); }
TEST(GradCore, ClosedFormQuadraticFloat) { CheckClosedFormQuadratic<simdpp::float32x8>(); }

TEST(GradCore, DoubleDirectionalDerivatives) { CheckDirections<simdpp::float64x4>(); }
TEST(GradCore, FloatDirectionalDerivatives) { CheckDirections<simdpp::float32x8>(); }
}

int main(int argc, char **argv) {
    bool erb = argc > 1 && std::string(argv[1]) == "--erb";
    if (erb) {
        --argc;
        for (int i = 1; i < argc; ++i) argv[i] = argv[i + 1];
        argv[argc] = nullptr;
    }
    phase_limiter::GradCoreSettings::GetInstance().set_erb_eval_func_weighting(erb);
    phase_limiter::GradCoreSettings::GetInstance().set_absolute_min_noise(1e-6f);
    ::testing::InitGoogleTest(&argc, argv);
    tbb::task_scheduler_handle scheduler{tbb::attach{}};
    const int result = RUN_ALL_TESTS();
    tbb::finalize(scheduler);
    return result;
}
