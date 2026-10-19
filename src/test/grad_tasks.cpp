#include <algorithm>
#include <stdexcept>
#include <vector>

#include "gtest/gtest.h"
#include "phase_limiter/GradCalculator.h"
#include "tbb/task_arena.h"

namespace {

class SourceCacheScope {
public:
    explicit SourceCacheScope(bool enabled): previous_(phase_limiter::GradCoreSettings::GetInstance().src_cache()) {
        phase_limiter::GradCoreSettings::GetInstance().set_src_cache(enabled);
    }
    ~SourceCacheScope() { phase_limiter::GradCoreSettings::GetInstance().set_src_cache(previous_); }
private:
    bool previous_;
};

// Compare the scheduler's observable results with the sequential execution of
// the same overlapping FFT windows. The hook writes the input and clears the
// gradient, so scheduling a boundary too early can change either result.
template <class Simd>
void CheckTasks(int length, bool cache, int workers) {
    typedef typename Simd::element_type Float;
    typedef phase_limiter::impl::TaskGroup<Simd> Group;
    SourceCacheScope settings(cache);
    phase_limiter::GradCalculator<Simd> calculator(length, 44100, 22050, workers,
        "linear", 1e-6, 1, 0.5, 100, 400, 1);
    std::vector<Float> source(2 * length, Float(0.125));
    Float *channels[2] = {source.data(), source.data() + 1};
    calculator.copyWaveSrcFrom(channels, 2);
    calculator.noise = 0.1;
    calculator.gradEnabled = true;

    phase_limiter::impl::Tasks<Simd> tasks(workers);
    for (int channel = 0; channel < 2; ++channel) {
        for (int window: {256, 1024, 16384}) {
            for (int begin = 0; begin + window <= length; begin += window / 2) {
                phase_limiter::impl::Task<Simd> task = {};
                task.calculator = &calculator;
                task.channel = channel;
                task.windowLen = window;
                task.bg = begin;
                task.ed = begin + window;
                tasks.addTask(task);
            }
        }
    }
    tasks.prepare();
    bakuage::AlignedPodVector<Float> wave[2], gradient[2];
    for (int ch = 0; ch < 2; ++ch) {
        wave[ch].resize(length);
        gradient[ch].resize(length);
    }
    int generation = 0;
    const auto hook = [&](Group *group) {
        for (int i = group->bg; i < group->ed; ++i) {
            wave[group->channel][i] = Float(((i + 17 * group->channel + generation) % 97 - 48) / 64.0);
            gradient[group->channel][i] = 0;
        }
        group->output_dot_product = group->bg * 0.25;
        group->output_norm_sqr = group->ed * 0.5;
    };
    const auto input = [&](int channel, int i) { return wave[channel].data() + i; };
    const auto output = [&](int channel, int i, const Simd &value) {
        Float *dest = gradient[channel].data() + i;
        const Simd previous = simdpp::load(dest);
        simdpp::store(dest, previous + value);
    };
    tbb::task_arena arena(workers);
    for (generation = 0; generation < 3; ++generation) {
        SCOPED_TRACE(generation);
        const auto expected = tasks.process(hook, input, output, true);
        std::vector<Float> expected_gradient[2];
        for (int ch = 0; ch < 2; ++ch) {
            expected_gradient[ch].assign(gradient[ch].data(), gradient[ch].data() + length);
            std::fill(wave[ch].data(), wave[ch].data() + length, Float(-7));
            std::fill(gradient[ch].data(), gradient[ch].data() + length, Float(11));
        }
        phase_limiter::impl::TaskResult actual;
        arena.execute([&] { actual = tasks.process(hook, input, output); });
        EXPECT_EQ(expected.eval, actual.eval);
        EXPECT_EQ(expected.dot_product, actual.dot_product);
        EXPECT_EQ(expected.norm_sqr, actual.norm_sqr);
        for (int ch = 0; ch < 2; ++ch)
            EXPECT_TRUE(std::equal(expected_gradient[ch].begin(), expected_gradient[ch].end(), gradient[ch].data()));

        calculator.gradEnabled = false;
        const auto serial_eval = tasks.process(hook, input, output, true);
        arena.execute([&] { actual = tasks.process(hook, input, output); });
        EXPECT_EQ(serial_eval.eval, actual.eval);
        calculator.gradEnabled = true;
    }

    // Hook-only processing must not run any FFT input/output callbacks.
    const auto forbidden_input = [](int, int) -> Float * { throw std::runtime_error("unexpected FFT"); };
    arena.execute([&] { EXPECT_NO_THROW(tasks.process(hook, forbidden_input, output, false, true)); });

    // A failed call must finish/cancel its tasks before unwinding and leave the
    // dependency counters reusable. Exercise exceptions from hooks and FFT input.
    const auto failing_hook = [](Group *) { throw std::runtime_error("hook failed"); };
    arena.execute([&] {
        EXPECT_THROW(tasks.process(failing_hook, input, output), std::runtime_error);
        EXPECT_THROW(tasks.process(hook, forbidden_input, output), std::runtime_error);
    });
    const auto expected = tasks.process(hook, input, output, true);
    arena.execute([&] { EXPECT_EQ(expected.eval, tasks.process(hook, input, output).eval); });
}

TEST(GradTasks, ParallelMatchesSerial) {
    for (int workers: {1, 2, 4}) {
        for (bool cache: {false, true}) {
            for (int length: {256, 16384, 16640, 65792}) {
                SCOPED_TRACE(::testing::Message() << "workers=" << workers << " cache=" << cache << " length=" << length);
                CheckTasks<simdpp::float32x8>(length, cache, workers);
                CheckTasks<simdpp::float64x4>(length, cache, workers);
            }
        }
    }
}

} // namespace
