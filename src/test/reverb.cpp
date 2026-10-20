#include "gtest/gtest.h"

#include <vector>
#include <tbb/task_arena.h>
#include "audio_analyzer/reverb.h"

TEST(AudioAnalyzerReverb, ShortAndBoundaryInputs) {
    // Reference outputs from the original full-array energy summation.
    // Cover overlapping head/tail ranges and both sides of their boundaries.
    const struct {
        int samples;
        double expected;
    } cases[] = {
        {1, 370.0},
        {7, 370.0},
        {11024, -5.7660424538645705},
        {11025, -5.2030460222828667},
        {11026, -4.9249275575512597},
        {22049, -7.3790066849598634},
        {22050, -7.3826370016067191},
        {22051, -7.3952901548970518},
    };
    for (const auto &test : cases) {
        SCOPED_TRACE(test.samples);
        std::vector<double> input(test.samples);
        for (int i = 0; i < test.samples; ++i) {
            input[i] = ((i * 37) % 257 - 128) / 128.0;
        }
        double drr;
        audio_analyzer::CalculateDirectReverbEnergyRatio(
            input.data(), 1, test.samples, 44100, &drr);
        EXPECT_NEAR(test.expected, drr, 1e-5);
    }
}

TEST(AudioAnalyzerReverb, ChannelSchedulingPreservesResult) {
    for (int channels : {1, 2, 3}) {
        SCOPED_TRACE(channels);
        const int samples = 24001;
        std::vector<float> input(channels * samples);
        for (int i = 0; i < channels * samples; ++i) {
            input[i] = ((i * 37) % 257 - 128) / 128.0f;
        }
        float serial, parallel;
        tbb::task_arena serial_arena(1);
        tbb::task_arena parallel_arena(4);
        serial_arena.execute([&] {
            audio_analyzer::CalculateDirectReverbEnergyRatio(
                input.data(), channels, samples, 48000, &serial);
        });
        parallel_arena.execute([&] {
            audio_analyzer::CalculateDirectReverbEnergyRatio(
                input.data(), channels, samples, 48000, &parallel);
        });
        EXPECT_EQ(serial, parallel);
    }
}
