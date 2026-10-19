#include "gtest/gtest.h"

#include <limits>
#include <stdexcept>
#include <vector>
#include "phase_limiter/resampling.h"

TEST(Resampling, RejectsInvalidArgumentsWithoutChangingInput) {
    const std::vector<float> original{0.25f, -0.25f, 0.5f, -0.5f};
    for (auto resample : {phase_limiter::Upsample, phase_limiter::Downsample}) {
        auto wave = original;
        for (int channels : {0, -1}) {
            EXPECT_THROW(resample(&wave, channels, 2), std::invalid_argument);
            EXPECT_THROW(resample(&wave, channels, 1), std::invalid_argument);
        }
        for (int factor : {0, -1}) {
            EXPECT_THROW(resample(&wave, 2, factor), std::invalid_argument);
        }
        EXPECT_THROW(resample(&wave, 3, 1), std::logic_error);
        EXPECT_THROW(resample(&wave, 2, std::numeric_limits<int>::max()), std::length_error);
        EXPECT_EQ(original, wave);
    }
    auto wave = original;
    EXPECT_THROW(phase_limiter::Downsample(&wave, 2, 3), std::logic_error);
    float frequency = -1;
    EXPECT_THROW(phase_limiter::CalcMaxAvailableNormalizedFreq(&wave, 0, &frequency), std::invalid_argument);
    EXPECT_THROW(phase_limiter::CalcMaxAvailableNormalizedFreq(&wave, -1, &frequency), std::invalid_argument);
    EXPECT_THROW(phase_limiter::CalcMaxAvailableNormalizedFreq(&wave, 3, &frequency), std::logic_error);
}

TEST(Resampling, EmptyInputAndIdentity) {
    std::vector<float> empty;
    float frequency = -1;
    phase_limiter::CalcMaxAvailableNormalizedFreq(&empty, 2, &frequency);
    EXPECT_EQ(0, frequency);
    phase_limiter::Upsample(&empty, 2, 2);
    phase_limiter::Downsample(&empty, 2, 2);
    EXPECT_TRUE(empty.empty());
    std::vector<float> wave{0.25f, -0.25f};
    const auto original = wave;
    phase_limiter::Upsample(&wave, 2, 1);
    phase_limiter::Downsample(&wave, 2, 1);
    EXPECT_EQ(original, wave);
}

TEST(Resampling, PreservesSilenceAndFrameCounts) {
    std::vector<float> wave(32, 0);
    phase_limiter::Upsample(&wave, 2, 2);
    EXPECT_EQ(std::vector<float>(64, 0), wave);
    phase_limiter::Downsample(&wave, 2, 2);
    EXPECT_EQ(std::vector<float>(32, 0), wave);
}
