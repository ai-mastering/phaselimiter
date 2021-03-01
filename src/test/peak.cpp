#include "gtest/gtest.h"
#include "bakuage/memory.h"
#include "audio_analyzer/peak.h"

TEST(AudioAnalyzerPeak, Impulse) {
    const int channels = 1;
    const int samples = 44100;
    const int oversample = 4;
    bakuage::AlignedPodVector<float> input(channels * samples);
    input[channels * (samples / 2)] = 1;
    
    float peak, rms, true_peak;
    audio_analyzer::CalculatePeakAndRMS(input.data(), channels, samples, &peak, &rms, oversample, &true_peak);
    
    EXPECT_NEAR(0, peak, 1e-7);
    EXPECT_NEAR(10 * std::log10(1.0 / (channels * samples)), rms, 1e-6);
    EXPECT_NEAR(0, true_peak, 1e-7);
}

TEST(AudioAnalyzerPeak, LowpassTruePeak) {
    const int channels = 1;
    const int samples = 44100;
    const int oversample = 4;
    bakuage::AlignedPodVector<float> input(channels * samples);
    input[channels * (samples / 2)] = 1;
    
    float true_peak;
    audio_analyzer::CalculateLowpassTruePeak<float>(input.data(), channels, samples, 44100, 15000, oversample, &true_peak);
    
    EXPECT_NEAR(20 * std::log10(15000.0 / 22050), true_peak, 1e-6);
}
