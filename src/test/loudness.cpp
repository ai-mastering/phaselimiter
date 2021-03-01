#include <iostream>
#include <vector>
#include <sstream>
#include <chrono>
#include <string>
#include <stdexcept>

#include "gtest/gtest.h"

#include "bakuage/utils.h"
#include "bakuage/memory.h"
#include "bakuage/loudness_ebu_r128.h"

TEST(Loudness, Test1KhzSine) {
    typedef float Float;
    using namespace bakuage;
    
    AlignedPodVector<Float> input(10 * 44100);
    for (int i = 0; i < input.size() / 2; i++) {
        const auto x = std::sin(2 * M_PI * 1000 * i / 44100);
        input[2 * i + 0] = x;
        input[2 * i + 1] = x;
    }
    
    Float loudness = 1e37;
    std::vector<int> histo;
    loudness_ebu_r128::CalculateLoudness(input.data(), 2, input.size() / 2, 44100, &loudness, &histo);
    
    EXPECT_NEAR(0, loudness, 0.01);
}

TEST(Loudness, Test1KhzSineYouTubeLoudness) {
    typedef float Float;
    using namespace bakuage;
    
    AlignedPodVector<Float> input(10 * 44100);
    for (int i = 0; i < input.size() / 2; i++) {
        const auto x = std::sin(2 * M_PI * 1000 * i / 44100);
        input[2 * i + 0] = x;
        input[2 * i + 1] = x;
    }
    
    Float loudness = 1e37;
    Float max_loudness = 1e37;
    std::vector<int> histo;
    loudness_ebu_r128::CalculateLoudnessCore<Float>(input.data(), 2, input.size() / 2, 44100, 0.4, 0.1, -70, -10, &loudness, nullptr, &histo, nullptr, nullptr, true, &max_loudness);
    
    EXPECT_NEAR(0, loudness, 0.01);
    EXPECT_NEAR(0, max_loudness, 0.01);
}
