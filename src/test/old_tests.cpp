#include <cstdio>
#include <iostream>
#include <vector>
#include <sstream>
#include <chrono>
#include <string>
#include <stdexcept>

#include "gtest/gtest.h"

#include "bakuage/utils.h"
#include "bakuage/biquad_iir_filter.h"
#include "bakuage/gammatone_filter.h"
#include "bakuage/pan_detect_filter.h"
#include "bakuage/time_varying_lowpass_filter.h"
#include "bakuage/delay_filter.h"
#include "bakuage/fir_filter.h"
#include "bakuage/fir_filter2.h"

typedef double Float;

using namespace bakuage;

constexpr int kSampleRate = 44100;

TEST(OldTests, GammatoneFilter) {
    GammatoneFilter<Float> filter(1000, 100, kSampleRate);
    Float original_energy = 0;
    Float filter_energy = 0;
    for (int i = 0; i < kSampleRate; i++) {
        Float x = std::cos(2 * M_PI * 1000 * i / kSampleRate);
        original_energy += Sqr(x);
        filter_energy += Sqr(filter.Clock(x));
    }
    std::printf("filter energy / original energy = %.3e / %.3e\n", filter_energy, original_energy);
}

TEST(OldTests, LowpassFilter) {
    BiquadIIRFilter<Float> filter(
                                  BiquadIIRCoef<Float>::CreateLowpass(1000.0 / kSampleRate, 1));
    Float original_energy = 0;
    Float filter_energy = 0;
    for (int i = 0; i < kSampleRate; i++) {
        Float x = std::cos(2 * M_PI * 50 * i / kSampleRate);
        original_energy += Sqr(x);
        filter_energy += Sqr(filter.Clock(x));
    }
    std::printf("filter energy / original energy = %.3e / %.3e\n", filter_energy, original_energy);
}

TEST(OldTests, PanDetectFilter) {
    PanDetectFilter<Float> filter(1000, 100, kSampleRate);
    Float pan = 0;
    Float original_energy = 0;
    Float filter_energy = 0;
    Float original_count = 0;
    Float filter_count = 0;
    for (int i = 0; i < 60 * kSampleRate; i++) {
        Float x = std::cos(2 * M_PI * 1000 * i / kSampleRate);
        original_energy += 2 * Sqr(x);
        original_count += 1;
        auto output = filter.Clock(x, x);
        if (output.valid) {
            pan += output.pan;
            filter_energy += output.energy;
            filter_count += 1;
        }
    }
    std::printf("filter energy / original energy = %.3e / %.3e = %.3e\n",
           filter_energy / filter_count, original_energy / original_count,
           filter_energy / filter_count / (original_energy / original_count));
}

TEST(OldTests, TimeVaryingLowpassFilter) {
    for (int i = 1; i < 6; i++) {
        int samples;
        float peak = std::pow(10, -i);
        float a = TimeVaryingLowpassFilter<Float>::CalculateAFromPeak(2, peak, &samples);
        std::printf("%.3e %.3e %d", peak, a, samples);
    }
}

TEST(OldTests, DelayFilter) {
    for (int j = 0; j < 4; j++) {
        DelayFilter<float> filter(j);
        for (int i = 0; i <= j; i++) {
            filter.Clock(i);
        }
        
        for (int i = j + 1; i < 100; i++) {
            EXPECT_EQ(i - j, filter.Clock(i));
        }
    }
}

template <typename Float>
void test_fir_filter() {
    for (int k = 0; k < 10; k++) {
        int n = 2 * k + 1;
        for (int j = 0; j < n; j++) {
            std::vector<Float> fir(n);
            fir[j] = 1;
            
            FirFilter<Float> fir_filter(fir.begin(), fir.end());
            for (int i = 0; i <= j; i++) {
                fir_filter.Clock(i);
            }
            for (int i = j + 1; i < 100; i++) {
                EXPECT_EQ(i - j, fir_filter.Clock(i));
            }
        }
    }
}
TEST(OldTests, FirFilterFloat) {
    test_fir_filter<float>();
}
TEST(OldTests, FirFilterDouble) {
    test_fir_filter<double>();
}

template <typename Float>
void test_fir_filter2() {
    for (int k = 0; k < 20; k++) {
        int n = k + 1;
        for (int j = 0; j < n; j++) {
            std::vector<Float> fir(n);
            fir[j] = 1;
            
            for (int step = 1; step <= 100; step++) {
                FirFilter2<Float> fir_filter(fir.begin(), fir.end());
                std::vector<Float> buffer(100);
                std::vector<Float> output(buffer.size());
                for (int i = 0; i < buffer.size(); i++) {
                    buffer[i] = i;
                }
                for (int i = 0; i < buffer.size(); i += step) {
                    const int ed = std::min<int>(i + step, buffer.size());
                    fir_filter.Clock(buffer.data() + i, buffer.data() + ed, output.data() + i);
                }
                for (int i = j + 1; i < 100; i++) {
                    EXPECT_LT(1e-4, std::abs(output[i] - (i - j)));
                }
            }
        }
    }
}
TEST(OldTests, FirFilter2Float) {
    test_fir_filter<float>();
}

