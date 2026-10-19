#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "audio_analyzer/peak.h"
#include "bakuage/delay_filter2.h"
#include "bakuage/fir_filter2.h"
#include "bakuage/fir_resample_filter.h"
#include "bakuage/vector_math.h"

// Exercise empty input and tails around AVX widths and common block sizes.
// Compare numerical results to a scalar contract, not just absence of crashes.
TEST(AudioBoundaries, VectorTails) {
    for (int size : {0, 1, 7, 8, 9, 15, 16, 17, 255, 256, 257}) {
        SCOPED_TRACE(size);
        // Guard the empty case with a valid pointer, as required by DSP backends.
        std::vector<float> input(std::max(1, size), 0.25f);
        std::vector<float> output(std::max(1, size), -99.0f);
        bakuage::VectorMulConstant(input.data(), 2.0f, output.data(), size);
        for (int i = 0; i < size; ++i) EXPECT_FLOAT_EQ(0.5f, output[i]);
        if (size == 0) { EXPECT_FLOAT_EQ(-99.0f, output[0]); }
    }
}

TEST(AudioBoundaries, DelayShortAndEmptyBlocks) {
    for (int delay : {0, 1, 8, 17}) {
        for (int size : {0, 1, 7, 8, 9, 16, 17, 31, 32, 33}) {
            SCOPED_TRACE(delay);
            SCOPED_TRACE(size);
            for (bool inplace : {false, true}) {
                std::vector<float> input(std::max(1, size), 1.0f);
                std::vector<float> output(std::max(1, size), -99.0f);
                bakuage::DelayFilter2<float> filter(delay);
                float *result = inplace ? input.data() : output.data();
                filter.Clock(input.data(), input.data(), result);
                for (int pos = 0; pos < size; pos += 7) {
                    const int count = std::min(7, size - pos);
                    filter.Clock(input.data() + pos, input.data() + pos + count, result + pos);
                }
                for (int i = 0; i < size; ++i) EXPECT_FLOAT_EQ(i < delay ? 0.0f : 1.0f, result[i]);
                if (size == 0) { EXPECT_FLOAT_EQ(inplace ? 1.0f : -99.0f, result[0]); }
            }
        }
    }
}

TEST(AudioBoundaries, FirMatchesScalarConvolution) {
    const std::vector<float> coefficients{0.25f, 0.5f, 0.25f};
    for (int size : {0, 1, 7, 8, 9, 15, 16, 17, 255, 256, 257}) {
        for (int block : {1, 7, 8, 9, 256}) {
            SCOPED_TRACE(size);
            SCOPED_TRACE(block);
            std::vector<float> input(std::max(1, size));
            std::vector<float> output(std::max(1, size), -99.0f);
            for (int i = 0; i < size; ++i) input[i] = (i % 5 - 2) * 0.125f;
            bakuage::FirFilter2<float> filter(coefficients.begin(), coefficients.end());
            filter.Clock(input.data(), input.data(), output.data());
            for (int pos = 0; pos < size; pos += block) {
                const int count = std::min(block, size - pos);
                filter.Clock(input.data() + pos, input.data() + pos + count, output.data() + pos);
            }
            for (int i = 0; i < size; ++i) {
                float expected = 0;
                for (int k = 0; k < 3 && k <= i; ++k) expected += coefficients[k] * input[i - k];
                EXPECT_NEAR(expected, output[i], 1e-6);
            }
            if (size == 0) { EXPECT_FLOAT_EQ(-99.0f, output[0]); }
        }
    }
}

TEST(AudioBoundaries, ResamplerMinimumBlocks) {
    for (int input_rate : {44100, 48000}) {
        for (int output_rate : {44100, 48000}) {
            SCOPED_TRACE(input_rate);
            SCOPED_TRACE(output_rate);
            bakuage::FirResampleFilter<float>::Config config;
            config.input_sample_rate = input_rate;
            config.output_sample_rate = output_rate;
            bakuage::FirResampleFilter<float> filter(config);
            // Clock consumes one complete block. Partial blocks must be padded
            // by the caller; exact-sized buffers expose any overread to ASan.
            std::vector<float> input(filter.input_process_size(), 0.0f);
            std::vector<float> output(filter.output_process_size(), -99.0f);
            for (int block = 0; block < 3; ++block) {
                filter.Clock(input.data(), output.data());
                for (float sample : output) EXPECT_FLOAT_EQ(0.0f, sample);
            }
        }
    }
}

TEST(AudioBoundaries, Small2DTransformPlans) {
    for (int width : {4, 8}) {
        SCOPED_TRACE(width);
        const int height = 4;
        const int size = width * height;
        std::vector<float> input(2 * size, 0.0f);
        std::vector<float> spectrum(2 * size);
        std::vector<float> restored(2 * size);
        input[0] = 1.0f;
        bakuage::Dft2D<float> dft(width, height);
        dft.Forward(input.data(), spectrum.data());
        dft.Backward(spectrum.data(), restored.data());
        for (int i = 0; i < 2 * size; ++i) EXPECT_NEAR(input[i] * size, restored[i], 1e-5);

        std::vector<float> silence(size, 0.0f);
        std::vector<float> output(size, -99.0f);
        bakuage::Dct2D<float> dct(width, height);
        dct.Forward(silence.data(), output.data());
        for (float sample : output) EXPECT_FLOAT_EQ(0.0f, sample);
    }
}


TEST(AudioBoundaries, ShortInterleavedPeakAndRms) {
    for (int channels : {1, 2}) {
        for (int frames : {1, 7, 8, 9, 255, 256, 257}) {
            SCOPED_TRACE(channels);
            SCOPED_TRACE(frames);
            std::vector<float> input(channels * frames);
            double energy = 0;
            for (int i = 0; i < channels * frames; ++i) {
                input[i] = i % 2 ? -0.5f : 0.25f;
                energy += input[i] * input[i];
            }
            float peak, rms;
            audio_analyzer::CalculatePeakAndRMS<float>(input.data(), channels, frames,
                &peak, &rms, 1, nullptr);
            const float expected_peak = input.size() == 1 ? 0.25f : 0.5f;
            EXPECT_NEAR(20 * std::log10(expected_peak), peak, 1e-5);
            EXPECT_NEAR(10 * std::log10(energy / input.size()), rms, 1e-5);
        }
        for (int rate : {44100, 48000}) {
            std::vector<float> silence(channels, 0.0f);
            float peak;
            audio_analyzer::CalculateLowpassTruePeak<float>(silence.data(), channels,
                1, rate, 15000.0f, 2, &peak);
            EXPECT_NEAR(-740.0f, peak, 1e-4);
        }
    }
}
