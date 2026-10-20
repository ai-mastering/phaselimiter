#include "gtest/gtest.h"

#include <vector>
#include <tbb/task_arena.h>
#include "audio_analyzer/reverb.h"

namespace {

std::vector<double> DirectSmoothPsd(const std::vector<double> &psd, int samples, int length) {
    std::vector<long double> weights(2 * length - 1);
    for (int j = 1 - length; j < length; ++j) {
        weights[j + length - 1] = (1 + std::cos(std::acos(-1.L) * j / length)) / (2 * length);
    }
    std::vector<double> result(psd.size());
    for (int k = 0; k < psd.size(); ++k) {
        long double sum = 0;
        for (int j = 1 - length; j < length; ++j) {
            int index = ((k + j) % samples + samples) % samples;
            if (index > samples / 2) index = samples - index;
            sum += psd[index] * weights[j + length - 1];
        }
        result[k] = sum;
    }
    return result;
}

// Independent smoothing and full-array energy summation reference. Keep the
// same float input/output DFTs so the comparison isolates the replaced stages.
double DirectDrr(const std::vector<double> &input, int sample_rate) {
    const int samples = input.size();
    const double scale = 1 / std::sqrt(samples);
    bakuage::AlignedPodVector<float> time(samples);
    bakuage::AlignedPodVector<std::complex<float>> spectrum(samples / 2 + 1);
    bakuage::RealDft<float> dft(samples);
    for (int i = 0; i < samples; ++i) time[i] = input[i] * scale;
    dft.Forward(time.data(), reinterpret_cast<float *>(spectrum.data()));
    std::vector<double> psd(spectrum.size());
    for (int k = 0; k < psd.size(); ++k) psd[k] = std::norm(spectrum[k]);
    const auto smooth = DirectSmoothPsd(psd, samples, std::max(1, int(1.0 * samples / sample_rate * 125)));
    for (int k = 0; k < psd.size(); ++k) {
        spectrum[k] = std::complex<float>(std::sqrt(psd[k] / std::max(1e-37, smooth[k])) * scale, 0);
    }
    dft.Backward(reinterpret_cast<float *>(spectrum.data()), time.data());
    double direct = 0, reverb = 0;
    const int direct_length = .001 * sample_rate;
    const int transition = .0001 * sample_rate;
    for (int i = 0; i < samples; ++i) {
        const int distance = std::min(i, samples - i);
        double d = 0, r = 0;
        if (distance < direct_length) d = time[i];
        else if (distance < direct_length + transition) {
            const double weight = .5 + .5 * std::cos(M_PI * (distance - direct_length) / transition);
            d = time[i] * weight;
            r = time[i] * (1 - weight);
        } else if (distance < int(.25 * sample_rate)) r = time[i];
        direct += d * d;
        reverb += r * r;
    }
    return -10 * std::log10(1e-37 + reverb / (1e-37 + direct));
}

}

TEST(AudioAnalyzerReverb, SmoothingMatchesDirectConvolution) {
    for (int samples : {1, 7, 4096, 4097, 44100}) {
        SCOPED_TRACE(samples);
        const int length = std::max(1, samples * 125 / 44100);
        std::vector<double> psd(samples / 2 + 1);
        for (int i = 0; i < psd.size(); ++i) psd[i] = ((i * 37) % 257) / 128.0;
        const auto expected = DirectSmoothPsd(psd, samples, length);
        const auto actual = audio_analyzer::impl::ReverbSmoothPsd(psd.data(), samples, length);
        for (int i = 0; i < psd.size(); ++i) EXPECT_NEAR(expected[i], actual[i], 1e-12);
    }
}

TEST(AudioAnalyzerReverb, SmoothingHighDynamicRangeAndBound) {
    const int samples = 1048576, length = 2972;
    std::vector<double> psd(samples / 2 + 1, 1e-12);
    const int spike = 123456;
    psd[spike] = 1e6;
    const auto actual = audio_analyzer::impl::ReverbSmoothPsd(psd.data(), samples, length);
    for (int i = 0; i < psd.size(); ++i) {
        ASSERT_TRUE(std::isfinite(actual[i]));
        ASSERT_GE(actual[i], psd[i] / length);
        const int distance = std::abs(i - spike);
        const long double weight = distance < length ?
            (1 + std::cos(std::acos(-1.L) * distance / length)) / (2 * length) : 0;
        const double expected = 1e-12 + (1e6 - 1e-12L) * weight;
        ASSERT_NEAR(expected, actual[i], 1e-12 + expected * 1e-12) << i;
    }
}

TEST(AudioAnalyzerReverb, ShortAndBoundaryInputs) {
    // Cover overlapping head/tail ranges and both sides of their boundaries.
    for (int samples : {1, 7, 11024, 11025, 11026, 22049, 22050, 22051}) {
        SCOPED_TRACE(samples);
        std::vector<double> input(samples);
        for (int i = 0; i < samples; ++i) input[i] = ((i * 37) % 257 - 128) / 128.0;
        double drr;
        audio_analyzer::CalculateDirectReverbEnergyRatio(input.data(), 1, samples, 44100, &drr);
        EXPECT_NEAR(DirectDrr(input, 44100), drr, 1e-5);
    }
}

TEST(AudioAnalyzerReverb, SilenceAndNarrowBandInputs) {
    const int samples = 6001;
    std::vector<double> input(samples, 0);
    double drr;
    audio_analyzer::CalculateDirectReverbEnergyRatio(input.data(), 1, samples, 48000, &drr);
    EXPECT_DOUBLE_EQ(370, drr);
    for (int i = 0; i < samples; ++i) input[i] = std::sin(2 * M_PI * 440 * i / 48000);
    audio_analyzer::CalculateDirectReverbEnergyRatio(input.data(), 1, samples, 48000, &drr);
    EXPECT_NEAR(DirectDrr(input, 48000), drr, 1e-5);
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
