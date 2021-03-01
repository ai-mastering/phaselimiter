#ifndef BAKUAGE_AUDIO_ANALYZER_PEAK_H_
#define BAKUAGE_AUDIO_ANALYZER_PEAK_H_

#include <algorithm>
#include <cmath>
#include <vector>

#include "bakuage/loudness_filter.h"
#include "bakuage/memory.h"
#include "bakuage/utils.h"
#include "bakuage/fir_design.h"
#include "bakuage/fir_filter2.h"

namespace audio_analyzer {
    
template <typename Float>
void CalculateLowpassTruePeak(const Float *input, int channels, int samples, int sample_rate, Float lowpass_freq, int true_peak_oversample, Float *true_peak) {
    using namespace bakuage;
    
    // calculate true peak
    if (true_peak) {
        double tp = 0;
        const auto fir = bakuage::CalculateBandPassFir<double>(0, lowpass_freq / sample_rate / true_peak_oversample, 1024 * true_peak_oversample + 1, 7);
        const auto max_process_size = bakuage::CeilInt<int>(bakuage::CeilPowerOf2(2 * fir.size()), true_peak_oversample);
        const auto max_src_process_size = max_process_size / true_peak_oversample;
        bakuage::AlignedPodVector<float> temp_input(max_process_size);
        bakuage::AlignedPodVector<float> temp_output(max_process_size);
        for (int ch = 0; ch < channels; ch++) {
            bakuage::FirFilter2<float> fir_filter(fir.begin(), fir.end());
            for (int i = 0; i < samples + fir.size(); i += max_src_process_size) {
                for (int j = 0; j < max_src_process_size; j++) {
                    temp_input[true_peak_oversample * j] = i + j < samples ? input[channels * (i + j) + ch] * true_peak_oversample : 0;
                }
                fir_filter.Clock(temp_input.data(), temp_input.data() + max_process_size, temp_output.data());
                for (int j = 0; j < max_process_size; j++) {
                    tp = std::max<Float>(tp, std::abs(temp_output[j]));
                }
            }
        }
        *true_peak = 20 * std::log10(tp + 1e-37);
    }
}
    
template <typename Float>
void CalculatePeakAndRMS(const Float *input, int channels, int samples,
                       Float *peak, Float *rms, int true_peak_oversample, Float *true_peak) {
    using namespace bakuage;

    double peak2 = 0;
    double rms2 = 0;
    int total_samples = channels * samples;

    for (int i = 0; i < total_samples; i++) {
        if (peak) peak2 = std::max<double>(peak2, std::abs(input[i]));
        if (rms) rms2 += Sqr(input[i]);
    }
    
    if (peak) *peak = 20 * std::log10(peak2 + 1e-37);
    if (rms) *rms = 10 * std::log10(rms2 / total_samples + 1e-37);
    
    // calculate true peak
    CalculateLowpassTruePeak<Float>(input, channels, samples, 1, 0.5, true_peak_oversample, true_peak);
}
    
}

#endif 
