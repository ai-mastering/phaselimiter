#ifndef PHASE_LIMITER_RESAMPLING_H_
#define PHASE_LIMITER_RESAMPLING_H_

#include <vector>

namespace phase_limiter {
    // Waves contain interleaved samples and must have a positive channel count.
    // Resampling factors must be >= 1. Invalid shapes/arguments throw logic_error
    // (including invalid_argument); sizes beyond supported limits throw length_error.
    // Empty waves are accepted; their maximum available frequency is zero.
    void CalcMaxAvailableNormalizedFreq(const std::vector<float> *wave, int channels, float *max_available_normalized_freq);
    
    // upsample 1 -> n
    void Upsample(std::vector<float> *wave, int channels, int n);
    // downsample n -> 1
    void Downsample(std::vector<float> *wave, int channels, int n);
}

#endif
