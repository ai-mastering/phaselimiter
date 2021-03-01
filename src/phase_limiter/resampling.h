#ifndef PHASE_LIMITER_RESAMPLING_H_
#define PHASE_LIMITER_RESAMPLING_H_

#include <vector>

namespace phase_limiter {
    void CalcMaxAvailableNormalizedFreq(const std::vector<float> *wave, int channels, float *max_available_normalized_freq);
    
    // upsample 1 -> n
    void Upsample(std::vector<float> *wave, int channels, int n);
    // downsample n -> 1
    void Downsample(std::vector<float> *wave, int channels, int n);
}

#endif
