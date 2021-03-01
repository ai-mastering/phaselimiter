#ifndef PHASE_LIMITER_AUTO_MASTERING_H_
#define PHASE_LIMITER_AUTO_MASTERING_H_

#include <functional>
#include <vector>

namespace phase_limiter {
    void AutoMastering(std::vector<float> *_wave, const float **irs, const int *ir_lens, const int sample_rate, const std::function<void (float)> &progress_callback);
    void AutoMastering2(std::vector<float> *_wave, const int sample_rate, const std::function<void(float)> &progress_callback);
    void AutoMastering3(std::vector<float> *_wave, const int sample_rate, const std::function<void(float)> &progress_callback);
    void AutoMastering5(std::vector<float> *_wave, const int sample_rate, const std::function<void (float)> &progress_callback);
}

#endif
