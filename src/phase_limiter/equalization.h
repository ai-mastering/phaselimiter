#ifndef PHASE_LIMITER_EQUALIZATION_H_
#define PHASE_LIMITER_EQUALIZATION_H_

#include <vector>

namespace phase_limiter {
void CutLowAndHighFreq(std::vector<float> *wave, int channels, float normalized_low_cut_off_freq, float normalized_high_cut_off_freq);
}

#endif
