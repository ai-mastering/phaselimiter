#ifndef PHASE_LIMITER_FREQ_EXPANDER_H_
#define PHASE_LIMITER_FREQ_EXPANDER_H_

#include <vector>

namespace phase_limiter {
void FreqExpand(std::vector<float> *wave, int channels, int sample_rate, float ratio);
}

#endif
