#ifndef PHASE_LIMITER_PRE_COMPRESSION_H_
#define PHASE_LIMITER_PRE_COMPRESSION_H_

#include <vector>

namespace phase_limiter {
void PreCompress(std::vector<float> *_wave, int sample_rate);
}

#endif