#ifndef PHASE_LIMITER_FIR_PROCESSING_H_
#define PHASE_LIMITER_FIR_PROCESSING_H_

#include <algorithm>
#include "bakuage/fir_filter2.h"
#include "bakuage/memory.h"

namespace phase_limiter {

// Filter interleaved audio, discarding the FIR delay and flushing with zeros.
// Input and output may be identical. Keep the original FFT block boundaries so that
// overlap-add arithmetic is unchanged, with scratch independent of track length.
inline void FilterWithDelay(bakuage::FirFilter2<float> *filter,
                            const float *input, int frames, int channels,
                            int delay, float *output) {
    const int total_length = frames + delay;
    const int block_size = std::min(total_length, filter->max_process_len());
    bakuage::AlignedPodVector<float> temp_input(block_size);
    bakuage::AlignedPodVector<float> temp_output(block_size);
    for (int ch = 0; ch < channels; ch++) {
        filter->Clear();
        for (int offset = 0; offset < total_length; offset += block_size) {
            const int count = std::min(block_size, total_length - offset);
            for (int i = 0; i < count; i++) {
                temp_input[i] = offset + i < frames ? input[channels * (offset + i) + ch] : 0;
            }
            filter->Clock(temp_input.data(), temp_input.data() + count, temp_output.data());
            for (int i = std::max(0, delay - offset); i < count; i++) {
                output[channels * (offset + i - delay) + ch] = temp_output[i];
            }
        }
    }
}

} // namespace phase_limiter

#endif
