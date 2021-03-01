#ifndef BAKUAGE_BAKUAGE_RNNOISE_H_
#define BAKUAGE_BAKUAGE_RNNOISE_H_

#include <cstdlib>
#include <memory>
#include "bakuage/memory.h"
#include "bakuage/vector_math.h"

namespace bakuage {
    class RnnoiseLibrary {
    public:
        RnnoiseLibrary(const char *librnnoise_path);
        void *rnnoise_create() const;
        void rnnoise_destroy(void *st) const;
        float rnnoise_process_frame(void *st, float *out, const float *in, int pitch_filter_enabled) const;
        float rnnoise_process_frame_multi(int num_channels, void **st, float **out, const float **in, int pitch_filter_enabled) const;
    private:
        std::shared_ptr<void> impl_;
    };
    
    class Rnnoise {
    public:
        Rnnoise(const RnnoiseLibrary &library, int channels): library_(library) {
            for (int ch = 0; ch < channels; ch++) {
                states_.push_back(library_.rnnoise_create());
            }
        }
        
        ~Rnnoise() {
            for (const auto st: states_) {
                if (st) {
                    library_.rnnoise_destroy(st);
                }
            }
        }
        
        void Process(const float **input, float **output, bool pitch_filter_enabled) {
            const float scale = (1 << 15) - 1;
            for (int ch = 0; ch < states_.size(); ch++) {
                bakuage::VectorMulConstant<float>(input[ch], scale, output[ch], frame_size());
            }
            library_.rnnoise_process_frame_multi(states_.size(), states_.data(), output, (const float **)output, pitch_filter_enabled);
            for (int ch = 0; ch < states_.size(); ch++) {
                bakuage::VectorMulConstant<float>(output[ch], 1.0 / scale, output[ch], frame_size());
            }
        }
        
        int frame_size() const { return 480; }
        int delay_samples() const { return frame_size(); }
    private:
        bakuage::AlignedPodVector<void *> states_;
        RnnoiseLibrary library_;
    };
}

#endif
