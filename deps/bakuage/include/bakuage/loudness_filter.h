#ifndef BAKUAGE_BAKUAGE_LOUDNESS_FILTER_H_
#define BAKUAGE_BAKUAGE_LOUDNESS_FILTER_H_

#include "bakuage/biquad_iir_filter.h"

namespace bakuage {
	template <typename Float = double>
    class LoudnessFilter {
    public:
        LoudnessFilter(int sample_freq) {
            BiquadIIRCoef<Float> coef;
            coef.a1 = -1.69065929318241; 
            coef.a2 = 0.73248077421585;
            coef.b0 = 1.53512485958697;
            coef.b1 = -2.69169618940638;
            coef.b2 = 1.19839281085285;                
            highshelf_filter_.set_coef(coef.ChangeSampleFreq(48000, sample_freq));
            
            coef.a1 = -1.99004745483398; 
            coef.a2 = 0.99007225036621;
            coef.b0 = 1;
            coef.b1 = -2;
            coef.b2 = 1;                
            highpass_filter_.set_coef(coef.ChangeSampleFreq(48000, sample_freq));
        }

        Float Clock(const Float &x) {
            return highpass_filter_.Clock(highshelf_filter_.Clock(x));
        };
    private:
        BiquadIIRFilter<Float> highpass_filter_;
        BiquadIIRFilter<Float> highshelf_filter_;
    };
}

#endif 