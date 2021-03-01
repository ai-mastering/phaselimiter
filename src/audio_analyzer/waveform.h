#ifndef BAKUAGE_AUDIO_ANALYZER_WAVEFORM_H_
#define BAKUAGE_AUDIO_ANALYZER_WAVEFORM_H_

#include <cmath>
#include <algorithm>
#include <limits>
#include <vector>

namespace audio_analyzer {
    
template <typename Float>
void CalculateWaveform(Float *input, int channels, int samples, int sample_freq, 
                       int max_waveform_length, std::vector<Float> *waveform) {   
    int waveform_length = (std::min)(max_waveform_length, samples);

    std::vector<Float> &wave = *waveform;

    wave.clear();
    wave.resize(2 * waveform_length);

    for (int k = 0; k < waveform_length; k++) {
        Float mi = (std::numeric_limits<Float>::max)();
        Float ma = -(std::numeric_limits<Float>::max)();

        int bg = std::floor((Float)k / waveform_length * samples);
        int ed = std::floor((Float)(k + 1) / waveform_length * samples);
        ed = (std::min)(ed, samples);
        if (k == waveform_length - 1) {
            ed = samples;
        }

        Float *input_ptr = input + channels * bg;
        for (int j = bg; j < ed; j++) {
            for (int k = 0; k < channels; k++) {
                mi = (std::min)(mi, *input_ptr);
                ma = (std::max)(ma, *input_ptr);
                input_ptr++;
            }
        }

        wave[2 * k + 0] = mi;
        wave[2 * k + 1] = ma;
    }
}
}

#endif 