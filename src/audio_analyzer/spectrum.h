#ifndef BAKUAGE_AUDIO_ANALYZER_SPECTRUM_H_
#define BAKUAGE_AUDIO_ANALYZER_SPECTRUM_H_

#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include "bakuage/memory.h"
#include "bakuage/utils.h"
#include "bakuage/dft.h"

namespace audio_analyzer {
    
template <typename Float>
void CalculateSpectrum(Float *input, int channels, int samples, int sample_freq, 
                       int spectrum_length, std::vector<Float> *spectrum) {   
                           using bakuage::Sqr;

    std::vector<Float> &spec = *spectrum;
    spec.clear();
    spec.resize(2 * (spectrum_length / 2 + 1));
    std::vector<double> powers(spectrum_length / 2 + 1);

    bakuage::RealDft<float> dft(spectrum_length);
    float *fft_input = (float *)bakuage::AlignedMalloc(sizeof(float) * spectrum_length);
    float *fft_output = (float *)bakuage::AlignedMalloc(2 * sizeof(float) * (spectrum_length / 2 + 1));

    // 総エネルギーがかわらないように、sqrt(hannng)窓
    // FFTの正規化も行う
    std::vector<Float> window(spectrum_length);
    for (int i = 0; i < spectrum_length; i++) {
        window[i] = std::sqrt((std::max)(0.0, 0.5 - 0.5 * std::cos(2.0 * M_PI * i / spectrum_length)))
            / std::sqrt(spectrum_length);
    }

    int pos = -spectrum_length / 2;
    while (pos < samples) {
        for (int channel = 0; channel < channels; channel++) {
            for (int j = 0; j < spectrum_length; j++) {
                int index = pos + j;
                if (index < 0 || samples <= index) {
                    fft_input[j] = 0;
                }
                else {
                    fft_input[j] = input[channels * index + channel] * window[j];
                }
            }
            
            dft.Forward(fft_input, fft_output);
            for (int j = 0; j <= spectrum_length / 2; j++) {
                powers[j] += Sqr(fft_output[2 * j + 0]) + Sqr(fft_output[2 * j + 1]);
            }
        }
        pos += spectrum_length / 2;
    }

    double r = (double)spectrum_length / (channels * samples);
    for (int j = 0; j <= spectrum_length / 2; j++) {
        spec[2 * j + 0] = (double)j / spectrum_length * sample_freq;
        spec[2 * j + 1] = 10 * std::log10(r * powers[j] + 1e-37);
    }

    bakuage::AlignedFree(fft_input);
    bakuage::AlignedFree(fft_output);
}

}

#endif 
