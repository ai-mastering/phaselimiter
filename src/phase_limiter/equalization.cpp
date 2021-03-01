#include "phase_limiter/equalization.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include "bakuage/fir_design.h"
#include "bakuage/fir_filter2.h"

namespace phase_limiter {

void CutLowAndHighFreq(std::vector<float> *wave, int channels, float normalized_low_cut_off_freq, float normalized_high_cut_off_freq) {
    if (normalized_low_cut_off_freq == 0 && normalized_high_cut_off_freq == 0) return;
    
    if (wave->size() % channels) {
        throw std::logic_error("input wave length must be multiple of channels");
    }
    int src_length = wave->size() / channels;
    const double transition_width = 5.0 / 44100; // normalized freq
    const double stopband_reduce_db = 70; // dB
    int filter_len;
    double alpha;
    bakuage::CalcKeiserFirParams(stopband_reduce_db, transition_width, &filter_len, &alpha);
    
#if 0
    std::cerr << "CutLowAndHighFreq\tstopband_reduce_db:" << stopband_reduce_db << "\ttransition_width:" << transition_width << "\tfilter_len:" << filter_len << "\talpha:" << alpha << "\tnormalized_low_cut_off_freq:" << normalized_low_cut_off_freq << "\tnormalized_high_cut_off_freq:" << normalized_high_cut_off_freq << std::endl;
#endif
    
    const auto fir = bakuage::CalculateBandPassFir<double>(normalized_low_cut_off_freq, normalized_high_cut_off_freq, filter_len, alpha);
    const int delay_samples = filter_len / 2;
    bakuage::FirFilter2<float> fir_filter(fir.begin(), fir.end());
    bakuage::AlignedPodVector<float> temp_input(src_length + delay_samples);
    bakuage::AlignedPodVector<float> temp_output(src_length + delay_samples);
    for (int ch = 0; ch < channels; ch++) {
        fir_filter.Clear();
        for (int i = 0; i < src_length; i++) {
            temp_input[i] = (*wave)[channels * i + ch];
        }
        fir_filter.Clock(temp_input.data(), temp_input.data() + src_length + delay_samples, temp_output.data());
        for (int i = 0; i < src_length; i++) {
            (*wave)[channels * i + ch] = temp_output[i + delay_samples];
        }
    }
}

}
