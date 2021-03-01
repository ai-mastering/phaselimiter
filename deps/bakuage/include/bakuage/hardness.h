#ifndef BAKUAGE_BAKUAGE_HARDNESS_H_
#define BAKUAGE_BAKUAGE_HARDNESS_H_

#include <cmath>
#include <complex>
#include <vector>
#include "bakuage/memory.h"
#include "bakuage/utils.h"
#include "bakuage/statistics.h"

namespace bakuage {
    // mfccはenergy sum modeを想定している
    template <class Float>
    void CalculateBandwidth(Float *input, int channels, int samples, int sample_freq, Float *bandwidth, Float *diff_bandwidth) {
        // calculate mfcc
        const int shift_resolution = 2;
        const int output_shift_resolution = 2;
        const int width = output_shift_resolution * ((16384 * sample_freq / 44100) / output_shift_resolution); // 0.372 sec, 4x only
        const int shift = width / shift_resolution;
        const int spec_len = width / 2 + 1;
        int pos = -width + shift;
        bakuage::AlignedPodVector<float> window(width);
        bakuage::CopyHanning(width, window.begin());
        bakuage::AlignedPodVector<float> fft_input(width);
        bakuage::AlignedPodVector<std::complex<float>> fft_output(spec_len);
        bakuage::AlignedPodVector<float> fft_energy(spec_len);
        bakuage::AlignedPodVector<float> prev_fft_energy(spec_len);
        bakuage::AlignedPodVector<float> freqs(spec_len);
        for (int i = 0; i < spec_len; i++) {
            freqs[i] = 1.0 * i * sample_freq / width;
        }
        bakuage::RealDft<float> dft(width);
        double sum_bandwidth = 0;
        double sum_diff_bandwidth = 0;
        double sum_energy = 0;
        double sum_diff_energy = 0;
        while (pos < samples) {
            bakuage::TypedFillZero(fft_energy.data(), fft_energy.size());
            for (int i = 0; i < channels; i++) {
                for (int j = 0; j < width; j++) {
                    int k = pos + j;
                    fft_input[j] = (0 <= k && k < samples) ? input[channels * k + i] * window[j] : 0;
                }
                dft.Forward(fft_input.data(), (float *)fft_output.data());
                for (int j = 0; j < spec_len; j++) {
                    fft_energy[j] += std::norm(fft_output[j]);
                }
            }
            
            {
                double energy = 0;
                for (int j = 0; j < spec_len; j++) {
                    energy += fft_energy[j];
                }
                sum_energy += energy;
                Statistics stats;
                for (int j = 0; j < spec_len; j++) {
                    double hz = 1.0 * (j + 0.5) * sample_freq / width;
                    stats.Add(bakuage::HzToMel(hz), fft_energy[j]);
                }
                sum_bandwidth += stats.mean() * energy;
            }
            
            {
                double diff_energy = 0;
                for (int j = 0; j < spec_len; j++) {
                    diff_energy += std::abs(fft_energy[j] - prev_fft_energy[j]);
                }
                sum_diff_energy += diff_energy;
                Statistics diff_stats;
                for (int j = 0; j < spec_len; j++) {
                    double hz = 1.0 * (j + 0.5) * sample_freq / width;
                    diff_stats.Add(bakuage::HzToMel(hz), std::abs(fft_energy[j] - prev_fft_energy[j]));
                }
                sum_diff_bandwidth += diff_stats.mean() * diff_energy;
            }
            
            bakuage::TypedMemcpy(prev_fft_energy.data(), fft_energy.data(), spec_len);
            
            pos += shift;
        }
        
        *bandwidth = sum_bandwidth / (1e-37 + sum_energy);
        *diff_bandwidth = sum_diff_bandwidth / (1e-37 + sum_diff_energy);
    }
}

#endif /* BAKUAGE_BAKUAGE_HARDNESS_H_ */
