#include "phase_limiter/resampling.h"

#include <cstring>
#include <stdexcept>
#include "bakuage/fir_filter2.h"
#include "bakuage/fir_design.h"
#include "bakuage/memory.h"
#include "bakuage/vector_math.h"

namespace {
int freq_len(int n) {
	return n / 2 + 1;
}
}

namespace phase_limiter{
    
    // dftはpoolを使わない。サイズが大きいので
    
    void CalcMaxAvailableNormalizedFreq(const std::vector<float> *wave, int channels, float *max_available_normalized_freq) {
        if (wave->size() % channels) {
            throw std::logic_error("input wave length must be multiple of channels");
        }
        const int src_length = wave->size() / channels;
        const int fft_len = bakuage::CeilPowerOf2(src_length);
        const int freq_length = freq_len(fft_len);
        
        bakuage::AlignedPodVector<float> energies(freq_length);
        {
            bakuage::AlignedPodVector<float> fft_input(fft_len);
            bakuage::AlignedPodVector<std::complex<float>> fft_output(freq_length);
            bakuage::AlignedPodVector<float> channel_energies(freq_length);
            bakuage::RealDft<float> dft(fft_len);
        
            for (int channel = 0; channel < channels; channel++) {
                // fft
                for (int i = 0; i < src_length; i++) {
                    fft_input[i] = (*wave)[channels * i + channel];
                }
                dft.Forward(fft_input.data(), (float *)fft_output.data());
                bakuage::VectorNorm(fft_output.data(), channel_energies.data(), freq_length);
                bakuage::VectorAddInplace(channel_energies.data(), energies.data(), freq_length);
            }
        }
        
        const double energy_threshold = 1e-7 * bakuage::VectorSum(energies.data(), freq_length) / freq_length;
        *max_available_normalized_freq = 0;
        for (int i = freq_length - 1; i >= 0; i--) {
            if (energies[i] > energy_threshold) {
                *max_available_normalized_freq = 1.0 * i / fft_len;
                break;
            }
        }
    }
    
    void Upsample(std::vector<float> *wave, int channels, int n) {
        if (n == 1) return;
        
        if (wave->size() % channels) {
            throw std::logic_error("input wave length must be multiple of channels");
        }
        int src_length = wave->size() / channels;
        int dest_length = src_length * n;
        wave->resize(channels * dest_length);
        
        const double transition_width = 20.0 / 44100 / n; // normalized freq
        const double stopband_reduce_db = 140; // dB
        int filter_len;
        double alpha;
        bakuage::CalcKeiserFirParams(stopband_reduce_db, transition_width, &filter_len, &alpha);
        
        // UpsampleフィルタはDownsampleよりもカットオフ周波数を低くする
        const auto fir = bakuage::CalculateBandPassFir<double>(0, 0.5 / n - 2 * transition_width, filter_len, alpha);
        bakuage::FirFilter2<float> fir_filter(fir.begin(), fir.end());
        const int delay_samples = filter_len / 2;
        bakuage::AlignedPodVector<float> temp_input(dest_length + delay_samples);
        bakuage::AlignedPodVector<float> temp_output(dest_length + delay_samples);
        for (int ch = 0; ch < channels; ch++) {
            fir_filter.Clear();
            for (int i = 0; i < src_length; i++) {
                temp_input[n * i] = (*wave)[channels * i + ch] * n;
            }
            fir_filter.Clock(temp_input.data(), temp_input.data() + dest_length + delay_samples, temp_output.data());
            for (int i = 0; i < dest_length; i++) {
                (*wave)[channels * i + ch] = temp_output[i + delay_samples];
            }
        }
    }

    void Downsample(std::vector<float> *wave, int channels, int n) {
        if (n == 1) return;
        
        if (wave->size() % channels) {
            throw std::logic_error("input wave length must be multiple of channels");
        }
        int src_length = wave->size() / channels;
        if (src_length % n) {
            throw std::logic_error("src_length must be multiple of n");
        }
        int dest_length = src_length / n;
        
        {
            const double transition_width = 20.0 / 44100 / n; // normalized freq
            const double stopband_reduce_db = 140; // dB
            int filter_len;
            double alpha;
            bakuage::CalcKeiserFirParams(stopband_reduce_db, transition_width, &filter_len, &alpha);
            
            const auto fir = bakuage::CalculateBandPassFir<double>(0, 0.5 / n - transition_width, filter_len, alpha);
            bakuage::FirFilter2<float> fir_filter(fir.begin(), fir.end());
            const int delay_samples = filter_len / 2;
            bakuage::AlignedPodVector<float> temp_input(src_length + delay_samples);
            bakuage::AlignedPodVector<float> temp_output(src_length + delay_samples);
            for (int ch = 0; ch < channels; ch++) {
                fir_filter.Clear();
                for (int i = 0; i < src_length; i++) {
                    temp_input[i] = (*wave)[channels * i + ch];
                }
                fir_filter.Clock(temp_input.data(), temp_input.data() + src_length + delay_samples, temp_output.data());
                for (int i = 0; i < dest_length; i++) {
                    (*wave)[channels * i + ch] = temp_output[i * n + delay_samples];
                }
            }
        }

        wave->resize(channels * dest_length);
    }

}
