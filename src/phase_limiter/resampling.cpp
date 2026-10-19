#include "phase_limiter/resampling.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include "bakuage/fir_filter2.h"
#include "bakuage/fir_design.h"
#include "bakuage/memory.h"
#include "bakuage/vector_math.h"

namespace {
size_t ValidateWave(const std::vector<float> &wave, int channels) {
    if (channels <= 0) {
        throw std::invalid_argument("channels must be positive");
    }
    if (wave.size() % channels) {
        throw std::logic_error("input wave length must be multiple of channels");
    }
    return wave.size() / channels;
}

void ValidateFactor(int n) {
    if (n < 1) {
        throw std::invalid_argument("resampling factor must be positive");
    }
    // FirFilter2 rounds twice the FIR length to a power of two and allocates
    // twice that FFT length. Keep those internal int calculations representable.
    const double order = (140.0 - 8) / (2.285 * (20.0 / 44100 / n));
    const double filter_len = 2 * std::ceil(order / 2) + 1;
    if (filter_len > (std::numeric_limits<int>::max() / 8 + 1)) {
        throw std::length_error("resampling filter exceeds DSP size limit");
    }
}

size_t BufferLength(size_t frames, size_t delay) {
    // AlignedPodVector and pointer arithmetic must both fit in address space.
    const size_t limit = static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(float);
    if (delay > limit || frames > limit - delay) {
        throw std::length_error("resampling buffer is too large");
    }
    return frames + delay;
}

int freq_len(int n) {
	return n / 2 + 1;
}
}

namespace phase_limiter{
    
    // dftはpoolを使わない。サイズが大きいので
    
    void CalcMaxAvailableNormalizedFreq(const std::vector<float> *wave, int channels, float *max_available_normalized_freq) {
        const size_t src_length = ValidateWave(*wave, channels);
        if (src_length == 0) {
            *max_available_normalized_freq = 0;
            return;
        }
        // The FFT backend accepts int lengths, including the rounded power of two.
        if (src_length > static_cast<size_t>(std::numeric_limits<int>::max() / 2 + 1)) {
            throw std::length_error("input exceeds FFT size limit");
        }
        const int fft_len = static_cast<int>(bakuage::CeilPowerOf2(static_cast<unsigned int>(src_length)));
        const int freq_length = freq_len(fft_len);
        
        bakuage::AlignedPodVector<float> energies(freq_length);
        {
            bakuage::AlignedPodVector<float> fft_input(fft_len);
            bakuage::AlignedPodVector<std::complex<float>> fft_output(freq_length);
            bakuage::AlignedPodVector<float> channel_energies(freq_length);
            bakuage::RealDft<float> dft(fft_len);
        
            for (int channel = 0; channel < channels; channel++) {
                // fft
                for (size_t i = 0; i < src_length; i++) {
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
        const size_t src_length = ValidateWave(*wave, channels);
        ValidateFactor(n);
        if (n == 1 || src_length == 0) return;
        if (wave->size() > wave->max_size() / static_cast<size_t>(n)) {
            throw std::length_error("upsampled wave is too large");
        }
        const size_t dest_length = src_length * static_cast<size_t>(n);
        
        const double transition_width = 20.0 / 44100 / n; // normalized freq
        const double stopband_reduce_db = 140; // dB
        int filter_len;
        double alpha;
        bakuage::CalcKeiserFirParams(stopband_reduce_db, transition_width, &filter_len, &alpha);
        
        // UpsampleフィルタはDownsampleよりもカットオフ周波数を低くする
        const auto fir = bakuage::CalculateBandPassFir<double>(0, 0.5 / n - 2 * transition_width, filter_len, alpha);
        bakuage::FirFilter2<float> fir_filter(fir.begin(), fir.end());
        const int delay_samples = filter_len / 2;
        const size_t buffer_length = BufferLength(dest_length, delay_samples);
        bakuage::AlignedPodVector<float> temp_input(buffer_length);
        bakuage::AlignedPodVector<float> temp_output(buffer_length);
        wave->resize(static_cast<size_t>(channels) * dest_length);
        for (int ch = 0; ch < channels; ch++) {
            fir_filter.Clear();
            for (size_t i = 0; i < src_length; i++) {
                temp_input[n * i] = (*wave)[channels * i + ch] * n;
            }
            fir_filter.Clock(temp_input.data(), temp_input.data() + dest_length + delay_samples, temp_output.data());
            for (size_t i = 0; i < dest_length; i++) {
                (*wave)[channels * i + ch] = temp_output[i + delay_samples];
            }
        }
    }

    void Downsample(std::vector<float> *wave, int channels, int n) {
        const size_t src_length = ValidateWave(*wave, channels);
        ValidateFactor(n);
        if (n == 1 || src_length == 0) return;
        if (src_length % n) {
            throw std::logic_error("src_length must be multiple of n");
        }
        const size_t dest_length = src_length / n;
        
        {
            const double transition_width = 20.0 / 44100 / n; // normalized freq
            const double stopband_reduce_db = 140; // dB
            int filter_len;
            double alpha;
            bakuage::CalcKeiserFirParams(stopband_reduce_db, transition_width, &filter_len, &alpha);
            
            const auto fir = bakuage::CalculateBandPassFir<double>(0, 0.5 / n - transition_width, filter_len, alpha);
            bakuage::FirFilter2<float> fir_filter(fir.begin(), fir.end());
            const int delay_samples = filter_len / 2;
            const size_t buffer_length = BufferLength(src_length, delay_samples);
            bakuage::AlignedPodVector<float> temp_input(buffer_length);
            bakuage::AlignedPodVector<float> temp_output(buffer_length);
            for (int ch = 0; ch < channels; ch++) {
                fir_filter.Clear();
                for (size_t i = 0; i < src_length; i++) {
                    temp_input[i] = (*wave)[channels * i + ch];
                }
                fir_filter.Clock(temp_input.data(), temp_input.data() + src_length + delay_samples, temp_output.data());
                for (size_t i = 0; i < dest_length; i++) {
                    (*wave)[channels * i + ch] = temp_output[i * n + delay_samples];
                }
            }
        }

        wave->resize(channels * dest_length);
    }

}
