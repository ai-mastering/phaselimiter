#include "phase_limiter/freq_expander.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#ifdef PHASELIMITER_ENABLE_FFTW
#include "fftw3.h"
#endif
#include "bakuage/utils.h"

namespace {
#ifdef PHASELIMITER_ENABLE_FFTW
int freq_len(int n) {
	// http://www.fftw.org/fftw3_doc/One_002dDimensional-DFTs-of-Real-Data.html
	return n / 2 + 1;
}
#endif
}

namespace phase_limiter {

// STFT overlap addで処理
void FreqExpand(std::vector<float> *wave, int channels, int sample_rate, float ratio){
#ifdef PHASELIMITER_ENABLE_FFTW
	if (wave->size() % channels) {
		throw std::logic_error("input wave length must be multiple of channels");
	}
	std::vector<float> result(wave->size());
	const int len = wave->size() / channels;
	const int windows_size = (int)(44100 * 0.02) & (~15);
	const int shift_size = windows_size / 2;
	std::vector<double> powers(windows_size / 2 + 1);

	float *fft_input = (float *)fftwf_malloc(sizeof(float) * windows_size);
	std::memset(fft_input, 0, sizeof(float) * windows_size);
	fftwf_complex *fft_output = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * freq_len(windows_size));
	std::memset(fft_output, 0, sizeof(fftwf_complex) * freq_len(windows_size));

	fftwf_plan plan = fftwf_plan_dft_r2c_1d(windows_size, fft_input, fft_output, FFTW_ESTIMATE);
	fftwf_plan inv_plan = fftwf_plan_dft_c2r_1d(windows_size, fft_output, fft_input, FFTW_ESTIMATE);

	// FFTの正規化も行う
	std::vector<float> window(windows_size);
	for (int i = 0; i < windows_size; i++) {
		window[i] = std::sqrt(0.5 - 0.5 * std::cos(2.0 * M_PI * i / windows_size)) / std::sqrt(windows_size);
	}

	float erb_scale = 1;
	std::vector<int> bin_groups;
	bin_groups.push_back(0);
	float next_freq = erb_scale * bakuage::GlasbergErb(0);
	for (int i = 1; i < freq_len(windows_size); i++) {
		float f = 1.0 * i / windows_size * sample_rate;
		if (f > next_freq) {
			bin_groups.push_back(i);
			next_freq = f + erb_scale * bakuage::GlasbergErb(f);
		}
	}

	int pos = -windows_size + shift_size;
	while (pos < len) {
		for (int channel = 0; channel < channels; channel++) {
			for (int j = 0; j < windows_size; j++) {
				int index = pos + j;
				if (index < 0 || len <= index) {
					fft_input[j] = 0;
				}
				else {
					fft_input[j] = (*wave)[channels * index + channel] * window[j];
				}
			}

			fftwf_execute(plan);

			for (int i = 0; i < bin_groups.size() / 2; i++) {
				int group1 = 2 * i;
				int group2 = 2 * i + 1;
				int group1_bg = bin_groups[group1];
				int group1_ed = bin_groups[group2];
				int group2_bg = bin_groups[group2];
				int group2_ed = group2 + 1 < bin_groups.size() ? bin_groups[group2 + 1] : freq_len(windows_size);
				double ener1 = 0;
				double ener2 = 0;
				for (int j = group1_bg; j < group1_ed; j++) {
					ener1 += bakuage::Sqr(fft_output[j][0]) + bakuage::Sqr(fft_output[j][1]);
				}
				for (int j = group2_bg; j < group2_ed; j++) {
					ener2 += bakuage::Sqr(fft_output[j][0]) + bakuage::Sqr(fft_output[j][1]);
				}
				ener1 /= (group1_ed - group1_bg);
				ener2 /= (group2_ed - group2_bg);
				double total_ener = ener1 + ener2;

				double loud1 = 10 * std::log10(1e-37 + ener1);
				double loud2 = 10 * std::log10(1e-37 + ener2);

				double gain1, gain2;
				if ((ratio - 1) * ener1 > (ratio - 1) * ener2) {
					double gain1_to_2 = std::pow(10, (1.0 / 20) * (ratio - 1) * (loud2 - loud1));
					gain1 = std::sqrt(total_ener / (1e-37 + ener1 + bakuage::Sqr(gain1_to_2) * ener2));
					gain2 = gain1 * gain1_to_2;
				}
				else {
					double gain2_to_1 = std::pow(10, (1.0 / 20) * (ratio - 1) * (loud1 - loud2));
					gain2 = std::sqrt(total_ener / (1e-37 + ener2 + bakuage::Sqr(gain2_to_1) * ener1));
					gain1 = gain2 * gain2_to_1;
				}
				for (int j = group1_bg; j < group1_ed; j++) {
					fft_output[j][0] *= gain1;
					fft_output[j][1] *= gain1;
				}
				for (int j = group2_bg; j < group2_ed; j++) {
					fft_output[j][0] *= gain2;
					fft_output[j][1] *= gain2;
				}
			}

			fftwf_execute(inv_plan);

			for (int j = 0; j < windows_size; j++) {
				int index = pos + j;
				if (index < 0 || len <= index) {
				}
				else {
					result[channels * index + channel] += fft_input[j] * window[j];
				}
			}
		}
		pos += shift_size;
	}

	for (int i = 0; i < result.size(); i++) {
		(*wave)[i] = result[i];
	}

	fftwf_destroy_plan(plan);
	fftwf_destroy_plan(inv_plan);
	fftwf_free(fft_input);
	fftwf_free(fft_output);
#endif
}

}
