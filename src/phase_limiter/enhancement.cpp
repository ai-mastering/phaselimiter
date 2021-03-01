#include "phase_limiter/equalization.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#ifdef PHASELIMITER_ENABLE_FFTW
#include "fftw3.h"
#endif
#include "bakuage/fir_design.h"
#include "bakuage/fir_filter.h"
#include "bakuage/time_varying_lowpass_filter.h"

namespace {
#ifdef PHASELIMITER_ENABLE_FFTW
int freq_len(int n) {
	// http://www.fftw.org/fftw3_doc/One_002dDimensional-DFTs-of-Real-Data.html
	return n / 2 + 1;
}
#endif
}

namespace phase_limiter {

// 作りかけ、うまくいかない
/*void Enhance(std::vector<float> *wave, int channels) {
	if (wave->size() % channels) {
		throw std::logic_error("input wave length must be multiple of channels");
	}
	int src_length = wave->size() / channels;
	int src_freq_length = freq_len(src_length);

	float *fft_input = (float *)fftwf_malloc(sizeof(float) * src_length);
	std::memset(fft_input, 0, sizeof(float) * src_length);
	fftwf_complex *fft_output = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * src_freq_length);
	fftwf_complex *fft_spec_temp = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * src_freq_length);
	std::memset(fft_output, 0, sizeof(fftwf_complex) * src_freq_length);
	fftwf_plan forward_plan = fftwf_plan_dft_r2c_1d(src_length, fft_input, fft_output, FFTW_ESTIMATE);
	fftwf_plan backward_plan = fftwf_plan_dft_c2r_1d(src_length, fft_output, fft_input, FFTW_ESTIMATE);

	const float scale = 1.0 / src_length;
	for (int channel = 0; channel < channels; channel++) {
		// fft
		for (int i = 0; i < src_length; i++) {
			fft_input[i] = (*wave)[channels * i + channel];
		}
		fftwf_execute(forward_plan);

		// enhance
		std::memset(fft_spec_temp, 0, sizeof(fftwf_complex) * src_freq_length);
		for (int i = 0; i < src_freq_length - 1; i++) {
			float norm = fft_output[i][0] * fft_output[i][0] + fft_output[i][1] * fft_output[i][1]
				+ fft_output[i + 1][0] * fft_output[i + 1][0] + fft_output[i + 1][1] * fft_output[i + 1][1];
			float scale = 1 / (1e-37 + std::sqrt(norm));

			int j = 2 * i;
			if (j < src_freq_length) {
				fft_spec_temp[j][0] += scale * (fft_output[i][0] * fft_output[i][0] - fft_output[i][1] * (-fft_output[i][1]));
				fft_spec_temp[j][1] += scale * (fft_output[i][0] * (-fft_output[i][1]) + fft_output[i][1] * fft_output[i][0]);
			}
			j = 2 * i + 1;
			if (j < src_freq_length) {
				fft_spec_temp[j][0] += scale * (fft_output[i][0] * fft_output[i + 1][0] - fft_output[i][1] * (-fft_output[i + 1][1]));
				fft_spec_temp[j][1] += scale * (fft_output[i][0] * (-fft_output[i + 1][1]) + fft_output[i][1] * fft_output[i + 1][0]);
			}
		}
		for (int i = 0; i < src_freq_length; i++) {
			fft_output[i][0] = 0.1 * fft_spec_temp[i][0];
			fft_output[i][1] = 0.1 * fft_spec_temp[i][1];
		}

		// ifft
		fftwf_execute(backward_plan);
		for (int i = 0; i < src_length; i++) {
			(*wave)[channels * i + channel] = fft_input[i] * scale;
		}
	}

	fftwf_destroy_plan(forward_plan);
	fftwf_destroy_plan(backward_plan);
	fftwf_free(fft_input);
	fftwf_free(fft_output);
	fftwf_free(fft_spec_temp);
}*/

float distort(float x) {
	// return x + 0.1 * x * x + 0.1 * x * x * x + 0.1 * x * x * x * x;
	const float a = 1;
	return a * x / (1 + std::abs(a * x));
}

void Enhance(std::vector<float> *wave, int channels) {
#ifdef PHASELIMITER_ENABLE_FFTW
	if (wave->size() % channels) {
		throw std::logic_error("input wave length must be multiple of channels");
	}
	int src_length = wave->size() / channels;
	int src_freq_length = freq_len(src_length);

	float *fft_input = (float *)fftwf_malloc(sizeof(float) * src_length);
	float *fft_wave_temp = (float *)fftwf_malloc(sizeof(float) * src_length);
	float *fft_wave_temp2 = (float *)fftwf_malloc(sizeof(float) * src_length);
	std::memset(fft_input, 0, sizeof(float) * src_length);
	std::memset(fft_wave_temp, 0, sizeof(float) * src_length);

	fftwf_complex *fft_output = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * src_freq_length);
	fftwf_complex *fft_spec_temp = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * src_freq_length);
	std::memset(fft_output, 0, sizeof(fftwf_complex) * src_freq_length);
	std::memset(fft_spec_temp, 0, sizeof(fftwf_complex) * src_freq_length);

	fftwf_plan forward_plan = fftwf_plan_dft_r2c_1d(src_length, fft_input, fft_output, FFTW_ESTIMATE);
	fftwf_plan backward_plan = fftwf_plan_dft_c2r_1d(src_length, fft_output, fft_input, FFTW_ESTIMATE);

	const float scale = 1.0 / src_length;
	for (int channel = 0; channel < channels; channel++) {
		// fft
		for (int i = 0; i < src_length; i++) {
			fft_input[i] = (*wave)[channels * i + channel];
		}
		fftwf_execute(forward_plan);

		// calculate filter param
		int delay_samples;
		int lowpass_filter_order = 2;
		float peak = std::min<float>(1.0, 1.0 / (44100.0 * 1 + 1e-30));
		float a = bakuage::TimeVaryingLowpassFilter<float>::CalculateAFromPeak(lowpass_filter_order, peak, &delay_samples);

		// enhance
		std::memcpy(fft_spec_temp, fft_output, sizeof(fftwf_complex) * src_freq_length);
		std::memset(fft_wave_temp, 0, sizeof(float) * src_length);
		for (int64_t k = 0; k < 250/*22050 / 2 - 1000*/; k += 50) {
			int low_freq = std::min<int64_t>(src_freq_length, k * src_length / 44100);
			int hi_freq = std::min<int64_t>(src_freq_length, (k + 50) * src_length / 44100);
			std::cerr << "AAAAA" << k << std::endl;
			std::memset(fft_output, 0, sizeof(fftwf_complex) * src_freq_length);
			for (int i = low_freq; i < hi_freq; i++) {
				fft_output[i][0] = fft_spec_temp[i][0] * scale;
				fft_output[i][1] = fft_spec_temp[i][1] * scale;
			}
			// ifft
			fftwf_execute(backward_plan);

			// sqr -> low pass
			/*for (int i = 0; i < src_length; i++) {
				fft_wave_temp2[i] = fft_input[i];
				fft_input[i] = fft_input[i] * fft_input[i];
			}
			fftwf_execute(forward_plan);
			for (int i = (int64_t)20 * src_length / 44100; i < src_freq_length - 1; i++) {
				fft_output[i][0] = 0;
				fft_output[i][1] = 0;
			}
			fftwf_execute(backward_plan);

			for (int i = 0; i < src_length; i++) {
				const float x = fft_wave_temp2[i];
				fft_input[i] = x * x / (1e-37 + std::sqrt(std::max<float>(1e-37, fft_input[i] * scale)));
			}

			fftwf_execute(forward_plan);
			for (int i = 0; i < low_freq; i++) {
				fft_output[i][0] = 0;
				fft_output[i][1] = 0;
			}
			for (int i = 2 * hi_freq; i < src_freq_length; i++) {
				fft_output[i][0] = 0;
				fft_output[i][1] = 0;
			}
			fftwf_execute(backward_plan);
			for (int i = 0; i < src_length; i++) {
				fft_wave_temp[i] += fft_input[i] * scale;
			}*/

			bakuage::TimeVaryingLowpassFilter<float> lowpass_filter(lowpass_filter_order, a);
			for (int i = 0; i < src_length + delay_samples; i++) {
				const float x = i < src_length ? fft_input[i] : 0;
				const float y = lowpass_filter.Clock(x * x);
				if (i - delay_samples >= 0) {
					const float scale2 = 1e-18 + std::sqrt(y);
					const float x2 = fft_input[i - delay_samples];
					const float n_x2 = x2 / scale2; // normalized x2
					fft_input[i - delay_samples] = scale2 * distort(n_x2);
				}
			}
			fftwf_execute(forward_plan);
			for (int i = 0; i < low_freq; i++) {
				fft_output[i][0] = 0;
				fft_output[i][1] = 0;
			}
			for (int i = 4 * hi_freq; i < src_freq_length; i++) {
				fft_output[i][0] = 0;
				fft_output[i][1] = 0;
			}
			fftwf_execute(backward_plan);
			for (int i = 0; i < src_length; i++) {
				fft_wave_temp[i] += fft_input[i] * scale;
			}
		}

		std::cerr << "bbbbbbbb" << std::endl;
		for (int i = 0; i < src_length; i++) {
			(*wave)[channels * i + channel] = fft_wave_temp[i];
		}
	}
	std::cerr << "ccccccc" << std::endl;

	fftwf_destroy_plan(forward_plan);
	fftwf_destroy_plan(backward_plan);
	fftwf_free(fft_input);
	fftwf_free(fft_wave_temp);
	fftwf_free(fft_wave_temp2);
	fftwf_free(fft_output);
	fftwf_free(fft_spec_temp);
	std::cerr << "ddddddd" << std::endl;
#endif
}

}
