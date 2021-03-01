#ifndef BAKUAGE_AUDIO_ANALYZER_SPECTROGRAM_H_
#define BAKUAGE_AUDIO_ANALYZER_SPECTROGRAM_H_

#include <cmath>
#include <cstring>
#include <limits>
#include <vector>
#include <algorithm>

#include "bakuage/mfcc.h"
#include "bakuage/utils.h"
#include "bakuage/window_func.h"
#include "CImg.h"

namespace audio_analyzer {

template <typename Float>
void WriteSpectrogramPng(Float *input, int channels, int samples, int sample_freq, int image_height, const char *output) {
	using namespace cimg_library;
	using namespace bakuage;

	// calculate mfcc
	int num_filters = image_height;
	bakuage::MfccCalculator<float> mfcc_calculator(sample_freq, 0, 22000, num_filters);
	int shift_resolution = 2;
	int output_shift_resolution = 2;
	int width = output_shift_resolution * ((16384 * sample_freq / 44100) / output_shift_resolution); // 0.372 sec, 4x only
	int shift = width / shift_resolution;
	int output_shift = width / output_shift_resolution;
	int mfcc_len = 13;
	int pos = -width + shift;
	int spec_len = width / 2 + 1;
	std::vector<std::complex<float>> complex_spec_mid(spec_len);
	std::vector<std::complex<float>> complex_spec_side(spec_len);
	std::vector<float> src_mid_mel_bands;
	std::vector<float> src_side_mel_bands;
	std::vector<float> window(width);
	bakuage::CopyHanning(width, window.begin());
    float *fft_input = (float *)bakuage::AlignedMalloc(sizeof(float) * width);
    std::complex<float> *fft_output = (std::complex<float> *)bakuage::AlignedMalloc(sizeof(std::complex<float>) * spec_len);
    bakuage::RealDft<float> dft(width);

	bool mono_mode = true;
	while (pos < samples) {
		// window and fft
		std::fill_n(complex_spec_mid.data(), spec_len, 0);
		std::fill_n(complex_spec_side.data(), spec_len, 0);
		for (int i = 0; i < channels; i++) {
			for (int j = 0; j < width; j++) {
				int k = pos + j;
				fft_input[j] = (0 <= k && k < samples) ? input[channels * k + i] * window[j] : 0;
			}
            dft.Forward(fft_input, (float *)fft_output);
			for (int j = 0; j < spec_len; j++) {
				const auto spec = fft_output[j];
				complex_spec_mid[j] += spec;
				complex_spec_side[j] += spec * (2.0f * i - 1);
			}
		}

		// 3dB/oct スロープ補正
		// mean モードを使うので、ピンクノイズは-3dB/octになることに注意
		for (int j = 0; j < spec_len; j++) {
			const auto freq = 1.0 * j / width * sample_freq;
			const auto scale = std::sqrt(freq); // linear空間なのでsqrt
			complex_spec_mid[j] *= scale;
			complex_spec_side[j] *= scale;
		}

		// calculate mel band (energy mean mode)
		src_mid_mel_bands.resize(src_mid_mel_bands.size() + num_filters);
		src_side_mel_bands.resize(src_side_mel_bands.size() + num_filters);
		mfcc_calculator.calculateMelSpectrumFromDFT((float *)complex_spec_mid.data(),
			width, true, true, &src_mid_mel_bands[src_mid_mel_bands.size() - num_filters]);
		mfcc_calculator.calculateMelSpectrumFromDFT((float *)complex_spec_side.data(),
			width, true, true, &src_side_mel_bands[src_side_mel_bands.size() - num_filters]);

		pos += shift;
	}
	const int count = src_mid_mel_bands.size() / num_filters;

	if (mono_mode) {
		for (int i = 0; i < src_mid_mel_bands.size(); i++) {
			src_mid_mel_bands[i] += src_side_mel_bands[i];
		}
	}

	const double max_band = std::max<double>(*std::max_element(src_mid_mel_bands.begin(), src_mid_mel_bands.end()),
		*std::max_element(src_side_mel_bands.begin(), src_side_mel_bands.end()));
	const double normalize_scale = 1.0 / (1e-37 + max_band);
	const double power =
		0.2;
		// std::log(2) / std::log(10); // energy -> sone

	const int image_width = count;
	const float color[] = { 0.525f, 1.0f, 1.0f, 1.0f };
	CImg<unsigned char> img(image_width, image_height, 1, 3, 0);
	for (int i = 0; i < count; i++) {
		for (int j = 0; j < num_filters; j++) {
			const int x = i;
			const int y = image_height - 1 - j;

			const auto normalized_mid = src_mid_mel_bands[i * num_filters + j] * normalize_scale;
			const auto normalized_side = src_side_mel_bands[i * num_filters + j] * normalize_scale;

			if (mono_mode) {
				const double v = 1.6 * std::pow(normalized_side, power);
				for (int k = 0; k < 3; k++) {
					img(x, y, 0, k) = 255 * std::min<double>(1.0, color[k] * v);
				}
			}
			else {
				img(x, y, 0, 0) = 0;// 255 * 0.5 * (std::pow(normalized_mid, power) + std::pow(normalized_side, power));
				img(x, y, 0, 1) = 255 * std::pow(normalized_side, power);// 255 * std::pow(normalized_mid, power);
				img(x, y, 0, 2) = 255 * std::pow(normalized_mid, power);
			}
		}
	}
	img.save_png(output);

    bakuage::AlignedFree(fft_input);
    bakuage::AlignedFree(fft_output);
}

}

#endif
