#ifndef BAKUAGE_AUDIO_ANALYZER_SPECTRUM_DISTRIBUTION_H_
#define BAKUAGE_AUDIO_ANALYZER_SPECTRUM_DISTRIBUTION_H_

#include <cmath>
#include <cstring>
#include <limits>
#include <vector>
#include <algorithm>

#include "CImg.h"
#include "bakuage/memory.h"
#include "bakuage/dft.h"
#include "bakuage/mfcc.h"
#include "bakuage/utils.h"
#include "bakuage/window_func.h"
#include "audio_analyzer/spectrum_distribution_base_image.h"

namespace audio_analyzer {

template <typename Float>
void WriteSpectrumDistributionPng(Float *input, int channels, int samples, int sample_freq, const char *output) {
	using namespace bakuage;
	using namespace cimg_library;

	// calculate mfcc
	const int image_scale = 1;
	const int image_left = 40;
	const int image_top = 28;
	const int image_width = (610 - image_left) / image_scale;
	const int image_height = (348 - image_top) / image_scale;
	const int num_filters = 40;
	const int shift_resolution = 2;
	const int width = shift_resolution * ((16384 * sample_freq / 44100) / shift_resolution); // 0.372 sec, 4x only
	const int shift = width / shift_resolution;
	int pos = -width + shift;
	const int spec_len = width / 2 + 1;
	std::vector<std::complex<float>> complex_spec_mid(spec_len);
	std::vector<std::complex<float>> complex_spec_side(spec_len);
	std::vector<float> window(width);
	bakuage::CopyHanning(width, window.begin());
    float *fft_input = (float *)bakuage::AlignedMalloc(sizeof(float) * width);
    std::complex<float> *fft_output = (std::complex<float> *)bakuage::AlignedMalloc(sizeof(std::complex<float>) * spec_len);
    bakuage::RealDft<float> dft(width);
    
	const float min_hz = 20;
	const float max_hz = 20000;
	const float min_db = -96;
	const float max_db = 0;
	const float pixel_per_db = 1.0 * image_height / (max_db - min_db);
	CImg<float> img(image_width, image_height, 1, 1, 0);
	int count = 0;
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
			const auto scale = std::sqrt(freq / 1000 / std::sqrt(width) * 1e-4); // linear空間なのでsqrt
			complex_spec_mid[j] *= scale;
			complex_spec_side[j] *= scale;
		}

		// draw spectrum
		CImg<int> points(num_filters, 2);

		for (int j = 0; j < num_filters; j++) {
			const double center_freq = min_hz * std::pow(max_hz / min_hz, 1.0 * j / (num_filters - 1));
			const double bg_freq = min_hz * std::pow(max_hz / min_hz, 1.0 * (j - 1) / (num_filters - 1));
			const double ed_freq = min_hz * std::pow(max_hz / min_hz, 1.0 * (j + 1) / (num_filters - 1));
			double sum = 0;
			double sum_weight = 0;
			const int center_k = std::max<int>(0, center_freq / sample_freq * width);
			const int bg = std::max<int>(0, bg_freq / sample_freq * width);
			const int ed = std::min<int>(spec_len, ed_freq / sample_freq * width);
			for (int k = bg; k < ed; k++) {
				const double weight = k < center_k ?
					1 - 1.0 * (center_k - k) / (1e-37 + center_k - bg) :
					1 - 1.0 * (k - center_k) / (1e-37 + ed - center_k);
				sum += (bakuage::Sqr(complex_spec_mid[k].real()) + bakuage::Sqr(complex_spec_mid[k].imag())
					+ bakuage::Sqr(complex_spec_side[k].real()) + bakuage::Sqr(complex_spec_side[k].imag())) * weight;
				sum_weight += weight;
			}
			sum /= 1e-37 + sum_weight;

			const double db = 10 * std::log10(1e-10 + sum);
			points(j, 0) = image_width * (std::log(center_freq) - std::log(min_hz)) / (std::log(max_hz) - std::log(min_hz));
			points(j, 1) = image_height - image_height * (db - min_db) / (max_db - min_db);
		}

		for (int j = 0; j < num_filters - 1; j++) {
			const double scale = 1.0 * (points(j + 1, 1) - points(j, 1)) / (points(j + 1, 0) - points(j, 0));
			for (int x = points(j, 0); x < points(j + 1, 0); x++) {
				if (0 <= x && x < image_width) {
					int y = scale * (x - points(j, 0)) + points(j, 1);
					if (0 <= y && y < image_height) {
						img(x, y) += 1;
					}
				}
			}
		}

		pos += shift;
		count++;
	}

	// blur
	{
		const double rate = std::exp(-1.0 / (1.0 * pixel_per_db));
		for (int x = 0; x < image_width; x++) {
			double sum = 0;
			for (int y = 0; y < image_height; y++) {
				sum = sum * rate + img(x, y) * (1.0 - rate);
				img(x, y) = sum;
			}
			sum = 0;
			for (int y = image_height - 1; y >= 0; y--) {
				sum = sum * rate + img(x, y) * (1.0 - rate);
				img(x, y) = sum;
			}
		}
	}
	if (image_scale != 1) {
		img.resize(image_scale * image_width, image_scale * image_height, 1, 1,
			5 // cubic
		);
	}

	// convert output image
	const float color[] = { 0.525f, 1.0f, 1.0f, 1.0f };
	const double scale = 20 * pixel_per_db / (1e-37 + count);
	CImg<unsigned char> output_img(image_scale * image_width, image_scale * image_height, 1, 4, 0);
	for (int i = 0; i < image_scale * image_width; i++) {
		for (int j = 0; j < image_scale * image_height; j++) {
			const double v = img(i, j) * scale;
			for (int k = 0; k < 4; k++) {
				output_img(i, j, 0, k) = 255 * std::min<double>(1.0, color[k] * v);
			}
		}
	}

	CImg<unsigned char> base_image(SpectrumDistributionBaseImageData(), 640, 387, 1, 3);
	base_image.draw_image(image_left, image_top, output_img, output_img.get_channel(3), 1, 255);

	base_image.save_png(output);
    
    bakuage::AlignedFree(fft_input);
    bakuage::AlignedFree(fft_output);
}

}

#endif 
