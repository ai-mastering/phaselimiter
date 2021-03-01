#ifndef BAKUAGE_AUDIO_ANALYZER_RHYTHM_SPECTROGRAM_H_
#define BAKUAGE_AUDIO_ANALYZER_RHYTHM_SPECTROGRAM_H_

#include <cmath>
#include <cstring>
#include <limits>
#include <vector>
#include <algorithm>

#include "bakuage/memory.h"
#include "bakuage/mfcc.h"
#include "bakuage/utils.h"
#include "bakuage/window_func.h"
#include "CImg.h"

namespace audio_analyzer {

template <typename Float>
void WriteRhythmSpectrogramPng(Float *input, int channels, int samples, int sample_freq, int image_height, const char *output_path) {
	using namespace cimg_library;
	using namespace bakuage;

	// calculate mel spectrum
	int num_filters = image_height;
	bakuage::MfccCalculator<float> mfcc_calculator(sample_freq, 0, 22000, num_filters);
    int width = 0.02 * sample_freq;
	int shift = width / 2;
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

    constexpr bool summarize_freq = true;

    std::vector<float> output;

#if 1
    // calculate short time fft
    const int window2 = 2 * static_cast<int>(4.0 * sample_freq / shift / 2); // 2.0 sec (even sample)
    const int shift2 = window2 / 8;
    bakuage::RealDft<float> dft2(window2);
    bakuage::AlignedPodVector<float> fft_input2(window2);
    bakuage::AlignedPodVector<std::complex<float>> fft_output2(window2 / 2 + 1);
    std::vector<float> window_func2(window2);
    bakuage::CopyHanning(window2, window_func2.begin());
    for (int i = window2 / 2; i < count - window2 / 2 - 1; i += shift2) {
        std::fill_n(fft_input2.begin(), window2, 0);
        for (int j = 0; j < window2; j++) {
            for (int k = 0; k < num_filters; k++) {
                float e = src_mid_mel_bands[num_filters * (i - window2 / 2 + j) + k];
                float e2 = src_mid_mel_bands[num_filters * (i - window2 / 2 + j + 1) + k];
                fft_input2[j] += (std::log10(e2 + 1e-7) - std::log10(e + 1e-7)) * window_func2[j];
            }
        }
        dft2.Forward(fft_input2.data(), (float *)fft_output2.data());

#if 1
        // 差分をとったスパイク自体が倍音成分を含んでしまうので、
        // FFTだとリズムによる倍音なのかスパイクの鋭さによる倍音なのか区別できない。
        // だから、自己相関が良いと思う。
        // 自己相関をとった後で、テンポ不変特徴量はどうやって計算する？
        for (int j = 0; j < window2 / 2 + 1; j++) {
            if (j == 0 || j == window2 / 2) {
                fft_output2[j] = 0;
            } else {
                fft_output2[j] = std::norm(fft_output2[j]);
            }
        }
        dft2.Backward((float *)fft_output2.data(), fft_input2.data());

#if 0
        double l2 = 0;
        double l1 = 0;
        for (int j = 2; j < window2 / 2; j++) {
            l2 += bakuage::Sqr(fft_input2[j]);
            l1 += std::abs(fft_input2[j]);
        }
        std::cerr << i * shift / sample_freq << " " << std::sqrt(l2) / (1e-37 + l1) << std::endl;
#endif

#if 1
        double sum = 0;
        double c = 0;
        for (int j = 10; j < window2 / 2; j++) {
            double hz = sample_freq / (1.0 * j * shift);
            sum += std::abs(fft_input2[j]) * hz * hz * hz;
            c += std::abs(fft_input2[j]) * hz * hz;
        }
        std::cerr << i * shift / sample_freq << " " << sum / (1e-37 + c) << std::endl;
#endif

#endif

#if 1
        output.resize(output.size() + window2);
        for (int j = 0; j < window2; j++) {
            output[output.size() - window2 + j] = fft_input2[j];
        }
#else
        output.resize(output.size() + window2 / 2 + 1);
        for (int j = 0; j < window2 / 2 + 1; j++) {
            if (j < 2) continue;
            // todo real fft compensation
            output[output.size() - (window2 / 2 + 1) + j] = std::norm(fft_output2[j]);
        }
#endif
    }
    const int height =
#if 1
    window2;
#else
    window2 / 2 + 1;
#endif

#else

    // calculate short time 2d fft
    const int window_2d = 2 * static_cast<int>(2.0 * sample_freq / shift / 2); // 2.0 sec (even sample)
    const int shift_2d = window_2d / 8;
    bakuage::Dft2D<float> dft_2d(window_2d, num_filters);
    bakuage::AlignedPodVector<std::complex<float>> fft_input_2d(window_2d * num_filters);
    bakuage::AlignedPodVector<std::complex<float>> fft_output_2d(window_2d * num_filters);
    std::vector<float> window_func_2d(window_2d);
    bakuage::CopyHanning(window_2d, window_func_2d.begin());
    for (int i = window_2d / 2; i < count - window_2d / 2 - 1; i += shift_2d) {
        for (int j = 0; j < window_2d; j++) {
            for (int k = 0; k < num_filters; k++) {
                float e = src_mid_mel_bands[num_filters * (i - window_2d / 2 + j) + k];
                float e2 = src_mid_mel_bands[num_filters * (i - window_2d / 2 + j + 1) + k];
                fft_input_2d[window_2d * k + j] =
#if 0
                e * window_func_2d[j];
#else
                (std::log10(e2 + 1e-7) - std::log10(e + 1e-7)) * window_func_2d[j];
#endif
            }
        }
        dft_2d.Forward((float *)fft_input_2d.data(), (float *)fft_output_2d.data());

        for (int j = 0; j < window_2d; j++) {
            for (int k = 0; k < num_filters; k++) {
                double freq = std::min(j, window_2d - j);
                fft_output_2d[window_2d * k + j] = std::norm(fft_output_2d[window_2d * k + j]) ;//* freq * freq;
            }
        }
        dft_2d.Backward((float *)fft_output_2d.data(), (float *)fft_input_2d.data());

        // 二次元自己相関 (原点を除く)
        if (summarize_freq) {
            output.resize(output.size() + window_2d);
            for (int j = 0; j < window_2d; j++) {
                for (int k = 0; k < num_filters; k++) {
                    if (j == 0) continue;
                    output[output.size() - window_2d + j] += fft_input_2d[window_2d * k + j].real();
                }
            }
        } else {
            output.resize(output.size() + window_2d * num_filters);
            for (int j = 0; j < window_2d; j++) {
                for (int k = 0; k < num_filters; k++) {
                    if (j == 0) continue;
                    output[output.size() - window_2d * num_filters + num_filters * j + k] = fft_input_2d[window_2d * k + j].real();
                }
            }
        }
    }

    const int height = summarize_freq ? window_2d : num_filters;
#endif

    const int image_width = output.size() / height;
	CImg<unsigned char> img(image_width, height, 1, 3, 0);
	for (int i = 0; i < image_width; i++) {
        const double min_output = *std::min_element(output.begin() + i * height, output.begin() + (i + 1) * height);
        const double max_output = *std::max_element(output.begin() + i * height, output.begin() + (i + 1) * height);
		for (int j = 0; j < height; j++) {
			const int x = i;
			const int y = height - 1 - j;

			const auto v = (output[i * height + j] - min_output) / (max_output - min_output);

            img(x, y, 0, 0) = 255 * v;
            img(x, y, 0, 1) = 255 * v;
            img(x, y, 0, 2) = 255 * v;
		}
	}
	img.save_png(output_path);

    bakuage::AlignedFree(fft_input);
    bakuage::AlignedFree(fft_output);
}

}

#endif
