#ifndef BAKUAGE_AUDIO_ANALYZER_NMF_SPECTROGRAM_H_
#define BAKUAGE_AUDIO_ANALYZER_NMF_SPECTROGRAM_H_

#include <cmath>
#include <cstring>
#include <limits>
#include <vector>
#include <algorithm>

#include "bakuage/memory.h"
#include "bakuage/mfcc.h"
#include "bakuage/utils.h"
#include "bakuage/window_func.h"
#include "bakuage/nmf.h"
#include "CImg.h"

namespace audio_analyzer {

    template <typename Float>
    void WriteNmfSpectrogramPng(Float *input, int channels, int samples, int sample_freq, int image_height, const char *output_path) {
        using namespace cimg_library;
        using namespace bakuage;

        // calculate mel spectrum
        int num_filters = image_height;
        bakuage::MfccCalculator<float> mfcc_calculator(sample_freq, 0, 22000, num_filters);
        int width = 2 * static_cast<int>(16384 * sample_freq / 44100 / 2);
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

        Eigen::MatrixXd v(num_filters, count);
        for (int i = 0; i < count; i++) {
            for (int j = 0; j < num_filters; j++) {
                v(j, i) = src_mid_mel_bands[num_filters * i + j];
                std::cerr << v(j, i) << std::endl;
            }
        }
        Eigen::MatrixXd w, h;
        int k = 1;//num_filters / 4;
        bakuage::Nmf(v, k, 1000, &w, &h);

        std::vector<float> output;
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < num_filters; j++) {
                std::cerr << w(j, i) << std::endl;
                output.push_back(w(j, i));
            }
        }

        const int height = num_filters;
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

