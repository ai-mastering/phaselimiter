#ifndef BAKUAGE_MFCC_H_
#define BAKUAGE_MFCC_H_

#include <cmath>
#include "bakuage/utils.h"

namespace bakuage {
template <class T>
class MfccCalculator {
public:
	MfccCalculator(int sample_rate, double min_freq, double max_freq, int num_filters) :
		sample_rate_(sample_rate), min_freq_(min_freq), max_freq_(max_freq), num_filters_(num_filters) {
		min_mel_ = bakuage::HzToMel(min_freq_);
		max_mel_ = bakuage::HzToMel(max_freq_);
	}

	double center_freq(int index) const {
		return calc_center_freq(index + 1);
	}

	/*void calculateMfccFromDFT(const T *input, int fft_size, T *output) const {
		calculateMelSpectrumFromDFT(input, fft_size, output);
		fft_.DCTForward(output);
	}

	void calculateMfccFromSpectrum(const T *input, int fft_size, T *output) const {
		calculateMelSpectrumFromSpectrum(input, fft_size, output);
		fft_.DCTForward(output);
	}*/

	// input: (real, image, real, image, ...)
	// meanだとエネルギー平均をとる。sumだとエネルギー合計をとる。
    template <class InputT>
	void calculateMelSpectrumFromDFT(const InputT input, int fft_size, bool mean, bool cover_all, T *output) const {
		int spec_len = fft_size / 2 + 1;
		std::vector<T> spectrum(spec_len);
		for (int i = 0; i < spec_len; i++) {
			spectrum[i] = Sqr(input[2 * i + 0]) + Sqr(input[2 * i + 1]);
		}
		calculateMelSpectrumFromSpectrum(spectrum.data(), fft_size, mean, cover_all, output);
	}
	// input: power spectrum
    template <class InputT>
	void calculateMelSpectrumFromSpectrum(const InputT input, int fft_size, bool mean, bool cover_all, T *output) const  {
		int input_len = fft_size / 2 + 1;

		for (int i = 0; i < num_filters_; i++) {
			const double center_freq = calc_center_freq(i);
			const double center_freq_prev = calc_center_freq(i - 1);
			const double center_freq_next = calc_center_freq(i + 1);
			// indexは少しはみだすようにする
			const int bg_index = std::max<int>(0, std::floor(fft_size * center_freq_prev / sample_rate_));
			const int ed_index = std::min<int>(input_len, std::ceil(fft_size * center_freq_next / sample_rate_));
			double sum = 0;
			double sum_weight = 0;
			for (int j = bg_index; j < ed_index; j++) {
				const double freq = sample_rate_ * (double)j / fft_size;
				double weight = calc_weight(freq, center_freq_prev, center_freq, center_freq_next);
				if (cover_all) {
					if (i == 0 && freq <= center_freq) weight = 1;
					if (i == num_filters_ - 1 && center_freq <= freq) weight = 1;
				}
				sum += input[j] * weight;
				sum_weight += weight;
			}
			output[i] = mean ? sum / (1e-100 + sum_weight) : sum;
		}
	}

    template <class InputT>
	void calculateSpectrumFromMelSpectrum(const InputT input, int fft_size, bool mean, bool cover_all, T *output) const {
		int output_len = fft_size / 2 + 1;

		std::fill_n(output, output_len, 0);

		for (int i = 0; i < num_filters_; i++) {
			const double center_freq = calc_center_freq(i);
			const double center_freq_prev = calc_center_freq(i - 1);
			const double center_freq_next = calc_center_freq(i + 1);
			// indexは少しはみだすようにする
			const int bg_index = std::max<int>(0, std::floor(fft_size * center_freq_prev / sample_rate_));
			const int ed_index = std::min<int>(output_len, std::ceil(fft_size * center_freq_next / sample_rate_));
			double sum_weight = 0;
			if (mean) {
				for (int j = bg_index; j < ed_index; j++) {
					const double freq = sample_rate_ * (double)j / fft_size;
					double weight = calc_weight(freq, center_freq_prev, center_freq, center_freq_next);
					if (cover_all) {
						if (i == 0 && freq <= center_freq) weight = 1;
						if (i == num_filters_ - 1 && center_freq <= freq) weight = 1;
					}
					sum_weight += weight;
				}
			}
			for (int j = bg_index; j < ed_index; j++) {
				const double freq = sample_rate_ * (double)j / fft_size;
				double weight = calc_weight(freq, center_freq_prev, center_freq, center_freq_next);
				if (cover_all) {
					if (i == 0 && freq <= center_freq) weight = 1;
					if (i == num_filters_ - 1 && center_freq <= freq) weight = 1;
				}
				const double normalized_weight = mean ? weight / (1e-100 + sum_weight) : weight;
				output[j] += normalized_weight * input[i];
			}
		}
	}
	int num_filters() const {
		return num_filters_;
	}
private:
	// filter_index: [-1, num_filters_]
	double calc_center_freq(int filter_index) const {
		return bakuage::MelToHz(min_mel_ + (max_mel_ - min_mel_) * (1.0 + filter_index) / (1.0 + num_filters_));
	}
	double calc_weight(double freq, double filter_center_freq_prev, double filter_center_freq, double filter_center_freq_next) const {
		// max(0ははみだしたとき用
		if (freq < filter_center_freq) {
			return std::max<double>(0.0, (freq - filter_center_freq_prev) / (filter_center_freq - filter_center_freq_prev));
		}
		else {
			return std::max<double>(0.0, (filter_center_freq_next - freq) / (filter_center_freq_next - filter_center_freq));
		}
	}

	int sample_rate_;
	double min_freq_;
	double max_freq_;
	double min_mel_;
	double max_mel_;
	int num_filters_;
};
}

#endif
