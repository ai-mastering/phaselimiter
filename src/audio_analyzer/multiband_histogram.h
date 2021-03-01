#ifndef BAKUAGE_AUDIO_ANALYZER_MULTIBAND_HISTOGRAM_H_
#define BAKUAGE_AUDIO_ANALYZER_MULTIBAND_HISTOGRAM_H_

#include <algorithm>
#include <vector>
#include <functional>

#include "bakuage/dft.h"
#include "bakuage/memory.h"
#include "bakuage/loudness_filter.h"
#include "bakuage/ms_compressor_filter.h"
#include "bakuage/utils.h"

namespace audio_analyzer {

template <typename Float>
class Band {
public:
	Float low_freq;
	Float high_freq;
	// ver 1
	Float loudness;
	Float loudness_range;
	Float mid_to_side_loudness;
	Float mid_to_side_loudness_range;
	//std::vector<int> histogram;
	//std::vector<int> mid_to_side_histogram;

	// ver 2
	Float mid_mean;
	Float side_mean;
};

template <typename Float>
std::vector<Band<Float>> CreateBandsByErb(int sample_rate, Float erb_scale) {
	std::vector<Band<Float>> bands;

	Float prev_freq = 0;
	while (1) {
		Float next_freq = prev_freq + erb_scale * bakuage::GlasbergErb(prev_freq);

		// 最後が短いときはスキップ
		if (next_freq >= sample_rate / 2) {
			if ((sample_rate / 2 - prev_freq) / (next_freq - prev_freq) < 0.5) {
				break;
			}
		}

		Band<Float> band = { 0 };
		band.low_freq = prev_freq;
		band.high_freq = next_freq;
		bands.push_back(band);

		if (next_freq >= sample_rate / 2) {
			break;
		}
		prev_freq = next_freq;
	}
	
	return bands;
}

template <typename Float>
Float Mean(const std::vector<Float> &v) {
	double sum = 0;
	for (int i = 0; i < v.size(); i++) {
		sum += v[i];
	}
	return sum / (1e-37 + v.size());
}

// ラウドネス規格: BS.1770
// Loudness Range http://www.abma-bvam.be/PDF/EBU_PLOUD/EBU_tech3342.pdf
// http://jp.music-group.com/TCE/Tech/LRA.pdf
// block_sec = 3, shift_sec = 2, relative_threshold_db = -20

// CompressorFilterのAnalyzeでやる方法は正確だけど遅い
// 遅いとユーザー体験が悪くなるし、実験のイテレーションも悪くなる
// だから、STFTを使った解析をする。
// BS.1770を少し拡張した感じ(bandが一つのときにBS.1770と等価になるようにする <- 窓関数かけないとダメだ)

template <typename Float>
void CalculateMultibandLoudness(const Float *input, const int channels, const int samples, const int sample_freq,
	const Float block_sec, const Float shift_sec, const Float relative_threshold_db,
	Band<Float> *bands, const int band_count, int *block_samples) {
	using namespace bakuage;

	// 400ms block
	int width = (int)(sample_freq * block_sec); // nearest samples
	int shift = (int)(sample_freq * shift_sec);

	std::vector<Float> filtered(channels * samples);
	std::vector<LoudnessFilter<double>> filters;
	for (int i = 0; i < channels; i++) {
		LoudnessFilter<double> filter(sample_freq);
		for (int j = 0; j < samples; j++) {
			int k = channels * j + i;
			filtered[k] = filter.Clock(input[k]);
		}
	}

	std::vector<std::vector<Float>> blocks(band_count);
	std::vector<std::vector<Float>> mid_to_side_blocks(band_count);

    const int spec_len = width / 2 + 1;
    float *fft_input = (float *)bakuage::AlignedMalloc(sizeof(float) * width);
	std::vector<std::complex<float> *> fft_outputs(channels);
    for (int ch = 0; ch < channels; ch++) {
        fft_outputs[ch] = (std::complex<float> *)bakuage::AlignedMalloc(sizeof(std::complex<float>) * spec_len);
    }
    bakuage::RealDft<float> dft(width);

	// FFTの正規化も行う (sqrt(hanning)窓)
	std::vector<float> window(width);
	for (int i = 0; i < width; i++) {
		window[i] = std::sqrt(0.5 - 0.5 * std::cos(2.0 * M_PI * i / width)) / std::sqrt(width);
	}

	int pos = 0;
	// 規格では最後のブロックは使わないけど、
	// 使ったほうが実用的なので使う
	while (pos < samples) {
		int end = std::min<int>(pos + width, samples);

		// FFT
		for (int ch = 0; ch < channels; ch++) {
			for (int i = 0; i < width; i++) {
				fft_input[i] = pos + i < end ? filtered[channels * (pos + i) + ch] * window[i] : 0;
			}
            dft.Forward(fft_input, (float *)fft_outputs[ch]);
		}

		// binをbandに振り分けていく
		for (int band_index = 0; band_index < band_count; band_index++) {
			int low_bin_index = std::floor(width * bands[band_index].low_freq / sample_freq);
			int high_bin_index = std::min<int>(std::floor(width * bands[band_index].high_freq / sample_freq), spec_len);

			// total
			double sum = 0;
			for (int ch = 0; ch < channels; ch++) {
				for (int i = low_bin_index; i < high_bin_index; i++) {
                    sum += std::norm(fft_outputs[ch][i]);
				}
			}
			double z = -0.691 + 10 * std::log10(1e-37 + sum / (0.5 * width)); // 0.5は窓関数の分
			blocks[band_index].push_back(z);

			// -70 <-> [-70, -69)
			/*int index = std::floor(z) + 70;
			if (0 <= index && index < histo.size()) {
			histo[index]++;
			}*/

			// mid to side
			if (channels == 2) {
				double mid_sum = 0;
				double side_sum = 0;
				for (int i = low_bin_index; i < high_bin_index; i++) {
                    mid_sum += std::norm(fft_outputs[0][i] + fft_outputs[1][i]);
                    side_sum += std::norm(fft_outputs[0][i] - fft_outputs[1][i]);
				}
				z = 10 * std::log10(1e-37 + side_sum / (1e-37 + mid_sum));
				mid_to_side_blocks[band_index].push_back(z);
			}
		}

		// 75% overlap
		pos += shift;
	}

	for (int band_index = 0; band_index < band_count; band_index++) {
		for (int calc_index = 0; calc_index < 2; calc_index++) {
			const auto &band_blocks = calc_index == 0 ? blocks[band_index] : mid_to_side_blocks[band_index];

			double threshold = -70;
			for (int k = 0; k < 2; k++) {
				double count = 0;
				double sum = 0;
				for (double z : band_blocks) {
					if (z < threshold) continue;
					count++;
					sum += z;
				}

				double mean = sum / (1e-37 + count);
				if (k == 0) {
					threshold = mean + relative_threshold_db;
				}
				else if (k == 1) {
					if (calc_index == 0) {
						bands[band_index].loudness = mean;
					}
					else {
						bands[band_index].mid_to_side_loudness = mean;
					}
				}
			}

			// loudness range
			std::vector<Float> sorted_blocks;
			for (double z : band_blocks) {
				if (z < threshold) continue;
				sorted_blocks.push_back(z);
			}
			std::sort(sorted_blocks.begin(), sorted_blocks.end());

			double q10 = 0;
			for (int i = 0; i < sorted_blocks.size(); i++) {
				if (10 * sorted_blocks.size() <= 100 * i) {
					q10 = sorted_blocks[i];
					break;
				}
			}
			double q95 = 0;
			for (int i = 0; i < sorted_blocks.size(); i++) {
				if (95 * sorted_blocks.size() <= 100 * i) {
					q95 = sorted_blocks[i];
					break;
				}
			}
			if (calc_index == 0) {
				bands[band_index].loudness_range = q95 - q10;
			}
			else {
				bands[band_index].mid_to_side_loudness_range = q95 - q10;
			}
		}
	}

	if (block_samples) {
		*block_samples = width;
	}
    
    bakuage::AlignedFree(fft_input);
    for (int ch = 0; ch < channels; ch++) {
        bakuage::AlignedFree(fft_outputs[ch]);
    }
}

// stereo only
template <typename Float>
void CalculateMultibandLoudness2(const Float *input, const int channels, const int samples, const int sample_freq,
	const Float block_sec, const Float shift_sec, const Float relative_threshold_db,
	Band<Float> *bands, const int band_count, std::vector<std::vector<Float>> *covariance, int *block_samples) {
	using namespace bakuage;

	assert(channels == 2);

	// 400ms block
	int width = (int)(sample_freq * block_sec); // nearest samples
	int shift = (int)(sample_freq * shift_sec);

	std::vector<Float> filtered(channels * samples);
	std::vector<LoudnessFilter<double>> filters;
	for (int i = 0; i < channels; i++) {
		LoudnessFilter<double> filter(sample_freq);
		for (int j = 0; j < samples; j++) {
			int k = channels * j + i;
			filtered[k] = filter.Clock(input[k]);
		}
	}

    const int spec_len = width / 2 + 1;
    float *fft_input = (float *)bakuage::AlignedMalloc(sizeof(float) * width);
    std::vector<std::complex<float> *> fft_outputs(channels);
    for (int ch = 0; ch < channels; ch++) {
        fft_outputs[ch] = (std::complex<float> *)bakuage::AlignedMalloc(sizeof(std::complex<float>) * spec_len);
    }
    bakuage::RealDft<float> dft(width);

	// FFTの正規化も行う (sqrt(hanning)窓)
	std::vector<float> window(width);
	for (int i = 0; i < width; i++) {
		window[i] = std::sqrt(0.5 - 0.5 * std::cos(2.0 * M_PI * i / width)) / std::sqrt(width);
	}

	int pos = 0;
	std::vector<std::vector<Float>> mid_blocks(band_count);
	std::vector<std::vector<Float>> side_blocks(band_count);
	// 規格では最後のブロックは使わないけど、
	// 使ったほうが実用的なので使う
	while (pos < samples) {
		int end = std::min<int>(pos + width, samples);

		// FFT
		for (int ch = 0; ch < channels; ch++) {
			for (int i = 0; i < width; i++) {
				fft_input[i] = pos + i < end ? filtered[channels * (pos + i) + ch] * window[i] : 0;
			}
            dft.Forward(fft_input, (float *)fft_outputs[ch]);
		}

		// binをbandに振り分けていく
		for (int band_index = 0; band_index < band_count; band_index++) {
			int low_bin_index = std::floor(width * bands[band_index].low_freq / sample_freq);
			int high_bin_index = std::min<int>(std::floor(width * bands[band_index].high_freq / sample_freq), spec_len);

			// mid
			double sum = 0;
			for (int i = low_bin_index; i < high_bin_index; i++) {
                sum += std::norm(fft_outputs[0][i] + fft_outputs[1][i]) ;
			}
			mid_blocks[band_index].push_back(10 * std::log10(1e-7//1e-37 
				+ sum / (0.5 * width)));
			// side
			sum = 0;
			for (int i = low_bin_index; i < high_bin_index; i++) {
                sum += std::norm(fft_outputs[0][i] - fft_outputs[1][i]);
			}
			side_blocks[band_index].push_back(10 * std::log10(1e-7//1e-37 
				+ sum / (0.5 * width)));
		}

		// 75% overlap
		pos += shift;
	}
	
	// calculate mean
	std::vector<Float> mid_threshold(band_count);
	std::vector<Float> side_threshold(band_count);
	for (int band_index = 0; band_index < band_count; band_index++) {
		for (int calc_index = 0; calc_index < 2; calc_index++) {
			const auto &band_blocks = calc_index == 0 ? mid_blocks[band_index] : side_blocks[band_index];

			double threshold = -1e10;//-70;
			for (int k = 0; k < 2; k++) {
				double count = 0;
				double sum = 0;
				for (double z : band_blocks) {
                    const bool valid = z >= threshold;
					count += valid;
					sum += valid * z;
				}

				double mean = sum / (1e-37 + count);
				if (k == 0) {
					threshold = mean + relative_threshold_db;
					if (calc_index == 0) {
						mid_threshold[band_index] = threshold;
					}
					else {
						side_threshold[band_index] = threshold;
					}
				}
				else if (k == 1) {
					if (calc_index == 0) {
						bands[band_index].mid_mean = mean;
					}
					else {
						bands[band_index].side_mean = mean;
					}
				}
			}
		}
	}
	
	// calculate covariance
	covariance->clear();
	for (int i = 0; i < 2 * band_count; i++) {
		covariance->emplace_back(std::vector<Float>(2 * band_count));
	}
	for (int band_index1 = 0; band_index1 < band_count; band_index1++) {
		for (int is_side1 = 0; is_side1 < 2; is_side1++) {
			for (int band_index2 = 0; band_index2 < band_count; band_index2++) {
				for (int is_side2 = 0; is_side2 < 2; is_side2++) {
					int row = 2 * band_index1 + is_side1;
					int col = 2 * band_index2 + is_side2;

					const double mean1 = is_side1 ? bands[band_index1].side_mean : bands[band_index1].mid_mean;
					const double mean2 = is_side2 ? bands[band_index2].side_mean : bands[band_index2].mid_mean;
					const double threshold1 = is_side1 ? side_threshold[band_index1] : mid_threshold[band_index1];
					const double threshold2 = is_side2 ? side_threshold[band_index2] : mid_threshold[band_index2];
					
					const auto &band_blocks1 = is_side1 ? side_blocks[band_index1] : mid_blocks[band_index1];
					const auto &band_blocks2 = is_side2 ? side_blocks[band_index2] : mid_blocks[band_index2];
					
					double v = 0;
					double c = 0;
					for (int i = 0; i < band_blocks1.size(); i++) {
                        const double x1 = band_blocks1[i];
                        const double x2 = band_blocks2[i];
                        const bool valid = (x1 >= threshold1) & (x2 >= threshold2); // not && instead & for performance
						v += valid * (x1 - mean1) * (x2 - mean2);
						c += valid;
					}
					(*covariance)[row][col] += v / (1e-37 + c);
				}
			}
		}
	}

	if (block_samples) {
		*block_samples = width;
	}
    
    bakuage::AlignedFree(fft_input);
    for (int ch = 0; ch < channels; ch++) {
        bakuage::AlignedFree(fft_outputs[ch]);
    }
}

}

#endif
