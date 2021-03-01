#ifndef BAKUAGE_AUDIO_ANALYZER_ACOUSTIC_ENTROPY_H_
#define BAKUAGE_AUDIO_ANALYZER_ACOUSTIC_ENTROPY_H_

#include <algorithm>
#include <vector>
#include <functional>
#include <random>

#ifdef __cplusplus
//extern "C" {
#endif
#ifdef __cplusplus
//}
#endif

#include "bakuage/dft.h"
#include "bakuage/memory.h"
#include "bakuage/utils.h"
#include "bakuage/mfcc.h"
#include "bakuage/statistics.h"
#include "bakuage/dct.h"

namespace audio_analyzer {

// 独自のAcousticEntropy計算

// 各種パラメータについて
// Essentia(AcousticBrainz)になるべくあわせる
// mel-spectrum計算パラメータは、http://essentia.upf.edu/documentation/reference/streaming_MFCC.html を参考にした
// コードを追ったが、この値が使われているみたい https://github.com/MTG/essentia/blob/ce060997688818fd3ea4e2429607ee74ba577b2a/src/algorithms/spectral/mfcc.h#L60
// window sizeとshift sizeは？
// ここで切り出されている https://github.com/MTG/essentia/blob/master/src/essentia/utils/extractor_music/MusicLowlevelDescriptors.cpp#L43
// ここにパラメータかいてる https://github.com/MTG/essentia/blob/412dc1c4ad06da855c5fe6d0b9f3acf79278c725/src/algorithms/standard/windowing.h#L40
// いや、これは窓掛けだけか
// こっちが切り出し担当か https://github.com/MTG/essentia/blob/master/src/essentia/utils/extractor_music/MusicLowlevelDescriptors.cpp#L39
// 普通にここに設定書いてた https://github.com/MTG/essentia/blob/7aa9dbb863c882a10f3abdb2ddbb5800ce0e174f/doc/sphinxdoc/streaming_extractor_music.rst
// ただし、window typeはhanningで妥協しよう

// entropy: bit

template <typename Float>
void CalculateAcousticEntropy(const Float *input, const int channels, const int samples, const int sample_freq,
	int *block_samples, Float *entropy, Float *damage) {
	using namespace bakuage;

	const double min_freq = 0;
	const double max_freq = 11000;
	const int num_filters = 40;
	const double block_sec = 2048.0 / 44100;
	const double shift_sec = 1024.0 / 44100;
	const double min_energy = 1e-14; // -140dB
	const double signal_to_noise_db = 20;
	const double noise_db_oct = -4.5;
	const double damage_compensation_db_oct = -3.0;
	const bool use_mfcc = false;

	const int width = (int)(sample_freq * block_sec); // nearest samples
	const int shift = (int)(sample_freq * shift_sec);

	// initialize fftw
    const int spec_len = width / 2 + 1;
    float *fft_input = (float *)bakuage::AlignedMalloc(sizeof(float) * width);
    std::complex<float> *fft_output = (std::complex<float> *)bakuage::AlignedMalloc(sizeof(std::complex<float>) * spec_len);
    bakuage::RealDft<float> dft(width);

	// FFTの正規化も行う (hanning窓)
	std::vector<float> window(width);
	for (int i = 0; i < width; i++) {
		window[i] = (0.5 - 0.5 * std::cos(2.0 * M_PI * i / width)) / std::sqrt(width);
	}

	// initialize mfcc calculator
	MfccCalculator<double> mfcc_calculator(sample_freq, min_freq, max_freq, num_filters);

	// initialize fft for mfcc calc
	bakuage::Dct dct(num_filters);

	// memory layout: [index][left 0, left 1 ..., right 0, right 1, ...]
	std::vector<std::vector<double>> mel_spectrum_dbs;

	int pos = 0;
	while (pos < samples) {
		int end = std::min<int>(pos + width, samples);

		// FFT and calculate mel spectrum
		std::vector<double> mel_spectrum_db(channels * num_filters);
		for (int ch = 0; ch < channels; ch++) {
			for (int i = 0; i < width; i++) {
				fft_input[i] = pos + i < end ? input[channels * (pos + i) + ch] * window[i] : 0;
			}
            dft.Forward(fft_input, (float *)fft_output);
			mfcc_calculator.calculateMelSpectrumFromDFT((float *)fft_output, width, true, false, &(mel_spectrum_db[ch * num_filters]));
		}
		for (int i = 0; i < mel_spectrum_db.size(); i++) {
			mel_spectrum_db[i] = 10 * std::log10(min_energy + mel_spectrum_db[i]);
		}
		mel_spectrum_dbs.emplace_back(std::move(mel_spectrum_db));

		pos += shift;
	}

	// calculate damage
	std::vector<Statistics> energy_stats(num_filters);
	for (int i = 0; i < mel_spectrum_dbs.size(); i++) {
		for (int j = 0; j < num_filters; j++) {
			double energy = 0;
			for (int k = 0; k < channels; k++) {
				energy += bakuage::DbToEnergy(mel_spectrum_dbs[i][k * num_filters + j]);
			}
			energy_stats[j].Add(energy / channels);
		}
	}
	*damage = -1e100;
	for (int j = 0; j < num_filters; j++) {
		double freq = mfcc_calculator.center_freq(j);
		double band_damage = 10 * std::log10(energy_stats[j].mean()) - std::log2(freq / 1000) * damage_compensation_db_oct;
		*damage = std::max<double>(*damage, band_damage);
	}

	// estimate entropy
	// std::vector<const double *> mel_spectrum_db_ptrs;
//	std::vector<int> kdpee_keys(mel_spectrum_dbs.size());
	VectorStatistics stats(channels * num_filters);
	for (int i = 0; i < mel_spectrum_dbs.size(); i++) {
		// mel_spectrum_db_ptrs.push_back(&mel_spectrum_dbs[i][0]);

		std::vector<double> spectrum_db_compensated(channels * num_filters);
		for (int k = 0; k < channels; k++) {
			std::vector<double> spectrum_db_compensated_part(num_filters);
			for (int j = 0; j < num_filters; j++) {
				double freq = mfcc_calculator.center_freq(j);
				double min_db = *damage - signal_to_noise_db + std::log2(freq / 1000) * noise_db_oct;
				double original = mel_spectrum_dbs[i][k * num_filters + j];
				spectrum_db_compensated_part[j] = std::max<double>(min_db, original);
			}
			if (use_mfcc) {
				dct.DctType2Replacing(spectrum_db_compensated_part.data());
				for (int j = 0; j < num_filters; j++) {
					spectrum_db_compensated_part[j] *= std::sqrt(2.0 / num_filters);
				}
				for (int j = 13; j < num_filters; j++) {
					spectrum_db_compensated_part[j] = 0;
				}
			}
			std::copy(spectrum_db_compensated_part.begin(), spectrum_db_compensated_part.end(), spectrum_db_compensated.begin() + k * num_filters);
		}
		stats.Add(spectrum_db_compensated.data());
	}

	// kdpeeだとうまく計算できなかった。
	//*entropy = kdpee(mel_spectrum_db_ptrs.data(), mel_spectrum_db_ptrs.size(), channels * num_filters,
	//	mel_spectrum_db_mins.data(), mel_spectrum_db_maxs.data(), 1.96, kdpee_keys.data()) / std::log(2);

	*entropy = stats.log_determinant(1) / std::log(2);

#if 0
	std::cerr << "acoustic entropy " << *entropy <<" damage " << *damage << " samples " << mel_spectrum_dbs.size() << " dimension " << channels * num_filters << std::endl;
	for (int i = 0; i < stats.dimension(); i++) {
		std::cerr << "acoustic entropy min max " << stats.min_vec()[i] << " " << stats.max_vec()[i] << std::endl;
	}
#endif

	if (block_samples) {
		*block_samples = width;
	}

    bakuage::AlignedFree(fft_input);
    bakuage::AlignedFree(fft_output);
}

}

#endif
