#include "phase_limiter/auto_mastering.h"

#include <cmath>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <chrono>
#include <string>
#include <random>
#include <stdexcept>
#include <algorithm>
#include <fstream>
#include <streambuf>
#include <thread>
#include <mutex>
#include "gflags/gflags.h"
#include "picojson.h"

#include "audio_analyzer/reverb.h"
#include "audio_analyzer/peak.h"
#include "bakuage/loudness_ebu_r128.h"
#include "audio_analyzer/multiband_histogram.h"
#include "audio_analyzer/statistics.h"
#include "bakuage/convolution.h"
#include "bakuage/sndfile_wrapper.h"
#include "bakuage/compressor_filter.h"
#include "bakuage/channel_wise_compressor_filter.h"
#include "bakuage/ms_compressor_filter.h"
#include "bakuage/file_utils.h"
#include "bakuage/utils.h"
#include "bakuage/fir_design.h"
#include "bakuage/fir_filter.h"
#include "bakuage/ffmpeg.h"
#include "bakuage/window_func.h"
#include "bakuage/mfcc.h"
#include "bakuage/dct.h"

DECLARE_string(mastering2_config_file);

typedef float Float;
using namespace bakuage;

namespace {

void raise(const std::string &message) {
	throw std::logic_error("auto mastering2 error: " + message);
}

class Band {
public:
	Band(const picojson::object &band) {
		using namespace picojson;
		gain = band.at("gain").get<double>();
		mean = band.at("mean").get<double>();
		ratio = band.at("ratio").get<double>();
		threshold_relative_to_mean = band.at("threshold_relative_to_mean").get<double>();
	}
	Float gain;
	Float mean;
	Float ratio;
	Float threshold_relative_to_mean;
};

enum class Mastering2Mode {
	kBand = 1,
	kMfcc = 2,
};

class Mastering2Config {
public:
	/*
	kBandの場合
	bandsの長さは40

	kMfccの場合
	40band -> 13coefで、bandsの長さは13
	*/
	Mastering2Config(const std::string &reference_file_path) {
		using namespace picojson;
		value v;
		std::ifstream reference_file(reference_file_path.c_str());
		std::string json_str((std::istreambuf_iterator<char>(reference_file)),
			std::istreambuf_iterator<char>());
		std::string err = parse(v, json_str);
		if (!err.empty()) {
			raise(err);
		}
		object root = v.get<object>();

		auto mode_str = root.at("mode").get<std::string>();
		if (mode_str == "band") {
			mode = Mastering2Mode::kBand;
		}
		else if (mode_str == "mfcc") {
			mode = Mastering2Mode::kMfcc;
		}
		else {
			raise("unknown mode " + mode_str);
		}

		array bands_json = root.at("bands").get<array>();
		for (const auto &band_json : bands_json) {
			bands.push_back(Band(band_json.get<object>()));
		}
	}

	Mastering2Mode mode;
	std::vector<Band> bands;
};
}

namespace phase_limiter {

void AutoMastering2(std::vector<float> *_wave, const int sample_rate, const std::function<void(float)> &progress_callback) {
#ifdef PHASELIMITER_ENABLE_FFTW
	Mastering2Config mastering2_config(FLAGS_mastering2_config_file);

	const int frames = _wave->size() / 2;
	const int channels = 2;
	const float *wave_ptr = &(*_wave)[0];

	std::mutex task_mtx;
	std::mutex result_mtx;
	std::mutex progression_mtx;
	std::vector<std::function<void()>> tasks;
	std::vector<Float> result(_wave->size());

	/*
	処理概要

	1. STFT
	2. calculate mel band or MFCC
	3. calculate gain for each band or each MFCC
	4. gainを1の結果に適用
	5. overlap addで出力

	*/

	int width = 2048;
	int shift = 1024;
	int num_filters = 40;
	int mfcc_len = 13;
	int pos = -width + shift;
	int spec_len = width / 2 + 1;
	std::vector<std::vector<std::complex<float>>> complex_spec(channels, std::vector<std::complex<float>>(spec_len));
	std::vector<std::complex<float>> complex_spec_mono(spec_len);
	std::vector<float> mel_bands_mono(num_filters);
	std::vector<float> mfcc_mono(num_filters);
	std::vector<float> mfcc_gains_mono(num_filters);
	std::vector<float> band_gains_mono(num_filters);
	std::vector<float> spec_gains_mono(spec_len);
	std::vector<float> window(width);
	bakuage::CopyHanning(width, window.begin());
	bakuage::MfccCalculator<float> mfcc_calculator(sample_rate, 0, 11000, num_filters);
	bakuage::Dct dct(num_filters);
	double *fft_input;
	fftw_complex * fft_output;
	fftw_plan plan;
	fftw_plan inv_plan;
	{
		std::lock_guard<std::recursive_mutex> lock(FFTW::mutex());

		fft_input = (double *)fftw_malloc(sizeof(double) * width);
		std::fill_n(fft_input, width, 0);
		fft_output = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * spec_len);
		std::fill_n((double *)fft_output, 2 * spec_len, 0);
		plan = fftw_plan_dft_r2c_1d(width, fft_input, fft_output, FFTW_ESTIMATE);
		inv_plan = fftw_plan_dft_c2r_1d(width, fft_output, fft_input, FFTW_ESTIMATE);
	}
	while (pos < frames) {
		// window and fft
		std::fill_n(complex_spec_mono.data(), spec_len, 0);
		for (int i = 0; i < channels; i++) {
			for (int j = 0; j < width; j++) {
				int k = pos + j;
				fft_input[j] = (0 <= k && k < frames) ? wave_ptr[channels * k + i] * window[j] : 0;
			}
			fftw_execute(plan);
			for (int j = 0; j < spec_len; j++) {
				complex_spec[i][j] = std::complex<float>(fft_output[j][0], fft_output[j][1]);
				complex_spec_mono[j] += complex_spec[i][j];
			}
		}

		// calculate mel band
		mfcc_calculator.calculateMelSpectrumFromDFT((float *)complex_spec_mono.data(), width, true, false, mel_bands_mono.data());
		for (int j = 0; j < num_filters; j++) {
			mel_bands_mono[j] = 10 * std::log10(1e-10 + mel_bands_mono[j]);
		}

		// calculate gain
		if (mastering2_config.mode == Mastering2Mode::kBand) {
			for (int j = 0; j < num_filters; j++) {
				const double threshold = mastering2_config.bands[j].mean + mastering2_config.bands[j].threshold_relative_to_mean;
				if (mel_bands_mono[j] < threshold) {
					band_gains_mono[j] = (threshold - mastering2_config.bands[j].mean) / mastering2_config.bands[j].ratio +
						mastering2_config.bands[j].mean + mastering2_config.bands[j].gain - threshold;
				}
				else {
					band_gains_mono[j] = (mel_bands_mono[j] - mastering2_config.bands[j].mean) / mastering2_config.bands[j].ratio +
						mastering2_config.bands[j].mean + mastering2_config.bands[j].gain - mel_bands_mono[j];
				}
			}
		}
		else if (mastering2_config.mode == Mastering2Mode::kMfcc) {
			// calculate mfcc
			dct.DctType2(mel_bands_mono.data(), mfcc_mono.data());

			// calculate gain in mfcc
			for (int j = 0; j < mfcc_len; j++) {
				const double threshold = mastering2_config.bands[j].mean + mastering2_config.bands[j].threshold_relative_to_mean;
				const double ratio = mastering2_config.bands[j].ratio;
				if (j == 0 && mfcc_mono[j] < threshold) {
					mfcc_gains_mono[j] = (threshold - mastering2_config.bands[j].mean) / ratio +
						mastering2_config.bands[j].mean + mastering2_config.bands[j].gain - threshold;
				}
				else {
					mfcc_gains_mono[j] = (mfcc_mono[j] - mastering2_config.bands[j].mean) / ratio +
						mastering2_config.bands[j].mean + mastering2_config.bands[j].gain - mfcc_mono[j];
				}
			}

			// calculate band gain
			for (int j = 0; j < mfcc_len; j++) {
				mfcc_gains_mono[j] *= 2.0 / num_filters;
			}
			dct.DctType3(mfcc_gains_mono.data(), band_gains_mono.data());
		}

		// apply gain
		mfcc_calculator.calculateSpectrumFromMelSpectrum(band_gains_mono.data(), width, true, false, spec_gains_mono.data());
		for (int i = 0; i < channels; i++) {
			for (int j = 0; j < spec_len; j++) {
				const double scale = bakuage::Pow(10, spec_gains_mono[j] / 20);
				complex_spec[i][j] *= scale;
			}
		}

		// ifft and output
		for (int i = 0; i < channels; i++) {
			for (int j = 0; j < spec_len; j++) {
				fft_output[j][0] = complex_spec[i][j].real();
				fft_output[j][1] = complex_spec[i][j].imag();
			}
			fftw_execute(inv_plan);
			for (int j = 0; j < width; j++) {
				int k = pos + j;
				if (0 <= k && k < frames) {
					result[channels * k + i] += fft_input[j];
				}
			}
		}

		pos += shift;
	}

	{
		std::lock_guard<std::recursive_mutex> lock(FFTW::mutex());

		fftw_destroy_plan(plan);
		fftw_destroy_plan(inv_plan);
		fftw_free(fft_output);
		fftw_free(fft_input);
	}

	*_wave = std::move(result);
#endif
}

}
