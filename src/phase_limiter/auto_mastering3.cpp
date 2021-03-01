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
#include <numeric>
#include "gflags/gflags.h"
#include "picojson.h"

#include "audio_analyzer/reverb.h"
#include "audio_analyzer/peak.h"
#include "bakuage/loudness_ebu_r128.h"
#include "audio_analyzer/multiband_histogram.h"
#include "audio_analyzer/statistics.h"
#include "bakuage/convolution.h"
#include "bakuage/compressor_filter.h"
#include "bakuage/channel_wise_compressor_filter.h"
#include "bakuage/ms_compressor_filter.h"
#include "bakuage/file_utils.h"
#include "bakuage/utils.h"
#include "bakuage/fir_design.h"
#include "bakuage/fir_filter.h"
#include "bakuage/window_func.h"
#include "bakuage/mfcc.h"
#include "bakuage/dct.h"
#include "bakuage/statistics.h"
#include "bakuage/mastering3_score.h"

DECLARE_int32(mastering3_iteration);
DECLARE_double(mastering3_target_sn);

typedef float Float;
using namespace bakuage;

namespace {

constexpr int verbose = 0;
constexpr int verbose2 = 0;
constexpr int verbose3 = 0;

struct State {
	int param_count() const {
		return 4 * comp_band_count;
	}
	int comp_band_count;
	std::vector<float> compressor_ratios;
	std::vector<float> compressor_thresholds;
	std::vector<float> compressor_wets;
	std::vector<float> compressor_gains;
	// std::vector<float> eq_gains;
	std::vector<float> mfcc_ratios;
};

// eval_funcを最小にするようなStateを探す
template <class InitializeState, class GenerateNeighbor, class Evaluate, class State, class Progress>
void SimulatedAnnealing(const InitializeState &initialize_state, const GenerateNeighbor &generate_neighbor,
	const Evaluate &evaluate, int iter_count, double initial_t, double t_rate, const Progress &progress, State *optimum_state, double *optimum_eval) {
	if (verbose) std::cerr << "SimulatedAnnealing start" << std::endl;

	double t = initial_t;
	State current_state, next_state;
	initialize_state(&current_state);
	double current_eval = evaluate(current_state);
	*optimum_state = current_state;
	*optimum_eval = current_eval;
	if (verbose) std::cerr << "SimulatedAnnealing initialized " << current_eval << std::endl;

	std::mt19937 engine(1);
	std::uniform_real_distribution<> uniform_dist(0.0, 1.0);

	const int progress_stride = iter_count / 10;
	for (int i = 0; i < iter_count; i++) {
		generate_neighbor(current_state, &next_state);
		double next_eval = evaluate(next_state);

		// update optimum
		if (next_eval < *optimum_eval) {
			if (verbose) std::cerr << "SimulatedAnnealing optimum updated " << next_eval << std::endl;
			*optimum_state = next_state;
			*optimum_eval = next_eval;
		}

		// update current
		double delta_eval = next_eval - current_eval;
		// double p = std::exp(-(next_eval - current_eval) / t);
		double p = delta_eval <= 0 ? 1 : 2.0 / (1 + std::exp(delta_eval / t));
		if (uniform_dist(engine) < p) {
			current_state = next_state;
			current_eval = next_eval;
		}

		// t = 1.0 / (1 + i);
		t *= t_rate;

		if (i % progress_stride == 0) {
			progress(1.0 * i / iter_count);
		}
	}
}

}

namespace phase_limiter {

void AutoMastering3(std::vector<float> *_wave, const int sample_rate, const std::function<void(float)> &progress_callback) {
#ifdef PHASELIMITER_ENABLE_FFTW
	progress_callback(0);

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

	3の処理1
    複数のmel bandをあわせたコンプバンド単位でコンプレッションを行う。
	各コンプバンドのパラメータ
	・ratio

	3の処理2
	各mel band単位でEQを行う。
	各mel bandのパラメータ
	・gain

	3の処理3
	最終的に各mel bandのゲインが計算されるので、それをlinear spectrumのゲインに変換
	(各mel bandをスケールして足し合わせた場合と等価になるように <- 計算量を減らすため)

	3の処理2後のmel bandに対して評価関数と評価関数計算のためのサブ指標を計算する。
	(CalculateMastering3Score)

	考え方
	Compatibility, Richness -> 両方あわせて、ノイズ下Acoustic Entropy最大化
	Loudness -> リミッターで最大化するから不要
	Softness = -耳ダメージであらわすが、ノイズのゲインを耳ダメージから求めているから、
	　　評価関数というよりも制約に含めている。

	評価関数 = ノイズ下Acoustic Entropy

	*/

	int shift_resolution = 2;
	int output_shift_resolution = 2;
	int width = output_shift_resolution * ((16384 * sample_rate / 44100) / output_shift_resolution); // 0.372 sec, 4x only
	int shift = width / shift_resolution;
	int output_shift = width / output_shift_resolution;
	int num_filters = 40;
	// int mfcc_len = 13;
	int pos = -width + shift;
	int spec_len = width / 2 + 1;
	std::vector<std::complex<float>> complex_spec_mid(spec_len);
	std::vector<std::complex<float>> complex_spec_side(spec_len);
	std::vector<float> src_mid_mel_bands;
	std::vector<float> src_side_mel_bands;
	std::vector<float> window(width);
	bakuage::CopyHanning(width, window.begin());
	bakuage::MfccCalculator<float> mfcc_calculator(sample_rate, 0, 22000, num_filters);
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
	if (verbose) std::cerr << "Mastering3 mel band calculation start" << std::endl;
	while (pos < frames) {
		// window and fft
		std::fill_n(complex_spec_mid.data(), spec_len, 0);
		std::fill_n(complex_spec_side.data(), spec_len, 0);
		for (int i = 0; i < channels; i++) {
			for (int j = 0; j < width; j++) {
				int k = pos + j;
				fft_input[j] = (0 <= k && k < frames) ? wave_ptr[channels * k + i] * window[j] : 0;
			}
			fftw_execute(plan);
			for (int j = 0; j < spec_len; j++) {
				auto spec = std::complex<float>(fft_output[j][0], fft_output[j][1]);
				complex_spec_mid[j] += spec;
				complex_spec_side[j] += spec * (2.0f * i - 1);
			}
		}

		// calculate mel band (energy sum mode)
		src_mid_mel_bands.resize(src_mid_mel_bands.size() + num_filters);
		src_side_mel_bands.resize(src_side_mel_bands.size() + num_filters);
		mfcc_calculator.calculateMelSpectrumFromDFT((float *)complex_spec_mid.data(),
			width, false, true, &src_mid_mel_bands[src_mid_mel_bands.size() - num_filters]);
		mfcc_calculator.calculateMelSpectrumFromDFT((float *)complex_spec_side.data(),
			width, false, true, &src_side_mel_bands[src_side_mel_bands.size() - num_filters]);

		pos += shift;
	}
	const int count = src_mid_mel_bands.size() / num_filters;

	// calculate noise melband (実際のノイズではなく、エネルギー平均)
	std::vector<std::complex<float>> complex_spec_noise(spec_len);
	std::vector<float> noise_mel_bands(num_filters);
	for (int i = 0; i < spec_len; i++) {
		double cutoff_freq = 50;
		double freq = 1.0 * (i + 0.5) / width * sample_rate;
		double noise_energy = std::pow(10, -4.5 * std::log2((cutoff_freq + freq) / 1000.0) * 0.1);
		complex_spec_noise[i] = std::sqrt(noise_energy);
	}
	mfcc_calculator.calculateMelSpectrumFromDFT((float *)complex_spec_noise.data(),
		width, false, true, noise_mel_bands.data());

	// 評価関数のエイリアス
	auto calculate_mastering3_score = [count, &noise_mel_bands, &mfcc_calculator](const float *mid_mel_bands, const float *side_mel_bands,
		float *loudness, float *ear_damage, float *acoustic_entropy_mfcc, float *acoustic_entropy_eigen) {
		auto speaker_compensation_flat = [](double hz) { return 0; };
#if 0
        auto speaker_compensation_typical = [](double hz) {
			if (hz < 200) {
				return 10 * std::log2(hz / 200);
			}
			else {
				return 0.0;
			}
		};
		auto speaker_compensation_phone = [](double hz) {
			if (hz < 500) {
				return -20;
			}
			else if (hz > 10000) {
				return -20;
			}
			else {
				return 0;
			}
		};
#endif

		float loudness_flat = 0, ear_damage_flat = 0, acoustic_entropy_mfcc_flat = 0, acoustic_entropy_eigen_flat = 0, diff_acoustic_entropy_eigen_flat = 0;
		CalculateMastering3Score(mid_mel_bands, side_mel_bands, noise_mel_bands.data(),
			count, mfcc_calculator, FLAGS_mastering3_target_sn, speaker_compensation_flat,
			speaker_compensation_flat, speaker_compensation_flat, &loudness_flat, &ear_damage_flat,
			&acoustic_entropy_mfcc_flat, &acoustic_entropy_eigen_flat, &diff_acoustic_entropy_eigen_flat);
		/*float loudness_phone = 0, ear_damage_phone = 0, acoustic_entropy_mfcc_phone = 0, acoustic_entropy_eigen_phone = 0;
		CalculateMastering3Score(mid_mel_bands, side_mel_bands, noise_mel_bands.data(),
			count, mfcc_calculator, FLAGS_mastering3_target_sn, speaker_compensation_phone, &loudness_phone, &ear_damage_phone,
			&acoustic_entropy_mfcc_phone, &acoustic_entropy_eigen_phone);*/

		*loudness = loudness_flat;
		*ear_damage = ear_damage_flat;
		*acoustic_entropy_mfcc = acoustic_entropy_mfcc_flat;
		*acoustic_entropy_eigen = acoustic_entropy_eigen_flat;
		/**loudness = 0.5 * (loudness_flat + loudness_phone);
		*ear_damage = 0.5 * (ear_damage_flat + ear_damage_phone);
		*acoustic_entropy_mfcc = 0.5 * (acoustic_entropy_mfcc_flat + acoustic_entropy_mfcc_phone);
		*acoustic_entropy_eigen = 0.5 * (acoustic_entropy_eigen_flat + acoustic_entropy_eigen_phone);*/
	};

	// before評価関数の計算
	float score_loudness_before;
	float score_ear_damage_before;
	float score_acoustic_entropy_mfcc_before;
	float score_acoustic_entropy_eigen_before;
	calculate_mastering3_score(src_mid_mel_bands.data(), src_side_mel_bands.data(),
		&score_loudness_before, &score_ear_damage_before,
		&score_acoustic_entropy_mfcc_before, &score_acoustic_entropy_eigen_before);

	// 最適化ループ
	if (verbose) std::cerr << "Mastering3 optimization start" << std::endl;
	std::mt19937 engine(std::time(NULL));
	std::normal_distribution<> norm_dist(0.0, 1.0);
	std::uniform_real_distribution<> uniform_dist(0.0, 1.0);
	std::vector<float> output_mid_mel_bands(src_mid_mel_bands.size());
	std::vector<float> output_side_mel_bands(src_side_mel_bands.size());
	std::vector<float> comp_band_border_freqs =
		// { 200, 400, 800, 1600, 3200, 6400, };
		{ 400, 1000, 5000, };
	int comp_band_count = comp_band_border_freqs.size() + 1;
	std::vector<int> comp_band_indicies;
	//comp_band_indicies.push_back(0); comp_band_indicies.push_back(num_filters);
	/*for (int j = 0; j < comp_band_count + 1; j++) {
		comp_band_indicies.push_back(j);
	}*/
	comp_band_indicies.push_back(0);
	for (int j = 0; j < comp_band_border_freqs.size(); j++) {
		int idx = num_filters;
		for (int k = 0; k < num_filters; k++) {
			if (mfcc_calculator.center_freq(k) > comp_band_border_freqs[j]) {
				idx = k;
				break;
			}
		}
		comp_band_indicies.push_back(idx);
	}
	comp_band_indicies.push_back(num_filters);
	if (verbose) {
		std::cerr << "Mastering3 comp_band_indicies ";
		for (int j = 0; j < comp_band_indicies.size(); j++) std::cerr << (int)comp_band_indicies[j] << " ";
		std::cerr << std::endl;
	}
	const int mfcc_comp_count = 4;
	auto initialize_state = [comp_band_count, &norm_dist, &engine](State *state) {
		if (verbose2) std::cerr << "Mastering3 initialize_state" << std::endl;
		state->compressor_ratios.resize(2 * comp_band_count);
		state->compressor_thresholds.resize(2 * comp_band_count); // dB relative to mean energy
		state->compressor_wets.resize(2 * comp_band_count); // 0-1 (for parallel compression)
		state->compressor_gains.resize(2 * comp_band_count);
		for (int j = 0; j < 2 * comp_band_count; j++) {
			state->compressor_ratios[j] = 1; // std::max<double>(1, std::pow(1, uniform_dist(engine)));
			state->compressor_thresholds[j] = -30;// +10 * norm_dist(engine);
			state->compressor_wets[j] = 1; // uniform_dist(engine);
			state->compressor_gains[j] = std::pow(10, 0 * norm_dist(engine) * 0.1);
		}
		//state->compressor_gains[2 * 3 + 0] = 1e-4;
		//state->compressor_gains[2 * 3 + 1] = 1e-4;
		state->mfcc_ratios.resize(2 * mfcc_comp_count);
		for (int j = 0; j < 2 * mfcc_comp_count; j++) {
			state->mfcc_ratios[j] = 1;
		}

		/*state->eq_gains.resize(2 * num_filters); // energy ratio
		for (int j = 0; j < 2 * num_filters; j++) {
			state->eq_gains[j] = std::pow(10, 0 * norm_dist(engine) * 0.1);
		}*/
	};
	auto generate_neighbor = [comp_band_count, &norm_dist, &uniform_dist, &engine](const State &input, State *output) {
		if (verbose2) std::cerr << "Mastering3 generate_neighbor" << std::endl;
		output->compressor_ratios.resize(2 * comp_band_count);
		output->compressor_thresholds.resize(2 * comp_band_count); // dB relative to mean energy
		output->compressor_wets.resize(2 * comp_band_count); // 0-1 (for parallel compression)
		output->compressor_gains.resize(2 * comp_band_count);

		int param_count = 4 * (2 * comp_band_count);
		double scale = uniform_dist(engine) * 1 / std::sqrt(param_count);

		for (int j = 0; j < 2 * comp_band_count; j++) {
			output->compressor_ratios[j] =
				input.compressor_ratios[j] * std::pow(1.1, norm_dist(engine) * scale);
				// std::max<double>(1, input.compressor_ratios[j] * std::pow(1.1, norm_dist(engine) * scale));
			output->compressor_thresholds[j] = input.compressor_thresholds[j] + 1 * norm_dist(engine) * scale;
			output->compressor_wets[j] = std::max<double>(0, std::min<double>(1, input.compressor_wets[j]
				* std::pow(10, 1 * norm_dist(engine) * scale * 0.1)));
			output->compressor_gains[j] = input.compressor_gains[j] * std::pow(10, 1.0 * norm_dist(engine) * scale * 0.1);

			//output->compressor_wets[j] = 1;
			//output->compressor_gains[j] = 1;
		}

		output->mfcc_ratios.resize(2 * mfcc_comp_count);
		for (int j = 0; j < 2 * mfcc_comp_count; j++) {
			output->mfcc_ratios[j] = input.mfcc_ratios[j] * std::pow(1.1, norm_dist(engine) * scale);
		}

		/*output->eq_gains.resize(2 * num_filters); // energy ratio
		for (int j = 0; j < 2 * num_filters; j++) {
			output->eq_gains[j] = input.eq_gains[j] * std::pow(10, 1 * norm_dist(engine) * scale * 0.1);
		}*/
	};
	// detect compressor mean energy
	std::vector<bakuage::Statistics> compressor_energy_stats(2 * comp_band_count);
	std::vector<bakuage::Statistics> compressor_db_stats(2 * comp_band_count);
	for (int i = 0; i < count; i++) {
		for (int j = 0; j < comp_band_count; j++) {
			const int bg_idx = i * num_filters + comp_band_indicies[j];
			const int ed_idx = i * num_filters + comp_band_indicies[j + 1];
			const double mid = std::accumulate(src_mid_mel_bands.begin() + bg_idx, src_mid_mel_bands.begin() + ed_idx, 0.0);
			const double side = std::accumulate(src_side_mel_bands.begin() + bg_idx, src_side_mel_bands.begin() + ed_idx, 0.0);
			if (verbose3) {
				std::cerr << mid << " " << side << std::endl;
			}
			compressor_energy_stats[2 * j + 0].Add(mid);
			compressor_energy_stats[2 * j + 1].Add(side);
			compressor_db_stats[2 * j + 0].Add(10 * std::log10(1e-10 + mid));
			compressor_db_stats[2 * j + 1].Add(10 * std::log10(1e-10 + side));
		}
	}
	std::vector<bakuage::Statistics> mid_mfcc_stats(num_filters);
	std::vector<bakuage::Statistics> side_mfcc_stats(num_filters);
	for (int i = 0; i < count; i++) {
		std::vector<float> log_mid_mel_bands(num_filters);
		std::vector<float> log_side_mel_bands(num_filters);
		for (int j = 0; j < num_filters; j++) {
			log_mid_mel_bands[j] = 10 * std::log10(1e-10 + src_mid_mel_bands[i * num_filters + j]);
			log_side_mel_bands[j] = 10 * std::log10(1e-10 + src_side_mel_bands[i * num_filters + j]);
		}
		std::vector<float> mid_mfcc(num_filters);
		std::vector<float> side_mfcc(num_filters);
		dct.DctType2(log_mid_mel_bands.data(), mid_mfcc.data());
		dct.DctType2(log_side_mel_bands.data(), side_mfcc.data());
		for (int j = 0; j < num_filters; j++) {
			mid_mfcc_stats[j].Add(mid_mfcc[j]);
			side_mfcc_stats[j].Add(side_mfcc[j]);
		}
	}

	auto apply_effects = [&comp_band_indicies, comp_band_count, count,
		num_filters, &compressor_energy_stats](const float *input_mid_mel_bands, const float *input_side_mel_bands,
		const State &state,
		float *output_mid_mel_bands, float *output_side_mel_bands) {
		if (verbose2) std::cerr << "Mastering3 apply_effects" << std::endl;

		auto compressor_gain_func = [](double ratio, double threshold, double mean, double x) {
			double relative_x = std::max<double>(threshold, std::min<double>(-threshold, x - mean));
			return (1.0 / ratio - 1) * relative_x;
		};

		for (int i = 0; i < count; i++) {
			// apply compressor
			for (int j = 0; j < comp_band_count; j++) {
				const int bg_idx = comp_band_indicies[j];
				const int ed_idx = comp_band_indicies[j + 1];
				const double input_mid = std::accumulate(input_mid_mel_bands + bg_idx, input_mid_mel_bands + ed_idx, 0.0);
				const double input_side = std::accumulate(input_side_mel_bands + bg_idx, input_side_mel_bands + ed_idx, 0.0);

				const double mean_mid = 1e-10 + compressor_energy_stats[2 * j + 0].mean();
				const double mean_side = 1e-10 + compressor_energy_stats[2 * j + 1].mean();

				// 合計エネルギーに両側スレッショルドのパラレルコンプ
				const double input_sum = input_mid + input_side;
				const double mean_sum = mean_mid + mean_side;
				const double input_sum_db = 10 * std::log10(1e-10 + input_sum);
				const double mean_sum_db = 10 * std::log10(1e-10 + mean_sum);

				const double sum_gain = (
					std::pow(10, 0.1 * compressor_gain_func(state.compressor_ratios[2 * j + 0],
						state.compressor_thresholds[2 * j + 0], mean_sum_db, input_sum_db))
					* state.compressor_wets[2 * j + 0] + (1 - state.compressor_wets[2 * j + 0])
					) * state.compressor_gains[2 * j + 0];

				// Side to Midにパラレルコンプ (side to midの理由は、パラレルコンプをしたときに、sideを圧縮せずに持ち上げる効果が期待できるから)
				const double side_to_mid_db = 10 * std::log10(std::pow(10, 0.1 * state.compressor_thresholds[2 * j + 1]) + mean_mid / mean_side);
				const double side_to_mid_gain =
					std::pow(10, 0.1 * side_to_mid_db * (1.0 / state.compressor_ratios[2 * j + 1] - 1))
					* state.compressor_wets[2 * j + 1] + (1 - state.compressor_wets[2 * j + 1]);

				// solve following
				// 1: input_mid * mid_gain + input_side * side_gain = sum_gain * (input_mid + input_side)
				// 2: mid_gain = side_to_mid_gain * side_gain
				const double side_gain = sum_gain *(input_mid + input_side) / (input_mid * side_to_mid_gain + input_side);
				const double mid_gain = sum_gain *(input_mid + input_side) / (input_mid + input_side / side_to_mid_gain);

				for (int m = bg_idx; m < ed_idx; m++) {
					output_mid_mel_bands[m] = input_mid_mel_bands[m] * mid_gain;
					output_side_mel_bands[m] = input_side_mel_bands[m] * side_gain;
				}

				/*mean = 1e-10 + compressor_energy_stats[2 * j + 0].mean();
				threshold = mean * std::pow(10, 0.1 * state.compressor_thresholds[2 * j + 0]);
				relative_db = 10 * std::log10((threshold + input_mid) / mean);
				gain_db = relative_db * (1.0 / state.compressor_ratios[2 * j + 0] - 1);
				for (int m = bg_idx; m < ed_idx; m++) {
					output_mid_mel_bands[m] = input_mid_mel_bands[m]
						 *(state.compressor_wets[2 * j + 0] * std::pow(10, 0.1 * gain_db) + (1 - state.compressor_wets[2 * j + 0]));
					output_mid_mel_bands[m] *= state.compressor_gains[2 * j + 0];
				}*/

				/*if (verbose3) {
					std::cerr
						<< input_mid << " "
						<< mean << " "
						<< threshold << " "
						<< relative_db << " "
						<< gain_db << " "
						<< output_mid_mel_bands[j] << " "
						<< input_mid_mel_bands[j] << std::endl;
				}*/

				/*int k = 1;
				mean = 1e-10 + compressor_energy_stats[2 * j + k].mean();
				threshold = mean * std::pow(10, 0.1 * state.compressor_thresholds[2 * j + k]);
				relative_db = 10 * std::log10((threshold + input_side) / mean);
				gain_db = relative_db * (1.0 / state.compressor_ratios[2 * j + k] - 1);
				for (int m = bg_idx; m < ed_idx; m++) {
					output_side_mel_bands[m] = input_side_mel_bands[m]
						 *(state.compressor_wets[2 * j + k] * std::pow(10, 0.1 * gain_db) + (1 - state.compressor_wets[2 * j + k]));
					output_side_mel_bands[m] *= state.compressor_gains[2 * j + k];
				}*/

				/*if (verbose3) {
					std::cerr
						<< input_side << " "
						<< mean << " "
						<< threshold << " "
						<< relative_db << " "
						<< gain_db << " "
						<< output_side_mel_bands[j] << " "
						<< input_side_mel_bands[j] << std::endl;
				}*/
			}

			// MFCC空間でいじる実験
			/*{
				std::vector<float> log_mid_mel_bands(num_filters);
				std::vector<float> log_side_mel_bands(num_filters);
				for (int j = 0; j < num_filters; j++) {
					log_mid_mel_bands[j] = 10 * std::log10(1e-10 + output_mid_mel_bands[j]);
					log_side_mel_bands[j] = 10 * std::log10(1e-10 + output_side_mel_bands[j]);
				}
				std::vector<float> mid_mfcc(num_filters);
				std::vector<float> side_mfcc(num_filters);
				dct.DctType2(log_mid_mel_bands.data(), mid_mfcc.data());
				dct.DctType2(log_side_mel_bands.data(), side_mfcc.data());

				for (int j = 1; j < mfcc_comp_count; j++) {
					mid_mfcc[j] = 1.0 / state.mfcc_ratios[2 * j + 0] * (mid_mfcc[j] - mid_mfcc_stats[j].mean()) + mid_mfcc_stats[j].mean();
					side_mfcc[j] = 1.0 / state.mfcc_ratios[2 * j + 1] * (side_mfcc[j] - side_mfcc_stats[j].mean()) + side_mfcc_stats[j].mean();
				}

				dct.DctType3(mid_mfcc.data(), log_mid_mel_bands.data());
				dct.DctType3(side_mfcc.data(), log_side_mel_bands.data());
				for (int j = 0; j < num_filters; j++) {
					log_mid_mel_bands[j] *= (2.0 / num_filters);
					log_side_mel_bands[j] *= (2.0 / num_filters);
				}

				for (int j = 0; j < num_filters; j++) {
					//std::cerr << output_mid_mel_bands[j] << " " << std::pow(10, 0.1 * log_mid_mel_bands[j]) << " " << input_mid_mel_bands[j] << std::endl;
					output_mid_mel_bands[j] = std::pow(10, 0.1 * log_mid_mel_bands[j]);
					output_side_mel_bands[j] = std::pow(10, 0.1 * log_side_mel_bands[j]);
					//std::cerr << output_mid_mel_bands[j] << " " << std::pow(10, 0.1 * log_mid_mel_bands[j]) << " " << input_mid_mel_bands[j] << std::endl;

				}
			}*/

			// apply eq
			for (int j = 0; j < comp_band_count; j++) {
				// int k = 1;
				//output_mid_mel_bands[j] *= state.eq_gains[2 * j + 0];
				//output_side_mel_bands[j] *= state.eq_gains[2 * j + k];

				/*std::cerr
					<< input_mid_mel_bands[j] << " "
					<< input_side_mel_bands[j] << " "
					<< output_mid_mel_bands[j] << " "
					<< output_side_mel_bands[j] << " "
					<< state.eq_gains[2 * j + 0] << " "
					<< state.eq_gains[2 * j + 1] << std::endl;*/
			}

			input_mid_mel_bands += num_filters;
			input_side_mel_bands += num_filters;
			output_mid_mel_bands += num_filters;
			output_side_mel_bands += num_filters;
		}

		output_mid_mel_bands -= count * num_filters;
		output_side_mel_bands -= count * num_filters;

		// ear guard (WIP)
		/*std::vector<float> loudness_weights(num_filters);
		std::vector<float> ear_damage_weights(num_filters);
		for (int j = 0; j < num_filters; j++) {
			auto freq = mfcc_calculator.center_freq(j);
			loudness_weights[j] = std::pow(10,
				(bakuage::loudness_contours::HzToSplAt60Phon(1000) - bakuage::loudness_contours::HzToSplAt60Phon(freq)) * 0.1);
			ear_damage_weights[j] = std::pow(10, (3.0 * std::log2(freq / 1000.0)) * 0.1);
		}
		bakuage::Statistics loudness_energy;
		std::vector<bakuage::Statistics> ear_damage_energy;
		for (int i = 0; i < count; i++) {
			for (int j = 0; j < num_filters; j++) {
				const double total_energy = output_mid_mel_bands[num_filters * i + j] + output_side_mel_bands[num_filters * i + j];
				loudness_energy.Add(total_energy * loudness_weights[j]);
				ear_damage_energy[j].Add(total_energy * ear_damage_weights[j]);
			}
		}
		const double loudness = 10 * std::log10(1e-10 + loudness_energy.mean());
		for (int i = 0; i < count; i++) {
			const double gain = ear_damage_energy[j].mean()
			for (int j = 0; j < num_filters; j++) {
				const double total_energy = output_mid_mel_bands[num_filters * i + j] + output_side_mel_bands[num_filters * i + j];
				const double max_energy =
				loudness_energy.Add(total_energy * loudness_weights[j]);
				ear_damage_energy[j].Add(total_energy * ear_damage_weights[j]);
			}
		}*/
	};
	auto evaluate = [comp_band_count, &apply_effects, &src_mid_mel_bands, &src_side_mel_bands,
		&output_mid_mel_bands, &output_side_mel_bands, score_acoustic_entropy_eigen_before,
		score_ear_damage_before, score_loudness_before, calculate_mastering3_score](const State &state, float regualization_coef = 1) {
		if (verbose2) std::cerr << "Mastering3 evaluate" << std::endl;

		// process
		apply_effects(src_mid_mel_bands.data(), src_side_mel_bands.data(), state,
			output_mid_mel_bands.data(), output_side_mel_bands.data());

		// regularization
		double regularization = 0;
		int param_count = 4 * (2 * comp_band_count);
		for (int j = 0; j < 2 * comp_band_count; j++) {
			// 知覚できる最小値が1になるのが目安
			regularization += bakuage::Sqr(4 * std::log2(state.compressor_ratios[j]));
			// regularization += state.compressor_wets[j];
			regularization += bakuage::Sqr(10 * std::log10(state.compressor_gains[j]));
			// regularization += std::abs(10 * std::log10(state.eq_gains[j]));
		}
		regularization /= param_count;

		// distance
		double dist = 0;
		for (int i = 0; i < output_mid_mel_bands.size(); i++) {
			dist += bakuage::Sqr(10 * std::log10((1e-10 + output_mid_mel_bands[i]) / (1e-10 + src_mid_mel_bands[i])));
			dist += bakuage::Sqr(10 * std::log10((1e-10 + output_side_mel_bands[i]) / (1e-10 + src_side_mel_bands[i])));
		}
		dist = dist / (2 * output_mid_mel_bands.size());

		// mastering score
		float score_loudness;
		float score_ear_damage;
		float score_acoustic_entropy_mfcc;
		float score_acoustic_entropy_eigen;
		calculate_mastering3_score(output_mid_mel_bands.data(), output_side_mel_bands.data(),
			&score_loudness, &score_ear_damage,
			&score_acoustic_entropy_mfcc, &score_acoustic_entropy_eigen);
		return
			-(score_acoustic_entropy_eigen - score_acoustic_entropy_eigen_before)
			- 1e5 * std::min<double>(0, score_acoustic_entropy_eigen - score_acoustic_entropy_eigen_before)
			+ 1e1 * ((score_ear_damage - score_loudness) - (score_ear_damage_before - score_loudness_before))
			+ 1e5 * std::max<double>(0, (score_ear_damage - score_loudness) - (score_ear_damage_before - score_loudness_before))
			// + regualization_coef * regularization
			+ regualization_coef * dist
			;
	};
	State optimum_state;
	double optimum_eval;
	SimulatedAnnealing(initialize_state, generate_neighbor, evaluate,
		FLAGS_mastering3_iteration, 1, std::pow(0.01, 1.0 / FLAGS_mastering3_iteration), progress_callback, &optimum_state, &optimum_eval);
	if (verbose) {
		State state;
		initialize_state(&state);
		std::cerr << "Mastering3 optimum eval " << evaluate(state) << " -> " << optimum_eval << std::endl;
		std::cerr << "Mastering3 optimum eval without regularization " << evaluate(state, 0) << " -> " << evaluate(optimum_state, 0) << std::endl;
		std::cerr << "Mastering3 optimum state" << std::endl;
		for (int j = 0; j < comp_band_count; j++) {
			std::cerr
				<< optimum_state.compressor_ratios[2 * j + 0] << " " << optimum_state.compressor_ratios[2 * j + 1]
				<< " " << optimum_state.compressor_thresholds[2 * j + 0] << " " << optimum_state.compressor_thresholds[2 * j + 1]
				<< " " << optimum_state.compressor_gains[2 * j + 0] << " " << optimum_state.compressor_gains[2 * j + 1]
				<< " " << optimum_state.compressor_wets[2 * j + 0] << " " << optimum_state.compressor_wets[2 * j + 1]
				//<< " " << optimum_state.eq_gains[2 * j + 0] << " " << optimum_state.eq_gains[2 * j + 1]
				<< std::endl;
		}
		std::cerr << "Mastering3 optimum state mfcc" << std::endl;
		for (int j = 0; j < mfcc_comp_count; j++) {
			std::cerr
				<< optimum_state.mfcc_ratios[2 * j + 0] << " " << optimum_state.mfcc_ratios[2 * j + 1]
				<< std::endl;
		}
		std::cerr << "Mastering3 optimum state diff" << std::endl;
		for (int j = 0; j < comp_band_count; j++) {
			State state1 = optimum_state;
			state1.compressor_ratios[2 * j + 0] *= 1.1;
			State state2 = optimum_state;
			state2.compressor_thresholds[2 * j + 0] += 1;
			State state3 = optimum_state;
			state3.compressor_gains[2 * j + 0] *= std::pow(10, 1.0 * 0.1);
			State state4 = optimum_state;
			state4.compressor_wets[2 * j + 0] /= std::pow(10, 1.0 * 0.1);
			std::cerr << "with reg "
				<< evaluate(state1) - evaluate(optimum_state)
				<< " " << evaluate(state2) - evaluate(optimum_state)
				<< " " << evaluate(state3) - evaluate(optimum_state)
				<< " " << evaluate(state4) - evaluate(optimum_state)
				<< std::endl;
			std::cerr << "without reg "
				<< evaluate(state1, 0) - evaluate(optimum_state, 0)
				<< " " << evaluate(state2, 0) - evaluate(optimum_state, 0)
				<< " " << evaluate(state3, 0) - evaluate(optimum_state, 0)
				<< " " << evaluate(state4, 0) - evaluate(optimum_state, 0)
				<< std::endl;
		}
		std::cerr << "Mastering3 optimum chart ratio vs eval(with reg)" << std::endl;
		for (int j = 0; j < comp_band_count; j++) {
			for (int k = 0; k < 10; k++) {
				State state1 = optimum_state;
				state1.compressor_ratios[2 * j + 0] = std::pow(2, 0.1 * k);
				std::cerr
					<< evaluate(state1) - evaluate(optimum_state) << " ";
			}
			std::cerr << std::endl;
		}
		std::cerr << "Mastering3 optimum chart ratio vs eval(without reg)" << std::endl;
		for (int j = 0; j < comp_band_count; j++) {
			for (int k = 0; k < 10; k++) {
				State state1 = optimum_state;
				state1.compressor_ratios[2 * j + 0] = std::pow(2, 0.1 * k);
				std::cerr
					<< evaluate(state1, 0) - evaluate(optimum_state, 0) << " ";
			}
			std::cerr << std::endl;
		}
		std::cerr << "Mastering3 optimum chart gain vs eval(with reg)" << std::endl;
		for (int j = 0; j < comp_band_count; j++) {
			for (int k = 0; k < 10; k++) {
				State state1 = optimum_state;
				state1.compressor_gains[2 * j + 0] = std::pow(10, (k - 5) * 0.1);
				std::cerr
					<< evaluate(state1) - evaluate(optimum_state) << " ";
			}
			std::cerr << std::endl;
		}
		std::cerr << "Mastering3 optimum chart gain vs eval(without reg)" << std::endl;
		for (int j = 0; j < comp_band_count; j++) {
			for (int k = 0; k < 10; k++) {
				State state1 = optimum_state;
				state1.compressor_gains[2 * j + 0] = std::pow(10, (k - 5) * 0.1);
				std::cerr
					<< evaluate(state1, 0) - evaluate(optimum_state, 0) << " ";
			}
			std::cerr << std::endl;
		}
		std::cerr << "Mastering3 optimum chart wet vs eval(without reg)" << std::endl;
		for (int j = 0; j < comp_band_count; j++) {
			for (int k = 0; k < 10; k++) {
				State state1 = optimum_state;
				state1.compressor_wets[2 * j + 0] = k * 0.1;
				std::cerr
					<< evaluate(state1, 0) - evaluate(optimum_state, 0) << " ";
			}
			std::cerr << std::endl;
		}

		std::cerr << "Mastering3 score" << std::endl;
		std::cerr << "before loudness " << score_loudness_before << " ear dmg " << score_ear_damage_before
			<< " ear dmg rel " << score_ear_damage_before - score_loudness_before << " ac. ent. " << score_acoustic_entropy_mfcc_before
			<< " ac. ent. eigen " << score_acoustic_entropy_eigen_before << std::endl;
		apply_effects(src_mid_mel_bands.data(), src_side_mel_bands.data(), optimum_state,
			output_mid_mel_bands.data(), output_side_mel_bands.data());
		float score_loudness;
		float score_ear_damage;
		float score_acoustic_entropy_mfcc;
		float score_acoustic_entropy_eigen;
		calculate_mastering3_score(output_mid_mel_bands.data(), output_side_mel_bands.data(),
			&score_loudness, &score_ear_damage,
			&score_acoustic_entropy_mfcc, &score_acoustic_entropy_eigen);
		std::cerr << "after loudness " << score_loudness << " ear dmg " << score_ear_damage
			<< " ear dmg rel " << score_ear_damage - score_loudness << " ac. ent. " << score_acoustic_entropy_mfcc
			<< " ac. ent. eigen " << score_acoustic_entropy_eigen << std::endl;
	}

	// 音源処理
	if (verbose) std::cerr << "Mastering3 output start" << std::endl;
	pos = -width + output_shift;
	std::vector<float> band_gains(num_filters);
	std::vector<float> spec_gains(spec_len);
	apply_effects(src_mid_mel_bands.data(), src_side_mel_bands.data(), optimum_state,
		output_mid_mel_bands.data(), output_side_mel_bands.data());
	int mel_band_index = 0;
	while (pos < frames) {
		// window and fft
		std::fill_n(complex_spec_mid.data(), spec_len, 0);
		std::fill_n(complex_spec_side.data(), spec_len, 0);
		for (int i = 0; i < channels; i++) {
			for (int j = 0; j < width; j++) {
				int k = pos + j;
				fft_input[j] = (0 <= k && k < frames) ? wave_ptr[channels * k + i] * window[j] : 0;
			}
			fftw_execute(plan);
			for (int j = 0; j < spec_len; j++) {
				auto spec = std::complex<float>(fft_output[j][0], fft_output[j][1]);
				complex_spec_mid[j] += spec;
				complex_spec_side[j] += spec * (2.0f * i - 1);
			}
		}

		// apply gain
		for (int j = 0; j < num_filters; j++) {
			//std::cerr << output_mid_mel_bands[mel_band_index * num_filters + j] << " " << src_mid_mel_bands[mel_band_index * num_filters + j] << std::endl;
			band_gains[j] = std::sqrt(output_mid_mel_bands[mel_band_index * num_filters + j] / (1e-37 + src_mid_mel_bands[mel_band_index * num_filters + j]));
		}
		mfcc_calculator.calculateSpectrumFromMelSpectrum(band_gains.data(), width, false, true, spec_gains.data());
		for (int j = 0; j < spec_len; j++) {
			complex_spec_mid[j] *= spec_gains[j];
		}
		for (int j = 0; j < num_filters; j++) {
			band_gains[j] = std::sqrt(output_side_mel_bands[mel_band_index * num_filters + j] / (1e-37 + src_side_mel_bands[mel_band_index * num_filters + j]));
		}
		mfcc_calculator.calculateSpectrumFromMelSpectrum(band_gains.data(), width, false, true, spec_gains.data());
		for (int j = 0; j < spec_len; j++) {
			complex_spec_side[j] *= spec_gains[j];
		}

		// ifft and output
		for (int i = 0; i < channels; i++) {
			for (int j = 0; j < spec_len; j++) {
				auto spec = 0.5f * (complex_spec_mid[j] + (2.0f * i - 1) * complex_spec_side[j]);
				fft_output[j][0] = spec.real();
				fft_output[j][1] = spec.imag();
			}
			fftw_execute(inv_plan);
			for (int j = 0; j < width; j++) {
				int k = pos + j;
				if (0 <= k && k < frames) {
					result[channels * k + i] += fft_input[j] * (output_shift_resolution / 2);
				}
			}
		}

		pos += output_shift;
		mel_band_index++;
	}

	{
		std::lock_guard<std::recursive_mutex> lock(FFTW::mutex());

		fftw_destroy_plan(plan);
		fftw_destroy_plan(inv_plan);
		fftw_free(fft_output);
		fftw_free(fft_input);
	}

	progress_callback(1);

	*_wave = std::move(result);
#endif
}

}
