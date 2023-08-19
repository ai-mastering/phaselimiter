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
#include "tbb/tbb.h"

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
#include "bakuage/fir_filter2.h"
#include "bakuage/ffmpeg.h"

DECLARE_bool(mastering_parallel_compression);
DECLARE_string(mastering_reference_file);
DECLARE_double(mastering_matching_level);
DECLARE_double(mastering_ms_matching_level);
DECLARE_bool(mastering_reverb);
DECLARE_double(mastering_reverb_gain);
DECLARE_double(mastering_reverb_drr_range);
DECLARE_double(mastering_reverb_target_drr);
DECLARE_double(mastering_reverb_predelay);
DECLARE_bool(mastering_reverb_ensure_monotone);

typedef float Float;
using namespace bakuage;

namespace {

Float Interpolate(const Float &from, const Float &to, const Float &t) {
	return from * (1 - t) + to * t;
}

class LoudnessMapping {
public:
    LoudnessMapping() {}
    LoudnessMapping(Float original_mean, Float original_stddev, 
        Float target_mean, Float target_stddev):
        original_mean_(original_mean),
		target_mean_(target_mean),
        threshold_(original_mean - 40) {
        inv_ratio_ = std::min<Float>(1, target_stddev / (1e-37 + original_stddev));
    }

    Float operator () (Float x) {
		Float y;
        if (threshold_ <= x) {
            y = (x - original_mean_) * inv_ratio_ + target_mean_;
        }
        else {
            Float gain = (threshold_ - original_mean_) * inv_ratio_ + target_mean_ - threshold_;
            y = x + gain;
        }
		if (FLAGS_mastering_parallel_compression) {
			Float z = x + target_mean_ - original_mean_;
			return 20 * std::log10(1e-37 + 0.5 * std::pow(10, (1.0 / 20) * y) + 0.5 * std::pow(10, (1.0 / 20) * z));
		}
		else {
			return y;
		}
    }

    Float threshold() const { return threshold_; }
private:    
    Float inv_ratio_;
    Float original_mean_;
    Float target_mean_;
    Float threshold_;
};
class MsLoudnessMapping {
public:
	MsLoudnessMapping() {}
	MsLoudnessMapping(Float original_mean, Float original_stddev,
		Float target_mean, Float target_stddev) :
		original_mean_(original_mean),
		target_mean_(target_mean),
		threshold_(original_mean - 40) {
		inv_ratio_ = std::min<Float>(1, target_stddev / (1e-37 + original_stddev));
	}

	Float operator () (Float x) {
		if (threshold_ <= x) {
			return (x - original_mean_) * inv_ratio_ + target_mean_;
		}
		else {
			Float gain = (threshold_ - original_mean_) * inv_ratio_ + target_mean_ - threshold_;
			return x + gain;
		}
	}

	Float threshold() const { return threshold_; }
private:
	Float inv_ratio_;
	Float original_mean_;
	Float target_mean_;
	Float threshold_;
};
typedef MsCompressorFilter<Float, LoudnessMapping, MsLoudnessMapping> Compressor;

class LoudnessMappingForReverb {
public:
	LoudnessMappingForReverb() {}
	LoudnessMappingForReverb(Float center, Float gain, Float sigma) :
		center_(center), gain_(gain), sigma_(sigma) {
		if (FLAGS_mastering_reverb_ensure_monotone) {
			Float diff_erf_at_0 = 2 / std::sqrt(M_PI);
			sigma_ = std::max<Float>(diff_erf_at_0 * FLAGS_mastering_reverb_drr_range * 0.5, sigma);
		}
	}

	Float operator () (Float x) {
		Float drr_range = FLAGS_mastering_reverb_drr_range;
		return x - drr_range * 0.5 * std::erf((x - center_) / (1e-37 + sigma_)) + gain_;
		/*if (x >= upper_bound_) {
			return -1 * (x - upper_bound_) + upper_bound_;
		}
		return std::min<Float>(upper_bound_, x + gain_);*/
	}
private:
	Float center_;
	Float gain_;
	Float sigma_;
};
typedef CompressorFilter<Float, LoudnessMappingForReverb> CompressorForReverb;
/*class LoudnessMappingForReverb {
public:
	LoudnessMappingForReverb() {}
	LoudnessMappingForReverb(Float upper_bound, Float gain) :
		upper_bound_(upper_bound), gain_(gain) {
	}

	void operator () (int num_channels, const Float *x, Float *y) {
		Float g = upper_bound_;
		for (int i = 0; i < num_channels; i++) {
			g = std::min<Float>(g, x[i] + gain_);
			// g = std::min<Float>(g, -1 * (x[i] - upper_bound_) + upper_bound_);
		}
		for (int i = 0; i < num_channels; i++) {
			y[i] = g;
		}
	}
private:
	Float upper_bound_;
	Float gain_;
};
typedef ChannelWiseCompressorFilter<Float, LoudnessMappingForReverb> CompressorForReverb;*/

void raise(const std::string &message) {
    throw std::logic_error("auto mastering error: " + message);
}

class Band {
public:
    Band(const picojson::object &band) {
        using namespace picojson;
        if (band.find("low_freq") == band.end()) {
			low_freq = 0;
        }
        else {
			low_freq = band.at("low_freq").get<double>();
        }
        if (band.find("high_freq") == band.end()) {
			high_freq = 0.5 * 44100;
        }
        else {
			high_freq = band.at("high_freq").get<double>();
        }
		loudness = band.at("loudness").get<double>();
		loudness_range = band.at("loudness_range").get<double>();
		mid_to_side_loudness = band.at("mid_to_side_loudness").get<double>();
		mid_to_side_loudness_range = band.at("mid_to_side_loudness_range").get<double>();
    }
    Float low_freq;
    Float high_freq;
	Float loudness;
	Float loudness_range;
	Float mid_to_side_loudness;
	Float mid_to_side_loudness_range;
};

class MasteringReference {
public:
    MasteringReference(const std::string &reference_file_path) {
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
        array bands_json = root.at("bands").get<array>();
        for (const auto band_json: bands_json) {
            bands.push_back(Band(band_json.get<object>()));
        }
    }

    std::vector<Band> bands;
};

void CalculateHistogramStatistics(const std::vector<Float> &histogram, Float *mean, Float *stddev) {
    audio_analyzer::Statistics stat;
    for (int i = 0; i < histogram.size(); i++) {
        stat.Add(i - 70, histogram[i]);
    }
    *mean = stat.mean();
    *stddev = stat.stddev();
}
}

namespace phase_limiter {

// ir は原点の1を含まない
void AutoMastering(std::vector<float> *_wave, const float **irs, const int *ir_lens, const int sample_rate, const std::function<void(float)> &progress_callback) {
    MasteringReference reference(FLAGS_mastering_reference_file);

    const int frames = _wave->size() / 2;
    const int channels = 2;

	std::mutex result_mtx;
	std::mutex progression_mtx;
	std::vector<std::function<void ()>> tasks;
    std::vector<Float> result(_wave->size());
	std::vector<Float> progressions(reference.bands.size());

	// calculate original statistics
	std::vector<audio_analyzer::Band<Float>> original_bands;
	{
		for (const auto band : reference.bands) {
			audio_analyzer::Band<Float> b;
			b.low_freq = band.low_freq;
			b.high_freq = band.high_freq;
			original_bands.push_back(b);
		}
		int block_samples;
		audio_analyzer::CalculateMultibandLoudness<Float>(_wave->data(), channels, frames, sample_rate,
			0.4, 0.1, -20, original_bands.data(), original_bands.size(), &block_samples);
	}

	const auto update_progression = [&progressions, &progression_mtx, progress_callback](int i, Float p) {
		std::lock_guard<std::mutex> lock(progression_mtx);
		Float total = 0;
		progressions[i] = p;
		for (const auto &a : progressions) {
			total += a;
		}
		progress_callback(total / progressions.size());
	};

	for (int band_index = 0; band_index < reference.bands.size(); band_index++) {
		const auto &band = reference.bands[band_index];
		const auto &original_band = original_bands[band_index];
		const auto update_progression_bound = std::bind(update_progression, band_index, std::placeholders::_1);
		tasks.push_back([band, original_band, sample_rate, frames, _wave, &result, &result_mtx, update_progression_bound, irs, ir_lens, channels]() {
			const float *wave_ptr = &(*_wave)[0];

			int fir_delay_samples;
			std::vector<Float> fir;
			{
				fir_delay_samples = static_cast<int>(0.002 * sample_rate);
				int n = 2 * fir_delay_samples + 1;
				Float freq1 = std::min<Float>(0.5, band.low_freq / sample_rate);
				Float freq2 = std::min<Float>(0.5, band.high_freq / sample_rate);
				fir = CalculateBandPassFir<Float>(freq1, freq2, n, 4);
			}
			update_progression_bound(0.1);

			const int len = frames + fir.size() - 1;
			bakuage::AlignedPodVector<float> filtered(channels * len);
            {
                FirFilter2<Float> fir_filter(fir.begin(), fir.end());
                bakuage::AlignedPodVector<Float> filter_temp_input(frames + fir_delay_samples);
                bakuage::AlignedPodVector<Float> filter_temp_output(frames + fir_delay_samples);
                for (int ch = 0; ch < channels; ch++) {
                    fir_filter.Clear();
                    for (int i = 0; i < frames; i++) {
                        filter_temp_input[i] = wave_ptr[channels * i + ch];
                    }
                    fir_filter.Clock(filter_temp_input.data(), filter_temp_input.data() + frames + fir_delay_samples, filter_temp_output.data());
                    for (int i = 0; i < frames; i++) {
                        filtered[channels * i + ch] = filter_temp_output[i + fir_delay_samples];
                    }
                }
            }
			update_progression_bound(0.2);

			Float original_mean, original_stddev, original_mid_to_side_mean, original_mid_to_side_stddev,
				reference_mean, reference_stddev, reference_mid_to_side_mean, reference_mid_to_side_stddev,
				target_mean, target_stddev, target_mid_to_side_mean, target_mid_to_side_stddev;
			{
				// originalの平均はCalculateHistogram由来のもので計算し、残りはbands, original_bandsから計算する
				// 補正(original_bandsのmeanとCalculateHistogram由来のmeanの差分を、referenceに上乗せするなど)をかける
				std::vector<Float> histogram;
				std::vector<Float> mid_to_side_histogram;
				bakuage::loudness_ebu_r128::CalculateHistogram(filtered.data(), channels, len, sample_rate, 0.2f, &histogram, &mid_to_side_histogram);

				CalculateHistogramStatistics(histogram, &original_mean, &original_stddev);
				CalculateHistogramStatistics(mid_to_side_histogram, &original_mid_to_side_mean, &original_mid_to_side_stddev);
				original_stddev = original_band.loudness_range;
				original_mid_to_side_stddev = original_band.mid_to_side_loudness_range;

				reference_mean = band.loudness + (original_mean - original_band.loudness);
				reference_stddev = band.loudness_range;
				reference_mid_to_side_mean = band.mid_to_side_loudness; //元々打ち消しあっているので補正しない +(original_mid_to_side_mean - original_band.mid_to_side_loudness);
				reference_mid_to_side_stddev = band.mid_to_side_loudness_range;

				target_mean = Interpolate(original_mean, reference_mean, FLAGS_mastering_matching_level);
				target_stddev = Interpolate(original_stddev, reference_stddev, FLAGS_mastering_matching_level);
				target_mid_to_side_mean = Interpolate(original_mid_to_side_mean, reference_mid_to_side_mean, FLAGS_mastering_ms_matching_level);
				target_mid_to_side_stddev = Interpolate(original_mid_to_side_stddev, reference_mid_to_side_stddev, FLAGS_mastering_ms_matching_level);
			}
			std::cerr << "original mean stddev mean(ms) stddev(ms) = "
				<< original_mean << ", " << original_stddev << ", "
				<< original_mid_to_side_mean << ", " << original_mid_to_side_stddev << std::endl;
			std::cerr << "reference mean stddev mean(ms) stddev(ms) = "
				<< reference_mean << ", " << reference_stddev << ", "
				<< reference_mid_to_side_mean << ", " << reference_mid_to_side_stddev << std::endl;
			std::cerr << "target mean stddev mean(ms) stddev(ms) = "
				<< target_mean << ", " << target_stddev << ", "
				<< target_mid_to_side_mean << ", " << target_mid_to_side_stddev << std::endl;
			update_progression_bound(0.4);

			const LoudnessMapping loudness_mapping(original_mean, original_stddev,
				target_mean, target_stddev);
			const MsLoudnessMapping ms_loudness_mapping(original_mid_to_side_mean, original_mid_to_side_stddev,
				target_mid_to_side_mean, target_mid_to_side_stddev);
			Compressor::Config compressor_config;
			compressor_config.loudness_mapping_func = loudness_mapping;
			compressor_config.ms_loudness_mapping_func = ms_loudness_mapping;
			compressor_config.max_mean_sec = 0.2;
			compressor_config.num_channels = channels;
			compressor_config.sample_rate = sample_rate;
			Compressor compressor(compressor_config);

			const int shift = compressor.delay_samples();
			const int len2 = frames + shift;
			filtered.resize(channels * len2);
			bakuage::AlignedPodVector<Float> temp_input(channels);
			bakuage::AlignedPodVector<Float> temp_output(channels);

			// filteredにin-placeで書き込んでから共有のresultに足しこむ
			for (int j = 0; j < len2; j++) {
				for (int i = 0; i < channels; i++) {
					temp_input[i] = filtered[channels * j + i];
				}
				compressor.Clock(&temp_input[0], &temp_output[0]);

				for (int i = 0; i < channels; i++) {
					filtered[channels * j + i] = temp_output[i];
				}
			}
			update_progression_bound(0.6);

			// flush filtered (dry sound)
			{
				std::lock_guard<std::mutex> lock(result_mtx);
				int len3 = frames * channels;
				int channels_shift = channels * shift;
				for (int j = 0; j < len3; j++) {
					result[j] += filtered[j + channels_shift];
				}
			}

			// ここからしたはfiltered is wet sound
			if (FLAGS_mastering_reverb) {
				// compression for reverb
				Float reverb_center;
				Float reverb_gain;
				Float reverb_sigma;
				{
					float sum_loudness = 0;
					float count_loudness = 0;
					LoudnessMappingForReverb loudness_mapping_for_reverb(0, 0, 1);
					CompressorForReverb::Config compressor_for_reverb_config;
					compressor_for_reverb_config.loudness_mapping_func = loudness_mapping_for_reverb;
					compressor_for_reverb_config.mean_sec = 0.02;
					compressor_for_reverb_config.num_channels = channels;
					compressor_for_reverb_config.sample_rate = sample_rate;
					CompressorForReverb compressor_for_reverb(compressor_for_reverb_config);
					for (int j = 0; j < filtered.size() / channels; j++) {
						for (int i = 0; i < channels; i++) {
							temp_input[i] = filtered[channels * j + i];
						}
						float loudness;
						compressor_for_reverb.Analyze(&temp_input[0], &loudness);

						if (loudness >= target_mean - 60) {
							sum_loudness += loudness;
							count_loudness += 1;
						}
						/*float loudness[2];
						compressor_for_reverb.Analyze(&temp_input[0], loudness);

						for (int i = 0; i < 2; i++) {
							if (loudness[i] >= target_mean - 60) {
								sum_loudness += loudness[i];
								count_loudness += 1;
							}
						}*/
					}

					float loudness = sum_loudness / (1e-37 + count_loudness);
					reverb_center = loudness;// +FLAGS_mastering_reverb_target_drr;
					reverb_gain = FLAGS_mastering_reverb_gain;
					reverb_sigma = target_stddev;
					fprintf(stderr, "reverb_center %f\n", reverb_center);
					fprintf(stderr, "reverb_gain %f\n", reverb_gain);
					fprintf(stderr, "reverb_sigma %f\n", reverb_sigma);
				}
				update_progression_bound(0.7);

				LoudnessMappingForReverb loudness_mapping_for_reverb(reverb_center, reverb_gain, reverb_sigma);
				CompressorForReverb::Config compressor_for_reverb_config;
				compressor_for_reverb_config.loudness_mapping_func = loudness_mapping_for_reverb;
				compressor_for_reverb_config.mean_sec = 0.02;
				compressor_for_reverb_config.num_channels = channels;
				compressor_for_reverb_config.sample_rate = sample_rate;
				CompressorForReverb compressor_for_reverb(compressor_for_reverb_config);
				filtered.resize(filtered.size() + channels * compressor_for_reverb.delay_samples());
				for (int j = 0; j < filtered.size() / channels; j++) {
					for (int i = 0; i < channels; i++) {
						temp_input[i] = filtered[channels * j + i];
					}
					compressor_for_reverb.Clock(&temp_input[0], &temp_output[0]);

					for (int i = 0; i < channels; i++) {
						filtered[channels * j + i] = temp_output[i];
					}
				}
				update_progression_bound(0.8);

				std::vector<double> ir_energies(channels);
				for (int i = 0; i < channels; i++) {
					for (int j = /*channels * sample_rate * 0.008*/ 0; j < channels * ir_lens[i]; j++) {
						ir_energies[i] += bakuage::Sqr(irs[i][j]);
					}
				}

				// reverb (energyは変化しない)
				for (int i = 0; i < channels; i++) {
					for (int ir_ch = 0; ir_ch < channels; ir_ch++) {
					//{ int ir_ch = i;
						int ir_index = i;
						bakuage::AlignedPodVector<Float> ir_splitted(ir_lens[ir_index]);
						Float scale = 1.0 / (1e-37 + std::sqrt(ir_energies[ir_index]));
						for (int j = 0; j < ir_splitted.size(); j++) {
							ir_splitted[j] = irs[ir_index][channels * j + ir_ch] * scale;
						}

						int filtered_len = filtered.size() / channels;
                        bakuage::AlignedPodVector<Float> filtered_splitted(filtered_len + ir_lens[ir_index] - 1);
						for (int j = 0; j < filtered_len; j++) {
							filtered_splitted[j] = filtered[channels * j + i];
								//0.5 * (compressed_for_reverb[channels * j + 0]
								//+ compressed_for_reverb[channels * j + 1]);
						}

						// compressed_for_reverb_splitted = ir convolute compressed_for_reverb_splitted
						bakuage::Convolute(filtered_splitted.data(), filtered_len,
							ir_splitted.data(), ir_lens[ir_index], filtered_splitted.data());

						{
							std::lock_guard<std::mutex> lock(result_mtx);
							int predelay = sample_rate * FLAGS_mastering_reverb_predelay;
							for (int j = 0; j < frames; j++) {
								int k = j + shift + compressor_for_reverb.delay_samples() - predelay;
								if (0 <= k && k < filtered_splitted.size()) {
									result[channels * j + ir_ch] += filtered_splitted[k];
								}
							}
						}
					}
				}
			}

			update_progression_bound(1);
		});
	}
    
    tbb::parallel_for(0, (int)tasks.size(), [&tasks](int task_i) {
        tasks[task_i]();
    });

    *_wave = std::move(result);
}

}
