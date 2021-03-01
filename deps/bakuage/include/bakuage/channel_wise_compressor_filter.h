#ifndef BAKUAGE_BAKUAGE_CHANNEL_WISE_COMPRESSOR_FILTER_H_
#define BAKUAGE_BAKUAGE_CHANNEL_WISE_COMPRESSOR_FILTER_H_

#include <algorithm>
#include <limits>
#include "bakuage/delay_filter.h"
#include "bakuage/loudness_filter.h"
#include "bakuage/time_varying_lowpass_filter.h"

namespace bakuage {

template <typename Float, typename LoudnessMappingFunc>
class ChannelWiseCompressorFilter {
public:
	class Config {
	public:
		Config() : num_channels(0), sample_rate(0), mean_sec(0) {}
		int num_channels;
		int sample_rate;
		float mean_sec;
		LoudnessMappingFunc loudness_mapping_func;
	};

	ChannelWiseCompressorFilter(const Config &config) :
		config_(config), loudness_(config.num_channels), mapped_loudness_(config.num_channels) {

		int lowpass_filter_order = 2;
		Float peak = std::min<Float>(1.0, 1.0 / (config_.sample_rate * config_.mean_sec
			+ 1e-30));
		Float a = TimeVaryingLowpassFilter<Float>::CalculateAFromPeak(
			lowpass_filter_order, peak, &delay_samples_);

		for (int i = 0; i < config_.num_channels; i++) {
			loudness_filters_.emplace_back(config_.sample_rate);
			lowpass_filters_.emplace_back(lowpass_filter_order, a);
			delay_filters_.emplace_back(delay_samples_);
		}
	}

	// inputとoutputが同じアドレスでもOK (一個ずれとかはダメ)
	void Clock(const Float *input, Float *output) {
		Analyze(input, loudness_.data());

		config_.loudness_mapping_func(loudness_.size(), (const Float *)loudness_.data(), mapped_loudness_.data());		

		for (int i = 0; i < config_.num_channels; i++) {
			Float gain = std::exp((mapped_loudness_[i] - loudness_[i]) * (std::log(10) / 20));
			output[i] = delay_filters_[i].Clock(input[i]) * gain;
		}
	};

	void Analyze(const Float *input, Float *loudness) {
		for (int i = 0; i < config_.num_channels; i++) {
			Float filtered = loudness_filters_[i].Clock(input[i]);
			Float rms = lowpass_filters_[i].Clock(filtered * filtered);
			loudness[i] = -0.691 + 10 * std::log10(rms + 1e-37);
		}		
	};

	int delay_samples() const { return delay_samples_; }
private:
	Config config_;
	int delay_samples_;
	std::vector<LoudnessFilter<Float>> loudness_filters_;
	std::vector<TimeVaryingLowpassFilter<Float>> lowpass_filters_;
	std::vector<DelayFilter<Float>> delay_filters_;
	std::vector<Float> loudness_;
	std::vector<Float> mapped_loudness_;
};
}

#endif 