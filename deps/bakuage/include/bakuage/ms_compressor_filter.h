#ifndef BAKUAGE_BAKUAGE_MS_COMPRESSOR_FILTER_H_
#define BAKUAGE_BAKUAGE_MS_COMPRESSOR_FILTER_H_

#include <algorithm>
#include <limits>
#include "bakuage/delay_filter.h"
#include "bakuage/loudness_filter.h"
#include "bakuage/time_varying_lowpass_filter.h"

namespace bakuage {

	template <
        typename Float, 
        typename LoudnessMappingFunc, 
        typename MsLoudnessMappingFunc
    >
    class MsCompressorFilter {
    public:
        class Config {
        public:
            Config(): num_channels(0), sample_rate(0), max_mean_sec(0) {}
            int num_channels;
            int sample_rate;
            float max_mean_sec;
            LoudnessMappingFunc loudness_mapping_func;
            MsLoudnessMappingFunc ms_loudness_mapping_func;
        };

        MsCompressorFilter(const Config &config):
            config_(config),
            last_loudness_(-1000),
            wet_(1),
            mean_sec_(config_.max_mean_sec),
            temp_filter_(2) {

            int lowpass_filter_order = 2;
            Float peak = std::min<Float>(1.0, 1.0 / (config_.sample_rate * config_.max_mean_sec 
                + 1e-30));
            Float a = TimeVaryingLowpassFilter<Float>::CalculateAFromPeak(
                lowpass_filter_order, peak, &delay_samples_, &temp_filter_);
                
            for (int i = 0; i < config_.num_channels; i++) {
                loudness_filters_.emplace_back(config_.sample_rate);
                lowpass_filters_.emplace_back(lowpass_filter_order, a);
                delay_filters_.emplace_back(delay_samples_);
            }
        }

        /*
            mid基準でsideを圧縮、その後普通に全体で圧縮(ただし、side圧縮分の全体音圧補正あり)
        */
        void Clock(const Float *input, Float *output) {    
            if (config_.num_channels != 2) {
                PlainClock(input, output);
                return;
            }

            static const float sqrt_0_5 = std::sqrt(0.5);
            static const float log10_div_20 = std::log(10) / 20;

            Float ms[2];
            ms[0] = (input[0] + input[1]) * sqrt_0_5;
            ms[1] = (input[0] - input[1]) * sqrt_0_5;

            Float rmss[2];
            for (int i = 0; i < 2; i++) {
                Float filtered = loudness_filters_[i].Clock(ms[i]);
                rmss[i] = lowpass_filters_[i].Clock(filtered * filtered);
            }

            Float total_loudness = -0.691 + 10 * std::log10(rmss[0] + rmss[1] + 1e-37);
            Float mapped_loudness = config_.loudness_mapping_func(total_loudness);
            last_loudness_ = total_loudness;

            Float mid_to_side_loudness = 
                10 * (std::log10(rmss[1] + 1e-37) - std::log10(rmss[0] + 1e-37));
            Float side_gain = std::exp(log10_div_20 * 
                (config_.ms_loudness_mapping_func(mid_to_side_loudness) - mid_to_side_loudness));

            Float total_loudness_with_side_gain = -0.691 + 10 * std::log10(rmss[0] + rmss[1] * Sqr(side_gain) + 1e-37);
            Float gain = wet_ * std::exp(log10_div_20 * (mapped_loudness - total_loudness_with_side_gain));
            
            // wet_をgainに含めることで乗算回数を減らしてるのに注意
            Float dry = 1 - wet_;
            ms[0] = delay_filters_[0].Clock(ms[0]) * (gain + dry);
            ms[1] = delay_filters_[1].Clock(ms[1]) * (side_gain * gain + dry);           

            output[0] = (ms[0] + ms[1]) * sqrt_0_5;
            output[1] = (ms[0] - ms[1]) * sqrt_0_5;
        };     
		void Analyze(const Float *input, Float *loudness, Float *mid_loudness, Float *side_loudness) {
			if (config_.num_channels != 2) {
				// TODO: implement
				*loudness = 0;
				*mid_loudness = 0;
				*side_loudness = 0;
				return;
			}

			static const float sqrt_0_5 = std::sqrt(0.5);
	
			Float ms[2];
			ms[0] = (input[0] + input[1]) * sqrt_0_5;
			ms[1] = (input[0] - input[1]) * sqrt_0_5;

			Float rmss[2];
			for (int i = 0; i < config_.num_channels; i++) {
				Float filtered = loudness_filters_[i].Clock(ms[i]);
				rmss[i] = lowpass_filters_[i].Clock(filtered * filtered);
			}

			*loudness = -0.691 + 10 * std::log10(rmss[0] + rmss[1] + 1e-37);
			*mid_loudness = -0.691 + 10 * std::log10(rmss[0] + 1e-37);
			*side_loudness = -0.691 + 10 * std::log10(rmss[1] + 1e-37);
		};

        void set_mean_sec(Float value) {
            value = std::max<Float>(0, std::min<Float>(config_.max_mean_sec, value));            
            if (value == mean_sec_) {
                return;
            }
            mean_sec_ = value;

            int lowpass_filter_order = 2;
            Float peak = std::min<Float>(1.0, 1.0 / (config_.sample_rate * value
                + 1e-30));
            Float a = TimeVaryingLowpassFilter<Float>::CalculateAFromPeak(
                lowpass_filter_order, peak, &delay_samples_, &temp_filter_);
                
            for (int i = 0; i < config_.num_channels; i++) {
                lowpass_filters_[i].set_a(a);
                delay_filters_[i].set_delay(delay_samples_);
            }
        }
        void set_wet(Float value) { wet_ = value; } 
        int delay_samples() const { return delay_samples_; }
        Float last_loudness() const { return last_loudness_; }
    private:
        void PlainClock(const Float *input, Float *output) {
            Float rms = 0;
            for (int i = 0; i < config_.num_channels; i++) {
                Float filtered = loudness_filters_[i].Clock(input[i]);
                rms += lowpass_filters_[i].Clock(filtered * filtered);
            }

            Float loudness = -0.691 + 10 * std::log10(rms + 1e-37);
            Float mapped_loudness = config_.loudness_mapping_func(loudness);
            Float gain = wet_ * std::exp((mapped_loudness - loudness) * (std::log(10) / 20));
            
            // dry, wet_をgainに含めることで乗算回数を減らしてるのに注意
            Float dry = 1 - wet_;
            gain += dry;
            for (int i = 0; i < config_.num_channels; i++) {
                output[i] = delay_filters_[i].Clock(input[i]) * gain;
            }
       };

       Config config_;
       int delay_samples_;
       Float last_loudness_;
       Float wet_;
       Float mean_sec_;
       std::vector<LoudnessFilter<Float>> loudness_filters_;
       std::vector<TimeVaryingLowpassFilter<Float>> lowpass_filters_;
       std::vector<DelayFilter<Float>> delay_filters_;
       TimeVaryingLowpassFilter<Float> temp_filter_;
    };
}

#endif 
