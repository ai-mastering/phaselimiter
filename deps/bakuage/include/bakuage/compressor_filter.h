#ifndef BAKUAGE_BAKUAGE_COMPRESSOR_FILTER_H_
#define BAKUAGE_BAKUAGE_COMPRESSOR_FILTER_H_

#include <algorithm>
#include <limits>
#include "bakuage/delay_filter.h"
#include "bakuage/loudness_filter.h"
#include "bakuage/time_varying_lowpass_filter.h"

namespace bakuage {

	template <typename Float, typename LoudnessMappingFunc>
    class CompressorFilter {
    public:
        class Config {
        public:
            Config(): num_channels(0), sample_rate(0), mean_sec(0) {}
            int num_channels;
            int sample_rate;
            float mean_sec;
            LoudnessMappingFunc loudness_mapping_func;
        };

        CompressorFilter(const Config &config):
            config_(config) {

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
			Float loudness;
			Analyze(input, &loudness);

            Float mapped_loudness = config_.loudness_mapping_func(loudness);
            Float gain = std::exp((mapped_loudness - loudness) * (std::log(10) / 20));
            
            for (int i = 0; i < config_.num_channels; i++) {
                output[i] = delay_filters_[i].Clock(input[i]) * gain;
            }
        };

        void Analyze(const Float *input, Float *loudness) {
            Float rms = 0;
            for (int i = 0; i < config_.num_channels; i++) {
                Float filtered = loudness_filters_[i].Clock(input[i]);
                rms += lowpass_filters_[i].Clock(filtered * filtered);
            }
            *loudness = -0.691 + 10 * std::log10(rms + 1e-37);
        };
        // MSコンプ
        /*void Clock(const Float *input, Float *output) {
            if (config_.num_channels != 2) {
                for (int i = 0; i < config_.num_channels; i++) {
                    output[i] = input[i];
                }
                return;
            }

            Float ms[2];
            ms[0] = (input[0] + input[1]) * std::sqrt(0.5);
            ms[1] = (input[0] - input[1]) * std::sqrt(0.5);

            Float rmss[2] = { 0, 0 };
            for (int i = 0; i < config_.num_channels; i++) {
                Float filtered = loudness_filters_[i].Clock(ms[i]);
                rmss[i] += lowpass_filters_[i].Clock(filtered * filtered);
            }

            
            for (int i = 0; i < config_.num_channels; i++) {
                Float loudness = -0.691 + 10 * std::log10(rmss[i] + 1e-37);
                Float mapped_loudness = config_.loudness_mapping_func(loudness);
                Float gain = std::exp((mapped_loudness - loudness) * (std::log(10) / 20));
            
                ms[i] = delay_filters_[i].Clock(ms[i]) * gain;
            }

            output[0] = (ms[0] + ms[1]) * std::sqrt(0.5);
            output[1] = (ms[0] - ms[1]) * std::sqrt(0.5);
        };*/

        // 左右の音量差を広げるエフェクト (なんかマスターにかけると違和感がある)
        /*void Clock(const Float *input, Float *output) {
            if (config_.num_channels != 2) {
                for (int i = 0; i < config_.num_channels; i++) {
                    output[i] = input[i];
                }
                return;
            }

            Float rmss[2] = {0, 0};
            for (int i = 0; i < config_.num_channels; i++) {
                Float filtered = loudness_filters_[i].Clock(input[i]);
                rmss[i] += lowpass_filters_[i].Clock(filtered * filtered);
            }

            Float original_loudnesses[2];
            original_loudnesses[0] = 10 * std::log10(rmss[0] + 1e-37);
            original_loudnesses[1] = 10 * std::log10(rmss[1] + 1e-37);
            Float diff = original_loudnesses[1] - original_loudnesses[0];             
            Float original_total_loudness = 10 * std::log10(
                std::pow(10, 0.1 * original_loudnesses[0])
                + std::pow(10, 0.1 * original_loudnesses[1]));
            
            Float after_loudnesses[2];
            Float a = std::max<Float>(-30, std::min<Float>(30, diff));
            after_loudnesses[0] = original_loudnesses[0] - a; 
            after_loudnesses[1] = original_loudnesses[1] + a; 
            Float after_total_loudness = 10 * std::log10(
                std::pow(10, 0.1 * after_loudnesses[0])
                + std::pow(10, 0.1 * after_loudnesses[1]));

            Float compensation = after_total_loudness - original_total_loudness;
            after_loudnesses[0] -= compensation;
            after_loudnesses[1] -= compensation;

            Float gains[2];
            gains[0] = std::exp((after_loudnesses[0] - original_loudnesses[0]) * (std::log(10) / 20));
            gains[1] = std::exp((after_loudnesses[1] - original_loudnesses[1]) * (std::log(10) / 20));
            
            for (int i = 0; i < config_.num_channels; i++) {
                output[i] = delay_filters_[i].Clock(input[i]) * gains[i];
            }
        }*/

        int delay_samples() const { return delay_samples_; }
    private:
       Config config_;
       int delay_samples_;
       std::vector<LoudnessFilter<Float>> loudness_filters_;
       std::vector<TimeVaryingLowpassFilter<Float>> lowpass_filters_;
       std::vector<DelayFilter<Float>> delay_filters_;
    };
}

#endif 