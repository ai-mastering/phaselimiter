#ifndef BAKUAGE_BAKUAGE_TRANSIENT_FILTER_H_
#define BAKUAGE_BAKUAGE_TRANSIENT_FILTER_H_

#include <algorithm>
#include <limits>
#include "bakuage/delay_filter.h"
#include "bakuage/loudness_filter.h"
#include "bakuage/time_varying_lowpass_filter.h"

namespace bakuage {
    
    template <typename Float, typename GainFunc>
    class TransientFilter {
    public:
        class Config {
        public:
            Config(): num_channels(0), sample_rate(0), long_mean_sec(0), short_mean_sec(0) {}
            int num_channels;
            int sample_rate;
            float long_mean_sec;
            float short_mean_sec;
            GainFunc gain_func;
        };
        
        TransientFilter(const Config &config):
        config_(config) {
            const int lowpass_filter_order = 2;
            const Float long_peak = std::min<Float>(1.0, 1.0 / (config_.sample_rate * config_.long_mean_sec
                                                                + 1e-30));
            const Float long_a = TimeVaryingLowpassFilter<Float>::CalculateAFromPeak(
                                                                                     lowpass_filter_order, long_peak, &long_delay_samples_);
            const Float short_peak = std::min<Float>(1.0, 1.0 / (config_.sample_rate * config_.short_mean_sec
                                                                 + 1e-30));
            const Float short_a = TimeVaryingLowpassFilter<Float>::CalculateAFromPeak(
                                                                                      lowpass_filter_order, short_peak, &short_delay_samples_);
            short_loudness_delay_filter_ = std::unique_ptr<DelayFilter<Float>>(new DelayFilter<Float>(long_delay_samples_ - short_delay_samples_));
            for (int i = 0; i < config_.num_channels; i++) {
                loudness_filters_.emplace_back(config_.sample_rate);
                long_lowpass_filters_.emplace_back(lowpass_filter_order, long_a);
                short_lowpass_filters_.emplace_back(lowpass_filter_order, short_a);
                delay_filters_.emplace_back(long_delay_samples_);
            }
        }
        
        // inputとoutputが同じアドレスでもOK (一個ずれとかはダメ)
        void Clock(const Float *input, Float *output) {
            Float long_loudness, short_loudness;
            Analyze(input, &long_loudness, &short_loudness);
            
            const Float gain_db = config_.gain_func(long_loudness, short_loudness);
            const Float gain = std::exp(gain_db * (std::log(10) / 20));
            
            for (int i = 0; i < config_.num_channels; i++) {
                output[i] = delay_filters_[i].Clock(input[i]) * gain;
            }
        };
        
        void Analyze(const Float *input, Float *long_loudness, Float *short_loudness) {
            Float long_rms = 0;
            Float short_rms = 0;
            for (int i = 0; i < config_.num_channels; i++) {
                const Float filtered = loudness_filters_[i].Clock(input[i]);
                const Float sqr_filtered = filtered * filtered;
                long_rms += long_lowpass_filters_[i].Clock(sqr_filtered);
                short_rms += short_lowpass_filters_[i].Clock(sqr_filtered);
            }
            *long_loudness = -0.691 + 10 * std::log10(long_rms + 1e-37);
            *short_loudness = -0.691 + 10 * std::log10(short_loudness_delay_filter_->Clock(short_rms) + 1e-37);
        };
        
        int delay_samples() const { return long_delay_samples_; }
    private:
        Config config_;
        int long_delay_samples_;
        int short_delay_samples_;
        std::vector<LoudnessFilter<Float>> loudness_filters_;
        std::vector<TimeVaryingLowpassFilter<Float>> long_lowpass_filters_;
        std::vector<TimeVaryingLowpassFilter<Float>> short_lowpass_filters_;
        std::unique_ptr<DelayFilter<Float>> short_loudness_delay_filter_;
        std::vector<DelayFilter<Float>> delay_filters_;
    };
}

#endif /* BAKUAGE_BAKUAGE_TRANSIENT_FILTER_H_ */
