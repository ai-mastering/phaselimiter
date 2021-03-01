#ifndef BAKUAGE_BAKUAGE_FIR_RESAMPLE_FILTER_H_
#define BAKUAGE_BAKUAGE_FIR_RESAMPLE_FILTER_H_

#include "bakuage/delay_filter.h"
#include "bakuage/fir_design.h"
#include "bakuage/memory.h"
#include "bakuage/utils.h"

namespace bakuage {
    
    template <class Float>
    class FirResampleFilter {
    public:
        struct Config {
            int input_sample_rate;
            int output_sample_rate;
        };
        
        FirResampleFilter(const Config &config): config_(config), bypass_(config.input_sample_rate == config.output_sample_rate) {
            
            if (bypass_) {
                return;
            }
            
            const int lcm1 = lcm(config.input_sample_rate, config.output_sample_rate);
            up_factor_ = lcm1 / config.input_sample_rate;
            down_factor_ = lcm1 / config.output_sample_rate;
            
            const double transition_width = 2050.0 / lcm1;
            const double stopband_reduce_db = 70; // dB
            int filter_len;
            double alpha;
            bakuage::CalcKeiserFirParams(stopband_reduce_db, transition_width, &filter_len, &alpha);
            filter_len = bakuage::CeilInt<int>(filter_len - 1, lcm(2, lcm(up_factor_, down_factor_))) + 1;
            
            const double cut_freq = 0.5 * std::min<int>(config.input_sample_rate, config.output_sample_rate) / lcm1 - transition_width;
            const auto fir = bakuage::CalculateBandPassFir<double>(0, cut_freq, filter_len, alpha);
            fir_.resize(fir.size());
            std::copy(fir.begin(), fir.end(), fir_.begin());
            
            delay_filter_ = std::unique_ptr<bakuage::DelayFilter<Float>>(new bakuage::DelayFilter<Float>(down_factor_ + fir_.size() / up_factor_));
        }
        
        void Clock(const Float *input, Float *output) {
            if (bypass_) {
                *output = *input;
                return;
            }
            
            for (int i = 0; i < down_factor_; i++) {
                delay_filter_->Clock(input[i]);
            }
            
            for (int i = 0; i < up_factor_; i++) {
                Float sum = 0;
                const int output_pos = i * down_factor_;
                int k = 0;
                while (1) {
                    const int input_pos = (down_factor_ - 1 - k) * up_factor_;
                    const int fir_idx = output_pos - input_pos;
                    if (fir_idx >= fir_.size()) break;
                    
                    if (fir_idx >= 0) {
                        sum += fir_[fir_idx] * (*delay_filter_)[k];
                    }
                    k++;
                }
                output[i] = sum;
            }
        }
        
        int input_process_size() const {
            return bypass_ ? 1 : down_factor_;
        }
        
        int output_process_size() const {
            return bypass_ ? 1 : up_factor_;
        }
        
        int input_delay_samples() const {
            return bypass_ ? 0 : fir_.size() / 2 / up_factor_;
        }
        
        int output_delay_samples() const {
            return bypass_ ? 0 : fir_.size() / 2 / down_factor_;
        }
    private:
        Config config_;
        bool bypass_;
        int up_factor_;
        int down_factor_;
        bakuage::AlignedPodVector<Float> fir_;
        std::unique_ptr<bakuage::DelayFilter<Float>> delay_filter_;
    };
    
}

#endif
