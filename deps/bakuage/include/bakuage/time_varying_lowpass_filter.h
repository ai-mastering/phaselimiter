#ifndef BAKUAGE_BAKUAGE_TIME_VARYING_LOWPASS_FILTER_H_
#define BAKUAGE_BAKUAGE_TIME_VARYING_LOWPASS_FILTER_H_

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>
#include <iostream>

namespace bakuage {
    template <typename Float = double>
    class TimeVaryingLowpassFilter1 {
    public:
        TimeVaryingLowpassFilter1(Float _a = 1, Float initial_value = 0): integ_(initial_value), a_(_a), eps_(0) {}

        Float a() const { return a_; }
        void set_a(Float value) { a_ = value; }

        Float Clock(const Float &x) {
            integ_ = a_ * x + (1 - a_) * integ_;
			if (std::abs(integ_) < eps_) integ_ = 0;
            return integ_;
        };

        void Clear() { integ_ = 0; }

		void set_eps(Float value) { eps_ = value; }
        Float output() const { return integ_; }
    private:
        Float integ_;
        Float a_;
		Float eps_; // for denormal
    };

	template <typename Float = double>
    class TimeVaryingLowpassFilter {
    public:
        static Float CalculateAFromPeak(int order, const Float peak, int *delay_samples,
                TimeVaryingLowpassFilter *temp_filter = nullptr) {
            const double eps = 1e-40;            

            double mi = 1e300;
            double mi_a = 1;
            int mi_delay = 0;

            double step = 1;
            double a = eps;

            TimeVaryingLowpassFilter *filter = nullptr;
            if (!temp_filter) {
                filter = new TimeVaryingLowpassFilter(order);
                temp_filter = filter;
            }

            for (int i = 0; i < 100; i++) {   
                int delay;
                double p = CalculatePeak(order, a + step, &delay, temp_filter);
                if (p > peak) {
                    step *= 0.5;
                }
                else {
                    a += step;
                }
                double error = std::abs(p - peak) / (peak + eps);
                if (error < mi) {
                    mi = error;
                    mi_a = a;
                    mi_delay = delay;
                }
            }

            if (filter) {
                delete filter;
            }           
            
            *delay_samples = mi_delay;
            return mi_a;
        }

        TimeVaryingLowpassFilter(int order, Float _a = 1): filters_(order), output_(0) {
            for (int i = 0; i < order; i++) {
                filters_.emplace_back(_a);
            }
        }

        Float a() const { return filters_[0].a(); }
        void set_a(Float value) { 
            for (auto &filter: filters_) {
                filter.set_a(value);
            }
        }

        void Clear() { 
            for (auto &filter: filters_) {
                filter.Clear();
            }
        }

        Float Clock(const Float &x) {
            Float res = x;
            for (auto &filter: filters_) {
                res = filter.Clock(res);
            }
            output_ = res;
            return res;
        };
        
        Float output() const { return output_; }
    private:
        static double CalculatePeak(int order, const double a, int *delay_samples,
            TimeVaryingLowpassFilter *temp_filter) {
            double result;
            // ピーク計算
            {
                temp_filter->set_a(a);
                temp_filter->Clear();
                double b = temp_filter->Clock(1);
                double prev = 0;
                while (b > prev) {
                    prev = b;
                    b = temp_filter->Clock(0);
                }
                result = prev;
            }

            // delay計算
            {
                temp_filter->set_a(a);
                temp_filter->Clear();
                double b = temp_filter->Clock(1);
                double sum = 0;
                int delay = 0;
                while (sum < 0.5 && delay < 44100) {
                    b = temp_filter->Clock(0);
                    sum += b;
                    delay++;
                }
                *delay_samples = std::max<int>(1, delay);
            }

            return result;
        }

        std::vector<TimeVaryingLowpassFilter1<Float>> filters_;
        Float output_;
    };
}

#endif 
