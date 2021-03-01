#ifndef BAKUAGE_BAKUAGE_DELAY_FILTER_H_
#define BAKUAGE_BAKUAGE_DELAY_FILTER_H_

#include <cstring>
#include <algorithm>
#include <vector>
#include "bakuage/memory.h"
#include "bakuage/utils.h"

namespace bakuage {
    // pod only
    template <typename Float = double>
    class DelayFilter {
    public:
        DelayFilter(int max_delay, int delay = -1): 
            max_delay_(max_delay),
            delay_(delay), 
            pos_(3 * max_delay - 1 - max_delay_), 
            buffer_(3 * max_delay) { 
            if (delay == -1) {
                delay_ = max_delay_;
            }
        }     

        Float Clock(const Float &x) {
            if (max_delay_ == 0) return x;

            pos_--;
            if (pos_ < 0) {
                pos_ = buffer_.size() - max_delay_ - 1;
                TypedMemmove(&buffer_[pos_ + 1], &buffer_[0], max_delay_);
            }

            buffer_[pos_] = x;
            return buffer_[pos_ + delay_];
        }

        Float operator [] (int delay) {
            return buffer_[pos_ + delay];
        }

        Float *data() {
            return buffer_.data() + pos_;
        }
        
        void set_delay(int delay) {
            delay_ = (std::max)(0, std::min(max_delay_, delay));
        }

        int max_delay() { return max_delay_; }
    private:
        int max_delay_;
        int delay_;
        int pos_;
        AlignedPodVector<Float> buffer_;
    };
}

#endif 
