#ifndef BAKUAGE_BAKUAGE_DELAY_FILTER2_H_
#define BAKUAGE_BAKUAGE_DELAY_FILTER2_H_

#include <cstring>
#include <algorithm>
#include <vector>
#include "bakuage/memory.h"
#include "bakuage/utils.h"

namespace bakuage {
    // DelayFilterの高速化版
    // dataや[]は実装していないが、実装するとしたらDelayFilterのように反転させないようにしたい
    // delayは固定
    // pod only
    template <typename Float = double>
    class DelayFilter2 {
    public:
        DelayFilter2(int max_delay, int delay = -1):
        max_delay_(max_delay),
        pos_(max_delay_ - 1),
        buffer_(3 * max_delay) {}
        
        void Clock(const Float *bg, const Float *ed, Float *output) {
            const auto len = ed - bg;
            if (max_delay_ == 0) {
                TypedMemmove(output, bg, len);
                return;
            }
            
            // inplaceに対応するために処理順を工夫している
            
            const auto buffer_to_output_len = std::min<int>(max_delay_, len);
            const auto input_to_output_len = len - buffer_to_output_len;
            
            // copy input to buffer
            if (pos_ + buffer_to_output_len >= buffer_.size()) {
                // まだbufferを出力していないので、max_delay_分コピーする
                TypedMemmove(&buffer_[0], &buffer_[pos_ - (max_delay_ - 1)], max_delay_);
                pos_ = max_delay_ - 1;
            }
            TypedMemcpy(&buffer_[pos_ + 1], bg + input_to_output_len, buffer_to_output_len);
            
            // copy input to output
            TypedMemmove(output + buffer_to_output_len, bg, input_to_output_len);
            
            // copy buffer to output
            TypedMemcpy(output, &buffer_[pos_ - (max_delay_ - 1)], buffer_to_output_len);
            
            pos_ += buffer_to_output_len;
        }
        
        int max_delay() { return max_delay_; }
    private:
        int max_delay_;
        int pos_;
        AlignedPodVector<Float> buffer_;
    };
}

#endif
