#ifndef BAKUAGE_BAKUAGE_FIR_FILTER3_H_
#define BAKUAGE_BAKUAGE_FIR_FILTER3_H_

#include <algorithm>
#include "bakuage/delay_filter.h"
#include "bakuage/memory.h"
#include "bakuage/vector_math.h"

// FirFilterと同じだが、vector_mathを使うので速い

namespace bakuage {
    template <typename Float = double>
    class FirFilter3 {
    public:
        FirFilter3(int len) : fir_(len), delay_filter_(len) {}
        template <typename Iterator>
        FirFilter3(Iterator bg, Iterator ed): fir_(bg, ed), delay_filter_(fir_.size()) {}
        
        template <typename Iterator>
        void UpdateFir(Iterator bg, Iterator ed) {
            std::copy(bg, ed, fir_.begin());
        }
        
        Float Clock(const Float &x) {
            delay_filter_.Clock(x);
            return VectorDot(fir_.data(), delay_filter_.data(), fir_.size());
        };
        
        void ClockWithoutResult(const Float &x) {
            delay_filter_.Clock(x);
        };
    private:
        bakuage::AlignedPodVector<Float> fir_;
        DelayFilter<Float> delay_filter_;
    };
}

#endif

