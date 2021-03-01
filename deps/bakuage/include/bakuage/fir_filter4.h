#ifndef BAKUAGE_BAKUAGE_FIR_FILTER4_H_
#define BAKUAGE_BAKUAGE_FIR_FILTER4_H_

#include <algorithm>
#include <complex>
#include <cstring>
#include "bakuage/memory.h"

namespace bakuage {
    // IPPのmulti-rate filterのラッパー
    // Float must be float, double, std::complex<float>, std::complex<double>
    
    // ClearMixer4の場合は、FirFilter2のほうが速いっぽい
    // あと、delay lineの扱いはまだちゃんと検証していないから、もし使う場合は、process size 1でうまく処理できるか確かめてから本番投入
    
    struct FirFilter4ImplBase {
        virtual ~FirFilter4ImplBase() {}
    };
    
    template <class Float = double>
    class FirFilter4 {
    public:
        FirFilter4(const FirFilter4& x): up_factor_(x.up_factor_), down_factor_(x.down_factor_), fir_(x.fir_) {
            PrepareFir(fir_.data(), fir_.data() + fir_.size(), up_factor_, down_factor_);
        }
        FirFilter4(FirFilter4&& x): up_factor_(x.up_factor_), down_factor_(x.down_factor_), fir_(std::move(x.fir_)), impl_(std::move(x.impl_)) {}
        FirFilter4& operator=(const FirFilter4& x) {
            up_factor_ = x.up_factor_;
            down_factor_ = x.down_factor_;
            fir_ = x.fir_;
            PrepareFir(fir_.data(), fir_.data() + fir_.size(), up_factor_, down_factor_);
            return *this;
        }
        FirFilter4& operator=(FirFilter4&& x) {
            up_factor_ = x.up_factor_;
            down_factor_ = x.down_factor_;
            fir_ = std::move(x.fir_);
            impl_ = std::move(x.impl_);
            return *this;
        }
        
        template <typename Iterator>
        FirFilter4(Iterator bg, Iterator ed, int up_factor = 1, int down_factor = 1): up_factor_(up_factor), down_factor_(down_factor), fir_(bg, ed) {
            PrepareFir(fir_.data(), fir_.data() + fir_.size(), up_factor, down_factor);
        }
        virtual ~FirFilter4() {}
        
        void Clock(const Float *bg, const Float *ed, Float *output);
    private:
        void PrepareFir(const Float *bg, const Float *ed, int up_factor, int down_factor);
        
        int up_factor_;
        int down_factor_;
        bakuage::AlignedPodVector<Float> fir_;
        std::unique_ptr<FirFilter4ImplBase> impl_;
    };
}

#endif
