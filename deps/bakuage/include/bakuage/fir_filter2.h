#ifndef BAKUAGE_BAKUAGE_FIR_FILTER2_H_
#define BAKUAGE_BAKUAGE_FIR_FILTER2_H_

#include <algorithm>
#include <complex>
#include <cstring>
#include "bakuage/dft.h"
#include "bakuage/memory.h"
#include "bakuage/vector_math.h"
#include "bakuage/utils.h"

namespace bakuage {
    // overlap addで高速にたたみこむフィルター
    // Float must be pod
    
    template <typename Float = double>
    class FirFilter2 {
    public:
        FirFilter2(int len):
        fir_size_(len),
        fir_spec_(ExtendedSpecSizeFromFirSize(len)),
        fir_spec_double_(ExtendedSpecSizeFromFirSize(len)),
        work_(ExtendedSizeFromFirSize(len)),
        work_double_(ExtendedSizeFromFirSize(len)),
        work_spec_(ExtendedSpecSizeFromFirSize(len)),
        output_buffer_(2 * ExtendedSizeFromFirSize(len)),
        output_buffer_pos_(0),
        dft_(ExtendedSizeFromFirSize(len)),
        dft_double_(ExtendedSizeFromFirSize(len))
        {}
        
        template <typename Iterator>
        FirFilter2(Iterator bg, Iterator ed): FirFilter2(std::distance(bg, ed)) {
            UpdateFir(bg, ed);
        }
        
        // サイズ変更はできない
        template <typename Iterator>
        void UpdateFir(Iterator bg, Iterator ed) {
            // calc fir_spec_ (doubleで計算してからfloatに戻すことで計算精度を上げる)
            std::copy(bg, ed, work_double_.begin());
            TypedFillZero(work_double_.data() + fir_size(), extended_size() - fir_size());
            
            // fir_spec_にはnormalize_scaleも含める
            dft_double_.ForwardPerm(work_double_.data(), (double *)fir_spec_double_.data());
            const double normalize_scale = std::pow(extended_size(), -1);
            for (int i = 0; i < fir_spec_double_.size(); i++) {
                fir_spec_[i] = fir_spec_double_[i] * normalize_scale;
            }
        }
        
        // clear state
        void Clear() {
            bakuage::TypedFillZero(output_buffer_.data(), output_buffer_.size());
            output_buffer_pos_ = 0;
        }
        
        void Clock(const Float *bg, const Float *ed, Float *output) {
            const int max_process_len = extended_size() - fir_size() + 1;
            while (ed - bg > max_process_len) {
                DoClock(bg, bg + max_process_len, output);
                bg += max_process_len;
                output += max_process_len;
            }
            if (ed - bg > 0) {
                DoClock(bg, ed, output);
            }
        };
    private:
        static int ExtendedSizeFromFirSize(int fir_size) {
            return CeilPowerOf2(2 * fir_size);
        }
        static int ExtendedSpecSizeFromFirSize(int fir_size) {
            // perm
            return ExtendedSizeFromFirSize(fir_size) / 2;// + 1;
        }
        
        // ed - bg <= max_process_len
        void DoClock(const Float *bg, const Float *ed, Float *output) {
            const int len = ed - bg;
            const int convolved_size = fir_size() + len - 1;
            
            // fft src
            std::memcpy(work_.data(), bg, sizeof(Float) * len);
            std::memset(work_.data() + len, 0, sizeof(Float) * (extended_size() - len));
#if 1
            dft_.ForwardPerm(work_.data(), (Float *)work_spec_.data());
#endif
            
#if 1
            // multiply spec
            VectorMulPermInplace(fir_spec_.data(), work_spec_.data(), extended_spec_size());
#endif
            
            // ifft
            dft_.BackwardPerm((Float *)work_spec_.data(), work_.data());
            
#if 1
            // shift if needed
            if (output_buffer_pos_ + convolved_size > output_buffer_.size()) {
                const int remaining_size = output_buffer_.size() - output_buffer_pos_;
                std::memmove(output_buffer_.data(), output_buffer_.data() + output_buffer_pos_, sizeof(Float) * remaining_size);
                std::memset(output_buffer_.data() + output_buffer_.size() - output_buffer_pos_, 0, sizeof(Float) * output_buffer_pos_);
                output_buffer_pos_ = 0;
            }
            
            // add and output
            VectorAdd(work_.data(), &output_buffer_[output_buffer_pos_], output, len);
            
            // add to buffer
            VectorAddInplace(work_.data() + len, &output_buffer_[output_buffer_pos_ + len], convolved_size - len);
            
            // bufferを進める
            output_buffer_pos_ += len;
#endif
        };
        
        int fir_size() const { return fir_size_; }
        int extended_size() const { return work_.size(); }
        int extended_spec_size() const { return fir_spec_.size(); }
        
        // doubleとついているのは計算精度を上げるために一時的に使う
        
        int fir_size_;
        AlignedPodVector<std::complex<Float>> fir_spec_; // extended_spec_size
        AlignedPodVector<std::complex<double>> fir_spec_double_; // extended_spec_size
        AlignedPodVector<Float> work_; // extended_size
        AlignedPodVector<double> work_double_; // extended_size
        AlignedPodVector<std::complex<Float>> work_spec_; // extended_spec_size
        AlignedPodVector<Float> output_buffer_; // fir_size
        int output_buffer_pos_;
        Float normalize_scale_;
        RealDft<Float> dft_;
        RealDft<double> dft_double_;
    };
}

#endif
