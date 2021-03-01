#ifndef BAKUAGE_BAKUAGE_PAN_DETECT_FILTER_H_
#define BAKUAGE_BAKUAGE_PAN_DETECT_FILTER_H_

#include <cstring>
#include <cmath>
#include <complex>
#include <vector>
#include "bakuage/memory.h"
#include "bakuage/utils.h"
#include "bakuage/biquad_iir_filter.h"
#include "bakuage/gammatone_filter.h"
#include "bakuage/dft.h"
#include "bakuage/vector_math.h"

namespace bakuage {

    template <typename Float = double>
    class PanDetectFilter {
    public:       
        struct Output {
            bool valid;
            Float energy;
            Float pan;
        };

        PanDetectFilter(Float center_freq, Float band_width, Float sample_rate): 
            center_freq_(center_freq),
            band_width_(band_width),
            sample_rate_(sample_rate)
        {
            mean_time_ = 0.2; //sec
            ild_range_db_ = 20; //dB
            itd_range_sec_ = 800e-6;  //sec
            window_size_sec_ = 0.020; //sec
            shift_size_sec_ = 0.010; //sec
            decimated_sample_rate_ = 6000;
            lowpass_freq_ = 1500;

            decimation_ = std::max<Float>(1, std::floor(sample_rate / decimated_sample_rate_));
            decimation_pos_ = 0;
            
            buffer_size_ = (int)(sample_rate / decimation_ * window_size_sec_);
            shift_size_ = (int)(sample_rate / decimation_ * shift_size_sec_);
            buffer_pos_ = 0;

            for (int i = 0; i < 2; i++) {                
                auditory_filter_[i] = GammatoneFilter<Float>(center_freq, band_width, sample_rate);

                lowpass_filter_[i] = BiquadIIRFilter<Float>(
                    BiquadIIRCoef<Float>::CreateLowpass(lowpass_freq_ / sample_rate, 1));
               
                filtered_buffer_[i].resize(buffer_size_);
            }

            // 総エネルギーがかわらないように、sqrt(hannng)窓 + (TODO: windowの正規化も行う)
            for (int i = 0; i < buffer_size_; i++) {
                window_.push_back(std::sqrt((std::max)(0.0, 0.5 - 0.5 * std::cos(2 * M_PI * i / buffer_size_))));
            }

            // 直流を取るのでエネルギーが0.5倍(対称)～1倍(正)になることの補正
            abs_compensation_ = std::sqrt(2);

            for (int i = 0; i < 3; i++) {
                fft_input_buf_[i].resize(buffer_size_);
                fft_output_buf_[i].resize(buffer_size_ / 2 + 1);
            }
            dft_ = std::unique_ptr<bakuage::RealDft<Float>>(new bakuage::RealDft<Float>(buffer_size_));
        }

        Output Clock(Float left, Float right) {
            Output output;
            output.valid = false;
            
            Float left_abs = lowpass_filter_[0].Clock(std::max<Float>(0, auditory_filter_[0].Clock(left)));   
            Float right_abs = lowpass_filter_[1].Clock(std::max<Float>(0, auditory_filter_[1].Clock(right)));         
            if (decimation_pos_ == 0) {
                filtered_buffer_[0][buffer_pos_] = left_abs * abs_compensation_;
                filtered_buffer_[1][buffer_pos_] = right_abs * abs_compensation_;
            }

            decimation_pos_++;
            if (decimation_pos_ >= decimation_) {
                decimation_pos_ = 0;
                buffer_pos_++;
                if (buffer_pos_ == buffer_size_) {
                    output = CalculatePan();

                    for (int i = 0; i < buffer_size_ - shift_size_; i++) {
                        filtered_buffer_[0][i] = filtered_buffer_[0][i + shift_size_];
                        filtered_buffer_[1][i] = filtered_buffer_[1][i + shift_size_];
                    }
                    buffer_pos_ -= shift_size_;
                }
            }

            return output;
        }

        Float band_width() const { return band_width_; }
		Float center_freq() const { return center_freq_; }

    private:
        PanDetectFilter(const PanDetectFilter& other); // non construction-copyable
        PanDetectFilter& operator=(const PanDetectFilter&); // non copyable

        int GetBuf3Index(int a) {
            return a + (a < 0 ? buffer_size_ : 0);
        }

        Output CalculatePan() {
            for (int i = 0; i < 2; i++) {
                bakuage::VectorMul(filtered_buffer_[i].data(), window_.data(), fft_input_buf_[i].data(), buffer_size_);
            }
            
            const Float left_energy = bakuage::Sqr(bakuage::VectorL2(fft_input_buf_[0].data(), buffer_size_));
            const Float right_energy = bakuage::Sqr(bakuage::VectorL2(fft_input_buf_[1].data(), buffer_size_));
            for (int i = 0; i < 2; i++) {
                dft_->Forward(fft_input_buf_[i].data(), (Float *)fft_output_buf_[i].data());
            }

            Float r = 1.0 / buffer_size_; // 相互相関関数を計算するためのFFT補正係数
            int spec_size = buffer_size_ / 2 + 1;

            // 相互相関関数を計算
#if 0
            for (int i = 0; i < spec_size; i++) {
                fft_output_buf_[2][i] = fft_output_buf_[0][i] * std::conj(fft_output_buf_[1][i]) * r;
            }
#else
            bakuage::VectorMulConj(fft_output_buf_[0].data(), fft_output_buf_[1].data(), fft_output_buf_[2].data(), spec_size);
            bakuage::VectorMulConstantInplace(r, fft_output_buf_[2].data(), spec_size);
#endif
            dft_->Backward((Float *)fft_output_buf_[2].data(), fft_input_buf_[2].data());

            // 相互相関関数の最大値を探す
            Float *buf3 = fft_input_buf_[2].data();
            Float ma = buf3[0];
            Float ma_index = 0;
            int range = (std::min)((int)(buffer_size_ * itd_range_sec_ / (buffer_size_ / sample_rate_ * decimation_)), 
                (buffer_size_ - 1) / 2 - 1);
            for (int i = -range; i <= range; i++) {
                Float w = buf3[GetBuf3Index(i)];
                if (ma < w) {
                    ma = w;
                    ma_index = i;
                }
            }
            Float c, s;
            Parabora(ma, buf3[GetBuf3Index(ma_index + 1)], buf3[GetBuf3Index(ma_index - 1)], &c, &s);
            ma = std::max<Float>(0, s); //負の相関は相関無しとみなす
            Float itd = -(ma_index + c) / buffer_size_ * (buffer_size_ / sample_rate_ * decimation_);
            itd = std::max<Float>(-itd_range_sec_, std::min<Float>(itd_range_sec_, itd)) * (90  / itd_range_sec_);

            Float w;
            if (center_freq_ > 8000) {
                w = 0.5;
            }
            else {
                w = 18 - std::log(center_freq_ / 50) / std::log(8000 / 50) * 17.5;
            }

            Output output;
            output.valid = true;
            output.energy = left_energy + right_energy;
            output.pan = -(90 * left_energy - 90 * right_energy + itd * ma * w)
                / (left_energy + right_energy + ma * w + 1e-15);   

            return output;
        }

        void Parabora(Float a, Float s, Float t, Float *x, Float *y) {
            Float b = (s - t) * 0.5;
            Float c = (s + t) * 0.5 - a;
            if (c == 0) {
                *x = 0;
                *y = a;
            }
            else {
                *x = -b / (2 * c);
                *y = a - (*x) * (*x) * c;
            }
        }

        BiquadIIRFilter<Float> lowpass_filter_[2];
        GammatoneFilter<Float> auditory_filter_[2];
        bakuage::AlignedPodVector<Float> filtered_buffer_[2];
        bakuage::AlignedPodVector<Float> window_;
        Float abs_compensation_;

        bakuage::AlignedPodVector<Float> fft_input_buf_[3];
        bakuage::AlignedPodVector<std::complex<Float>> fft_output_buf_[3];
        std::unique_ptr<bakuage::RealDft<Float>> dft_;

        Float center_freq_;
        Float band_width_;
        Float sample_rate_;

        Float mean_time_; //sec
        Float ild_range_db_; //dB
        Float itd_range_sec_;  //sec
        Float window_size_sec_; //sec
        Float shift_size_sec_; //sec
        int decimated_sample_rate_;
        Float lowpass_freq_;

        int decimation_;
        int decimation_pos_;
        int buffer_size_;
        int shift_size_;
        int buffer_pos_;
    };
}

#endif 
