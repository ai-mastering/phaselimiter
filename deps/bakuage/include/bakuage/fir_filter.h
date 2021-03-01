#ifndef BAKUAGE_BAKUAGE_FIR_FILTER_H_
#define BAKUAGE_BAKUAGE_FIR_FILTER_H_

#include <algorithm>
#include <cstdint>
#include <immintrin.h>
#include "bakuage/delay_filter.h"
#include "bakuage/memory.h"

namespace bakuage {
    template <typename Float = double>
    class FirFilter {
    public:
		FirFilter(int len) : fir_(len), delay_filter_(len) {}
        template <typename Iterator>
        FirFilter(Iterator bg, Iterator ed): fir_(bg, ed), delay_filter_(fir_.size()) {}

		template <typename Iterator>
		void UpdateFir(Iterator bg, Iterator ed) {
			std::copy(bg, ed, fir_.begin());
		}

        Float Clock(const Float &x) {
            delay_filter_.Clock(x);

            Float result = 0;
            for (int i = 0; i < fir_.size(); i++) {
                result += delay_filter_[i] * fir_[i];
            }
            return result;
        };
        
        void ClockWithoutResult(const Float &x) {
            delay_filter_.Clock(x);
        };
    private:
        std::vector<Float> fir_;
        DelayFilter<Float> delay_filter_;
    };

#ifdef __AVX__
//http://stackoverflow.com/questions/23189488/horizontal-sum-of-32-bit-floats-in-256-bit-avx-vector            
inline float _mm256_reduce_add_ps(__m256 x) {
    /* ( x3+x7, x2+x6, x1+x5, x0+x4 ) */
    const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    /* ( -, -, x1+x3+x5+x7, x0+x2+x4+x6 ) */
    const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    /* ( -, -, -, x0+x1+x2+x3+x4+x5+x6+x7 ) */
    const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    /* Conversion to float is a no-op on x86-64 */
    return _mm_cvtss_f32(x32);
}
#endif

inline float _mm_reduce_add_ps(__m128 x128) {
	/* ( -, -, x1+x3+x5+x7, x0+x2+x4+x6 ) */
	const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
	/* ( -, -, -, x0+x1+x2+x3+x4+x5+x6+x7 ) */
	const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
	/* Conversion to float is a no-op on x86-64 */
	return _mm_cvtss_f32(x32);
}

    template<>
    class FirFilter<float> {
    public:
		FirFilter(int len): fir_(len), delay_filter_(len) {}

        template <typename Iterator>
        FirFilter(Iterator bg, Iterator ed): fir_(bg, ed), delay_filter_(fir_.size()) {}

		template <typename Iterator>
		void UpdateFir(Iterator bg, Iterator ed) {
			std::copy(bg, ed, fir_.data());
		}

        float Clock(const float &x) {
            delay_filter_.Clock(x);

#ifdef __AVX__
            int len = 8 * (fir_.size() / 8);
		    __m256 sum = _mm256_set1_ps(0.0f);
            for (int i = 0; i < len; i += 8) {
                __m256 x = _mm256_loadu_ps(delay_filter_.data() + i);
                __m256 y = _mm256_load_ps(fir_.data() + i);
		        __m256 z = _mm256_mul_ps(x, y);
                sum = _mm256_add_ps(sum, z);
            }
            
            float result = 0;
            for (int i = len; i < fir_.size(); i++) {
                result += delay_filter_[i] * fir_[i];
            }
            return result + _mm256_reduce_add_ps(sum);
#else
			int len = 4 * (fir_.size() / 4);
			__m128 sum = _mm_set1_ps(0.0f);
			for (int i = 0; i < len; i += 4) {
				__m128 x = _mm_loadu_ps(delay_filter_.data() + i);
				__m128 y = _mm_load_ps(fir_.data() + i);
				__m128 z = _mm_mul_ps(x, y);
				sum = _mm_add_ps(sum, z);
			}

			float result = 0;
			for (int i = len; i < fir_.size(); i++) {
				result += delay_filter_[i] * fir_[i];
			}
			return result + _mm_reduce_add_ps(sum);
#endif
        };
        
        void ClockWithoutResult(const float &x) {
            delay_filter_.Clock(x);
        };
    private:
        AlignedPodVector<float> fir_;
        DelayFilter<float> delay_filter_;
    };
}

#endif 
