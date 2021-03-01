
#ifndef phase_limiter_config_h
#define phase_limiter_config_h

// 必ずconfig.hを経由してロードする (src/bakuageからも使うので、CMakeで指定するようにした)
// #define SIMDPP_ARCH_X86_AVX
// #define SIMDPP_ARCH_X86_AVX2
// #define SIMDPP_ARCH_X86_FMA3

#include <simdpp/simd.h>
#include "bakuage/memory.h"

#define PL_FFT_MAX_LEN (1 << 14)
#define PL_MAX_WORKER_COUNT 32
#define PL_PERFORMANCE_COUNTER
#define PL_CACHE_LINE_SIZE 64
#define PL_MEMORY_ALIGN PL_CACHE_LINE_SIZE

namespace phase_limiter {
    typedef simdpp::float32x8 DefaultSimdType;
    // typedef simdpp::float64x4 DefaultSimdType;
    
    inline void *Malloc(int size) {
        return bakuage::AlignedMalloc(size, PL_CACHE_LINE_SIZE);
    }
    
    template <class T>
    T *TypedMalloc(int len) {
        return (T *)Malloc(sizeof(T) * len);
    }
    
#if 0
    template <class T>
    T *TypedRealloc(T *ptr, int len) {
        return (T *)scalable_aligned_realloc(ptr, sizeof(T) * len, PL_MEMORY_ALIGN);
    }
#endif
    
    inline void Free(void *ptr) {
        return bakuage::AlignedFree(ptr);
    }
    
#if 0
    // streaming loadはhaswellでは意味ないらしい
    template <class SimdType>
    inline SimdType StreamLoad(const void *memory) {
        return simdpp::load(memory);
    }
    
    template <>
    inline simdpp::float32x8 StreamLoad(const void *memory) {
        return (__m256)_mm256_stream_load_si256((const __m256i *)memory);
    }
    
    template <>
    inline simdpp::float64x4 StreamLoad(const void *memory) {
        return (__m256d)_mm256_stream_load_si256((const __m256i *)memory);
    }
    
    template <class SimdType>
    inline void StreamLoadPacked2(SimdType &v1, SimdType &v2, const void *memory) {
        simdpp::load_packed2(v1, v2, memory);
    }
    
    template <>
    inline void StreamLoadPacked2(simdpp::float32x8 &v1, simdpp::float32x8 &v2, const void *memory) {
        simdpp::float32x8 raw1, raw2;
        raw1 = (__m256)_mm256_stream_load_si256((const __m256i *)memory);
        raw2 = (__m256)_mm256_stream_load_si256((const __m256i *)memory + 1);
        v1 = simdpp::unzip4_lo(raw1, raw2);
        v2 = simdpp::unzip4_hi(raw1, raw2);
    }
    
    template <>
    inline void StreamLoadPacked2(simdpp::float64x4 &v1, simdpp::float64x4 &v2, const void *memory) {
        simdpp::float64x4 raw1, raw2;
        raw1 = (__m256d)_mm256_stream_load_si256((const __m256i *)memory);
        raw2 = (__m256d)_mm256_stream_load_si256((const __m256i *)memory + 1);
        v1 = simdpp::unzip2_lo(raw1, raw2);
        v2 = simdpp::unzip2_hi(raw1, raw2);
    }
#endif
    
    // あまりパフォーマンスかわらないからいいや(一応少し速くなる) (そもそもlinuxでバグるので使わない)
    // アセンブラを確認したらsimdpp::fmadd(a, b, c)でfmaが生成されない。自前で書いてみる
    // -> っと思ったらコンパイラオプションで-mfmaを指定していなかっただけだった
    // なぜか #ifdef SIMDPP_ARCH_X86_FMA3だとうまくいかないので注意
    template <class SimdType>
    inline SimdType Fmadd(const SimdType a, const SimdType b, const SimdType c) {
#ifdef BA_FMA_ENABLED
        return simdpp::fmadd(a, b, c);
#else
        return a * b + c;
#endif
    }
    
    template <class SimdType>
    inline SimdType Fmsub(const SimdType a, const SimdType b, const SimdType c) {
#ifdef BA_FMA_ENABLED
        return simdpp::fmsub(a, b, c);
#else
        return a * b - c;
#endif
    }
 
#if 0
    // 逆に遅くなる
    inline void ClFlush(void *memory) {
        _mm_clflush((char *)memory);
    }
#endif
    
#if defined (__GNUC__)
#define IACA_SSC_MARK( MARK_ID )                        \
__asm__ __volatile__ (                                    \
"\n\t  movl $"#MARK_ID", %%ebx"    \
"\n\t  .byte 0x64, 0x67, 0x90"    \
: : : "memory" );
    
#else
#define IACA_SSC_MARK(x) {__asm  mov ebx, x\
__asm  _emit 0x64 \
__asm  _emit 0x67 \
__asm  _emit 0x90 }
#endif
    
#define IACA_START {IACA_SSC_MARK(111)}
#define IACA_END {IACA_SSC_MARK(222)}
    
#ifdef _WIN64
#include <intrin.h>
#define IACA_VC64_START __writegsbyte(111, 111);
#define IACA_VC64_END   __writegsbyte(222, 222);
#endif
}

#endif
