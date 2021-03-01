#include <vector>
#include <iostream>
#include <random>
#include "bakuage/memory.h"
#include "phase_limiter/GradCore.h"

void TestGradCoreFft();

template <class SimdType>
void TestGradCoreFft() {
#if 0
    typedef typename SimdType::element_type Float;
    
    auto x = &phase_limiter::impl::ThreadVar1<SimdType>::GetThreadInstance();
    const int len = 256;
    
    // FFTが直交変換であることを確かめる
    const double normalize_scale = 1.0 / std::sqrt(len);
    for (int i = 0; i < len; i++) {
        for (int j = 0; j < len; j++) {
            x->windowed()[j] = 0;
        }
        x->windowed()[i] = 1 * normalize_scale;
        x->doFFT(len);
        double ene = bakuage::Sqr(x->spec()[0]) + bakuage::Sqr(x->spec()[len]);
        for (int j = 1; j < len / 2; j++) {
            ene += 2 * (bakuage::Sqr(x->spec()[2 * j]) + bakuage::Sqr(x->spec()[2 * j + 1])); // 共役分もあるので2倍
        }
        if (std::abs(ene - 1) > 1e-7) {
            std::cerr << "grad core fft test failed " << i << " " << ene - 1 << std::endl;
        }
    }
    
    // IFFTが直交変換であることを確かめる
    for (int i = 0; i <= len; i++) {
        if (i == 1) continue;
        
        for (int j = 0; j <= len + 1; j++) {
            x->spec()[j] = 0;
        }
        x->spec()[i] = ((i == 0 || i == len) ? std::sqrt(1.0) : std::sqrt(0.5)) * normalize_scale; // 共役分もあるので半分
        x->doIFFT(len);
        double ene = 0;
        for (int j = 0; j < len; j++) {
            ene += bakuage::Sqr(x->windowed()[j]);
        }
        if (std::abs(ene - 1) > 2e-7) {
            std::cerr << "grad core ifft test failed " << i << " " << ene  -1 << std::endl;
        }
    }
    
#endif
    
    std::cerr << "grad core fft ifft test finished" << std::endl;
}

template <class SimdType>
void TestGradImpl() {
    typedef typename SimdType::element_type Float;
    using namespace phase_limiter;
    
    const int windowLen = 256;
    Float *wave = (Float *)bakuage::AlignedMalloc(sizeof(Float) * windowLen, PL_MEMORY_ALIGN);
    Float *waveSrc = (Float *)bakuage::AlignedMalloc(sizeof(Float) * windowLen, PL_MEMORY_ALIGN);
    Float *grad = (Float *)bakuage::AlignedMalloc(sizeof(Float) * windowLen, PL_MEMORY_ALIGN);
    Float *gradNumerical = (Float *)bakuage::AlignedMalloc(sizeof(Float) * windowLen, PL_MEMORY_ALIGN);
    const Float noise = 1e-6;
    
    std::mt19937 engine(1);
    std::normal_distribution<double> dist;
    for (int i = 0; i < windowLen; i++) {
#if 1
        wave[i] = dist(engine);
        waveSrc[i] = dist(engine);
#else
        wave[i] = 1 + i;
        waveSrc[i] = 2 + i;
#endif
    }
    
    const auto calc = [=]() {
        phase_limiter::GradContext<SimdType> context;
        return GradCore<SimdType>::calcEval23(GradOptions::Default(windowLen), wave, waveSrc, noise, &context);
    };
    
    const auto calcGrad = [=]() {
        phase_limiter::GradContext<SimdType> context;
        return GradCore<SimdType>::calcEvalGrad23(GradOptions::Default(windowLen), wave, waveSrc, grad, noise, &context);
    };
    
    const auto calcGradNumerically = [=](double delta) {
        const double waveEval = calc();
        Float *temp = (Float *)bakuage::AlignedMalloc(sizeof(Float) * windowLen, PL_MEMORY_ALIGN);
        for (int i = 0; i < windowLen; i++) {
            for (int j = 0; j < windowLen; j++) {
                temp[j] = wave[j];
            }
            temp[i] += delta;
            phase_limiter::GradContext<SimdType> context;
            const double neighborEval = GradCore<SimdType>::calcEval23(GradOptions::Default(windowLen), temp, waveSrc, noise, &context);
            //std::cerr << "result1:" << neighborEval << "\tresult2:" << waveEval << std::endl;
            gradNumerical[i] = (neighborEval - waveEval) / delta;
        }
        bakuage::AlignedFree(temp);
    };
    
    // test the output of calc and calcGrad is same
    {
        Float result1 = calc();
        Float result2 = calc();
        std::cerr << "result1:" << result1 << "\tresult2:" << result2 << std::endl;
    }
    
    // test the grad is correct
    {
        calcGrad();
        calcGradNumerically(0.01);
        double error = 0;
        for (int i = 0; i < windowLen; i++) {
            //std::cerr << "grad1:" << grad[i] << "\tgrad2:" << gradNumerical[i] << std::endl;
            error += bakuage::Sqr(grad[i] - gradNumerical[i]);
        }
        std::cerr << "grad_error:" << error << std::endl;
    }
    
    // test the output of calc and calcGrad is same
    {
        Float result1 = calc();
        Float result2 = calc();
        std::cerr << "result1:" << result1 << "\tresult2:" << result2 << std::endl;
    }
    
    // test the grad returns same value
    {
        calcGrad();
        std::vector<Float> v(grad, grad + windowLen);
        for (int i = 0; i < windowLen; i++) {
            grad[i] = 0;
        }
        calcGrad();
        double error = 0;
        for (int i = 0; i < windowLen; i++) {
            error += bakuage::Sqr(grad[i] - v[i]);
        }
        std::cerr << "grad twice error:" << error << std::endl;
    }
    
    // パフォーマンス
    {
        phase_limiter::GradContext<SimdType> context;
        GradCore<SimdType>::calcEvalGrad23(GradOptions::Default(windowLen), wave, waveSrc, grad, noise, &context);
        
        bakuage::StopWatch sw;
        sw.Start();
        for (int i = 0; i < 1000 * 100; i++) {
            GradCore<SimdType>::calcEvalGrad23(GradOptions::Default(windowLen), wave, waveSrc, grad, noise, &context);
        }
        std::cerr << "performance_sec:" << sw.Stop() << std::endl;
    }
    
    // splatのテスト
    {
        SimdType v1 = simdpp::splat(1);
        SimdType v2 = simdpp::splat(2);
        std::cerr << simdpp::extract<0>(v1) << " ";
        std::cerr << simdpp::extract<1>(v1) << " ";
        std::cerr << simdpp::extract<0>(v2) << " ";
        std::cerr << simdpp::extract<1>(v2) << " ";
        std::cerr << std::endl;
    }
}

void TestGrad() {
    TestGradImpl<simdpp::float32x4>();
    TestGradImpl<simdpp::float32x8>();
    TestGradImpl<simdpp::float64x2>();
    TestGradImpl<simdpp::float64x4>();
    TestGradCoreFft<simdpp::float32x4>();
    TestGradCoreFft<simdpp::float32x8>();
    TestGradCoreFft<simdpp::float64x2>();
    TestGradCoreFft<simdpp::float64x4>();
}
