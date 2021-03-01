#include <vector>
#include <iostream>
#include <random>
#include "bakuage/memory.h"
#include "bakuage/utils.h"
#include "GradCalculator.h"

template <class SimdType>
void TestGradCalculatorImpl() {
    typedef typename SimdType::element_type Float;
    
    const int waveLen = 44100 * 10;
    Float *wave = (Float *)bakuage::AlignedMalloc(sizeof(Float) * waveLen, PL_MEMORY_ALIGN);
    Float *waveProx = (Float *)bakuage::AlignedMalloc(sizeof(Float) * waveLen, PL_MEMORY_ALIGN);
    Float *grad1 = (Float *)bakuage::AlignedMalloc(sizeof(Float) * waveLen, PL_MEMORY_ALIGN);
    Float *grad2 = (Float *)bakuage::AlignedMalloc(sizeof(Float) * waveLen, PL_MEMORY_ALIGN);
    
    std::mt19937 engine(1);
    std::normal_distribution<double> dist;
    for (int i = 0; i < waveLen; i++) {
        wave[i] = dist(engine);
        waveProx[i] = dist(engine);
        grad1[i] = 0;
        grad2[i] = 0;
    }
    
    phase_limiter::GradCalculator<SimdType> calculator(waveLen / 2, 44100, 22050, 0, "linear", 1e-6, 1, 0.5, 800, 1200, 1);
    
    Float *ptr_array[2];
    ptr_array[0] = &wave[0];
    ptr_array[1] = &wave[1];
    calculator.copyWaveSrcFrom(ptr_array, 2);
    ptr_array[0] = &waveProx[0];
    ptr_array[1] = &waveProx[1];
    calculator.copyWaveProxFrom(ptr_array, 2);
    
    // 2回読んでも結果が変わらないことの確認
    {
        double eval1 = calculator.calcEval(0.01, calculator.emptyBeforeHook(), calculator.defaultWaveInputFunc()).eval;
        double eval2 = calculator.calcEval(0.01, calculator.emptyBeforeHook(), calculator.defaultWaveInputFunc()).eval;
        std::cerr << "result1:" << eval1 << "\tresult2:" << eval2 << std::endl;
    }
        
   // 2回読んでもgradが変わらないことの確認
    {
        Float *ptr_array1[2];
        ptr_array1[0] = &grad1[0];
        ptr_array1[1] = &grad1[1];
        Float *ptr_array2[2];
        ptr_array2[0] = &grad2[0];
        ptr_array2[1] = &grad2[1];
        
        double eval1 = calculator.calcEvalGrad(0.01, calculator.emptyBeforeHook(), calculator.defaultWaveInputFunc()).eval;
        calculator.copyGradTo(ptr_array1, 2);
        double eval2 = calculator.calcEvalGrad(0.01, calculator.emptyBeforeHook(), calculator.defaultWaveInputFunc()).eval;
        calculator.copyGradTo(ptr_array2, 2);
        
        double energy = 0;
        double error = 0;
        for (int i = 0; i < waveLen; i++) {
            energy += bakuage::Sqr(grad1[i]);
            error += bakuage::Sqr(grad1[i] - grad2[i]);
        }
        
        std::cerr << "result1:" << eval1 << "\tresult2:" << eval2 << std::endl;
        std::cerr << "energy:" << energy << "\terror:" << error << std::endl;
    }
    
    // パフォーマンス
    {
        bakuage::StopWatch sw;
        sw.Start();
        for (int i = 0; i < 10; i++) {
            calculator.calcEvalGrad(0.01, calculator.emptyBeforeHook(), calculator.defaultWaveInputFunc());
        }
        std::cerr << "performance_sec:" << sw.Stop() << std::endl;
    }
}


void TestGradCalculator() {
    TestGradCalculatorImpl<simdpp::float32x4>();
    TestGradCalculatorImpl<simdpp::float32x8>();
    TestGradCalculatorImpl<simdpp::float64x2>();
    TestGradCalculatorImpl<simdpp::float64x4>();
}
