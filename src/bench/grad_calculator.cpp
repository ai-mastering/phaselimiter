#include <vector>
#include <iostream>
#include <random>
#include <benchmark/benchmark.h>
#include <tbb/tbb.h>
#include "bakuage/memory.h"
#include "bakuage/utils.h"
#include "phase_limiter/GradCalculator.h"
#include "bench/utils.h"

template <class SimdType>
void BM_GradCalculatorEval(benchmark::State& state) {
    typedef typename SimdType::element_type Float;
    
    tbb::task_scheduler_init tbb_init(state.range(0));
    
    const int waveLen = 44100 * 10;
    Float *wave = (Float *)bakuage::AlignedMalloc(sizeof(Float) * waveLen, PL_MEMORY_ALIGN);
    Float *waveProx = (Float *)bakuage::AlignedMalloc(sizeof(Float) * waveLen, PL_MEMORY_ALIGN);
    
    std::mt19937 engine(1);
    std::normal_distribution<Float> dist;
    for (int i = 0; i < waveLen; i++) {
        wave[i] = dist(engine);
        waveProx[i] = dist(engine);
    }
    
    phase_limiter::GradCalculator<SimdType> calculator(waveLen / 2, 44100, 44100, 0, "linear", 1e-6, 1, 0.5, 800, 1200, 1);
    
    Float *ptr_array[2];
    ptr_array[0] = &wave[0];
    ptr_array[1] = &wave[1];
    calculator.copyWaveSrcFrom(ptr_array, 2);
    ptr_array[0] = &waveProx[0];
    ptr_array[1] = &waveProx[1];
    calculator.copyWaveProxFrom(ptr_array, 2);
    // warmup
    for (int i = 0; i < 10; i++) {
        calculator.calcEval(0.01, calculator.emptyBeforeHook(), calculator.defaultWaveInputFunc());
    }
    
    for (auto _ : state) {
        calculator.calcEval(0.01, calculator.emptyBeforeHook(), calculator.defaultWaveInputFunc());
    }
    
    bakuage::AlignedFree(wave);
    bakuage::AlignedFree(waveProx);
    
    state.SetComplexityN(1.0 * waveLen / 44100 / state.range(0));
}
MY_SIMD_BENCH(BM_GradCalculatorEval, ->Arg(1)->Arg(2)->Arg(4)->Arg(6)->Arg(8)->Arg(12)->Complexity(benchmark::oN));

template <class SimdType>
static void BM_GradCalculatorGrad(benchmark::State& state) {
    typedef typename SimdType::element_type Float;
    
    tbb::task_scheduler_init tbb_init(state.range(0));
    
    const int waveLen = 44100 * 10;
    Float *wave = (Float *)bakuage::AlignedMalloc(sizeof(Float) * waveLen, PL_MEMORY_ALIGN);
    Float *waveProx = (Float *)bakuage::AlignedMalloc(sizeof(Float) * waveLen, PL_MEMORY_ALIGN);
    
    std::mt19937 engine(1);
    std::normal_distribution<Float> dist;
    for (int i = 0; i < waveLen; i++) {
        wave[i] = dist(engine);
        waveProx[i] = dist(engine);
    }
    
    phase_limiter::GradCalculator<SimdType> calculator(waveLen / 2, 44100, 44100, 0, "linear", 1e-6, 1, 0.5, 800, 1200, 1);
    
    Float *ptr_array[2];
    ptr_array[0] = &wave[0];
    ptr_array[1] = &wave[1];
    calculator.copyWaveSrcFrom(ptr_array, 2);
    ptr_array[0] = &waveProx[0];
    ptr_array[1] = &waveProx[1];
    calculator.copyWaveProxFrom(ptr_array, 2);
    calculator.calcEvalGrad(0.01, calculator.emptyBeforeHook(), calculator.defaultWaveInputFunc());
    // warmup
    for (int i = 0; i < 10; i++) {
        calculator.calcEvalGrad(0.01, calculator.emptyBeforeHook(), calculator.defaultWaveInputFunc());
    }
    
    for (auto _ : state) {
        calculator.calcEvalGrad(0.01, calculator.emptyBeforeHook(), calculator.defaultWaveInputFunc());
    }
    
    bakuage::AlignedFree(wave);
    bakuage::AlignedFree(waveProx);
    
    state.SetComplexityN(1.0 * waveLen / 44100 / state.range(0));
}
MY_SIMD_BENCH(BM_GradCalculatorGrad, ->Arg(1)->Arg(2)->Arg(4)->Arg(6)->Arg(8)->Arg(12)->Complexity(benchmark::oN));


