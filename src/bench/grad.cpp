#include <vector>
#include <iostream>
#include <random>
#include <benchmark/benchmark.h>
#include "bakuage/memory.h"
#include "phase_limiter/GradCore.h"
#include "bench/utils.h"

template <class SimdType>
void BM_Eval(benchmark::State& state) {
    typedef typename SimdType::element_type Float;
    
    const int windowLen = state.range(0);
    Float *wave = (Float *)bakuage::AlignedMalloc(sizeof(Float) * windowLen, PL_MEMORY_ALIGN);
    Float *waveSrc = (Float *)bakuage::AlignedMalloc(sizeof(Float) * windowLen, PL_MEMORY_ALIGN);
    const Float noise = 0.01;
    std::mt19937 engine(1);
    std::normal_distribution<Float> dist;
    for (int i = 0; i < windowLen; i++) {
        wave[i] = dist(engine);
        waveSrc[i] = dist(engine);
    }
    phase_limiter::GradContext<SimdType> context;
    phase_limiter::GradCore<SimdType>::calcEval23(phase_limiter::GradOptions::Default(windowLen), wave, waveSrc, noise, &context);
    
    for (auto _ : state) {
        phase_limiter::GradCore<SimdType>::calcEval23(phase_limiter::GradOptions::Default(windowLen), wave, waveSrc, noise, &context);
    }
    
    bakuage::AlignedFree(wave);
    bakuage::AlignedFree(waveSrc);
    
    state.SetComplexityN(state.range(0));
}
MY_SIMD_BENCH(BM_Eval, ->Arg(1 << 8)->Arg(1 << 10)->Arg(1 << 12)->Arg(1 << 14)->Complexity(benchmark::oN));

template <class SimdType>
static void BM_Grad(benchmark::State& state) {
    typedef typename SimdType::element_type Float;
    
    const int windowLen = state.range(0);
    Float *wave = (Float *)bakuage::AlignedMalloc(sizeof(Float) * windowLen, PL_MEMORY_ALIGN);
    Float *waveSrc = (Float *)bakuage::AlignedMalloc(sizeof(Float) * windowLen, PL_MEMORY_ALIGN);
    Float *grad = (Float *)bakuage::AlignedMalloc(sizeof(Float) * windowLen, PL_MEMORY_ALIGN);
    const Float noise = 0.01;
    std::mt19937 engine(1);
    std::normal_distribution<Float> dist;
    for (int i = 0; i < windowLen; i++) {
        wave[i] = dist(engine);
        waveSrc[i] = dist(engine);
    }
    phase_limiter::GradContext<SimdType> context;
    phase_limiter::GradCore<SimdType>::calcEvalGrad23(phase_limiter::GradOptions::Default(windowLen), wave, waveSrc, grad, noise, &context);
    
    for (auto _ : state) {
        phase_limiter::GradCore<SimdType>::calcEvalGrad23(phase_limiter::GradOptions::Default(windowLen), wave, waveSrc, grad, noise, &context);
    }
    
    bakuage::AlignedFree(wave);
    bakuage::AlignedFree(waveSrc);
    bakuage::AlignedFree(grad);
    
    state.SetComplexityN(state.range(0));
}
MY_SIMD_BENCH(BM_Grad, ->Arg(1 << 8)->Arg(1 << 10)->Arg(1 << 12)->Arg(1 << 14)->Complexity(benchmark::oN));
