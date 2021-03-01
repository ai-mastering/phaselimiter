
#include <benchmark/benchmark.h>
#include "bakuage/memory.h"

void BM_TypedMemcpy(benchmark::State& state) {
    bakuage::AlignedPodVector<uint8_t> src(state.range(0));
    bakuage::AlignedPodVector<uint8_t> dest(state.range(0));
    
    for (auto _ : state) {
        bakuage::TypedMemcpy(dest.data(), src.data(), state.range(0));
    }
    
}
BENCHMARK(BM_TypedMemcpy)->Ranges({{ 1 << 10, 1 << 20 }});

void BM_TypedFillZero(benchmark::State& state) {
    bakuage::AlignedPodVector<uint8_t> dest(state.range(0));
    
    for (auto _ : state) {
        bakuage::TypedFillZero(dest.data(), state.range(0));
    }
    
}
BENCHMARK(BM_TypedFillZero)->Ranges({{ 1 << 10, 1 << 20 }});

