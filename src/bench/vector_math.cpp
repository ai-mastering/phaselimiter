
#include <benchmark/benchmark.h>
#include "bakuage/memory.h"
#include "bakuage/vector_math.h"

void BM_VectorMove(benchmark::State& state) {
    bakuage::AlignedPodVector<float> src(state.range(0) / 4);
    bakuage::AlignedPodVector<float> dest(state.range(0) / 4);
    
    for (auto _ : state) {
        bakuage::VectorMove(src.data(), dest.data(), state.range(0) / 4);
    }
    
}
BENCHMARK(BM_VectorMove)->Ranges({{ 1 << 10, 1 << 20 }});

void BM_VectorZero(benchmark::State& state) {
    bakuage::AlignedPodVector<float> dest(state.range(0) / 4);
    
    for (auto _ : state) {
        bakuage::VectorZero(dest.data(), state.range(0) / 4);
    }
    
}
BENCHMARK(BM_VectorZero)->Ranges({{ 1 << 10, 1 << 20 }});


