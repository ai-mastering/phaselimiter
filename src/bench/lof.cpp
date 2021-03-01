
#include <benchmark/benchmark.h>
#include "bakuage/memory.h"
#include "bakuage/vector_math.h"
#include "bakuage/lof.h"

namespace {
    typedef float Float;
    typedef bakuage::AlignedPodVector<Float> Point;
    
    struct DistFunc {
        double operator () (const Point &a, const Point &b) {
            return bakuage::VectorNormDiffL2(a.data(), b.data(), a.size());
        }
    };
}

void BM_LofPrepare(benchmark::State& state) {
    const int point_count = state.range(0);
    const int dim = 200;
    std::vector<Point> points(point_count);
    for (int i = 0; i < point_count; i++) {
        points[i].resize(dim);
        for (int j = 0; j < dim; j++) {
            points[i][j] = rand();
        }
    }
    for (auto _ : state) {
        DistFunc func;
        bakuage::Lof<Point, double, DistFunc> lof(func);
        lof.Prepare(points.begin(), points.end(), 100);
    }
}
BENCHMARK(BM_LofPrepare)->Ranges({{ 1 << 12, 1 << 13 }});

void BM_LofQuery(benchmark::State& state) {
    const int point_count = state.range(0);
    const int dim = 200;
    std::vector<Point> points(point_count);
    for (int i = 0; i < point_count; i++) {
        points[i].resize(dim);
        for (int j = 0; j < dim; j++) {
            points[i][j] = rand();
        }
    }
    DistFunc func;
    bakuage::Lof<Point, double, DistFunc> lof(func);
    lof.Prepare(points.begin(), points.end(), 100);
    int k = 0;
    for (auto _ : state) {
        lof.CalculateLof(points[(k++) % points.size()]);
    }
}
BENCHMARK(BM_LofQuery)->Ranges({{ 1 << 12, 1 << 13 }});
