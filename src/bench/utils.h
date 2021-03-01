//
//  utils.h
//  bakuage_native_executables
//
//  Created by mac2 on 2018/09/30.
//

#ifndef bench_utils_h
#define bench_utils_h

#define MY_FLOAT_BENCH(name, args) \
BENCHMARK_TEMPLATE(name, float)args; \
BENCHMARK_TEMPLATE(name, double)args;

#define MY_SIMD_BENCH(name, args) \
BENCHMARK_TEMPLATE(name, simdpp::float32x4)args; \
BENCHMARK_TEMPLATE(name, simdpp::float32x8)args; \
BENCHMARK_TEMPLATE(name, simdpp::float64x2)args; \
BENCHMARK_TEMPLATE(name, simdpp::float64x4)args;

#endif /* utils_h */
