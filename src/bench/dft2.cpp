#include <benchmark/benchmark.h>
#include <tbb/tbb.h>
#include "bakuage/dft.h"
#include "bakuage/memory.h"
#include "bakuage/vector_math.h"
#include "bench/utils.h"

namespace {
template <class Float>
struct RealDftTask {
    enum RealDftType {
        kCCSOutOfPlace = 1,
        kPermOutOfPlace = 2,
        kPermInPlace = 3,
        kPackOutOfPlace = 4,
        kPackInPlace = 5,
    };
    RealDftTask(int len, bool forward, RealDftType type): forward_(forward), type_(type), dft(len), input1(len), output1(len + 2){
        std::mt19937 engine(1);
        std::normal_distribution<Float> dist;
        for (int i = 0; i < len; i++) {
            input1[i] = dist(engine);
        }
    }
    void Execute() {
        switch (type_) {
            case kCCSOutOfPlace:
                if (forward_) {
                    dft.Forward(input1.data(), output1.data());
                } else {
                    dft.Backward(input1.data(), output1.data());
                }
                break;
            case kPermOutOfPlace:
                if (forward_) {
                    dft.ForwardPerm(input1.data(), output1.data());
                } else {
                    dft.BackwardPerm(input1.data(), output1.data());
                }
                break;
            case kPermInPlace:
                if (forward_) {
                    dft.ForwardPerm(input1.data(), input1.data());
                } else {
                    dft.BackwardPerm(input1.data(), input1.data());
                }
                break;
            case kPackOutOfPlace:
                if (forward_) {
                    dft.ForwardPack(input1.data(), output1.data());
                } else {
                    dft.BackwardPack(input1.data(), output1.data());
                }
                break;
            case kPackInPlace:
                if (forward_) {
                    dft.ForwardPack(input1.data(), input1.data());
                } else {
                    dft.BackwardPack(input1.data(), input1.data());
                }
                break;
        }
    }
private:
    bool forward_;
    RealDftType type_;
    bakuage::RealDft<Float> dft;
    bakuage::AlignedPodVector<Float> input1;
    bakuage::AlignedPodVector<Float> output1;
};
    
    template <class Float>
    void DoBench(benchmark::State& state, bool forward, typename RealDftTask<Float>::RealDftType type) {
        const int len = state.range(0);
        
        RealDftTask<Float> task(len, forward, type);
        
        // warmup
        task.Execute();
        
        for (auto _ : state) {
            task.Execute();
        }
        
        // float
        state.SetComplexityN(len);
    }
}

template <class Float>
static void BM_Dft2RealDftCCSForward(benchmark::State& state) {
    DoBench<Float>(state, true, RealDftTask<Float>::kCCSOutOfPlace);
}
MY_FLOAT_BENCH(BM_Dft2RealDftCCSForward, ->Arg(1 << 8)->Arg(1 << 9)->Arg(1 << 10)->Arg(1 << 11)->Arg(1 << 12)->Arg(1 << 13)->Arg(1 << 14)->Complexity(benchmark::oN));

template <class Float>
static void BM_Dft2RealDftCCSBackward(benchmark::State& state) {
    DoBench<Float>(state, false, RealDftTask<Float>::kCCSOutOfPlace);
}
MY_FLOAT_BENCH(BM_Dft2RealDftCCSBackward, ->Arg(1 << 8)->Arg(1 << 9)->Arg(1 << 10)->Arg(1 << 11)->Arg(1 << 12)->Arg(1 << 13)->Arg(1 << 14)->Complexity(benchmark::oN));

template <class Float>
static void BM_Dft2RealDftPermForward(benchmark::State& state) {
    DoBench<Float>(state, true, RealDftTask<Float>::kPermOutOfPlace);
}
MY_FLOAT_BENCH(BM_Dft2RealDftPermForward, ->Arg(1 << 8)->Arg(1 << 9)->Arg(1 << 10)->Arg(1 << 11)->Arg(1 << 12)->Arg(1 << 13)->Arg(1 << 14)->Complexity(benchmark::oN));

template <class Float>
static void BM_Dft2RealDftPermBackward(benchmark::State& state) {
    DoBench<Float>(state, false, RealDftTask<Float>::kPermOutOfPlace);
}
MY_FLOAT_BENCH(BM_Dft2RealDftPermBackward, ->Arg(1 << 8)->Arg(1 << 9)->Arg(1 << 10)->Arg(1 << 11)->Arg(1 << 12)->Arg(1 << 13)->Arg(1 << 14)->Complexity(benchmark::oN));

template <class Float>
static void BM_Dft2RealDftPermInplaceForward(benchmark::State& state) {
    DoBench<Float>(state, true, RealDftTask<Float>::kPermInPlace);
}
MY_FLOAT_BENCH(BM_Dft2RealDftPermInplaceForward, ->Arg(1 << 8)->Arg(1 << 9)->Arg(1 << 10)->Arg(1 << 11)->Arg(1 << 12)->Arg(1 << 13)->Arg(1 << 14)->Complexity(benchmark::oN));

template <class Float>
static void BM_Dft2RealDftPermInplaceBackward(benchmark::State& state) {
    DoBench<Float>(state, false, RealDftTask<Float>::kPermInPlace);
}
MY_FLOAT_BENCH(BM_Dft2RealDftPermInplaceBackward, ->Arg(1 << 8)->Arg(1 << 9)->Arg(1 << 10)->Arg(1 << 11)->Arg(1 << 12)->Arg(1 << 13)->Arg(1 << 14)->Complexity(benchmark::oN));

template <class Float>
static void BM_Dft2RealDftPackForward(benchmark::State& state) {
    DoBench<Float>(state, true, RealDftTask<Float>::kPackOutOfPlace);
}
MY_FLOAT_BENCH(BM_Dft2RealDftPackForward, ->Arg(1 << 8)->Arg(1 << 9)->Arg(1 << 10)->Arg(1 << 11)->Arg(1 << 12)->Arg(1 << 13)->Arg(1 << 14)->Complexity(benchmark::oN));

template <class Float>
static void BM_Dft2RealDftPackBackward(benchmark::State& state) {
    DoBench<Float>(state, false, RealDftTask<Float>::kPackOutOfPlace);
}
MY_FLOAT_BENCH(BM_Dft2RealDftPackBackward, ->Arg(1 << 8)->Arg(1 << 9)->Arg(1 << 10)->Arg(1 << 11)->Arg(1 << 12)->Arg(1 << 13)->Arg(1 << 14)->Complexity(benchmark::oN));

template <class Float>
static void BM_Dft2RealDftPackInplaceForward(benchmark::State& state) {
    DoBench<Float>(state, true, RealDftTask<Float>::kPackInPlace);
}
MY_FLOAT_BENCH(BM_Dft2RealDftPackInplaceForward, ->Arg(1 << 8)->Arg(1 << 9)->Arg(1 << 10)->Arg(1 << 11)->Arg(1 << 12)->Arg(1 << 13)->Arg(1 << 14)->Complexity(benchmark::oN));

template <class Float>
static void BM_Dft2RealDftPackInplaceBackward(benchmark::State& state) {
    DoBench<Float>(state, false, RealDftTask<Float>::kPackInPlace);
}
MY_FLOAT_BENCH(BM_Dft2RealDftPackInplaceBackward, ->Arg(1 << 8)->Arg(1 << 9)->Arg(1 << 10)->Arg(1 << 11)->Arg(1 << 12)->Arg(1 << 13)->Arg(1 << 14)->Complexity(benchmark::oN));
