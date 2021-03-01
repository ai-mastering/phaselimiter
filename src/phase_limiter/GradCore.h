#ifndef PHASE_LIMITER_GRAD_CORE_H_
#define PHASE_LIMITER_GRAD_CORE_H_

// 方針は将来的にはopenclに移行するから
// これはなるべく触らないで壊さない方針で行く

#define PHASE_LIMITER_GRAD_CORE_USE_FFT_PERM

#include <cmath>
#include <stdint.h>
#include <cstring>
#include <immintrin.h>
#include <algorithm>
#include <iostream>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <tbb/scalable_allocator.h>
#include "bakuage/dft.h"
#include "bakuage/memory.h"
#include "bakuage/utils.h"
#include "bakuage/vector_math.h"
#include "phase_limiter/config.h"

namespace phase_limiter {

template <class SimdType>
struct GradContext {
    GradContext()
    //, nonZeroSpecSrcCacheLen(0)
    {}
    void clear() {
        specSrcCache.resize(0);
        // nonZeroSpecSrcCacheLen = 0;
    }
    bakuage::AlignedPodVector<typename SimdType::element_type> specSrcCache; // メモリはspecSrcCacheで一括alloc
    //int nonZeroSpecSrcCacheLen;
};

struct GradOptions {
    static GradOptions Default(int len) {
        GradOptions options;
        options.len = len;
        options.sample_rate = 44100;
        options.max_available_freq = 44100;
        return options;
    }
    GradOptions(): len(0), sample_rate(0), max_available_freq(0) {}
    int len;
    int sample_rate;
    int max_available_freq; // これ以上は0とみなす。高域が無い音源やオーバーサンプル時に使う
};

class GradCoreSettings {
public:
    GradCoreSettings(): erb_eval_func_weighting_(false), src_cache_(false), absolute_min_noise_(0) {}
    static GradCoreSettings &GetInstance() {
        static GradCoreSettings instance;
        return instance;
    }
    void set_erb_eval_func_weighting(bool value) { erb_eval_func_weighting_ = value; }
    bool erb_eval_func_weighting() const { return erb_eval_func_weighting_; }
    void set_src_cache(bool value) { src_cache_ = value; }
    bool src_cache() const { return src_cache_; }
    void set_absolute_min_noise(float value) { absolute_min_noise_ = value; }
    float absolute_min_noise() const { return absolute_min_noise_; }
private:
    bool erb_eval_func_weighting_;
    bool src_cache_;
    float absolute_min_noise_;
    static GradCoreSettings instance_;
};

    namespace impl {
        // SimdType + 長さごとに一個
        template <class SimdType>
        struct PreCalc1 {
            typedef typename SimdType::element_type Float;
            int len;
            Float *window;
            Float *evalFunc1Weights; // 評価関数1(位相込みのもの)ようの重み
            Float *evalFunc2Weights; // 評価関数2(位相込みのもの)ようの重み
            Float *noiseWeights; // noise用の重み(外部から指定したnoise以下であることを保証するために、この重みは1以下)
            Float *sumWeights; // evalFunc1Weights + evalFunc2Weights
            PreCalc1(int _len, int sample_rate): len(_len), memory_(len + 4 * (len / 2 + SimdType::length)) {
                using namespace bakuage;
                std::stringstream ss;
                ss << "PreCalc1 constructor" << std::endl;
                std::cerr << ss.str();
                
                // + SimdType::lengthは、FFTの最後の項を処理するためのもの
                window = memory_.data();
                evalFunc1Weights = window + len;
                evalFunc2Weights = evalFunc1Weights + len / 2 + SimdType::length;
                noiseWeights = evalFunc2Weights + len / 2 + SimdType::length;
                sumWeights = noiseWeights + len / 2 + SimdType::length;
                
                const double normalization_scale = 1.0 / std::sqrt(len);
                for (int i = 0; i < len; i++) {
                    window[i] = (Float)((0.5 - 0.5 * std::cos(2.0 * M_PI * (double)i / (double)len)) * normalization_scale);
                }
                
                for (int i = 0; i < len / 2 + SimdType::length; i++) {
                    double freq = sample_rate * (double)i / (double)len;
                    
                    const auto evalFunc1Ratio = 1.0 / bakuage::Sqr(1.0 + freq / (1.0 / 0.008));
                    double totalWeight;
                    if (GradCoreSettings::GetInstance().erb_eval_func_weighting()) {
                        totalWeight = 1.0 / (1.0 + freq / 100.0);
                    } else {
                        totalWeight = 1.0;
                    }
                    const double fft_compensation = (0 < i && i < len / 2) ? 2 : 1; // (共役分の補正、evalにのみかかる。詳細は後述)
                    
                    evalFunc1Weights[i] = totalWeight * evalFunc1Ratio * fft_compensation;
                    evalFunc2Weights[i] = totalWeight * (1 - evalFunc1Ratio) * fft_compensation;
                    
                    // noise weighting (評価関数を最適化しやすい形にする(球に近くする))
                    if (GradCoreSettings::GetInstance().erb_eval_func_weighting()) {
                        noiseWeights[i] = std::sqrt(1.0 / (1.0 + freq / 100.0));
                    } else {
                        noiseWeights[i] = 1.0;
                    }
                    
                    sumWeights[i] = evalFunc1Weights[i] + evalFunc2Weights[i];
                    
                    // 注意 (ややこしいから良く考えて)
                    // fft_compensationは本来はevalのみにきくもので、gradにはかけるべきではない
                    // でも、たまたまgradに微分の結果生まれる定数項(2)があり、
                    // これはi == 0以外はfft_compensationと一致する
                    // なので、gradにもかける。i == 0の場合は、微分の定数項がかからなくなってしまうので、
                    // ループ後に個別に調整する(2をかける)
                    // なんでこんなことをやるかというと、命令数削減のため
                }
            }
            ~PreCalc1() {
                std::stringstream ss;
                ss << "PreCalc1 destructor" << std::endl;
                std::cerr << ss.str();
            }
        private:
            bakuage::AlignedPodVector<Float> memory_;
        };
        
        template <class SimdType>
        struct PreCalc1Manager {
            PreCalc1Manager() {
                std::stringstream ss;
                ss << "PreCalc1Manager constructor" << std::endl;
                std::cerr << ss.str();
            }
            ~PreCalc1Manager() {
                std::stringstream ss;
                ss << "PreCalc1Manager destructor" << std::endl;
                std::cerr << ss.str();
            }
            static PreCalc1Manager &GetInstance() {
                static PreCalc1Manager instance;
                return instance;
            }
            
            // slow
            std::shared_ptr<PreCalc1<SimdType>> GetPreCalc1(int len, int sample_rate) {
                std::lock_guard<std::mutex> lock(mtx_);
                
                const int key = 32 * sample_rate + bakuage::IntLog2(len);
                auto found = pre_calcs_[key].lock();
                if (found) {
                    return found;
                } else {
                    auto pre_calc1 = std::allocate_shared<PreCalc1<SimdType>>(pre_calc1_allocator_, len, sample_rate);
                    pre_calcs_[key] = pre_calc1;
                    return pre_calc1;
                }
            }
        private:
            std::mutex mtx_;
            tbb::scalable_allocator<PreCalc1<SimdType>> pre_calc1_allocator_;
            std::unordered_map<int, std::weak_ptr<PreCalc1<SimdType>>> pre_calcs_;
        };
        
        // スレッドごとに一つ
        template <class SimdType>
        class ThreadVar1 {
        public:
            typedef typename SimdType::element_type Float;
            
            ThreadVar1(): dft_pool_(&bakuage::ThreadLocalDftPool<bakuage::RealDft<Float>>::GetThreadInstance()) {
                std::stringstream ss;
                ss << "ThreadVar1 constructor" << std::endl;
                std::cerr << ss.str();
            }
            ~ThreadVar1() {
                std::stringstream ss;
                ss << "ThreadVar1 destructor" << std::endl;
                std::cerr << ss.str();
            }
            
            static ThreadVar1& GetThreadInstance() {
                static thread_local ThreadVar1 instance;
                return instance;
            }
            
            void Reserve(int len) {
                // 実数FFTの分も含む
                // FFTの最後の項を処理するために2 * SimdType::length(8)を足している
                windowed_.resize(std::max(windowed_.size(), len * sizeof(typename SimdType::element_type)));
                spec_.resize(std::max(spec_.size(), (len + 2 * SimdType::length) * sizeof(typename SimdType::element_type)));
                specSrc_.resize(std::max(specSrc_.size(), (len + 2 * SimdType::length) * sizeof(typename SimdType::element_type)));
            }
            
            bakuage::RealDft<Float> *GetDft(int len) { return dft_pool_->Get(len); }
            void *GetDftWork() { return dft_pool_->work(); }
            
            // thread_localの呼び出し回数を減らすための工夫
            PreCalc1<SimdType> *GetPreCalc1(int len, int sample_rate) {
                const int key = 32 * sample_rate + bakuage::IntLog2(len);
                auto result = pre_calcs_[key].get();
                if (!result) {
                    pre_calcs_[key] = PreCalc1Manager<SimdType>::GetInstance().GetPreCalc1(len, sample_rate);
                    return pre_calcs_[key].get();
                }
                return result;
            }
            
            Float *windowed() { return windowed_.data(); }
            Float *spec() { return spec_.data(); }
            Float *specSrc() { return specSrc_.data(); }
        private:
            bakuage::ThreadLocalDftPool<bakuage::RealDft<Float>> *dft_pool_;
            bakuage::AlignedPodVector<Float> windowed_;
            bakuage::AlignedPodVector<Float> spec_;
            bakuage::AlignedPodVector<Float> specSrc_;
            std::unordered_map<int, std::shared_ptr<PreCalc1<SimdType>>> pre_calcs_;
        };

        //http://stackoverflow.com/questions/992471/how-to-query-ift-int-with-template-class
        struct GradEnabled {
            static constexpr bool value = true;
        };
        struct GradDisabled {
            static constexpr bool value = false;
        };

        struct NoiseWeightingEnabled {
            static constexpr bool value = true;
        };
        struct NoiseWeightingDisabled {
            static constexpr bool value = false;
        };

        template <class SimdType>
        inline SimdType calcNormSqr(const SimdType r, const SimdType i) {
            return Fmadd<SimdType>(r, r, i * i);
        }
        
#if 1
        template <class SimdType>
        inline SimdType FastRcp(const SimdType x) {
            return simdpp::splat<SimdType>(1.0) / x;
        }
     
        // Skylakeではあまり速くならなかった (IACAで解析)。Haswellでは、sqrtはそのまま、こっちの最適化だけで速くなる
        // でも、実際に全体としては速くならない。(前後のFFTによってsqrtやdivのレイテンシーが隠れている？ -> 十分長い長さでも速くならない)
        // 2.3.5 VDIV / VSQRT Latency in https://software.intel.com/sites/default/files/managed/3d/23/intel-architecture-code-analyzer-3.0-users-guide.pdf
        // vdivとvsqrtは保守的に見積もるらしい
        // -> これら単独ではボトルネックになっていなくて、instruction数を削るしかないということか？
        template <>
        inline simdpp::float32x8 FastRcp<simdpp::float32x8>(const simdpp::float32x8 x) {
            simdpp::float32x8 rcp = simdpp::rcp_e(x);
#if 0
            return rcp;
#else
            return simdpp::rcp_rh(rcp, x);
#endif
        }
        
#if 0
        template <>
        inline simdpp::arch_avx::float32<8u> FastRcp<simdpp::arch_avx::float32<8u>>(const simdpp::arch_avx::float32<8u> x) {
            auto rcp = simdpp::rcp_e(x);
#if 0
            return rcp;
#else
            return simdpp::rcp_rh(rcp, x);
#endif
        }
#endif
#endif
        
#if 1
        template <class SimdType>
        inline SimdType FastSqrt(const SimdType x) {
            return simdpp::sqrt(x);
        }
        
        // あまり速くならなかった
        template <>
        inline simdpp::float32x8 FastSqrt<simdpp::float32x8>(const simdpp::float32x8 x) {
            return simdpp::rcp_e(simdpp::rsqrt_e(x));
        }
#endif

        template <class SimdType, class Grad, class NoiseWeighting, class WaveInputFunc, class GradOutputFunc>
        typename SimdType::element_type func(const GradOptions &options, const WaveInputFunc &wave_input_func, const typename SimdType::element_type *waveSrc, const GradOutputFunc &grad_output_func, typename SimdType::element_type _noise,
                                             std::vector<int> *histogram, GradContext<SimdType> *context) {
            const int len = options.len;
            const int sample_rate = options.sample_rate;
            
            typedef typename SimdType::element_type Float;

            auto tv = &impl::ThreadVar1<SimdType>::GetThreadInstance();
            tv->Reserve(len);

            // windowedは共有している。つまり、FFTでwindowed <-> spec, windowed <-> specSrcになるようにしている
            Float *windowed = tv->windowed();
            Float *spec = tv->spec();
            Float *specSrc = tv->specSrc();
            auto dft = tv->GetDft(len);
            auto dft_work = tv->GetDftWork();
            const auto *pre1 = tv->GetPreCalc1(len, sample_rate);

            // wave 窓関数 + FFT
#if 1
            bakuage::VectorMul(wave_input_func(0), pre1->window, windowed, len);
            std::memset(spec + len, 0, 2 * SimdType::length * sizeof(typename SimdType::element_type));
#endif
#if 1
            // for (int k = 0; k < 2; k++)
            // CCSかPermのout-of-placeが最速 (bench/dft2.cpp調べ)
            dft->Forward(windowed, spec, dft_work);
#endif

            // waveSrc 窓関数 + FFT
#if 1
            const int nonZeroSpecSrcLen = bakuage::CeilInt<int>(std::min<int>(len + 2, 2 * std::ceil((double)len * options.max_available_freq / sample_rate + 1)), 2 * SimdType::length);
#else
            const int nonZeroSpecSrcLen = len + 2 * SimdType::length;
#endif
#if 0
            std::stringstream ss;
            ss << nonZeroSpecSrcLen << "\t" << len << "\t";
            std::cout << ss.str() << std::endl;
#endif
#if 1
            if (GradCoreSettings::GetInstance().src_cache()) {
                if (!context->specSrcCache.size()) {
                    bakuage::VectorMul(waveSrc, pre1->window, windowed, len);
                    std::memset(specSrc + len, 0, 2 * SimdType::length * sizeof(typename SimdType::element_type));
                    dft->Forward(windowed, specSrc, dft_work);
                    
#if 1
                    context->specSrcCache.resize(nonZeroSpecSrcLen);
                    bakuage::TypedMemcpy(context->specSrcCache.data(), specSrc, nonZeroSpecSrcLen);
#else
                    // STFTだから高域が無いはずでも実質non zeroがほとんど出ない
                    context->nonZeroSpecSrcCacheLen = 0;
                    for (int i = len / 2; i >= 0; i--) {
                        if (std::norm(specSrc[i]) > 1e-12) {
                            context->nonZeroSpecSrcCacheLen = bakuage::CeilInt<int>(2 * (i + 1), 2 * SimdType::length);
                            std::cerr << context->nonZeroSpecSrcCacheLen << " " << len + 2 * SimdType::length << std::endl;
                            break;
                        }
                    }
                    context->specSrcCache = TypedMalloc<Float>(context->nonZeroSpecSrcCacheLen);
                    bakuage::TypedMemcpy(context->specSrcCache, specSrc, nonZeroSpecSrcLen);
#endif
                }
                specSrc = context->specSrcCache.data();
            }
            else {
                bakuage::VectorMul(waveSrc, pre1->window, windowed, len);
                std::memset(specSrc + len, 0, 2 * SimdType::length * sizeof(typename SimdType::element_type));
                dft->Forward(windowed, specSrc, dft_work);
            }
#endif

            if (histogram) {
                histogram->resize(200);
                for (int i = 0; i < len / 2; i++) {
                    double normSqr = specSrc[2 * i + 0] * specSrc[2 * i + 0]
                    + specSrc[2 * i + 1] * specSrc[2 * i + 1];
                    double log2NormSqr = std::log(normSqr + 1e-37) / std::log(2.0);
                    int index = std::max(0, std::min(199, 100 + (int)log2NormSqr));
                    (*histogram)[index]++;
                }
            }

            SimdType eval = simdpp::splat<SimdType>(0.0);
            const SimdType absolute_min_noise = simdpp::splat<SimdType>(GradCoreSettings::GetInstance().absolute_min_noise());
#if 1
            for (int i = 0; i < nonZeroSpecSrcLen; i += 2 * SimdType::length) {
#else
            for (int i = 0; i < len + 2 * SimdType::length; i += 2 * SimdType::length) {
            // for (int i = 0; i < 0; i += 2 * SimdType::length) {
#endif
                // IACA_START
                
                // ループの内側においたほうが速いらしい (測定誤差かも)
                const SimdType one = simdpp::splat<SimdType>(1.0);
                const SimdType eps = simdpp::splat<SimdType>(1e-37);
                // const __m256 eps2 = _mm256_set1_ps(1e-20f);
                //__m256 absMask = _mm256_set1_epi32(0x7FFFFFFF);
                const SimdType noise = simdpp::splat<SimdType>(_noise);

                //ロードしてシャッフル (0がreal, 1がimage)
                SimdType grad0, grad1, spec0, spec1, specSrc0, specSrc1;
                simdpp::load_packed2(spec0, spec1, spec + i);
#if 0
                simdpp::load_packed2(specSrc0, specSrc1, specSrc + i);
#else
                if (i < nonZeroSpecSrcLen) {
                    simdpp::load_packed2(specSrc0, specSrc1, specSrc + i);
                } else {
                    specSrc0 = simdpp::splat<SimdType>(0);
                    specSrc1 = simdpp::splat<SimdType>(0);
                }
#endif

                //ノルム関係
                const SimdType normSqr = calcNormSqr(spec0, spec1);
                const SimdType normSrcSqr = calcNormSqr(specSrc0, specSrc1);
#if 0
                const SimdType norm = FastSqrt(normSqr);
                const SimdType normSrc = FastSqrt(normSrcSqr);
#else
                const SimdType norm = simdpp::sqrt(normSqr);
                const SimdType normSrc = simdpp::sqrt(normSrcSqr);
#endif

                //差の計算
                const SimdType diff0 = spec0 - specSrc0;
                const SimdType diff1 = spec1 - specSrc1;

                //重み
                const SimdType evalFunc1Weight = simdpp::load(pre1->evalFunc1Weights + i / 2);
                const SimdType evalFunc2Weight = simdpp::load(pre1->evalFunc2Weights + i / 2);
                SimdType baseWeight;
                if (NoiseWeighting::value) {
                    baseWeight = Fmadd<SimdType>(noise, simdpp::load<SimdType>(pre1->noiseWeights + i / 2), normSrc + absolute_min_noise);
                } else {
                    baseWeight = noise + normSrc + absolute_min_noise;
                }

#if 0
                baseWeight = FastRcp<SimdType>(baseWeight * baseWeight);
#else
                baseWeight = one / (baseWeight * baseWeight);
#endif
                const SimdType weight1 = baseWeight * evalFunc1Weight;
                const SimdType weight2 = baseWeight * evalFunc2Weight;

                //評価関数1
#if 1
                const SimdType diffNormSqr = calcNormSqr(diff0, diff1);
                eval = Fmadd<SimdType>(diffNormSqr, weight1, eval);
                if (Grad::value) {
                    grad0 = diff0 * weight1;
                    grad1 = diff1 * weight1;
                }
#else
                grad0 = simdpp::splat<SimdType>(0.0f);
                grad1 = simdpp::splat<SimdType>(0.0f);
#endif

                //評価関数2
#if 1
                const SimdType normDiff = norm - normSrc;
                const SimdType normDiffSqr = normDiff * normDiff;
                eval = Fmadd<SimdType>(normDiffSqr, weight2, eval);
                if (Grad::value) {
                    /*//ゲタ処理をやる(AVX2が無いと辛い)
                     __m256 normGrad = _mm256_div_ps(_mm256_mul_ps(normDiff, weight2), _mm256_add_ps(norm, eps));
                     grad0 = _mm256_add_ps(grad0, _mm256_mul_ps(spec0, normGrad));
                     grad1 = _mm256_add_ps(grad1, _mm256_mul_ps(spec1, normGrad));*/

                    //とりあえずの解決
#if 0
                    // したのコードの最適化 (挙動変わる。epsを少し大きくすると似た挙動になる。安定性を重視して、したのコードのままにするか)
                    const __m256 normGrad = _mm256_div_ps(_mm256_mul_ps(normDiff, weight2), _mm256_add_ps(norm, eps));
                    grad0 = _mm256_add_ps(grad0, _mm256_mul_ps(spec0, normGrad));
                    grad1 = _mm256_add_ps(grad1, _mm256_mul_ps(spec1, normGrad));
#else
#if 0
                    const SimdType invNorm = FastRcp<SimdType>(norm + eps);
#else
                    const SimdType invNorm = one / (norm + eps);
#endif
                    spec0 = spec0 * invNorm;
                    spec1 = spec1 * invNorm;
                    const SimdType normGrad = normDiff * weight2;
                    grad0 = Fmadd<SimdType>(spec0, normGrad, grad0);
                    grad1 = Fmadd<SimdType>(spec1, normGrad, grad1);
#endif
                }
#endif

                //勾配書き込み
                if (Grad::value) {
#if 0
                    // 微分の定数項(2倍) -> weightに含めたから実行は不要
                    grad0 = grad0 + grad0;
                    grad1 = grad1 + grad1;
#endif
                    simdpp::store_packed2(spec + i, grad0, grad1);
                }
                
                // IACA_END
            }
           
#if 1
            // 上のループより3倍くらい速い
            // for (int i = 0; i < len + 2 * SimdType::length; i += 2 * SimdType::length) {
            for (int i = nonZeroSpecSrcLen; i < len + 2 * SimdType::length; i += 2 * SimdType::length) {
                // IACA_START
                
                const SimdType noise = simdpp::splat<SimdType>(_noise);
                
                //ロードしてシャッフル (0がreal, 1がimage)
                SimdType grad0, grad1, spec0, spec1;//, specSrc0, specSrc1;
                simdpp::load_packed2(spec0, spec1, spec + i);
                
                //ノルム関係
                const SimdType normSqr = calcNormSqr(spec0, spec1);
                
                //差の計算
                // const SimdType diff0 = spec0;
                // const SimdType diff1 = spec1;
                
                //重み
                SimdType baseWeight;
                if (NoiseWeighting::value) {
                    baseWeight = Fmadd<SimdType>(noise, simdpp::load<SimdType>(pre1->noiseWeights + i / 2), absolute_min_noise);
                } else {
                    baseWeight = noise + absolute_min_noise;
                }
                const SimdType totalWeight = simdpp::load<SimdType>(pre1->sumWeights + i / 2) / (baseWeight * baseWeight);
                
                // 評価関数1 + 評価関数2
                // const SimdType diffNormSqr = normSqr;
                eval = Fmadd<SimdType>(normSqr, totalWeight, eval);
                if (Grad::value) {
                    grad0 = spec0 * totalWeight;
                    grad1 = spec1 * totalWeight;
                }
                
                //勾配書き込み
                if (Grad::value) {
#if 0
                    // 微分の定数項(2倍) -> weightに含めたから実行は不要
                    grad0 = grad0 + grad0;
                    grad1 = grad1 + grad1;
#endif
                    simdpp::store_packed2(spec + i, grad0, grad1);
                }
                
                // IACA_END
            }
#endif

            if (Grad::value) {
                // 最後の項はweight = 0とみなして使わない
                // spec[len] = 0;
                // spec[len + 1] = 0;
                // 微分の定数項とFFTの補正を合わせたテクニカルな補正。詳細は重み計算のところのコメント参照
                spec[0] *= 2;
                spec[1] *= 2;

                //IFFT
#if 1
#ifdef PHASE_LIMITER_GRAD_CORE_USE_FFT_PERM
                spec[1] = spec[len];
                // for (int k = 0; k < 2; k++)
                // bench/dft2.cppのデータによると、IFFTはPerm In-placeが最速
                dft->BackwardPerm(spec, spec, dft_work);
                windowed = spec;
#else
                dft->Backward(spec, windowed, dft_work);
#endif
#endif

                for (int i = 0; i < len; i += SimdType::length) {
                    const SimdType grad = simdpp::load(windowed + i);
                    const SimdType w = simdpp::load(pre1->window + i);
                    grad_output_func(i, grad * w);
                }
            }

            Float res = simdpp::reduce_add(eval);
            return res;
        }
    }

template <class SimdType>
class GradCore {
public:
    typedef impl::NoiseWeightingEnabled NoiseWeighting;
    typedef typename SimdType::element_type Float;

    static Float calcEval23(const GradOptions &options, const Float *wave, const Float *waveSrc, Float noise, GradContext<SimdType> *context) {
        return impl::func<SimdType, impl::GradDisabled, NoiseWeighting>(options, [wave](int i) { return wave + i; }, waveSrc, [](int i, SimdType grad){}, noise, nullptr, context);
    }

    static Float calcEval23WithHistogram(const GradOptions &options, const Float *wave, const Float *waveSrc, Float noise,
                                         std::vector<int> *histogram, GradContext<SimdType> *context) {
        return impl::func<SimdType, impl::GradDisabled, NoiseWeighting>(options, [wave](int i) { return wave + i; }, waveSrc, [](int i, SimdType grad){}, noise, histogram, context);
    }

    static Float calcEvalGrad23(const GradOptions &options, const Float *wave, const Float *waveSrc, typename SimdType::element_type *grad, typename SimdType::element_type noise, GradContext<SimdType> *context) {
        return impl::func<SimdType, impl::GradEnabled, NoiseWeighting>(options, [wave](int i) { return wave + i; } , waveSrc, [grad](int i, const SimdType &g) { simdpp::store(grad + i, g); }, noise, nullptr, context);
    }

    template <class WaveInputFunc>
    static Float calcEval23FilterWithHistogram(const GradOptions &options, const WaveInputFunc &wave_input_func, const Float *waveSrc, Float noise,
                                         std::vector<int> *histogram, GradContext<SimdType> *context) {
        return impl::func<SimdType, impl::GradDisabled, NoiseWeighting, WaveInputFunc>(options, wave_input_func, waveSrc, [](int i, SimdType grad){}, noise, histogram, context);
    }

    template <class WaveInputFunc, class GradOutputFunc>
    static Float calcEvalGrad23Filter(const GradOptions &options, const WaveInputFunc &wave_input_func, const Float *waveSrc, const GradOutputFunc &grad_output_func, typename SimdType::element_type noise, GradContext<SimdType> *context) {
        return impl::func<SimdType, impl::GradEnabled, NoiseWeighting, WaveInputFunc, GradOutputFunc>(options, wave_input_func, waveSrc, grad_output_func, noise, nullptr, context);
    }
private:

};
}

#endif // PHASE_LIMITER_GRAD_CORE_H_
