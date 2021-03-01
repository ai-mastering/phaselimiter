#ifndef PHASE_LIMITER_GRAD_CALCULATOR_H_
#define PHASE_LIMITER_GRAD_CALCULATOR_H_

#include <array>
#include <cmath>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>
#include <list>
#include <immintrin.h>
#include <random>
#include <map>
#include <chrono>
#include "tbb/tbb.h"
#include "tbb/pipeline.h"
#include "tbb/scalable_allocator.h"
#include "tbb/cache_aligned_allocator.h"
#include "bakuage/memory.h"
#include "bakuage/utils.h"
#include "bakuage/fir_design.h"
#include "bakuage/vector_math.h"
#include "phase_limiter/config.h"
#include "phase_limiter/GradCore.h"

namespace phase_limiter {
    template<class SimdType> class GradCalculator;
    
    namespace impl {
        template <typename T>
        int sgn(T val) {
            return (T(0) < val) - (val < T(0));
        }
        
        inline int int_cond_or(int a, int b, int c) {
            if (a) return a;
            if (b) return b;
            return c;
        }
        
        template <class SimdType>
        inline SimdType ProxOperator(const SimdType x) {
            const SimdType one = simdpp::splat<SimdType>(1.0f);
            const SimdType minusOne = simdpp::splat<SimdType>(-1.0f);
            return simdpp::max(simdpp::min(x, one), minusOne); //NaNの関係で多分順番が大事
        }
        
        inline int GetWorkerCount(int worker_count) {
            if (worker_count > 0) {
                return std::min(worker_count, PL_MAX_WORKER_COUNT);
            }
            else {
                return std::thread::hardware_concurrency();
            }
        }
        
        class PerformanceCounter {
        public:
            static PerformanceCounter &GetInstance() {
                static PerformanceCounter instance;
                return instance;
            }
            void Start(const char *tag) {
#ifdef PL_PERFORMANCE_COUNTER
                data[tag].Start();
#endif
            }
            double Pause(const char *tag) {
#ifdef PL_PERFORMANCE_COUNTER
                return data[tag].Pause();
#else
                return 0;
#endif
            }
            double Time(const char *tag) {
#ifdef PL_PERFORMANCE_COUNTER
                return data[tag].time();
#else
                return 0;
#endif
            }
        private:
            std::map<std::string, bakuage::StopWatch> data;
        };
        
        template <class SimdType>
        class ThreadVar2 {
        public:
            typedef typename SimdType::element_type Float;
            
            static ThreadVar2 &GetThreadInstance() {
                static thread_local ThreadVar2 instance;
                return instance;
            }
            void Reserve(int len, int oversample) {
                work.resize(std::max<int>(work.size(), len));
                work_downsample_spec.resize(std::max<int>(work_downsample_spec.size(), (len / oversample) / 2 + 1));
                work_hi_spec.resize(std::max<int>(work_hi_spec.size(), len / 2 + 1));
            }
            bakuage::AlignedPodVector<Float> work;
            bakuage::AlignedPodVector<std::complex<Float>> work_downsample_spec;
            bakuage::AlignedPodVector<std::complex<Float>> work_hi_spec;
        private:
            std::unique_ptr<bakuage::RealDft<Float>> dfts_[32];
        };
        
        template <class SimdType>
        struct Task {
            typedef typename SimdType::element_type Float;
            
            void clearCache() {
                for (auto &context: contexts) {
                    context.clear();
                }
            }
            
            void prepare() {
                contexts.resize((ed - bg) / windowLen);
            }
            
            template <class WaveInputFunc, class GradOutputFunc>
            double doTask(const WaveInputFunc &wave_input_func, const GradOutputFunc &grad_output_func) {
                const int ch = channel;
                const double noise = calculator->noise;
                const auto bg_downsample = bg / calculator->oversample();
                const auto ed_downsample = ed / calculator->oversample();
                const auto windowLen_downsample = windowLen / calculator->oversample();
                const auto localWaveSrc = calculator->oversample() == 1 ? calculator->waveSrc : calculator->waveSrc_downsample;
                double sumEval = 0;
                GradOptions options;
                options.len = windowLen_downsample;
                options.sample_rate = calculator->sample_rate_downsample();
                options.max_available_freq = calculator->max_available_freq();
                if (calculator->gradEnabled) {
                    for (int i = bg_downsample, j = 0; i < ed_downsample; i += windowLen_downsample, j++) {
                        sumEval += GradCore<SimdType>::calcEvalGrad23Filter(options, [ch, i, &wave_input_func](int j) { return wave_input_func(ch, i + j); }, localWaveSrc[ch] + i, [ch, i, &grad_output_func](int j, const SimdType &g) { grad_output_func(ch, i + j, g); }, noise, &contexts[j]);
                    }
                }
                else {
                    for (int i = bg_downsample, j = 0; i < ed_downsample; i += windowLen_downsample, j++) {
                        sumEval += GradCore<SimdType>::calcEval23FilterWithHistogram(options, [ch, i, &wave_input_func](int j) { return wave_input_func(ch, i + j); }, localWaveSrc[channel] + i, noise,
                                                           calculator->histogram, &contexts[j]);
                    }
                }
                return sumEval;
            };
            
            int windowLen;
            int bg, ed, channel;
            GradCalculator<SimdType> *calculator;
            std::vector<GradContext<SimdType>, tbb::scalable_allocator<GradContext<SimdType>>> contexts;
        };
        
        // Taskを入出力が近いものどうしでまとめたもの。
        // コア間通信を減らすために使う
        template <class SimdType>
        struct TaskGroup {
            typedef typename SimdType::element_type Float;
            
            TaskGroup(): bg(0), ed(0), channel(0), output_eval(0), output_dot_product(0), output_norm_sqr(0) {}
            
            template <class WaveInputFunc, class GradOutputFunc>
            void doTask(const WaveInputFunc &wave_input_func, const GradOutputFunc &grad_output_func) {
                double sumEval = 0;
                for (const auto task: tasks) {
                    sumEval += task->doTask(wave_input_func, grad_output_func);
                }
                output_eval = sumEval;
            };
            
            int bg, ed;
            int channel;
            std::vector<Task<SimdType> *, tbb::scalable_allocator<Task<SimdType> *>> tasks;
            double output_eval;
            double output_dot_product;
            double output_norm_sqr;
            bakuage::AlignedPodVector<std::complex<Float>> wave_hi_spec;
        };
        
        struct TaskResult {
            double eval;
            double dot_product;
            double norm_sqr;
        };
        
        template <class SimdType>
        class Tasks {
        public:
            typedef typename SimdType::element_type Float;
            typedef Task<SimdType> TaskType;
            typedef TaskGroup<SimdType> TaskGroupType;
            typedef std::vector<TaskGroupType, tbb::cache_aligned_allocator<TaskGroupType>> TaskGroupVector;
            
            Tasks(int worker_count): task_groups1(task_group_allocator_), task_groups2(task_group_allocator_), worker_count_(GetWorkerCount(worker_count)) {
            }
            
            virtual ~Tasks() {
                clear();
            }
            
            void clearCache() {
                for (auto task: tasks) {
                    task->clearCache();
                }
            }
            
            void prepare() {
                for (auto task: tasks) {
                    task->prepare();
                }
                
                std::stable_sort(tasks.begin(), tasks.end(), [](TaskType *a, TaskType *b) {
                    return int_cond_or(sgn(a->bg - b->bg), sgn(a->windowLen - b->windowLen), sgn(a->channel - b->channel)) < 0;
                });
                
                // それぞれのコアのキャッシュに収まるくらいのサイズで区切る
                auto temp_tasks = tasks;
                const int stride = tasks[0]->calculator->fft_max_len(); // 1 * (1 << 14);
                int total_ed = tasks[0]->bg;
                for (const auto task: tasks) {
                    total_ed = std::max<int>(total_ed, task->ed);
                }
                
                task_groups1.clear();
                task_groups2.clear();
                
                for (int idx = 0; idx < 2; idx++) {
                    for (int channel = 0; channel < 2; channel++) {
                        for (int bg = tasks[0]->bg + (stride / 2) * idx; bg < total_ed; bg += stride) {
                            TaskGroupType task_group;
                            
                            auto it = temp_tasks.begin();
                            while (it != temp_tasks.end()) {
                                bool is_inside = bg <= (*it)->bg && (*it)->ed <= bg + stride && (*it)->channel == channel;
                                if (is_inside) {
                                    task_group.tasks.push_back(*it);
                                    it = temp_tasks.erase(it);
                                } else {
                                    ++it;
                                };
                                
                                // 明らかに範囲外の場合はループを抜ける
                                if (bg + stride <= (*it)->bg) break;
                            }
                            
                            // 1ループ目はbefore_hook用に全体を舐める必要があるので、中身が無くても追加する
                            if (task_group.tasks.size() || idx == 0) {
                                task_group.bg = bg;
                                task_group.ed = std::min<int>(bg + stride, total_ed);
                                task_group.channel = channel;
                                if (idx == 0) {
                                    task_groups1.push_back(task_group);
                                }
                                else {
                                    task_groups2.push_back(task_group);
                                }
                            }
                        }
                    }
                }
                
                if (temp_tasks.size()) {
                    std::cerr << "TaskGroup prepare failed" << std::endl;
                }
            };
            
            void addTask(const TaskType &task) {
                if (task.bg == task.ed)
                    return;
                TaskType *t = task_allocator_.allocate(1);
                task_allocator_.construct(t, task);
                tasks.push_back(t);
            }
            
            template <class BeforeHook, class WaveInputFunc, class GradOutputFunc>
            TaskResult process(const BeforeHook &before_hook, const WaveInputFunc &wave_input_func, const GradOutputFunc &grad_output_func, bool serial = false, bool before_hook_only = false) {
                typedef typename TaskGroupVector::iterator Iterator;
                // gradに書き込む領域に被りが無いように2回に分けて処理を行う。
                // 1回目は全体をもれなく一回だけ舐めることが保証されている。
                // (これを利用して、まったく関係の無いメモリ書き込みなど、別タスクをやらせることが可能)
                // gradはGradCalculatorがクリア済み
                
                // serial = true;
                
#if 0
                注意: task_groupsの分け方が間違えてるかも (task_groups2の最後の要素が空になっていないかも)
                // 分割統治法 (parallel_forより遅くなった)
                // イメージ図
                // 左と右を並行で処理し、最後に継ぎ目を処理する
                // task_groups1[0] | task_groups1[1]
                //          task_groups1[0] | task_groups2[1]
                const std::function<void(int, int, int)> execute_div_conq = [this, &execute_div_conq, &before_hook, &wave_input_func, &grad_output_func](int channel, int bg_idx, int ed_idx) {
                    constexpr int grain_size = 1;
                    const int gap = ed_idx - bg_idx;
                    const int center = (ed_idx + bg_idx) / 2;
                    if (gap <= grain_size) {
                        // 一個目
                        before_hook(&task_groups1[channel * task_groups1.size() / 2 + bg_idx]);
                        task_groups1[channel * task_groups1.size() / 2 + bg_idx].doTask(wave_input_func, grad_output_func);
                        
                        // 二個目以降は継ぎ目も処理する
                        for (int i = bg_idx + 1; i < ed_idx; i++) {
                            before_hook(&task_groups1[channel * task_groups1.size() / 2 + i]);
                            task_groups1[channel * task_groups1.size() / 2 + i].doTask(wave_input_func, grad_output_func);
                            task_groups2[channel * task_groups2.size() / 2 + i - 1].doTask(wave_input_func, grad_output_func);
                        }
                    }
                    else {
#if 1
                        tbb::parallel_invoke(std::bind(execute_div_conq, channel, bg_idx, center),
                                             std::bind(execute_div_conq, channel, center, ed_idx));
#else
                        execute_div_conq(channel, bg_idx, center);
                        execute_div_conq(channel, center, ed_idx);
#endif
                        // 継ぎ目
                        task_groups2[channel * task_groups2.size() / 2 + (center - 1) ].doTask(wave_input_func, grad_output_func);
                    }
                };
                tbb::parallel_invoke(std::bind(execute_div_conq, 0, 0, task_groups1.size() / 2),
                                     std::bind(execute_div_conq, 1, 0, task_groups1.size() / 2));
                //execute_div_conq(0, task_groups1.size());
                
#else
                // 1回目のループを行う (全体をもれなく一回だけ舐める、before_hookあり)
                const auto execute_with_before_hook = [&before_hook, &wave_input_func, &grad_output_func, before_hook_only](TaskGroupType *task_group) {
                    before_hook(task_group);
                    if (!before_hook_only) {
                        task_group->doTask(wave_input_func, grad_output_func);
                    }
                };
                if (serial) {
                    for (auto &task_group: task_groups1) {
                        execute_with_before_hook(&task_group);
                    }
                } else {
                    // task_groups1の0番目は
                    
#if 0
                    tbb::parallel_for(0, (int)task_groups1.size(),
                                      [this, &execute_with_before_hook](int i) {
                                          execute_with_before_hook(&task_groups1[i]);
                                      }
                                      );
#else
                    tbb::parallel_for(tbb::blocked_range<Iterator>(task_groups1.begin(), task_groups1.end()),
                                  [&execute_with_before_hook](const tbb::blocked_range<Iterator>& r) {
                                      for(auto it = r.begin(); it != r.end(); ++it) {
                                          execute_with_before_hook(&(*it));
                                      }
                                  }
                             );
#endif
                }
                
                // 2回目のループを行う (つぎはぎ部分、before_hookなし)
                if (!before_hook_only) {
                    const auto execute_without_before_hook = [&wave_input_func, &grad_output_func](TaskGroupType *task_group) {
                        task_group->doTask(wave_input_func, grad_output_func);
                    };
                    if (serial) {
                        for (auto &task_group: task_groups2) {
                            execute_without_before_hook(&task_group);
                        }
                    } else {
#if 0
                        tbb::parallel_for(0, (int)task_groups2.size(),
                        [this, &execute_without_before_hook](int i) {
                            execute_without_before_hook(&task_groups2[i]);
                        }
                        );
#else
                        tbb::parallel_for(tbb::blocked_range<Iterator>(task_groups2.begin(), task_groups2.end()),
                                      [&execute_without_before_hook](const tbb::blocked_range<Iterator>& r) {
                                          for(auto it = r.begin(); it != r.end(); ++it) {
                                              execute_without_before_hook(&(*it));
                                          }
                                      }
                        );
#endif
                    }
                }
                
#endif
                
                TaskResult result = { 0 };
                if (!before_hook_only) {
                    for (const auto &task_group: task_groups1) {
                        result.eval += task_group.output_eval;
                        result.dot_product += task_group.output_dot_product;
                        result.norm_sqr += task_group.output_norm_sqr;
                    }
                    for (const auto &task_group: task_groups2) {
                        result.eval += task_group.output_eval;
                        // 2回目のループは、dot_productとかは無いはず
                    }
                }
                
                return result;
            }
            
        private:
            tbb::cache_aligned_allocator<TaskType> task_allocator_;
            tbb::cache_aligned_allocator<TaskGroupType> task_group_allocator_;
            TaskGroupVector task_groups1;
            TaskGroupVector task_groups2;
            std::vector<TaskType *, tbb::scalable_allocator<Task<SimdType> *>> tasks; // own
            void clear() {
                for (size_t j = 0; j < tasks.size(); j++) {
                    tasks[j]->clearCache();
                    task_allocator_.destroy(tasks[j]);
                    task_allocator_.deallocate(tasks[j], 1);
                }
                tasks.clear();
            }
            const int worker_count_;
        };
    }
    
    template <class SimdType>
    class GradCalculator {
    public:
        typedef typename SimdType::element_type Float;
        typedef impl::TaskGroup<SimdType> TaskGroupType;
        
        /*
         オーバーサンプルの方法は二種類ある。
         1. 外部でoversampleしてそのまま計算する方法 (遅い、メモリ喰う)
           sample_rateはoversample後を与える
        　　oversampleは1を与える
         2. 外部でoversampleして、内部で評価関数計算のときだけダウンサンプリングする方法 (速い、省メモリ)
           sample_rateはoversample後を与える
           oversampleは倍率を与える。
         
         オーバーサンプルフィルタについて
         FIRでフィルタする。task_groupの(ed - bg) * oversampleが4だとしたら、
         タップ数は3。一般的にはタップ数 = (ed - bg) * oversample / 2 + 1
         012
         --2345
         -----567
         このタップ数で窓関数法で帯域分割フィルタを作る。
         (エイリアシングノイズ除去フィルタは不要。理由は、grad計算をするときにanalyzeフィルタを再度かけるから)
         
         メモリ配置について
         長さや位置とかでdownsample suffixがついていないものはオーバーサンプル後のもの
         
         0---bg---bg+len--ed---memLen
         bgとedはFISTAとかの処理対象 (SIMDで処理するので、bg, edは(SimdType::length * oversample)アライン)
         gradははみ出さないようにbg+len~edの間は0クリアする
         0~bg, ed~memLenは評価関数計算ではみ出す分
         
         FISTAとかはoversample後の長さで行って、
         downsample suffixがついているものとか、評価関数計算はdownsample後の長さで行う
         
         実装方針
         lowpass truepeakを抑える対応を同時に考えると複雑なので、あとで考える。
         かなり複雑だがテストを書いてなんとかする
         
         現状 (2019/02/11)
         内部oversampleには対応してない(コードはところどころ存在するが、完成していない。現状oversample == 1の場合のみ動く想定)
         */
        
        GradCalculator(int len, int sample_rate, int max_available_freq, int workerCount, const char *noise_update_mode, double noise_update_min_noise, double noise_update_initial_noise, double noise_update_fista_enable_ratio, int max_iter1, int max_iter2, int oversample):
        histogram(NULL), last_iter(0), last_iter2(0), sample_rate_(sample_rate), sample_rate_downsample_(sample_rate / oversample), max_available_freq_(max_available_freq), len_(len), noise_update_mode_(noise_update_mode), noise_update_min_noise_(noise_update_min_noise), noise_update_initial_noise_(noise_update_initial_noise), noise_update_fista_enable_ratio_(noise_update_fista_enable_ratio), max_iter1_(max_iter1), max_iter2_(max_iter2), oversample_(oversample) {
            using namespace bakuage;
            
            if (sample_rate != 44100 * bakuage::CeilPowerOf2(sample_rate / 44100)) {
                throw std::logic_error("sample_rate must be 44100 * 2^x");
            }
            if (oversample != bakuage::CeilPowerOf2(oversample)) {
                throw std::logic_error("oversample must be 2^x");
            }
            
            // 前後に評価関数用の余白を設ける
            bg = fft_max_len() / 2;
            ed = bg + CeilInt<int>(len, SimdType::length * oversample_);
            memLen = CeilInt<int>(ed + fft_max_len() / 2, fft_max_len() / 2);
            for (int i = 0; i < 2; i++) {
                waveProx[i] = TypedMalloc<Float>(memLen);
                wavePrev[i] = TypedMalloc<Float>(memLen);
                waveOut[i] = TypedMalloc<Float>(memLen);
                waveSrc[i] = TypedMalloc<Float>(memLen);
                grad[i] = TypedMalloc<Float>(memLen);
                
                if (oversample_ > 1) {
                    waveSrc_downsample[i] = TypedMalloc<Float>(memLen / oversample_);
                    wave_downsampled[i] = TypedMalloc<Float>(memLen / oversample_);
                    grad_downsampled[i] = TypedMalloc<Float>(memLen / oversample_);
                }
            }
            const int blockSize = fft_max_len() / 2;
            tasks = new impl::Tasks<SimdType>(workerCount);
            
            for (int channel = 0; channel < 2; channel++)
                for (int w = fft_min_len(); w <= fft_max_len(); w = 2 * w)
                    for (int shift = 0; shift < w; shift += w / 2) {
                        impl::Task<SimdType> task = { 0 };
                        task.calculator = this;
                        task.channel = channel;
                        task.windowLen = w;
                        task.bg = bg - shift;
                        task.ed = task.bg;
                        for (int j = task.bg; j < memLen; j += task.windowLen) {
                            if (task.bg / blockSize + 1 < j / blockSize) {
                                tasks->addTask(task);
                                if (j >= ed)
                                    break;
                                task.bg = task.ed;
                            }
                            task.ed = j;
                        }
                        if (task.bg < task.ed) {
                            tasks->addTask(task);
                        }
                    }
            tasks->prepare();
            
            // initialize filters
            if (oversample_ > 1) {
                const auto fir_samples = (oversample_filter_fft_len() - fft_max_len()) / 2 + 1;
                const int center = (fir_samples - 1) / 2;
                
                // 原点対称
                const auto lowpass_fir = bakuage::CalculateBandPassFir<Float>(0, 0.5 - 9.0 / 2 / fir_samples, fir_samples, 7);
                bakuage::AlignedPodVector<Float> work(oversample_filter_fft_len());
                work[0] = lowpass_fir[center];
                for (int i = 1; i < center + 1; i++) {
                    work[i] = lowpass_fir[center + i];
                    work[oversample_filter_fft_len() - i] = work[i];
                }
                
                auto &pool = bakuage::ThreadLocalDftPool<bakuage::RealDft<Float>>::GetThreadInstance();
                const auto dft = pool.Get(oversample_filter_fft_len());
                
                const Float normalization_scale = 1.0 / oversample_filter_fft_len();
                bakuage::AlignedPodVector<std::complex<Float>> work_spec(oversample_filter_fft_len() / 2 + 1);
                dft->Forward(work.data(), (Float *)work_spec.data(), pool.work());
                oversample_lowpass_spec_.resize(work_spec.size());
                for (int i = 0; i < work_spec.size(); i++) {
                    oversample_lowpass_spec_[i] = work_spec[i].real() * normalization_scale;
                }
                
                for (int i = 0; i < oversample_filter_fft_len(); i++) {
                    work[i] = (i == 0 ? 1 : 0) - work[i];
                }
                dft->Forward(work.data(), (Float *)work_spec.data(), pool.work());
                oversample_hipass_spec_.resize(work_spec.size());
                for (int i = 0; i < work_spec.size(); i++) {
                    oversample_hipass_spec_[i] = work_spec[i].real() * normalization_scale * 0;
                }
            }
            
            std::cerr << "GradCalculator initialized" << std::endl;
        }
        
        virtual ~GradCalculator() {
            using namespace bakuage;
            
            delete tasks;
            for (int i = 0; i < 2; i++) {
                Free(waveProx[i]);
                Free(wavePrev[i]);
                Free(waveOut[i]);
                Free(waveSrc[i]);
                Free(grad[i]);
                
                if (oversample_ > 1) {
                    Free(waveSrc_downsample[i]);
                    Free(wave_downsampled[i]);
                    Free(grad_downsampled[i]);
                }
            }
        }
        
        int fft_min_len() const { return (1 << 8) * (sample_rate_ / 44100); }
        int fft_max_len() const { return PL_FFT_MAX_LEN * (sample_rate_ / 44100); }
        int oversample_filter_fft_len() const { return 2 * fft_max_len(); }
        
        // input
        template <class DoublePtr>
        void copyWaveSrcFrom(DoublePtr src, int stride) { CopyFromImpl<DoublePtr>(src, waveSrc, stride); }
        template <class DoublePtr>
        void copyWaveProxFrom(DoublePtr src, int stride) { CopyFromImpl<DoublePtr>(src, waveProx, stride); }
        
        // output
        template <class DoublePtr>
        void copyGradTo(DoublePtr dest, int stride) { CopyToImpl<DoublePtr>(grad, dest, stride); }
        template <class DoublePtr>
        void copyWaveProxTo(DoublePtr dest, int stride) { CopyToImpl<DoublePtr>(waveProx, dest, stride); }
        
        std::function<void (TaskGroupType *)> emptyBeforeHook() const {
            return [](TaskGroupType *) {};
        }
        
        std::function<Float *(int, int)> defaultWaveInputFunc() const {
            return [this](int channel, int i){ return waveProx[channel] + i; };
        }
        
        // optimization
        template <class BeforeHook, class WaveInputFunc>
        impl::TaskResult calcEvalGrad(double _noise, const BeforeHook &before_hook, const WaveInputFunc &wave_input_func) {
            noise = _noise;
            gradEnabled = true;
            
            auto local_grad = oversample_ == 1 ? grad : grad_downsampled;
            const auto local_oversample = oversample_;
            return tasks->process(
                                  [local_grad, local_oversample, &before_hook](impl::TaskGroup<SimdType> *task_group) {
                                      before_hook(task_group);
                                      const int channel = task_group->channel;
                                      bakuage::TypedFillZero(local_grad[channel] + task_group->bg / local_oversample, (task_group->ed - task_group->bg) / local_oversample);
                                  },
                                  wave_input_func,
                                  [local_grad](int channel, int i, const SimdType &g) {
                                      const SimdType x = simdpp::load(local_grad[channel] + i);
                                      simdpp::store(local_grad[channel] + i, x + g);
                                  }
                                  );
        }
        template <class BeforeHook, class WaveInputFunc>
        impl::TaskResult calcEval(double _noise, const BeforeHook &before_hook, const WaveInputFunc wave_input_func, bool serial = false) {
            noise = _noise;
            gradEnabled = false;
            return tasks->process(before_hook, wave_input_func, [](int channel, int i, const SimdType &g) {}, serial);
        }
        
#if 0
        template <class BeforeHook>
        impl::TaskResult executeBeforeHook(const BeforeHook &before_hook, bool serial = false) {
            return tasks->process(before_hook, [](int channel, int i) { return simdpp::splat<SimdType>(0); }, [](int channel, int i, const SimdType &g) {}, serial, true);
        }
#endif
        
        // 単位評価関数値を計算 (ホワイトノイズが1dBずれた場合の誤差値)
        // waveSrc, waveOutを汚染するので注意
        double outputUnitEval(const std::string &mode) {
            tasks->clearCache();
            double unit_eval = 0;
            {
                std::mt19937 engine(1);
                std::normal_distribution<double> dist(0.0, 1.0);
                const double scale_1db = std::pow(10, 1.0 / 20);
                bakuage::RealDft<Float> dft(memLen);
                bakuage::AlignedPodVector<std::complex<Float>> spec(memLen / 2 + 1);
                for (int channel = 0; channel < 2; channel++) {
                    if (mode == "src_with_cut") {
                        dft.Forward(waveSrc[channel], (Float *)spec.data());
                        for (int i = 0; i < spec.size(); i++) {
                            const auto freq = 1.0 * i / spec.size() * sample_rate();
                            if (freq > max_available_freq()) {
                                spec[i] = 0;
                            }
                        }
                        dft.Backward((Float *)spec.data(), waveSrc[channel]);
                        const double scale = 1.0 / memLen;
                        for (int i = 0; i < memLen; i++) {
                            waveSrc[channel][i] *= scale;
                            waveOut[channel][i] = waveSrc[channel][i] * scale_1db;
                        }
                    } else if (mode == "noise_with_cut") {
                        for (int i = 0; i < spec.size(); i++) {
                            const auto freq = 1.0 * i / spec.size() * sample_rate();
                            if (freq <= max_available_freq()) {
                                spec[i].real(dist(engine) / std::sqrt(100 + freq));
                                spec[i].imag(dist(engine) / std::sqrt(100 + freq));
                            }
                            if (i == 0 || i == spec.size() - 1) {
                                spec[i].imag(0);
                            }
                        }
                        dft.Backward((Float *)spec.data(), waveSrc[channel]);
                        bakuage::VectorMulConstantInplace(scale_1db, spec.data(), spec.size());
                        dft.Backward((Float *)spec.data(), waveOut[channel]);
                    } else if (mode == "noise") { // リミッター誤差導入時から使っているもの
                        for (int i = 0; i < memLen; i++) {
                            const double x = dist(engine);
                            waveSrc[channel][i] = x;
                            waveOut[channel][i] = waveSrc[channel][i] * scale_1db;
                        }
                    } else if (mode == "src") {
                        for (int i = 0; i < memLen; i++) {
                            waveOut[channel][i] = waveSrc[channel][i] * scale_1db;
                        }
                    }
                }
                const auto local_waveOut = waveOut;
                const auto wave_input_func = [local_waveOut](int channel, int i) { return local_waveOut[channel] + i; };
                unit_eval = calcEval(noise_update_min_noise_, emptyBeforeHook(), wave_input_func).eval; // min noise
            }
            std::cerr << "unit_eval:" << unit_eval << std::endl;
            return unit_eval;
        }
        
        template <class ProgressCallback>
        void optimizeWithProgressCallback(const ProgressCallback &callback, double unit_eval) {
            using namespace impl;
            
            std::cerr << "optimizeWithProgressCallback" << std::endl;
            tasks->clearCache();
            
            //waveSrcをwaveProxとwavePrevとwaveOutにコピー
            for (int channel = 0; channel < 2; channel++) {
                for (int i = 0; i < memLen; i += SimdType::length) {
                    SimdType src = simdpp::load(waveSrc[channel] + i);
                    SimdType prox = ProxOperator(src);
                    simdpp::store(waveProx[channel] + i, prox);
                    simdpp::store(wavePrev[channel] + i, prox);
                    simdpp::store(waveOut[channel] + i, prox);
                }
            }
            
            // calculate histogram
#if 0
            std::vector<int> histo(200);
            histogram = &histo;
            calcEval(1, true);
            histogram = nullptr;
            for (int i = 0; i < histo.size(); i++) {
                histo[i] = std::log(1.0 + histo[i] * std::pow(2.0, i));
            }
            int totalHistoCount = 0;
            for (int i = 0; i < histo.size(); i++) {
                totalHistoCount += histo[i];
            }
#endif
            
            const int kMaxIter1 = max_iter1_;
            const int kMaxIter2 = max_iter2_;
            
            int iter = 1;
            int iter2 = 1;
            double t = 1;
            const double t_beta = 0.5;
            const double noise_beta = 0.5;
            double prevEvalProx = 1e100;
            double prevNormalizedEvalProx = 1e100;
            double prevEvalProx2 = 1e100;
            double noise = noise_update_initial_noise_;
            std::cerr << "optimizeWithProgressCallback loop start" << std::endl;
            while (iter <= kMaxIter1 && iter2 <= kMaxIter2) {
                PerformanceCounter::GetInstance().Start("optimize_loop1");
                
                if (noise_update_mode_ == "linear") {
                    const double target_noise = noise_update_initial_noise_ * std::pow(noise_update_min_noise_ / noise_update_initial_noise_, std::pow((double)iter / kMaxIter1, 1));
                    //double target_noise = std::pow(t / 0.01, 0.5);
                    //const double iter_pow = std::log(noise_update_initial_noise_ / noise_update_min_noise_) / std::log(kMaxIter1);
                    //const double target_noise = noise_update_initial_noise_ / std::pow(iter, iter_pow);
                    // const double target_noise = noise_update_min_noise_ * kMaxIter1 / iter;
                    /*
                    double target_noise = noise_update_min_noise_;
                    const double ratio = 1.0 * iter / kMaxIter1;
                    int cumHistoCount = 0;
                    for (int i = histo.size() - 1; i >= 0; i--) {
                        cumHistoCount += histo[i];
                        if (cumHistoCount > ratio * totalHistoCount) {
                            target_noise = std::max(noise_update_min_noise_, 1.4 * std::sqrt(std::pow(2, i - 100)));
                            break;
                        }
                    }*/
                    
#if 1
                    noise = target_noise;
                    //noise = 2 * std::sqrt(t);
                    //t *= 1.1;
#else
                    if (std::abs(target_noise / noise - 1) > 0.1) {
                        noise = target_noise;
                        tasks->clearNoiseCache();
                        prevEvalProx = prevEvalProx * bakauge::Sqr(noise / target_noise);//1e100;
                    }
#endif
                }
                else if (noise_update_mode_ == "adaptive") {
                    if (prevEvalProx > prevEvalProx2 * 0.999 && prevEvalProx < prevEvalProx2) {
                        noise *= noise_beta;
                        prevEvalProx /= bakuage::Sqr(noise_beta);//1e100;
                        prevEvalProx2 = prevEvalProx;//1e100;
                    }
                    else {
                        prevEvalProx2 = prevEvalProx;
                    }
                    if (noise < noise_update_min_noise_) {
                        break;
                    }
                } else {
                    throw std::logic_error("unknown noise update mode " + noise_update_mode_);
                }
                callback(std::max(0.0, std::min(1.0, std::log10(noise) / std::log10(noise_update_min_noise_))));
                
                // FISTAのfactor計算
                SimdType factor1 = simdpp::splat<SimdType>((double)(iter - 2) / (double)(iter + 1));
                // 最後のほうでFISTAをやめてみる
                if (std::log(noise_update_initial_noise_ / noise) > noise_update_fista_enable_ratio_ * std::log(noise_update_initial_noise_ / noise_update_min_noise_)) {
                    factor1 = simdpp::splat<SimdType>(0);
                }
                PerformanceCounter::GetInstance().Pause("optimize_loop1");
                
                PerformanceCounter::GetInstance().Start("calcEvalGrad");
                //std::cerr << "AAA" << std::endl;
                double evalOutHi = 0;
                auto evalTargetWave1 = oversample_ == 1 ?
                wavePrev : // wavePrevに保存してcalcEvalGrad後にwaveOutにrenameする (キャッシュ効率のため)
                wave_downsampled;
                const auto eval_out_result = calcEvalGrad(noise, [this, factor1/*, &evalOutHi*/](impl::TaskGroup<SimdType> *task_group) {
                    const int channel = task_group->channel;
                    
                    if (oversample_ > 1) {
                        // 領域確保
                        auto &tv = impl::ThreadVar2<SimdType>::GetThreadInstance();
                        const auto fft_len = 2 * fft_max_len();
                        const auto fft_len_downsample = fft_len / oversample_;
                        const auto bg_in_work = fft_len / 4;
                        const auto ed_in_work = task_group->ed - task_group->bg + bg_in_work;
                        tv.Reserve(fft_len, oversample_);
                        task_group->wave_hi_spec.resize(fft_len);
                        
                        // スレッドごとの一時領域(fft_max_len() * oversampleの倍の長さ)にprox結果を保存
                        auto work = tv.work.data();
                        const auto bg_with_margin = task_group->bg - bg_in_work;
                        const auto ed_with_margin = bg_with_margin + fft_len;
                        for (int i = std::max(0, bg_with_margin); i < std::min(memLen, ed_with_margin); i += SimdType::length) {
                            const SimdType prox = simdpp::load<SimdType>(waveProx[channel] + i);
                            const SimdType prev = simdpp::load<SimdType>(wavePrev[channel] + i);
                            const SimdType z = Fmadd<SimdType>(factor1, prox - prev, prox);
                            simdpp::store(work + (i - bg_with_margin), z);
                        }
                        bakuage::TypedFillZero(work, std::max(0, bg_with_margin) - bg_with_margin);
                        bakuage::TypedFillZero(work + std::min(memLen, ed_with_margin) - bg_with_margin, ed_with_margin - std::min(memLen, ed_with_margin));
                        
                        // FFT
                        auto &pool = bakuage::ThreadLocalDftPool<bakuage::RealDft<Float>>::GetThreadInstance();
                        const auto dft = pool.Get(fft_len);
                        dft->Forward(work, (Float *)task_group->wave_hi_spec.data(), pool.work());
                        
                        // 一時領域からwaveOutにコピー
                        // oversampleが無い場合は、wavePrevに保存して、calcEvalGrad後にwaveOutにrenameできるが (キャッシュ効率のため)、ダウンサンプルのFIRをかけるときにtaskグループの範囲外を読み取るので、別領域に保存する必要があるので、waveOutに保存
                        bakuage::TypedMemcpy(waveOut[channel] + task_group->bg, work + bg_in_work, ed_in_work - bg_in_work);
                        
                        // FIR + IFFTでwave_downsampledを作る
                        bakuage::VectorMul(oversample_lowpass_spec_.data(), task_group->wave_hi_spec.data(), tv.work_downsample_spec.data(), fft_len_downsample / 2 + 1);
                        const auto dft_downsample = pool.Get(fft_len / oversample_);
                        dft_downsample->Backward((Float *)tv.work_downsample_spec.data(), work, pool.work());
                        bakuage::TypedMemcpy(wave_downsampled[channel] + task_group->bg / oversample_, work + bg_in_work / oversample_, (ed_in_work - bg_in_work) / oversample_);
                        
                        // FIR + IFFTでwave_hiを作る (specのままにしておきたい)
                        bakuage::VectorMulInplace(oversample_hipass_spec_.data(), task_group->wave_hi_spec.data(), fft_len / 2 + 1);
                    }
                    else {
                        for (int i = task_group->bg; i < task_group->ed; i += SimdType::length) {
                            // normal fista
                            //std::cerr << i << " " << channel << std::endl;
                            SimdType prox = simdpp::load<SimdType>(waveProx[channel] + i);
                            SimdType prev = simdpp::load<SimdType>(wavePrev[channel] + i);
                            SimdType z = Fmadd<SimdType>(factor1, prox - prev, prox);
                            simdpp::store(wavePrev[channel] + i, z); // wavePrevに保存してcalcEvalGrad後にwaveOutにrenameする (キャッシュ効率のため)
                        }
                    }
                }, [evalTargetWave1](int channel, int i) {
                    return evalTargetWave1[channel] + i; // wavePrevに保存してcalcEvalGrad後にwaveOutにrenameする (キャッシュ効率のため)
                });
                //std::cerr << "bbb" << std::endl;
                const auto evalOut = eval_out_result.eval + evalOutHi;
                if (oversample_ == 1) {
                    // waveOut = old wavePrev
                    // waveProx = old waveOut
                    // wavePrev = old waveProx
                    std::swap(waveOut, wavePrev);
                    std::swap(waveProx, wavePrev);
                } else {
                    // waveOut = old waveOut
                    // waveProx = old wavePrev
                    // wavePrev = old waveProx
                    std::swap(waveProx, wavePrev);
                }
                if (oversample_ > 1) {
                    // gradをアップサンプリングして合成する
                    // task_groupごとに、task_groupに保存させてあるwave_hi_specをベースに、
                    // gradをinterpolate -> FFT -> エイリアシングノイズ除去フィルタしたものを足しこんで
                    // IFFTして配置すれば、gradのできあがり
                    
#if 0
                    executeBeforeHook([this](impl::TaskGroup<SimdType> *task_group) {
                        const int channel = task_group->channel;
                        
                        // 領域確保
                        auto &tv = impl::ThreadVar2<SimdType>::GetThreadInstance();
                        const auto fft_len = 2 * fft_max_len();
                        const auto fft_len_downsample = fft_len / oversample_;
                        const auto bg_in_work_downsample = fft_len_downsample / 4;
                        const auto ed_in_work_downsample = (task_group->ed - task_group->bg) / oversample_ + bg_in_work_downsample;
                        
                        // スレッドごとの一時領域(fft_max_len()の倍の長さ)にdownsample gradを保存
                        auto work = tv.work.data();
                        const auto bg_with_margin_downsample = task_group->bg / oversample_ - bg_in_work_downsample;
                        const auto ed_with_margin_downsample = bg_with_margin_downsample + fft_len_downsample;
                        const auto memLen_downsample = memLen / oversample_;
                        for (int i = std::max(0, bg_with_margin_downsample); i < std::min(memLen_downsample, ed_with_margin_downsample); i += SimdType::length) {
                            const SimdType g = simdpp::load<SimdType>(grad[channel] + i);
                            simdpp::store(work + (i - bg_with_margin_downsample), g);
                        }
                        bakuage::TypedFillZero(work, std::max(0, bg_with_margin_downsample) - bg_with_margin_downsample);
                        bakuage::TypedFillZero(work + std::min(memLen, ed_in_work_downsample) - bg_with_margin_downsample, ed_with_margin_downsample - std::min(memLen_downsample, ed_in_work_downsample));
                        
                        // FFT grad downsample
                        auto &pool = bakuage::ThreadLocalDftPool<bakuage::RealDft<Float>>::GetThreadInstance();
                        const auto dft_downsample = pool.Get(fft_len_downsample);
                        dft_downsample->Forward(work, (Float *)tv.work_downsample_spec.data(), pool.work());
                        
                        // grad + work_hi由来のgrad
                        bakuage::VectorAddInplace(tv.work_downsample_spec.data(), task_group->wave_hi_spec.data(), fft_len_downsample / 2 + 1);
                        
                        // IFFT grad oversample
                        const auto dft = pool.Get(fft_len);
                        dft->Backward((Float *)task_group->wave_hi_spec.data(), work, pool.work());
                        
                        // copy grad
                        const auto bg_in_work = fft_len / 4;
                        const auto ed_in_work = task_group->ed - task_group->bg + bg_in_work;
                        bakuage::TypedMemcpy(grad[channel] + task_group->bg, work + bg_in_work, ed_in_work - bg_in_work);
                    });
#endif
                }
                // gradだけははみ出てくるので、ここでちゃんとはみ出た分をカットする
                for (int channel = 0; channel < 2; channel++) {
                    bakuage::TypedFillZero(grad[channel] + bg + len_, ed - (bg + len_));
                }
                PerformanceCounter::GetInstance().Pause("calcEvalGrad");
                
                // FISTA line search
                // 毎回tを少し増やす。理由は回数が多くなると(12800回とか)偶然tが減少してしまうことがあるので
                // (理由違うかも、局所解に落ちていて、tの減少は妥当なものかも)
                // あまり結果に影響を与えないので、オッカムの剃刀理論にもとづいて、オフにする
                // t *= 1.1;
                while (iter2 <= kMaxIter2) {
                    // callback(std::max<double>((double)iter / kMaxIter1, (double)iter2 / kMaxIter2));
                    PerformanceCounter::GetInstance().Start("calcEval");
                    double evalProxHi = 0;
                    auto evalTargetWave2 = oversample_ == 1 ?
                    waveProx :
                    wave_downsampled;
                    const auto eval_prox_result = calcEval(noise, [this, t/*, &evalProxHi*/](impl::TaskGroup<SimdType> *task_group) {
                        const int channel = task_group->channel;
                        SimdType dot_product = simdpp::splat<SimdType>(0);
                        SimdType norm_sqr = simdpp::splat<SimdType>(0);
                        const SimdType minusTVec = simdpp::splat<SimdType>(-t);
                        
                        if (oversample_ > 1) {
                            // 領域確保
                            auto &tv = impl::ThreadVar2<SimdType>::GetThreadInstance();
                            const auto fft_len = 2 * fft_max_len();
                            const auto fft_len_downsample = fft_len / oversample_;
                            const auto bg_in_work = fft_len / 4;
                            const auto ed_in_work = task_group->ed - task_group->bg + bg_in_work;
                            
                            // スレッドごとの一時領域(fft_max_len() * oversampleの倍の長さ)にprox結果を保存
                            auto work = tv.work.data();
                            const auto bg_with_margin = task_group->bg - bg_in_work;
                            const auto ed_with_margin = bg_with_margin + fft_len;
                            for (int i = std::max(0, bg_with_margin); i < std::min(memLen, ed_with_margin); i += SimdType::length) {
                                const SimdType x = simdpp::load<SimdType>(waveOut[channel] + i);
                                const SimdType y = simdpp::load<SimdType>(grad[channel] + i);
                                const SimdType prox = ProxOperator(Fmadd<SimdType>(minusTVec, y, x)); //NaNの関係で多分順番が大事
                                if (task_group->bg <= i && i < task_group->ed) {
                                    const SimdType diff = prox - x;
                                    dot_product = Fmadd<SimdType>(diff, y, dot_product);
                                    norm_sqr = Fmadd<SimdType>(diff, diff, norm_sqr);
                                }
                                simdpp::store(work + (i - bg_with_margin), prox);
                            }
                            bakuage::TypedFillZero(work, std::max(0, bg_with_margin) - bg_with_margin);
                            bakuage::TypedFillZero(work + std::min(memLen, ed_with_margin) - bg_with_margin, ed_with_margin - std::min(memLen, ed_with_margin));
                            
                            // FFT
                            auto &pool = bakuage::ThreadLocalDftPool<bakuage::RealDft<Float>>::GetThreadInstance();
                            const auto dft = pool.Get(fft_len);
                            dft->Forward(work, (Float *)task_group->wave_hi_spec.data(), pool.work());
                            
                            // 一時領域からwaveProxにコピー
                            bakuage::TypedMemcpy(waveProx[channel] + task_group->bg, work + bg_in_work, ed_in_work - bg_in_work);
                            
                            // FIR + IFFTでwave_downsampledを作る
                            bakuage::VectorMul(oversample_lowpass_spec_.data(), task_group->wave_hi_spec.data(), tv.work_downsample_spec.data(), fft_len_downsample / 2 + 1);
                            const auto dft_downsample = pool.Get(fft_len / oversample_);
                            dft_downsample->Backward((Float *)tv.work_downsample_spec.data(), work, pool.work());
                            bakuage::TypedMemcpy(wave_downsampled[channel] + task_group->bg / oversample_, work + bg_in_work / oversample_, (ed_in_work - bg_in_work) / oversample_);
                        }
                        else {
                            // ここでちゃんとmax(bg, x), min(ed, x)によって、はみ出た分をカットする
                            for (int i = std::max(bg, task_group->bg); i < std::min(ed, task_group->ed); i += SimdType::length) {
                                const SimdType x = simdpp::load<SimdType>(waveOut[channel] + i);
                                const SimdType y = simdpp::load<SimdType>(grad[channel] + i);
                                const SimdType prox = ProxOperator(Fmadd<SimdType>(minusTVec, y, x)); //NaNの関係で多分順番が大事
                                const SimdType diff = prox - x;
                                dot_product = Fmadd<SimdType>(diff, y, dot_product);
                                norm_sqr = Fmadd<SimdType>(diff, diff, norm_sqr);
                                simdpp::store(waveProx[channel] + i, prox);
                            }
                        }
                        task_group->output_dot_product = simdpp::reduce_add(dot_product);
                        task_group->output_norm_sqr = simdpp::reduce_add(norm_sqr);
                    }, [evalTargetWave2](int channel, int i) {
                        return evalTargetWave2[channel] + i;
                    });
                    PerformanceCounter::GetInstance().Pause("calcEval");
                    std::cerr << "evalProx:" << eval_prox_result.eval + evalProxHi << "\tevalOut:" << evalOut << "\tdot_product:" << eval_prox_result.dot_product << "\tnorm_sqr:" << eval_prox_result.norm_sqr << std::endl;
                    if (eval_prox_result.eval + evalProxHi <= evalOut + eval_prox_result.dot_product + (0.5f / t) * eval_prox_result.norm_sqr) {
                        const double normalized_eval = ((eval_prox_result.eval + evalProxHi) * bakuage::Sqr(noise)) / (1e-37 + unit_eval * bakuage::Sqr(noise_update_min_noise_));
                        prevEvalProx = eval_prox_result.eval + evalProxHi;
                        prevNormalizedEvalProx = normalized_eval;
                        std::cerr << "eval:" << prevEvalProx << "\tt:" << t << "\tnoise:" << noise << "\tnormalized_eval:" << normalized_eval << std::endl;
                        break;
                    }
                    t *= t_beta;
                    iter2++;
                }
                
                iter++;
            }
            last_iter = iter;
            last_iter2 = iter2;
            
            std::cerr << "optimization finished. calc eval func of the final result" << std::endl;
            const auto eval_prox_result = calcEval(noise, [](impl::TaskGroup<SimdType> *task_group) {}, [this](int channel, int i) {
                return waveProx[channel] + i;
            });
            const double evalProx = eval_prox_result.eval;
            const double normalized_eval = evalProx / (1e-37 + unit_eval);
            std::cerr << "eval:" << evalProx << "\tnoise:" << noise_update_min_noise_ << "\tnormalized_eval:" << normalized_eval << "\tlast_iter:" << last_iter << "\tlast_iter2:" << last_iter2 << std::endl;
            
            std::cerr << "calcEval:" << PerformanceCounter::GetInstance().Time("calcEval") << std::endl;
            std::cerr << "calcEvalGrad:" << PerformanceCounter::GetInstance().Time("calcEvalGrad") << std::endl;
            std::cerr << "optimize_loop1:" << PerformanceCounter::GetInstance().Time("optimize_loop1") << std::endl;
                
            callback(1);
        }
        
        // for limiting_error calculation
        double CalcEvalFromProx(double _noise, double unit_eval) {
            tasks->clearCache();
            const auto eval_result = calcEval(_noise, [](impl::TaskGroup<SimdType> *task_group) {}, [this](int channel, int i) {
                return waveProx[channel] + i;
            });
            return eval_result.eval;
        }
        
        double CalcEvalGradFromProx(double _noise, double unit_eval) {
            tasks->clearCache();
            const auto eval_result = calcEvalGrad(_noise, [](impl::TaskGroup<SimdType> *task_group) {}, [this](int channel, int i) {
                return waveProx[channel] + i;
            });
            return eval_result.eval;
        }
        
        int sample_rate() const { return sample_rate_; }
        int sample_rate_downsample() const { return sample_rate_downsample_; }
        int max_available_freq() const { return max_available_freq_; }
        int oversample() const { return oversample_; }
        
        // variables
        std::vector<int> *histogram;
        bool gradEnabled;
        Float noise;
        Float *waveSrc[2];
        Float *waveSrc_downsample[2];
        Float *grad[2];
        int last_iter;
        int last_iter2;
    private:
        template <class DoublePtr>
        void CopyFromImpl(DoublePtr src, Float **dest, int stride) const {
            for (int channel = 0; channel < 2; channel++) {
                for (int i = 0; i < bg; i++)
                    dest[channel][i] = 0;
                for (int i = bg; i < bg + len_; i++)
                    dest[channel][i] = src[channel][(i - bg) * stride];
                for (int i = bg + len_; i < memLen; i++)
                    dest[channel][i] = 0;
            }
        }
        
        template <class DoublePtr>
        void CopyToImpl(Float **src, DoublePtr dest, int stride) const {
            for (int channel = 0; channel < 2; channel++) {
                for (int i = 0; i < len_; i++)
                    dest[channel][i * stride] = src[channel][i + bg];
            }
        }
        
        int sample_rate_;
        int sample_rate_downsample_;
        int max_available_freq_;
        int len_;
        int memLen;
        int bg, ed;
        impl::Tasks<SimdType> *tasks;
        Float *waveProx[2]; //x in fista(fgrad.pdf)
        Float *wavePrev[2]; //x[-1] in fista(fgrad.pdf)
        Float *waveOut[2]; //y in fista(fgrad.pdf)
        Float *wave_downsampled[2]; // oversample時にeval計算を効率良く行うための一時的な領域。oversample時のみ使われる
        Float *grad_downsampled[2]; // oversample時にeval計算を効率良く行うための領域。oversample時のみ使われる
        bakuage::AlignedPodVector<Float> oversample_lowpass_spec_; // 原点対称にするので実部だけで良い
        bakuage::AlignedPodVector<Float> oversample_hipass_spec_; // 原点対称にするので実部だけで良い
        std::string noise_update_mode_;
        double noise_update_min_noise_;
        double noise_update_initial_noise_;
        double noise_update_fista_enable_ratio_;
        int max_iter1_;
        int max_iter2_;
        int oversample_;
    };
    
}

class Tasks;



#endif 
