#include "bakuage/dissonance.h"

#include <cmath>
#include <complex>
#include <vector>
#include "tbb/tbb.h"
#include "bakuage/memory.h"
#include "bakuage/utils.h"
#include "bakuage/vector_math.h"
#include "bakuage/dft.h"
#include "bakuage/window_func.h"

namespace {
    template <class Float>
    struct DissonanceSethares1993Cache {
        DissonanceSethares1993Cache(const Float *freqs, int count): ds(count) {
            const double a = 3.5;
            const double b = 5.75;
            const double d_max = 0.24;
            const double s1 = 0.0207;
            const double s2 = 18.96;
            for (int i = 0; i < count; i++) {
                bakuage::AlignedPodVector<Float> row(count);
                const double s = d_max / (s1 * freqs[i] + s2);
                for (int j = i + 1; j < count; j++) {
                    const double x = s * (freqs[j] - freqs[i]);
                    row[j - i - 1] = std::exp(-a * x) - std::exp(-b * x);
                    
                    if (x > 10 || j == count - 1) { // d < 6.4e-16 when x > 10
                        row.resize(j - i);
                        row.shrink_to_fit();
                        ds[i] = std::move(row);
                        break;
                    }
                }
            }
        }
        std::vector<bakuage::AlignedPodVector<Float>> ds;
    };
    
    struct CalculateDissonanceResult {
        CalculateDissonanceResult operator + (const CalculateDissonanceResult &other) const {
            CalculateDissonanceResult result;
            result.dissonance = dissonance + other.dissonance;
            result.energy = energy + other.energy;
            return result;
        }
        double dissonance;
        double energy;
    };
    
    struct CalculateDissonanceThreadVar {
        CalculateDissonanceThreadVar(int width): spec_len(width / 2 + 1), fft_input(width), fft_output(spec_len), fft_energy(spec_len) {}
        
        static std::shared_ptr<CalculateDissonanceThreadVar> GetThreadInstance(int width) {
            static thread_local std::weak_ptr<CalculateDissonanceThreadVar> instance;
            auto locked = instance.lock();
            if (locked) {
                return locked;
            } else {
                locked = std::make_shared<CalculateDissonanceThreadVar>(width);
                instance = locked;
                return locked;
            }
        }
        
        int spec_len;
        bakuage::AlignedPodVector<float> fft_input;
        bakuage::AlignedPodVector<std::complex<float>> fft_output;
        bakuage::AlignedPodVector<float> fft_energy;
    };
    
    // spec contains energy (freqs must be ASC sorted)
    template <class Float>
    double DissonanceSethares1993(const Float *freqs, const Float *amps, int count, const DissonanceSethares1993Cache<Float> *cache) {
        double sum = 0;
        if (cache) {
            for (int i = 0; i < count; i++) {
                double s = 0;
#if 1
                s = bakuage::VectorDot(amps + i + 1, cache->ds[i].data(), cache->ds[i].size());
#else
                for (int j = 0; j < cache->ds[i].size(); j++) {
                    s += amps[i + j + 1] * cache->ds[i][j];
                }
#endif
                sum += amps[i] * s;
            }
        }
        else {
            // before optimization
            for (int i = 0; i < count; i++) {
                for (int j = i + 1; j < count; j++) {
                    sum += bakuage::DissonancePairSethares1993(freqs[i], freqs[j], amps[i], amps[j]);
                }
            }
        }
        return sum;
    }
}

namespace bakuage {
    // reference
    // https://pypi.org/project/dissonant/
    // https://github.com/bzamecnik/dissonant/blob/master/dissonant/tuning.py (Hz)
    // https://essentia.upf.edu/documentation/reference/streaming_Dissonance.html
    // amp^2 = energy
    
    double DissonancePairSethares1993(double hz1, double hz2, double amp1, double amp2) {
        const double a = 3.5;
        const double b = 5.75;
        const double d_max = 0.24;
        const double s1 = 0.0207;
        const double s2 = 18.96;
        
        if (hz2 < hz1) return DissonancePairSethares1993(hz2, hz1, amp2, amp1);
        
        const double s = d_max / (s1 * hz1 + s2);
        const double x = s * (hz2 - hz1);
        const double spl = amp1 * amp2;
        const double d = std::exp(-a * x) - std::exp(-b * x);
        return spl * d;
    }
    
    // mfccはenergy sum modeを想定している
    template <class Float>
    void CalculateDissonance(Float *input, int channels, int samples, int sample_freq, Float *dissonance, bool tbb_parallel) {
        // calculate mfcc
        const int shift_resolution = 2;
        const int output_shift_resolution = 2;
        const int width = output_shift_resolution * ((16384 * sample_freq / 44100) / output_shift_resolution); // 0.372 sec, 4x only
        const int shift = width / shift_resolution;
        const int spec_len = width / 2 + 1;
        
        // common const data
        bakuage::AlignedPodVector<float> window(width);
        bakuage::CopyHanning(width, window.begin());
        bakuage::AlignedPodVector<float> freqs(spec_len);
        for (int i = 0; i < spec_len; i++) {
            freqs[i] = 1.0 * (i + 0.5) * sample_freq / width;
        }
        DissonanceSethares1993Cache<float> cache(freqs.data(), spec_len);
        
        // for parallel_for
        bakuage::AlignedPodVector<int> pos_list;
        {
            int pos = -width + shift;
            while (pos < samples) {
                pos_list.push_back(pos);
                pos += shift;
            }
        }
        std::vector<std::shared_ptr<CalculateDissonanceThreadVar>> allocated_thread_vars(pos_list.size());
        
        const auto reduce_loop_func = [&](tbb::blocked_range<int> r, CalculateDissonanceResult init) {
            for (int pos_list_i = r.begin(); pos_list_i != r.end(); pos_list_i++) {
                const int pos = pos_list[pos_list_i];
                auto tv = CalculateDissonanceThreadVar::GetThreadInstance(width);
                allocated_thread_vars[pos_list_i] = tv;
                
                bakuage::TypedFillZero(tv->fft_energy.data(), tv->fft_energy.size());
                for (int i = 0; i < channels; i++) {
                    for (int j = 0; j < width; j++) {
                        int k = pos + j;
                        tv->fft_input[j] = (0 <= k && k < samples) ? input[channels * k + i] * window[j] : 0;
                    }
                    auto &pool = bakuage::ThreadLocalDftPool<bakuage::RealDft<float>>::GetThreadInstance();
                    const auto dft = pool.Get(width);
                    dft->Forward(tv->fft_input.data(), (float *)tv->fft_output.data(), pool.work());
                    for (int j = 0; j < spec_len; j++) {
                        tv->fft_energy[j] += std::norm(tv->fft_output[j]);
                    }
                }
                
                init.energy += bakuage::VectorSum(tv->fft_energy.data(), spec_len);
                bakuage::VectorSqrtInplace(tv->fft_energy.data(), spec_len);
                init.dissonance += DissonanceSethares1993(freqs.data(), tv->fft_energy.data(), spec_len, &cache);
            }
            return init;
        };
        
        const auto reduce_func = [](const CalculateDissonanceResult &x, const CalculateDissonanceResult &y) {
            return x + y;
        };
        
        CalculateDissonanceResult result;
        result.dissonance = 0;
        result.energy = 0;
        
        if (tbb_parallel) {
            result = tbb::parallel_reduce(tbb::blocked_range<int>(0, pos_list.size()), result, reduce_loop_func, reduce_func);
        } else {
            result = reduce_loop_func(tbb::blocked_range<int>(0, pos_list.size()), result);
        }
        
        *dissonance = result.dissonance
        * (1.0 * sample_freq / width) // compensation
        / (1e-37 + result.energy);
    }
    
    template
    void CalculateDissonance<float>(float *input, int channels, int samples, int sample_freq, float *dissonance, bool tbb_parallel);
    template
    void CalculateDissonance<double>(double *input, int channels, int samples, int sample_freq, double *dissonance, bool tbb_parallel);
}
