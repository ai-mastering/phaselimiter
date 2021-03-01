#include "phase_limiter/auto_mastering.h"

#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <fstream>
#include <mutex>
#include "gflags/gflags.h"
#include "picojson.h"
#include "tbb/tbb.h"
#include <optim.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include "bakuage/sound_quality2.h"
#include "bakuage/ms_compressor_filter.h"
#include "bakuage/utils.h"
#include "bakuage/fir_design.h"
#include "bakuage/fir_filter2.h"

DECLARE_string(sound_quality2_cache);
DECLARE_string(mastering5_optimization_algorithm);
DECLARE_int32(mastering5_optimization_max_eval_count);
DECLARE_double(mastering5_mastering_level);
DECLARE_string(mastering5_mastering_reference_file);

typedef float Float;
using namespace bakuage;

namespace {
    // compress(x) -> wet_gain -> output
    // x -> dry_gain -> output
    class LoudnessMapping {
    public:
        LoudnessMapping() {}
        LoudnessMapping(Float original_mean, Float relative_threshold,
                        Float wet_gain, Float relative_dry_gain, Float ratio):
        original_mean_(original_mean),
        target_mean_(original_mean + wet_gain),
        threshold_(original_mean + relative_threshold),
        dry_gain_(wet_gain + relative_dry_gain), inv_ratio_(1.0 / ratio) {}
        
        Float operator () (Float x) const {
            static const float log10_div_20 = std::log(10) / 20;
            Float w = std::max(threshold_, x);
            Float gain = (w - original_mean_) * inv_ratio_ + target_mean_ - w;
            Float y = x + gain;
            Float z = x + dry_gain_;
            return 20 * std::log10(1e-37 + 0.5 * std::exp(log10_div_20 * y) + 0.5 * std::exp(log10_div_20 * z));
        }
        
        Float threshold() const { return threshold_; }
    private:
        Float original_mean_;
        Float target_mean_;
        Float threshold_;
        Float dry_gain_;
        Float inv_ratio_;
    };
    typedef MsCompressorFilter<Float, LoudnessMapping, LoudnessMapping> Compressor;
    
    typedef arma::vec EffectParams;
    
    struct BandEffect {
        LoudnessMapping loudness_mapping;
        LoudnessMapping ms_loudness_mapping;
    };
    
    // 空間は重要。最適化のしやすさに関わる (あきらかに同じ値を返すような点が複数あると効率が悪い)
    // param変換。param in [a, b]。0で恒等変換になるようにする
    // 両側に均等なものはparam in [-1, 1]だが、そうでないものは[0, 1]とかもありえる
    Float ToRelThreshold(Float x) {
        return 20 * x;
    }
    Float ToWetGain(Float x) {
        return 10 * x;
    }
    Float ToRelativeDryGain(Float x) {
        return 10 * x;
    }
    Float ToRatio(Float x) {
        return std::pow(5, x);
    }
    
    // すべてゼロで恒等変換になるようにする
    struct Effect {
        Effect(const Eigen::VectorXd &original_mean, const EffectParams& params) {
            const int band_count = params.size() / 8;
            band_effects.resize(band_count);
            for (int i = 0; i < band_count; i++) {
                band_effects[i].loudness_mapping = LoudnessMapping(original_mean[2 * i + 0], ToRelThreshold(params(8 * i + 0)), ToWetGain(params(8 * i + 1)), ToRelativeDryGain(params(8 * i + 2)), ToRatio(params(8 * i + 3)));
                band_effects[i].ms_loudness_mapping = LoudnessMapping(original_mean[2 * i + 1], ToRelThreshold(params(8 * i + 4)), ToWetGain(params(8 * i + 5)), ToRelativeDryGain(params(8 * i + 6)), ToRatio(params(8 * i + 7)));
            }
        }
        std::vector<BandEffect> band_effects;
    };
    
}

namespace phase_limiter {
    
    // audio_analyzer(CalculateMultibandLoudness2)の仕様に合わせて、mean, covを計算する。
    // エフェクトはloudness vector上でシミュレーションする
    // 基準ラウドネスの違いとかはホワイトノイズを処理して補正値を計算して補正する
    void AutoMastering5(std::vector<float> *_wave, const int sample_rate, const std::function<void(float)> &progress_callback) {
        const int frames = _wave->size() / 2;
        const int channels = 2;
        const float block_sec = 0.4;
        
        // initialize sound quality calculator
        bakuage::SoundQuality2Calculator calculator;
        {
            std::ifstream ifs(FLAGS_sound_quality2_cache);
            boost::archive::binary_iarchive ia(ifs);
            ia >> calculator;
        }
        const auto band_count = calculator.band_count();
        const auto bands = calculator.bands();
        
        // initialize reference
        bakuage::MasteringReference2 mastering_reference;
        if (!FLAGS_mastering5_mastering_reference_file.empty()) {
            Eigen::VectorXd mean;
            Eigen::MatrixXd cov;
            bakuage::SoundQuality2CalculatorUnit::ParseReference(bakuage::LoadStrFromFile(FLAGS_mastering5_mastering_reference_file.c_str()).c_str(), &mean, &cov);
            mastering_reference = bakuage::MasteringReference2(mean, cov);
        }
        
        // calculate original band loudness vectors
        std::vector<bakuage::AlignedPodVector<float>> band_loudnesses;
        {
            // 400ms block
            const int sample_freq = sample_rate;
            const int width = bakuage::CeilInt<int>(sample_freq * block_sec, 4);
            const int shift = width / 2; // 本当は75% overlapだけど、高速化のために50% overlap
            const int samples = frames;
            
            bakuage::AlignedPodVector<Float> filtered(channels * samples);
            for (int i = 0; i < channels; i++) {
                LoudnessFilter<float> filter(sample_freq);
                for (int j = 0; j < samples; j++) {
                    int k = channels * j + i;
                    filtered[k] = filter.Clock((*_wave)[k]);
                }
            }
            
            const int spec_len = width / 2 + 1;
            
            // FFTの正規化も行う (sqrt(hanning)窓)
            bakuage::AlignedPodVector<float> window(width);
            bakuage::CopyHanning(width, window.data(), 1.0 / std::sqrt(width));
            
            // 規格では最後のブロックは使わないけど、
            // 使ったほうが実用的なので使う
            band_loudnesses.resize(bakuage::CeilInt(samples, shift) / shift);
            tbb::parallel_for<int>(0, band_loudnesses.size(), [&](int pos_idx) {
                const int pos = pos_idx * shift;
                bakuage::AlignedPodVector<float> band_loudness(2 * band_count);
                int end = std::min<int>(pos + width, samples);
                
                auto &pool = bakuage::ThreadLocalDftPool<bakuage::RealDft<Float>>::GetThreadInstance();
                const auto dft = pool.Get(width);
                
                bakuage::AlignedPodVector<float> fft_input(width);
                std::vector<bakuage::AlignedPodVector<std::complex<float>>> fft_outputs(channels);
                for (int ch = 0; ch < channels; ch++) {
                    fft_outputs[ch].resize(spec_len);
                }
                
                // FFT
                for (int ch = 0; ch < channels; ch++) {
                    for (int i = 0; i < width; i++) {
                        fft_input[i] = pos + i < end ? filtered[channels * (pos + i) + ch] * window[i] : 0;
                    }
                    dft->Forward(fft_input.data(), (float *)fft_outputs[ch].data(), pool.work());
                }
                
                // binをbandに振り分けていく
                for (int band_index = 0; band_index < band_count; band_index++) {
                    int low_bin_index = std::floor(width * bands[band_index].low_freq / sample_freq);
                    int high_bin_index = std::min<int>(std::floor(width * (bands[band_index].high_freq == 0 ? 0.5 : bands[band_index].high_freq / sample_freq)), spec_len);
                    
                    // mid
                    double sum = 0;
                    for (int i = low_bin_index; i < high_bin_index; i++) {
                        sum += std::norm(fft_outputs[0][i] + fft_outputs[1][i]) ;
                    }
                    band_loudness[2 * band_index + 0] = 10 * std::log10(1e-7//1e-37
                                                                     + sum / (0.5 * width));
                    // side
                    sum = 0;
                    for (int i = low_bin_index; i < high_bin_index; i++) {
                        sum += std::norm(fft_outputs[0][i] - fft_outputs[1][i]);
                    }
                    band_loudness[2 * band_index + 1] = 10 * std::log10(1e-7//1e-37
                                                                      + sum / (0.5 * width));
                }
                
                band_loudnesses[pos_idx] = std::move(band_loudness);
            });
        }
        progress_callback(0.1);
        
        // エフェクト補正値計算
        // ホワイトノイズに窓掛けFFTで上記処理をしたラウドネスと、
        // ホワイトノイズをコンプレッサーで解析したラウドネスを比較する
        // -> いったんなしで、差分を出力してみて大きいなら考える。
        
        // エフェクト適用シミュレーション (ms compressor filter)
        const auto apply_effect = [](const Effect &effect, const Float *input, Float *output) {
            for (int i = 0; i < effect.band_effects.size(); i++) {
                const auto &band_effect = effect.band_effects[i];
                // static const float sqrt_0_5 = std::sqrt(0.5); // 元のMsCompressorFilterはmsにするときにこれをかけていた。誤差が問題になるなら考える
                static const float log10_div_20 = std::log(10) / 20;
                
                Float input_m = input[2 * i + 0];
                Float input_s = input[2 * i + 1];
                Float rms_m = std::pow(10, 0.1 * input_m);
                Float rms_s = std::pow(10, 0.1 * input_s);
                
                Float total_loudness = -0.691 + 10 * std::log10(rms_m + rms_s + 1e-37);
                Float mapped_loudness = band_effect.loudness_mapping(total_loudness);
                
                Float mid_to_side_loudness = input_s - input_m;
                Float side_gain = std::exp(log10_div_20 * (band_effect.ms_loudness_mapping(mid_to_side_loudness) - mid_to_side_loudness));
                
                Float total_loudness_with_side_gain = -0.691 + 10 * std::log10(rms_m + rms_s * bakuage::Sqr(side_gain) + 1e-37);
                Float gain = std::exp(log10_div_20 * (mapped_loudness - total_loudness_with_side_gain));
                
                output[2 * i + 0] = 10 * std::log10(rms_m * bakuage::Sqr(gain));
                output[2 * i + 1] = 10 * std::log10(rms_s * bakuage::Sqr(side_gain * gain));
            }
        };
        
        // エフェクトパラメータからエフェクト適用後のmean, covと各バンドの平均誤差を計算する
        const auto calc_mean_cov = [apply_effect, band_count, &band_loudnesses](const Effect *effect, Eigen::VectorXd *mean_vec, Eigen::MatrixXd *cov, float *mse) {
            const auto relative_threshold_db = -20;
            mean_vec->resize(2 * band_count);
            cov->resize(2 * band_count, 2 * band_count);
            *mse = 0;
            
            // apply effect
            bakuage::AlignedPodVector<float> applied(2 * band_count);
            std::vector<bakuage::AlignedPodVector<float>> loudness_blocks(2 * band_count);
            for (int i = 0; i < 2 * band_count; i++) {
                loudness_blocks[i].resize(band_loudnesses.size());
            }
            for (int i = 0; i < band_loudnesses.size(); i++) {
                if (effect) {
                    apply_effect(*effect, band_loudnesses[i].data(), applied.data());
                } else {
                    bakuage::TypedMemcpy(applied.data(), band_loudnesses[i].data(), applied.size());
                }
                for (int j = 0; j < applied.size(); j++) {
                    *mse += bakuage::Sqr(band_loudnesses[i][j] - applied[j]);
                    loudness_blocks[j][i] = applied[j];
                }
            }
            *mse /= band_loudnesses.size() * applied.size();
            
            // calculate mean
            bakuage::AlignedPodVector<Float> thresholds(2 * band_count);
            for (int band_index = 0; band_index < 2 * band_count; band_index++) {
                const auto &band_blocks = loudness_blocks[band_index];
                
                double threshold = -1e10;//-70;
                for (int k = 0; k < 2; k++) {
                    Float count = 0;
                    Float sum = 0;
                    for (const auto &z: band_blocks) {
                        const bool valid = z >= threshold; // for easy optimization
                        count += valid;
                        sum += valid ? z : 0;
                    }
                    
                    double mean = sum / (1e-37 + count);
                    if (k == 0) {
                        threshold = mean + relative_threshold_db;
                        thresholds[band_index] = threshold;
                    }
                    else if (k == 1) {
                        (*mean_vec)[band_index] = mean;
                    }
                }
            }
            
            // calculate covariance
            for (int band_index1 = 0; band_index1 < 2 * band_count; band_index1++) {
                for (int band_index2 = band_index1; band_index2 < 2 * band_count; band_index2++) {
                    const Float mean1 = (*mean_vec)[band_index1];
                    const Float mean2 = (*mean_vec)[band_index2];
                    const Float threshold1 = thresholds[band_index1];
                    const Float threshold2 = thresholds[band_index2];
                    
                    const auto &band_blocks1 = loudness_blocks[band_index1];
                    const auto &band_blocks2 = loudness_blocks[band_index2];
                    
                    Float v = 0;
                    Float c = 0;
                    for (int i = 0; i < band_blocks1.size(); i++) {
                        const auto x1 = band_blocks1[i];
                        const auto x2 = band_blocks2[i];
                        const bool valid = (x1 >= threshold1) & (x2 >= threshold2); // for easy optimization (not && instead &)
                        v += valid * (x1 - mean1) * (x2 - mean2);
                        c += valid;
                    }
                    v  /= (1e-37 + c);
                    (*cov)(band_index1, band_index2) = v;
                    (*cov)(band_index2, band_index1) = v;
                }
            }
        };
        
        // calculate original
        Eigen::VectorXd original_mean;
        Eigen::MatrixXd original_cov;
        float original_mse;
        calc_mean_cov(nullptr, &original_mean, &original_cov, &original_mse);
        
        // define bound
        arma::vec lower_bounds(8 * band_count);
        arma::vec upper_bounds(8 * band_count);
        for (int i = 0; i < band_count; i++) {
            lower_bounds(8 * i + 0) = -1; // rel threshold
            upper_bounds(8 * i + 0) = 0.01;
            lower_bounds(8 * i + 1) = -1; // wet gain
            upper_bounds(8 * i + 1) = 1;
            lower_bounds(8 * i + 2) = -1; // relative dry gain
            upper_bounds(8 * i + 2) = 0.01;
            lower_bounds(8 * i + 3) = -0.01; // ratio
            upper_bounds(8 * i + 3) = 1;
            lower_bounds(8 * i + 4) = -1;
            upper_bounds(8 * i + 4) = 0.01;
            lower_bounds(8 * i + 5) = -1;
            upper_bounds(8 * i + 5) = 1;
            lower_bounds(8 * i + 6) = -1;
            upper_bounds(8 * i + 6) = 0.01;
            lower_bounds(8 * i + 7) = -0.01;
            upper_bounds(8 * i + 7) = 1;
        }
        {
            const double scale = 1e-2 + FLAGS_mastering5_mastering_level;
            lower_bounds *= scale;
            upper_bounds *= scale;
        }
        
        // エフェクトパラメータから評価関数を計算する
        std::mutex eval_mtx;
        float min_eval = 1e100;
        int eval_count = 0;
        const auto calc_eval = [calc_mean_cov, &calculator, &original_mean, &min_eval, &eval_count, &eval_mtx, &progress_callback, &lower_bounds, &upper_bounds, &mastering_reference](const EffectParams &params) -> double {
            Eigen::VectorXd mean;
            Eigen::MatrixXd cov;
            float msp = 0;
            for (int i = 0; i < params.size(); i++) {
                msp += bakuage::Sqr(params(i));
            }
            msp /= params.size();
            float bound_error = 0;
            for (int i = 0; i < params.size(); i++) {
                bound_error += bakuage::Sqr(std::max<float>(0, lower_bounds[i] - params[i]));
                bound_error += bakuage::Sqr(std::max<float>(0, params[i] - upper_bounds[i]));
            }
            float mse;
            Effect effect(original_mean, params);
            calc_mean_cov(&effect, &mean, &cov, &mse);
            
            const bakuage::MasteringReference2 target(mean, cov);
            float main_eval = 0;
            if (FLAGS_mastering5_mastering_reference_file.empty()) {
                float sound_quality;
                calculator.CalculateSoundQuality(target, &sound_quality, nullptr);
                main_eval = -sound_quality;
            } else {
                main_eval = calculator.CalculateDistance(mastering_reference, target);
            }
            
            // 事前の実験でalphaをいろいろ変えて最適化してmseを見たところ
            // 最適解のところで sound_qualityの微分とmseの微分が一致していると仮定すると、
            // alpha = 0.02 / sqrt(mse)になった。
            // mseの許容量(target_mse)からalphaを逆算して決める。
            // target_mseはFLAGS_mastering5_mastering_levelを元に適当に決める
            // betaはエフェクトのパラメータを変えたときにどのくらいmseが変わるかを考えて、alphaに比例させて決める。(paramを1変化させて、特徴量が10dB変化するなら、beta = 10^2 * alpha -> いろいろ試して微調整)
            // 実際これでやってみてtarget_mseになるかどうかで妥当性検証可能
            const float target_mse = bakuage::Sqr(4 * (1e-2 + FLAGS_mastering5_mastering_level));
            const float alpha = 0.02 / std::sqrt(target_mse);
            const float beta = bakuage::Sqr(10.0) * alpha;
            const float eval = main_eval + alpha * mse + beta * msp + bound_error * 1e4;
            {
                std::lock_guard<std::mutex> lock(eval_mtx);
                eval_count++;
                if (eval_count % (FLAGS_mastering5_optimization_max_eval_count / 10) == 0 && eval_count < FLAGS_mastering5_optimization_max_eval_count) {
                    progress_callback(0.1 + 0.5 * eval_count / FLAGS_mastering5_optimization_max_eval_count);
                }
                if (min_eval > eval) {
                    min_eval = eval;
                    std::cerr << "optimization " << eval_count << "\t" << min_eval << "\t" << main_eval << "\t" << mse << "\t" << msp << std::endl;
                }
            }
            return eval;
        };
        
        // calc initial eval
        EffectParams zero_params(8 * band_count);
        for (int i = 0; i < zero_params.size(); i++) {
            zero_params(i) = 0;
        }
        const auto initial_eval = calc_eval(zero_params);
        std::cerr << "optimization initial_eval: " << initial_eval << std::endl;
        
        // エフェクトパラメータ探索
        const auto find_params = [calc_eval, band_count, &zero_params, initial_eval, &lower_bounds, &upper_bounds]() {
            optim::algo_settings_t settings;
#if 0
            // boundしているとうまく最適化できないので使わない。
            // grep inv_transform in optim
            // optimは有界な値を実数全体に変換して最適化している
            // 境界そのものはinfに変換されてしまうからうまく処理できない
            // 原点が境界そのものにならないように注意。
            // 最初の分布も重要。
            // boundして最適化すると境界付近の値になりやすい気がする (変換の形的に境界付近の存在確率が高まる？次元の呪い？)
            // 正則化強めだと、boundなしのほうが最適化性能が高い。正則化弱めだと少しだけboundありのほうが性能高いが大差無い
            // だから、boundなしで行こう。こっちのほうが安定している。
            // 評価関数側でなんとかする
            settings.lower_bounds = lower_bounds;
            settings.upper_bounds = upper_bounds;
            settings.vals_bound = true;
#endif
#if 1
            settings.de_initial_lb = lower_bounds;
            settings.de_initial_ub = upper_bounds;
            settings.pso_initial_lb = lower_bounds;
            settings.pso_initial_ub = upper_bounds;
#endif
            auto result = zero_params;
            bool success;
            // それぞれ最大繰り返し回数の指定方法が違うので注意。ソースコード参照
            if (FLAGS_mastering5_optimization_algorithm == "nm") {
                settings.iter_max = FLAGS_mastering5_optimization_max_eval_count / lower_bounds.size();
                success = optim::nm(result, [calc_eval](const arma::vec& vec, arma::vec* grad_out, void *opt_data) {
                    return calc_eval(vec);
                }, nullptr, settings);
            }
            else if (FLAGS_mastering5_optimization_algorithm == "pso") {
                settings.pso_n_gen = FLAGS_mastering5_optimization_max_eval_count / settings.pso_n_pop;
                success = optim::pso(result, [calc_eval](const arma::vec& vec, arma::vec* grad_out, void *opt_data) {
                    return calc_eval(vec);
                }, nullptr, settings);
            } else if (FLAGS_mastering5_optimization_algorithm == "pso_dv") {
                settings.pso_n_gen = FLAGS_mastering5_optimization_max_eval_count / settings.pso_n_pop;
                success = optim::pso_dv(result, [calc_eval](const arma::vec& vec, arma::vec* grad_out, void *opt_data) {
                    return calc_eval(vec);
                }, nullptr, settings);
            } else if (FLAGS_mastering5_optimization_algorithm == "de") {
                settings.de_max_fn_eval = FLAGS_mastering5_optimization_max_eval_count;
                success = optim::de(result, [calc_eval](const arma::vec& vec, arma::vec* grad_out, void *opt_data) {
                    return calc_eval(vec);
                }, nullptr, settings);
            } else if (FLAGS_mastering5_optimization_algorithm == "de_prmm") {
                settings.de_max_fn_eval = FLAGS_mastering5_optimization_max_eval_count;
                success = optim::de_prmm(result, [calc_eval](const arma::vec& vec, arma::vec* grad_out, void *opt_data) {
                    return calc_eval(vec);
                }, nullptr, settings);
            } else {
                throw std::logic_error(std::string("unknown FLAGS_mastering5_optimization_algorithm " + FLAGS_mastering5_optimization_algorithm));
            }
            const auto result_eval = calc_eval(result);
            std::cerr << "optimization success: " << success << std::endl;
            std::cerr << "optimization solution y: " << result_eval << std::endl;
            for (int i = 0; i < band_count; i++) {
                std::cerr << "optimization solution x " << i << "\t" << result(8 * i + 0) << "\t" << result(8 * i + 1) << "\t" << result(8 * i + 2) << "\t" << result(8 * i + 3) << std::endl;
                std::cerr << "optimization solution x ms " << i << "\t" << result(8 * i + 4) << "\t" << result(8 * i + 5) << "\t" << result(8 * i + 6) << "\t" << result(8 * i + 7) << std::endl;
            }
            if (result_eval < initial_eval) {
                return result;
            } else {
                return zero_params;
            }
        };
        
        const auto effect_params = find_params();
        const Effect effect(original_mean, effect_params);
        
        std::mutex result_mtx;
        std::mutex progression_mtx;
        std::vector<std::function<void ()>> tasks;
        std::vector<Float> result(_wave->size());
        bakuage::AlignedPodVector<Float> progressions(band_count);
        
        const auto update_progression = [&progressions, &progression_mtx, progress_callback](int i, Float p) {
            std::lock_guard<std::mutex> lock(progression_mtx);
            Float total = 0;
            progressions[i] = p;
            for (const auto &a : progressions) {
                total += a;
            }
            progress_callback(0.6 + 0.4 * total / progressions.size());
        };
        
        for (int band_index = 0; band_index < band_count; band_index++) {
            const auto &band = calculator.bands()[band_index];
            const auto update_progression_bound = std::bind(update_progression, band_index, std::placeholders::_1);
            const auto &band_effect = effect.band_effects[band_index];
            tasks.push_back([band, band_effect, sample_rate, frames, _wave, &result, &result_mtx, update_progression_bound]() {
                const float *wave_ptr = &(*_wave)[0];
                
                int fir_delay_samples;
                std::vector<Float> fir;
                {
                    fir_delay_samples = static_cast<int>(0.2 * sample_rate);
                    const int n = 2 * fir_delay_samples + 1;
                    Float freq1 = std::min<Float>(0.5, band.low_freq / sample_rate);
                    Float freq2 = std::min<Float>(0.5, band.high_freq == 0 ? 0.5 : band.high_freq / sample_rate);
                    fir = CalculateBandPassFir<Float>(freq1, freq2, n, 4);
                }
                update_progression_bound(0.1);
                
                const int len = frames + fir.size() - 1;
                bakuage::AlignedPodVector<float> filtered(channels * len);
                {
                    FirFilter2<Float> fir_filter(fir.begin(), fir.end());
                    bakuage::AlignedPodVector<Float> filter_temp_input(frames + fir_delay_samples);
                    bakuage::AlignedPodVector<Float> filter_temp_output(frames + fir_delay_samples);
                    for (int ch = 0; ch < channels; ch++) {
                        fir_filter.Clear();
                        for (int i = 0; i < frames; i++) {
                            filter_temp_input[i] = wave_ptr[channels * i + ch];
                        }
                        fir_filter.Clock(filter_temp_input.data(), filter_temp_input.data() + frames + fir_delay_samples, filter_temp_output.data());
                        for (int i = 0; i < frames; i++) {
                            filtered[channels * i + ch] = filter_temp_output[i + fir_delay_samples];
                        }
                    }
                }
                update_progression_bound(0.2);
                
                Compressor::Config compressor_config;
                compressor_config.loudness_mapping_func = band_effect.loudness_mapping;
                compressor_config.ms_loudness_mapping_func = band_effect.ms_loudness_mapping;
                compressor_config.max_mean_sec = 0.2;
                compressor_config.num_channels = channels;
                compressor_config.sample_rate = sample_rate;
                Compressor compressor(compressor_config);
                
                const int shift = compressor.delay_samples();
                const int len2 = frames + shift;
                filtered.resize(channels * len2);
                bakuage::AlignedPodVector<Float> temp_input(channels);
                bakuage::AlignedPodVector<Float> temp_output(channels);
                
                // filteredにin-placeで書き込んでから共有のresultに足しこむ
                for (int j = 0; j < len2; j++) {
                    for (int i = 0; i < channels; i++) {
                        temp_input[i] = filtered[channels * j + i];
                    }
                    compressor.Clock(&temp_input[0], &temp_output[0]);
                    for (int i = 0; i < channels; i++) {
                        filtered[channels * j + i] = temp_output[i];
                    }
                }
                update_progression_bound(0.8);
                
                // flush filtered (dry sound)
                {
                    std::lock_guard<std::mutex> lock(result_mtx);
                    const int len3 = frames * channels;
                    const int channels_shift = channels * shift;
                    bakuage::VectorAddInplace(filtered.data() + channels_shift, result.data(), len3);
                }
                update_progression_bound(1);
            });
        }
        
        tbb::parallel_for(0, (int)tasks.size(), [&tasks](int task_i) {
            tasks[task_i]();
        });
        
        *_wave = std::move(result);
    }
    
}
