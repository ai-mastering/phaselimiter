#ifndef BAKUAGE_BAKUAGE_CLEAR_MIXER_FILTER4_H_
#define BAKUAGE_BAKUAGE_CLEAR_MIXER_FILTER4_H_

// FirFilter2のほうが速い
// #define BAKUAGE_CLEAR_MIXER_FILTER4_USE_FIR_FILTER4

#include <algorithm>
#include <limits>
#include <numeric>
#include <vector>
#include "bakuage/dft.h"
#include "bakuage/memory.h"
#include "bakuage/utils.h"
#include "bakuage/window_func.h"
#include "bakuage/fir_design.h"
#ifdef BAKUAGE_CLEAR_MIXER_FILTER4_USE_FIR_FILTER4
#include "bakuage/fir_filter4.h"
#else
#include "bakuage/fir_filter2.h"
#endif
#include "bakuage/fir_filter_bank.h"
#include "bakuage/time_varying_lowpass_filter.h"
#include "bakuage/delay_filter2.h"

// ClearMixer3の高速化版
// シンプルにするためにkFilterTimeVaryingや細かいifdef分岐は削除してある

/*
 さらに最適化したい場合のメモ
 重いのは、FirFilter2とFirFilterBank。

 可能性1: メモリ最適化
 メモリボトルネックかどうかはわからないが、
 プロファイラで見ると結構メモリコピーとかのウェイトが大きいから、
 メモリボトルネックの可能性はあるかも。
 そうだとしたら、キャッシュヒット率を上げれば良い。
 具体的には、重い計算とメモリアクセスを抱き合わせにするとか、
 FIRをの前計算結果を共有するとか。
 
 可能性2: FIRの間引きを周波数空間でやる
 FirFilter2をup sample, down sampleに対応させて、
 間引きとかを周波数空間でやれば、
 アルゴリズム的に高速化できる。
 面倒だからやっていないだけ。
 
 可能性3: VectorMathではなく自前のSIMDで書く
 VectorMathは演算ユニット使用効率が悪い。
 divとかsqrtとかがもったいないので、
 自前のSIMDで書けば早くなるかも。
 */

namespace bakuage {
    // Delphiから移植
    // FirFilterbank方式
    template <
    typename Float
    >
    class ClearMixerFilter4 {
    public:
        typedef FirFilterBank<Float> FilterBank;
        
        enum {
            // kFilterTimeVarying = 1,
            kFilterFir = 2,
        };
        
        enum {
            kNoiseReductionDisabled = 0,
            kNoiseReductionFlat = 1, // -3dB/octで一律カット
            kNoiseReductionFixedSpectrum = 2, // 与えた静的な周波数特性でカット
            kNoiseReductionFixedSpectrumLearn = 3, // 与えた静的な周波数特性でカット (学習), これを使った場合lock freeではないので注意
        };
        
        struct NoiseReductionFixedSpectrumProfile {
            std::vector<bakuage::AlignedPodVector<Float>> energy_thresholds; // [ch][band]
        };
        
        class Config {
        public:
            Config(): num_tracks(0), num_channels(0), sample_rate(0), fir_samples(0), gain_decimation(0), filter(0), energy_mean_sec(0), scale_mean_sec(0), noise_reduction(0), noise_reduction_threshold(0) {}
            int num_tracks;
            int num_channels;
            int sample_rate;
            int fir_samples;
            int gain_decimation;
            int filter;
            float energy_mean_sec;
            float scale_mean_sec;
            int noise_reduction;
            float noise_reduction_threshold; // energy, kNoiseReductionFixedSpectrumLearnではシフト量
            NoiseReductionFixedSpectrumProfile noise_reduction_fixed_spectrum_profile;
        };
        
        // 狭帯域ノイズを入力したときのenergyはサンプル周波数に依存しないようにする。
        // 細かいスケーリングは気にしないが
        // 振幅1の正弦波を入力したときに、どこかの帯域のエネルギーが1として出力されれば良い
        // 平均エネルギーではなく合計エネルギーを出力する
        // 表示するときはエネルギー/Hzという単位で表示すれば良い
        // ちなみに、ClearMixer2方式だと、表示するときにbin数で平均しているから、FFT sizeが大きくなると表示されるエネルギーが小さくなるはず
        struct AnalysisResult {
            std::vector<AlignedPodVector<Float>> *pre_scales; // [ch][band][track] sqrt(energy)
            std::vector<AlignedPodVector<Float>> *post_scales; // [ch][band][track] sqrt(energy)
            AlignedPodVector<int> *primary_tracks; // [ch][band]
            int frames;
        };
        
        ClearMixerFilter4(const Config &config):
        config_(config),
        wet_scale_(1),
        dry_scale_(0),
        ratio_(1),
        primary_scale_(1),
        eps_(0),
        track_scales_(config.num_tracks, 1.0),
        bands_(CreateBandsByErb(config.sample_rate, 1.0)),
        temp_scale_(config.num_channels, std::vector<AlignedPodVector<Float>>(bands_.size(), AlignedPodVector<Float>(config.num_tracks))),
        temp_original_scale_(config.num_channels, std::vector<AlignedPodVector<Float>>(bands_.size(), AlignedPodVector<Float>(config.num_tracks))),
        temp_primary_track_(config.num_channels, AlignedPodVector<int>(bands_.size())),
        temp_silent_len_(config.num_tracks, AlignedPodVector<uint64_t>(config.num_channels)),
        analysis_size_(CeilInt(2 * config.fir_samples, config.gain_decimation)),
        input_buffer_(config.num_tracks, std::vector<AlignedPodVector<Float>>(config.num_channels, AlignedPodVector<Float>(analysis_size_))),
        output_buffer_(config.num_channels, AlignedPodVector<Float>(analysis_size_)),
        temp_synthesized_(analysis_size_),
        buffer_pos_(0),
        splitted_(config.num_tracks, std::vector<AlignedPodVector<std::complex<Float>>>(bands_.size(), AlignedPodVector<std::complex<Float>>(analysis_size_))),
        splitted_ptrs_(bands_.size()),
        splitted_const_ptrs_(bands_.size()),
        temp_original_scale2_(config.num_tracks, AlignedPodVector<Float>(analysis_size_)), // この辺は過剰に領域をとっているので注意
        temp_scale2_(config.num_tracks, AlignedPodVector<Float>(analysis_size_)),
        temp_gain2_(config.num_tracks, AlignedPodVector<Float>(analysis_size_)),
        temp_original_l2_(analysis_size_),
        temp_l2_(analysis_size_),
        temp_inv_max_scale_(analysis_size_),
        temp_general_(analysis_size_),
        temp_mi_(analysis_size_)
        {
            typename FilterBank::Config filter_bank_config;
            for (const auto &band: bands_) {
                typename FilterBank::BandConfig band_config;
                const auto normalized_band_width = 1.0 * (band.high_freq - band.low_freq) / config.sample_rate;
                // alpha = 7のカイザー窓のケース
                // http://www.mk.ecei.tohoku.ac.jp/jspmatlab/pdf/matdsp4.pdf
                const auto normalized_transition_width = (9.0 / 2.0) / config.fir_samples;
#if 0
                band_config.decimation = config.gain_decimation;
#else
                // 適当な係数。エイリアシングノイズは本質的にFIRから生じるものと、FFTの処理上(decimationでゼロにする由来)発生するものの二種類ある
                // 後者のほうを抑えるために、decimationは小さめにしておく。1だとかなり発生するが、0.5だとかなり抑えられる
                // これをよりうまく抑えるのはFuture work
                // gain_decimationが大きいと、これに気付かないから注意
                const double deci_scale = 0.5;
                band_config.decimation = std::max<int>(1, CeilPowerOf2(std::min<int>(config.gain_decimation, deci_scale / (normalized_band_width + 2 * normalized_transition_width))));
#endif
                const auto normalized_center_freq = 0.5 * (band.low_freq + band.high_freq) / config.sample_rate;
                const auto normalized_bg_freq = band.low_freq / config.sample_rate;
                const auto normalized_ed_freq = band.high_freq / config.sample_rate;
                const auto normalized_min_freq = normalized_center_freq - 0.5 / band_config.decimation;
                const auto normalized_max_freq = normalized_center_freq + 0.5 / band_config.decimation;
                band_config.nonzero_base_normalized_freq = normalized_min_freq;
                
                const auto analysis_fir = CalculateBandPassFirComplex<Float>(normalized_bg_freq, normalized_ed_freq, config.fir_samples, 7);
                band_config.analysis_fir = AlignedPodVector<std::complex<Float>>(analysis_fir.begin(), analysis_fir.end());
                const auto synthesis_fir = CalculateBandPassFirComplex<Float>((normalized_bg_freq + normalized_min_freq) / 2, (normalized_ed_freq + normalized_max_freq) / 2, config.fir_samples, 7);
                band_config.synthesis_fir = AlignedPodVector<std::complex<Float>>(synthesis_fir.begin(), synthesis_fir.end());
                
                filter_bank_config.bands.emplace_back(std::move(band_config));
            }
            
            for (int i = 0; i < config.num_tracks; i++) {
                std::vector<FilterBank> banks;
                for (int j = 0; j < config.num_channels; j++) {
                    banks.emplace_back(filter_bank_config);
                }
                filter_banks_.emplace_back(std::move(banks));
            }
            
            filter_bank_config_ = filter_bank_config;
            
            {
                min_energy_delay_samples_ = CeilInt<int>(config.sample_rate * config.energy_mean_sec / 2, config.gain_decimation);
                min_scale_delay_samples_ = CeilInt<int>(config.sample_rate * config.scale_mean_sec / 2, config.gain_decimation);
                
#ifdef BAKUAGE_CLEAR_MIXER_FILTER4_USE_FIR_FILTER4
                std::vector<FirFilter4<Float>> energy_filters;
                std::vector<FirFilter4<Float>> scale_filters;
#else
                std::vector<FirFilter2<Float>> energy_filters;
                std::vector<FirFilter2<Float>> scale_filters;
#endif
                for (int i = 0; i < bands_.size(); i++) {
                    const auto &band = filter_bank_config.bands[i];
                    const int samples_per_gain = config_.gain_decimation / band.decimation;
                    
                    const int energy_n = 2 * min_energy_delay_samples_ / band.decimation + 1;
                    AlignedPodVector<Float> energy_fir(energy_n);
                    CopyHanning(energy_n, energy_fir.begin(), 1.0 / energy_n);
#ifdef BAKUAGE_CLEAR_MIXER_FILTER4_USE_FIR_FILTER4
                    energy_filters.emplace_back(energy_fir.begin(), energy_fir.end(), 1, samples_per_gain);
#else
                    energy_filters.emplace_back(energy_fir.begin(), energy_fir.end());
#endif
                    
                    const int scale_n = 2 * min_scale_delay_samples_ / band.decimation + 1;
                    AlignedPodVector<Float> scale_fir(scale_n);
                    CopyHanning(scale_n, scale_fir.begin(), 1.0 / scale_n);
                    // apply sample hold
                    AlignedPodVector<Float> scale_fir_with_sample_hold(scale_n + samples_per_gain - 1);
                    for (int j = 0; j < scale_n; j++) {
                        for (int k = 0; k < samples_per_gain; k++) {
                            scale_fir_with_sample_hold[j + k] += scale_fir[j];
                        }
                    }
#ifdef BAKUAGE_CLEAR_MIXER_FILTER4_USE_FIR_FILTER4
                    scale_filters.emplace_back(scale_fir_with_sample_hold.begin(), scale_fir_with_sample_hold.end(), samples_per_gain, 1);
#else
                    scale_filters.emplace_back(scale_fir_with_sample_hold.begin(), scale_fir_with_sample_hold.end());
#endif
                }
                energy_lowpass_fir_filters_ = Create2DVector(config_.num_tracks, config_.num_channels, energy_filters);
                scale_lowpass_fir_filters_ = Create2DVector(config_.num_tracks, config_.num_channels, scale_filters);
            }
            
            splitted_delay_filters_ = Create3DVector(config_.num_tracks, config_.num_channels, bands_.size(), DelayFilter2<std::complex<Float>>(1));
            for (int i = 0; i < bands_.size(); i++) {
#if 0
                min_energy_delay_samples_ = min_scale_delay_samples_ = 0;
                DelayFilter<std::complex<Float>> delay_filter(0);
#else
                DelayFilter2<std::complex<Float>> delay_filter((min_energy_delay_samples_ + min_scale_delay_samples_) / filter_bank_config.bands[i].decimation);
#endif
                for (int track = 0; track < config.num_tracks; track++) {
                    for (int ch = 0; ch < config.num_channels; ch++) {
                        splitted_delay_filters_[track][ch][i] = delay_filter;
                    }
                }
            }
            
#ifdef BA_CLEAR_MIXER_FILTER3_TRUE_DRY_ENABLED
            // input_bufferをoutput_bufferにコピーするので、analysis_size_は引く必要がある
            dry_delay_filters_ = Create2DVector(config_.num_tracks, config_.num_channels, DelayFilter<Float>(delay_samples() - analysis_size_));
#endif
            
            if (config.noise_reduction == kNoiseReductionFixedSpectrumLearn) {
                noise_reduction_fixed_spectrum_learn_ = std::vector<std::vector<bakuage::AlignedPodVector<Float>>>(config_.num_channels, std::vector<bakuage::AlignedPodVector<Float>>(bands_.size()));
            }
        }
        
        NoiseReductionFixedSpectrumProfile CalculateNoiseReductionFixedSpectrumProfile() const {
            NoiseReductionFixedSpectrumProfile profile;
            profile.energy_thresholds = std::vector<bakuage::AlignedPodVector<Float>>(config_.num_channels, bakuage::AlignedPodVector<Float>(bands_.size()));
            
            for (int ch = 0; ch < config_.num_channels; ch++) {
                for (int band = 0; band < bands_.size(); band++) {
                    auto energies = noise_reduction_fixed_spectrum_learn_[ch][band];
                    if (energies.size()) {
                        std::sort(energies.begin(), energies.end());
                        profile.energy_thresholds[ch][band] = energies[0.01 * energies.size()] * config_.noise_reduction_threshold;
                    } else {
                        profile.energy_thresholds[ch][band] = 0;
                    }
                }
            }
            
            return profile;
        }
        
        // input[track][channel][frame], output[channel][frame]
        void Clock(Float ***input, int frames, Float **output) {
            Clock(input, frames, output, [](const AnalysisResult &result) {});
        }
        
        // callback内のポインタは、callback呼び出し中のみ有効
        // input[track][ch]がnullの場合は、無音とみなす
        template <class AnalysisCallback>
        void Clock(Float ***input, int frames, Float **output, const AnalysisCallback &analysis_callback) {
            int base_frame = 0;
            
            while (base_frame < frames) {
                const int remaining_buffer_size = analysis_size_ - buffer_pos_;
                const int fill_size = std::min<int>(remaining_buffer_size, frames - base_frame);
                
                // 入力 + 出力 (無音検知処理付き)
                for (int ch = 0; ch < config_.num_channels; ch++) {
                    for (int track = 0; track < config_.num_tracks; track++) {
                        if (input[track][ch]) {
                            std::memcpy(input_buffer_[track][ch].data() + buffer_pos_, input[track][ch] + base_frame, sizeof(Float) * fill_size);
                            const bool is_input_silent = std::all_of(input[track][ch] + base_frame, input[track][ch] + base_frame + fill_size, [this](Float x) { return std::abs(x) < eps_; });
                            if (is_input_silent) {
                                temp_silent_len_[track][ch] += fill_size;
                            }
                            else {
                                temp_silent_len_[track][ch] = 0;
                            }
                        }
                        else {
                            if (temp_silent_len_[track][ch] < analysis_size_) {
                                std::memset(input_buffer_[track][ch].data() + buffer_pos_, 0, sizeof(Float) * fill_size);
                            }
                            temp_silent_len_[track][ch] += fill_size;
                        }
                    }
                    std::memcpy(output[ch] + base_frame, output_buffer_[ch].data() + buffer_pos_, sizeof(Float) * fill_size);
                }
                base_frame += fill_size;
                buffer_pos_ += fill_size;
                if (buffer_pos_ != analysis_size_) break;
                // input_bufferがたまり、output_bufferの削除可能領域ができたので処理
                
                for (int ch = 0; ch < config_.num_channels; ch++) {
                    // 帯域分割
                    for (int track = 0; track < config_.num_tracks; track++) {
                        if (temp_silent_len_[track][ch] >= memory_samples()) continue;
                        
                        for (int band = 0; band < bands_.size(); band++) {
                            splitted_ptrs_[band] = splitted_[track][band].data();
                        }
                        filter_banks_[track][ch].AnalysisClock(input_buffer_[track][ch].data(), input_buffer_[track][ch].data() + analysis_size_, splitted_ptrs_.data());
                    }
                    for (int band = 0; band < num_bands(); band++) {
                        const int samples_per_gain = config_.gain_decimation / filter_bank_config_.bands[band].decimation;
                        
                        // 各トラックに対して
                        // 2乗(vector化)してローパスしてscaleを計算 (scaleを保存する領域が必要 [track][frame])
                        // 各frameに対してエネルギー合計と(vector)、scale maxを計算(vector) (エネルギー合計とscale maxの領域が必要 [frame])
                        // 各トラックに対して、scale補正値を計算 (pow(s / m, ratio - 1) & normalize)
                        
                        const int gain_decimated_analysis_size = analysis_size_ / config_.gain_decimation;
                        
#if 1
                        // エネルギー計算
                        for (int track = 0; track < config_.num_tracks; track++) {
                            if (temp_silent_len_[track][ch] >= memory_samples()) continue;
                            
                            auto original_scale_vec = temp_original_scale2_[track].data();
                            auto &energy_lowpass_filter = energy_lowpass_fir_filters_[track][ch][band];
                            
                            // 2乗してローパスしてsqrt
                            const auto &sp = splitted_[track][band];
                            VectorNorm(sp.data(), temp_general_.data(), samples_per_gain * gain_decimated_analysis_size);
                            
#ifdef BAKUAGE_CLEAR_MIXER_FILTER4_USE_FIR_FILTER4
                            // decimationmも兼ねる
                            energy_lowpass_filter.Clock(temp_general_.data(), temp_general_.data() + samples_per_gain * gain_decimated_analysis_size, original_scale_vec);
#else
                            energy_lowpass_filter.Clock(temp_general_.data(), temp_general_.data() + samples_per_gain * gain_decimated_analysis_size, temp_general_.data());
                            VectorDecimate(temp_general_.data(), samples_per_gain * gain_decimated_analysis_size, original_scale_vec, samples_per_gain);
#endif
                            
                            VectorEnsureNonnegativeInplace(original_scale_vec, gain_decimated_analysis_size);
                            VectorSqrtInplace(original_scale_vec, gain_decimated_analysis_size);
                        }
#endif
                        
#if 1
                        // L2とmax scaleを計算
                        if (config_.noise_reduction == kNoiseReductionFlat) {
                            VectorSet<Float>(config_.noise_reduction_threshold, temp_original_l2_.data(), gain_decimated_analysis_size);
                            VectorSet<Float>(std::sqrt(config_.noise_reduction_threshold), temp_inv_max_scale_.data(), gain_decimated_analysis_size);
                        } else if (config_.noise_reduction == kNoiseReductionFixedSpectrum) {
                            VectorSet(config_.noise_reduction_fixed_spectrum_profile.energy_thresholds[ch][band], temp_original_l2_.data(), gain_decimated_analysis_size);
                            VectorSet(std::sqrt(config_.noise_reduction_fixed_spectrum_profile.energy_thresholds[ch][band]), temp_inv_max_scale_.data(), gain_decimated_analysis_size);
                        } else {
                            TypedFillZero(temp_original_l2_.data(), gain_decimated_analysis_size);
                            TypedFillZero(temp_inv_max_scale_.data(), gain_decimated_analysis_size);
                        }
                        VectorSet(-1, temp_mi_.data(), gain_decimated_analysis_size);
                        
                        for (int track = 0; track < config_.num_tracks; track++) {
                            if (temp_silent_len_[track][ch] >= memory_samples()) continue;
                            
                            const auto original_scale_vec = temp_original_scale2_[track].data();
                            
                            VectorMadInplace(original_scale_vec, original_scale_vec, temp_original_l2_.data(), gain_decimated_analysis_size);
                            
                            // うまくvector化できなかったので普通に計算
                            for (int gain_frame = 0; gain_frame < gain_decimated_analysis_size; gain_frame++) {
                                if (temp_inv_max_scale_[gain_frame] < original_scale_vec[gain_frame]) {
                                    temp_inv_max_scale_[gain_frame] = original_scale_vec[gain_frame];
                                    temp_mi_[gain_frame] = track;
                                }
                            }
                        }
                        VectorAddConstantInplace<Float>(1e-37, temp_inv_max_scale_.data(), gain_decimated_analysis_size);
                        VectorInvInplace(temp_inv_max_scale_.data(), gain_decimated_analysis_size);
                        
                        // ノイズ分析
                        if (config_.noise_reduction == kNoiseReductionFixedSpectrumLearn) {
                            for (int gain_frame = 0; gain_frame < gain_decimated_analysis_size; gain_frame++) {
                                const auto l2 = temp_original_l2_[gain_frame];
                                if (l2 > 0) { // ignore silence
                                    noise_reduction_fixed_spectrum_learn_[ch][band].push_back(l2);
                                }
                            }
                        }
        
                        // スパース化
                        if (config_.noise_reduction == kNoiseReductionFlat) {
                            const double scale = std::sqrt(config_.noise_reduction_threshold);
                            VectorMulConstant<Float>(temp_inv_max_scale_.data(), scale, temp_general_.data(), gain_decimated_analysis_size);
                            VectorPowConstant<Float>(temp_general_.data(), ratio_ - 1, temp_l2_.data(), gain_decimated_analysis_size);
                            VectorMulConstantInplace<Float>(scale, temp_l2_.data(), gain_decimated_analysis_size);
                        } else if (config_.noise_reduction == kNoiseReductionFixedSpectrum) {
                            const double scale = std::sqrt(config_.noise_reduction_fixed_spectrum_profile.energy_thresholds[ch][band]);
                            VectorMulConstant<Float>(temp_inv_max_scale_.data(), scale, temp_general_.data(), gain_decimated_analysis_size);
                            VectorPowConstant<Float>(temp_general_.data(), ratio_ - 1, temp_l2_.data(), gain_decimated_analysis_size);
                            VectorMulConstantInplace<Float>(scale, temp_l2_.data(), gain_decimated_analysis_size);
                        } else {
                            TypedFillZero(temp_l2_.data(), gain_decimated_analysis_size);
                        }
                        for (int track = 0; track < config_.num_tracks; track++) {
                            if (temp_silent_len_[track][ch] >= memory_samples()) continue;
                            // calc gain
                            VectorMul(temp_original_scale2_[track].data(), temp_inv_max_scale_.data(), temp_general_.data(), gain_decimated_analysis_size);
                            VectorPowConstant(temp_general_.data(), ratio_ - 1, temp_gain2_[track].data(), gain_decimated_analysis_size);
                            // L2 after
                            VectorMul(temp_original_scale2_[track].data(), temp_gain2_[track].data(), temp_scale2_[track].data(), gain_decimated_analysis_size);
                            VectorMadInplace(temp_scale2_[track].data(), temp_scale2_[track].data(), temp_l2_.data(), gain_decimated_analysis_size);
                        }
                        
                        // store normalize scale to temp_original_l2_
                        if (wet_scale_ != 0) {
                            VectorAddConstantInplace<Float>(1e-37, temp_l2_.data(), gain_decimated_analysis_size);
                            VectorDivInplace(temp_l2_.data(), temp_original_l2_.data(), gain_decimated_analysis_size);
                            VectorSqrtInplace(temp_original_l2_.data(), gain_decimated_analysis_size);
                        }
#endif
                        
#if 1
                        // temp_original_l2_に予め必要な係数をかける
                        // 2 * 2は正の周波数だけにしている分の補正と、よくわからない補正。なぜかこれでゲインが合う
                        if (wet_scale_ != 0) {
                            VectorMulConstantInplace(2 * 2 * wet_scale_, temp_original_l2_.data(), gain_decimated_analysis_size);
                        }
                        
                        // うまくvector化できなかったので普通に計算
                        if (primary_scale_ != 1) {
                            for (int gain_frame = 0; gain_frame < gain_decimated_analysis_size; gain_frame++) {
                                if (temp_mi_[gain_frame] >= 0) {
                                    temp_gain2_[temp_mi_[gain_frame]][gain_frame] *= primary_scale_;
                                }
                            }
                        }
                        
                        // 帯域ごとにゲインを掛け算する (帯域によってはサンプリングレートがさらに細かいのでゲインにもアップサンプリング用のローパス)
                        for (int track = 0; track < config_.num_tracks; track++) {
                            if (temp_silent_len_[track][ch] >= memory_samples()) continue;
                            
                            auto &delay = splitted_delay_filters_[track][ch][band];
                            auto &sp = splitted_[track][band];
                            auto gain_vec = temp_gain2_[track].data();
                            
                            if (wet_scale_ != 0) {
                                VectorMulInplace(temp_original_l2_.data(), gain_vec, gain_decimated_analysis_size);
                            } else {
                                TypedFillZero(gain_vec, gain_decimated_analysis_size);
                            }
#ifndef BA_CLEAR_MIXER_FILTER3_TRUE_DRY_ENABLED
                            // 2 * 2は正の周波数だけにしている分の補正と、よくわからない補正。なぜかこれでゲインが合う
                            if (dry_scale_ != 0) {
                                VectorAddConstantInplace(2 * 2 * dry_scale_, gain_vec, gain_decimated_analysis_size);
                            }
#endif
                            
                            // lowpass gain
#ifdef BAKUAGE_CLEAR_MIXER_FILTER4_USE_FIR_FILTER4
                            // interpolateも兼ねる
                            scale_lowpass_fir_filters_[track][ch][band].Clock(gain_vec, gain_vec + gain_decimated_analysis_size, gain_vec);
#else
                            VectorInterpolate(gain_vec, gain_decimated_analysis_size, temp_general_.data(), samples_per_gain);
                            scale_lowpass_fir_filters_[track][ch][band].Clock(temp_general_.data(), temp_general_.data() + samples_per_gain * gain_decimated_analysis_size, gain_vec);
#endif
                            
                            delay.Clock(sp.data(), sp.data() + samples_per_gain * gain_decimated_analysis_size, sp.data());
                            VectorMulInplace(gain_vec, sp.data(), samples_per_gain * gain_decimated_analysis_size);
                        }
#endif
                        
                        // 解析用の値をセット
                        temp_primary_track_[ch][band] = temp_mi_[gain_decimated_analysis_size - 1];
                        for (int track = 0; track < config_.num_tracks; track++) {
                            temp_original_scale_[ch][band][track] = temp_original_scale2_[track][gain_decimated_analysis_size - 1];
                            temp_scale_[ch][band][track] = temp_scale2_[track][gain_decimated_analysis_size - 1];
                        }
                    }
                    
                    // 帯域合成 + 出力バッファに出力
                    TypedFillZero(output_buffer_[ch].data(), analysis_size_);
                    for (int track = 0; track < config_.num_tracks; track++) {
                        if (temp_silent_len_[track][ch] >= memory_samples()) continue;
                        if (track_scales_[track] == 0) continue;
                        
                        for (int band = 0; band < bands_.size(); band++) {
                            splitted_const_ptrs_[band] = splitted_[track][band].data();
                        }
                        filter_banks_[track][ch].SynthesisClock(splitted_const_ptrs_.data(), analysis_size_, temp_synthesized_.data());
                        
#if 1
                        VectorMadConstantInplace(temp_synthesized_.data(), track_scales_[track], output_buffer_[ch].data(), analysis_size_);
#else
                        for (int frame = 0; frame < analysis_size_; frame++) {
                            output_buffer_[ch][frame] += track_scales_[track] * (temp_synthesized_[frame]
#ifdef BA_CLEAR_MIXER_FILTER3_TRUE_DRY_ENABLED
                                                                                 + dry_scale_ * dry_delay_filters_[track][ch].Clock(input_buffer_[track][ch][frame])
#endif
                                                                                 );
                        }
#endif
                    }
                }
                
                AnalysisResult analysis_result = { 0 };
                analysis_result.pre_scales = temp_original_scale_.data();
                analysis_result.post_scales = temp_scale_.data();
                analysis_result.primary_tracks = temp_primary_track_.data();
                analysis_result.frames = analysis_size_;
                analysis_callback(analysis_result);
                
                buffer_pos_ = 0;
            }
        };
        
        void set_ratio(Float value) { ratio_ = value; }
        Float ratio() const { return ratio_; }
        void set_wet_scale(Float value) { wet_scale_ = value; }
        Float wet_scale() const { return wet_scale_; }
        void set_dry_scale(Float value) { dry_scale_ = value; }
        Float dry_scale() const { return dry_scale_; }
        void set_primary_scale(Float value) { primary_scale_ = value; }
        Float primary_scale() const { return primary_scale_; }
        void set_track_scale(int i, Float value) { track_scales_[i] = value; }
        Float track_scale(int i) const { return  track_scales_[i]; }
        void set_eps(Float value) { eps_ = value; }
        Float eps() const { return eps; } // 絶対値がこれより小さい入力値はゼロとみなされる
        int delay_samples() const { return analysis_size_ + min_energy_delay_samples_ + min_scale_delay_samples_ + 2 * (config_.fir_samples / 2); }
        int num_bands() const { return bands_.size(); }
        Float band_freqs(int i) const {
            return i < bands_.size() - 1 ? bands_[i].low_freq : bands_[bands_.size() - 1].high_freq;
        }
    private:
        // 記憶される長さ。これ以上無音が続いたらdisableにできる
        // delay_samplesとは異なるので注意
        // いろいろな変数をゼロにフラッシュするために、analysis_size_は必ず含める必要がある
        // また、処理前に無音長計算を行うので、analysis_size_をもう一個分含める必要がある
        // TimeVaryingを使う場合は成立しないので注意
        int memory_samples() const {
            return 2 * analysis_size_ + 2 * min_energy_delay_samples_ + 1 + 2 * min_scale_delay_samples_ + 1 + 2 * config_.fir_samples;
        }
        
        class Band {
        public:
            Float low_freq;
            Float high_freq;
        };
        
        std::vector<Band> CreateBandsByErb(int sample_rate, Float erb_scale) {
            std::vector<Band> bands;
            
            Float prev_freq = 0;
            while (1) {
                Float next_freq = prev_freq + erb_scale * bakuage::GlasbergErb(prev_freq);
                
                // 最後が短いときはスキップ
                if (next_freq >= sample_rate / 2) {
                    if ((sample_rate / 2 - prev_freq) / (next_freq - prev_freq) < 0.5) {
                        break;
                    }
                }
                
                Band band = { 0 };
                band.low_freq = prev_freq;
                band.high_freq = next_freq;
                bands.push_back(band);
                
                if (next_freq >= sample_rate / 2) {
                    break;
                }
                prev_freq = next_freq;
            }
            
            bands.back().high_freq = sample_rate / 2;
            
            return bands;
        }
        
        Config config_;
        Float wet_scale_;
        Float dry_scale_;
        Float ratio_;
        Float primary_scale_;
        Float eps_;
        AlignedPodVector<Float> track_scales_;
        std::vector<Band> bands_;
        std::vector<std::vector<AlignedPodVector<Float>>> temp_scale_; // [ch][band][track]
        std::vector<std::vector<AlignedPodVector<Float>>> temp_original_scale_; // [ch][band][track]
        std::vector<AlignedPodVector<int>> temp_primary_track_; // [ch][band]
        std::vector<AlignedPodVector<uint64_t>> temp_silent_len_; // [track][ch] 無音最適化用、クロックをまたいで記憶される
        
        const int analysis_size_;
        std::vector<std::vector<AlignedPodVector<Float>>> input_buffer_; // [track][ch][frame] analysis_size_
        std::vector<AlignedPodVector<Float>> output_buffer_; // [ch] analysis_size_
        AlignedPodVector<Float> temp_synthesized_;
        int buffer_pos_;
        
        std::vector<std::vector<AlignedPodVector<std::complex<Float>>>> splitted_; // [track][band][frame] analysis_size_
        std::vector<std::complex<Float> *> splitted_ptrs_; // [band][frame]
        std::vector<const std::complex<Float> *> splitted_const_ptrs_; // [band][frame]
        
        std::vector<std::vector<FilterBank>> filter_banks_; // [track][ch]
        std::vector<std::vector<std::vector<DelayFilter2<std::complex<Float>>>>> splitted_delay_filters_; // [track][ch][band]
        std::vector<std::vector<DelayFilter2<Float>>> dry_delay_filters_; // [track][ch]
        
        // kFilterFir用
#ifdef BAKUAGE_CLEAR_MIXER_FILTER4_USE_FIR_FILTER4
        std::vector<std::vector<std::vector<FirFilter4<Float>>>> energy_lowpass_fir_filters_; // [track][ch][band]
        std::vector<std::vector<std::vector<FirFilter4<Float>>>> scale_lowpass_fir_filters_; // [track][ch][band]
#else
        std::vector<std::vector<std::vector<FirFilter2<Float>>>> energy_lowpass_fir_filters_; // [track][ch][band]
        std::vector<std::vector<std::vector<FirFilter2<Float>>>> scale_lowpass_fir_filters_; // [track][ch][band]
#endif
        
        // noise reduction learn
        std::vector<std::vector<bakuage::AlignedPodVector<Float>>> noise_reduction_fixed_spectrum_learn_; // [ch][band][index]
        
        typename FilterBank::Config filter_bank_config_;
        int min_energy_delay_samples_;
        int min_scale_delay_samples_;
        
        // 高速化のための領域
        std::vector<AlignedPodVector<Float>> temp_original_scale2_; // [track][frame]
        std::vector<AlignedPodVector<Float>> temp_scale2_; // [track][frame]
        std::vector<AlignedPodVector<Float>> temp_gain2_; // [track][frame] (temp_scale2_ / temp_original_scale2_)
        AlignedPodVector<Float> temp_original_l2_; // [frame]
        AlignedPodVector<Float> temp_l2_; // [frame]
        AlignedPodVector<Float> temp_inv_max_scale_; // [frame]
        AlignedPodVector<Float> temp_general_; // [frame] (汎用的に使えるテンポラリー)
        AlignedPodVector<int> temp_mi_; // [frame]
    };
}

#endif


