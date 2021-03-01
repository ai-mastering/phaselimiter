#ifndef BAKUAGE_BAKUAGE_CLEAR_MIXER_FILTER3_H_
#define BAKUAGE_BAKUAGE_CLEAR_MIXER_FILTER3_H_

#include <algorithm>
#include <limits>
#include <numeric>
#include <vector>
#include "bakuage/dft.h"
#include "bakuage/memory.h"
#include "bakuage/utils.h"
#include "bakuage/window_func.h"
#include "bakuage/fir_design.h"
#include "bakuage/fir_filter3.h"
#include "bakuage/fir_filter_bank.h"
#include "bakuage/time_varying_lowpass_filter.h"

// ClearMixer3では、FirFilterBankを通すだけでも音が変化するので、実はドライがドライにならない
// 本当のドライを使うモード。でも、本当のドライを使うと今度は、ratio 1のときにWet - Dryが無音にならないので、
// 混乱する。使ってみてわかりづらかったので、本当のドライを使わないことにした
// #define BA_CLEAR_MIXER_FILTER3_TRUE_DRY_ENABLED

namespace bakuage {
    // Delphiから移植
    // FirFilterbank方式
    template <
    typename Float
    >
    class ClearMixerFilter3 {
    public:
        typedef FirFilterBank<Float> FilterBank;
        
        enum {
            kFilterTimeVarying = 1,
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
        
        ClearMixerFilter3(const Config &config):
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
        splitted_const_ptrs_(bands_.size())
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
            
			if (config.filter == kFilterTimeVarying) {
				constexpr int lowpass_filter_order = 2;
				energy_lowpass_filters_ = std::vector<std::vector<std::vector<TimeVaryingLowpassFilter<Float>>>>(config_.num_tracks, std::vector<std::vector<TimeVaryingLowpassFilter<Float>>>(config_.num_channels, std::vector<TimeVaryingLowpassFilter<Float>>(bands_.size(), TimeVaryingLowpassFilter<Float>(lowpass_filter_order, 1))));
				scale_lowpass_filters_ = std::vector<std::vector<std::vector<TimeVaryingLowpassFilter<Float>>>>(config_.num_tracks, std::vector<std::vector<TimeVaryingLowpassFilter<Float>>>(config_.num_channels, std::vector<TimeVaryingLowpassFilter<Float>>(bands_.size(), TimeVaryingLowpassFilter<Float>(lowpass_filter_order, 1))));

				// 全体を統一する必要があるので
				min_energy_delay_samples_ = 1 << 30;
				min_scale_delay_samples_ = 1 << 30;
				for (int i = 0; i < bands_.size(); i++) {
					const auto &band = filter_bank_config.bands[i];

					const Float energy_peak = std::min<Float>(1.0, 1.0 / (1.0 * config_.sample_rate / band.decimation * config_.energy_mean_sec + 1e-37));
					int energy_delay_samples = 0;
					Float energy_a = TimeVaryingLowpassFilter<Float>::CalculateAFromPeak(lowpass_filter_order, energy_peak, &energy_delay_samples);

					const Float scale_peak = std::min<Float>(1.0, 1.0 / (1.0 * config_.sample_rate / band.decimation * config_.scale_mean_sec + 1e-37));
					int scale_delay_samples = 0;
					Float scale_a = TimeVaryingLowpassFilter<Float>::CalculateAFromPeak(lowpass_filter_order, scale_peak, &scale_delay_samples);

					for (int track = 0; track < config.num_tracks; track++) {
						for (int ch = 0; ch < config.num_channels; ch++) {
							energy_lowpass_filters_[track][ch][i].set_a(energy_a);
							scale_lowpass_filters_[track][ch][i].set_a(scale_a);
						}
					}

					min_energy_delay_samples_ = std::min<int>(min_energy_delay_samples_, energy_delay_samples * band.decimation);
					min_scale_delay_samples_ = std::min<int>(min_scale_delay_samples_, scale_delay_samples * band.decimation);
				}
			}
			else {
				min_energy_delay_samples_ = CeilInt<int>(config.sample_rate * config.energy_mean_sec / 2, config.gain_decimation);
				min_scale_delay_samples_ = CeilInt<int>(config.sample_rate * config.scale_mean_sec / 2, config.gain_decimation);
				
				std::vector<FirFilter3<Float>> energy_filters;
				std::vector<FirFilter3<Float>> scale_filters;
				for (int i = 0; i < bands_.size(); i++) {
					const auto &band = filter_bank_config.bands[i];

					const int energy_n = 2 * min_energy_delay_samples_ / band.decimation + 1;
					std::vector<Float> energy_fir(energy_n);
					CopyHanning(energy_n, energy_fir.begin(), 1.0 / energy_n);
					energy_filters.emplace_back(energy_fir.begin(), energy_fir.end());

					const int scale_n = 2 * min_scale_delay_samples_ / band.decimation + 1;
					std::vector<Float> scale_fir(scale_n);
					CopyHanning(scale_n, scale_fir.begin(), 1.0 / scale_n);
					scale_filters.emplace_back(scale_fir.begin(), scale_fir.end());
				}
				energy_lowpass_fir_filters_ = Create2DVector(config_.num_tracks, config_.num_channels, energy_filters);
				scale_lowpass_fir_filters_ = Create2DVector(config_.num_tracks, config_.num_channels, scale_filters);
			}
            
            splitted_delay_filters_ = Create3DVector(config_.num_tracks, config_.num_channels, bands_.size(), DelayFilter<std::complex<Float>>(1));
            for (int i = 0; i < bands_.size(); i++) {
#if 0
				min_energy_delay_samples_ = min_scale_delay_samples_ = 0;
				DelayFilter<std::complex<Float>> delay_filter(0);
#else
				DelayFilter<std::complex<Float>> delay_filter((min_energy_delay_samples_ + min_scale_delay_samples_) / filter_bank_config.bands[i].decimation);
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
                        
                        for (int gain_frame = 0; gain_frame < analysis_size_ / config_.gain_decimation; gain_frame++) {
                            // 2乗してローパス (帯域によってはさらに細かいサンプリングレートで)
                            for (int track = 0; track < config_.num_tracks; track++) {
								if (temp_silent_len_[track][ch] >= memory_samples()) continue;

                                const auto &sp = splitted_[track][band];
#if 0
								const auto x = bakuage::Sqr(sp[gain_frame * samples_per_gain].real());
#else
								const auto x = std::norm(sp[gain_frame * samples_per_gain]);
#endif
                                const Float scale = config_.filter == kFilterTimeVarying ?
									std::sqrt(energy_lowpass_filters_[track][ch][band].Clock(x)) :
									std::sqrt(std::max<Float>(0, energy_lowpass_fir_filters_[track][ch][band].Clock(x)));
                                temp_original_scale_[ch][band][track] = scale;
                                temp_scale_[ch][band][track] = scale;
                                
                                const int ed = (gain_frame + 1) * samples_per_gain;
                                for (int sample_frame = gain_frame * samples_per_gain + 1; sample_frame < ed; sample_frame++) {
#if 0
									const auto x = bakuage::Sqr(sp[sample_frame].real());
#else
									const auto x = std::norm(sp[sample_frame]);
#endif
                                    if (config_.filter == kFilterTimeVarying) {
                                        energy_lowpass_filters_[track][ch][band].Clock(x);
                                    } else {
										energy_lowpass_fir_filters_[track][ch][band].ClockWithoutResult(x);
                                    }
                                }
                            }
                            
#if 1
                            // スパース化をして帯域ごとのゲイン計算 (今の所expander方式のみ)
                            double l2 = 0;
                            double ma = 0;
                            int mi = -1;
                            if (config_.noise_reduction == kNoiseReductionFlat) {
                                l2 = config_.noise_reduction_threshold;
                                ma = std::sqrt(config_.noise_reduction_threshold);
                            } else if (config_.noise_reduction == kNoiseReductionFixedSpectrum) {
                                l2 = config_.noise_reduction_fixed_spectrum_profile.energy_thresholds[ch][band];
                                ma = std::sqrt(config_.noise_reduction_fixed_spectrum_profile.energy_thresholds[ch][band]);
                            }
                            for (int track = 0; track < config_.num_tracks; track++) {
								if (temp_silent_len_[track][ch] >= memory_samples()) continue;

                                l2 += bakuage::Sqr(temp_scale_[ch][band][track]);
								//const Float priority = track == 0 ? 2 : 1;
                                if (ma < temp_scale_[ch][band][track]/* * priority*/) {
                                    ma = temp_scale_[ch][band][track]/* * priority*/;
                                    mi = track;
                                }
                            }
                            const double inv_ma = 1.0 / (1e-37 + ma);
                            if (config_.noise_reduction == kNoiseReductionFixedSpectrumLearn) {
                                if (l2 > 0) { // ignore silence
                                    noise_reduction_fixed_spectrum_learn_[ch][band].push_back(l2);
                                }
                            }
                            temp_primary_track_[ch][band] = mi;
                            double l2_2 = 0;
                            if (config_.noise_reduction == kNoiseReductionFlat) {
                                const double scale = std::sqrt(config_.noise_reduction_threshold);
                                l2_2 = bakuage::Sqr(scale * std::pow(scale * inv_ma, ratio_ - 1));
                            } else if (config_.noise_reduction == kNoiseReductionFixedSpectrum) {
                                const double scale = std::sqrt(config_.noise_reduction_fixed_spectrum_profile.energy_thresholds[ch][band]);
                                l2_2 = bakuage::Sqr(scale * std::pow(scale * inv_ma, ratio_ - 1));
                            }
                            for (int track = 0; track < config_.num_tracks; track++) {
								if (temp_silent_len_[track][ch] >= memory_samples()) continue;

#if 0
								const double orig_relative = temp_scale_[ch][band][track] * inv_ma;
								double target_relative = std::pow(orig_relative, ratio_);
								target_relative = std::max<double>(0.1, target_relative);
								temp_scale_[ch][band][track] = target_relative * ma;
#else
                                temp_scale_[ch][band][track] *= std::pow(temp_scale_[ch][band][track] * inv_ma, ratio_ - 1);
#endif
                                l2_2 += bakuage::Sqr(temp_scale_[ch][band][track]);
                            }
                            double normalize_scale = std::sqrt(l2 / (1e-37 + l2_2));
                            for (int track = 0; track < config_.num_tracks; track++) {
                                temp_scale_[ch][band][track] *= normalize_scale;
                            }
#endif
                            
#if 1
                            // 帯域ごとにゲインを掛け算する (帯域によってはサンプリングレートがさらに細かいのでゲインにもアップサンプリング用のローパス)
                            for (int track = 0; track < config_.num_tracks; track++) {
								if (temp_silent_len_[track][ch] >= memory_samples()) continue;

                                auto &delay = splitted_delay_filters_[track][ch][band];
                                auto &sp = splitted_[track][band];
                                
                                // 瞬間値
                                double s = wet_scale_ * (temp_scale_[ch][band][track] / (1e-37 + temp_original_scale_[ch][band][track]));
#ifndef BA_CLEAR_MIXER_FILTER3_TRUE_DRY_ENABLED
								s += dry_scale_;
#endif
                                if (track == temp_primary_track_[ch][band]) {
                                    s *= primary_scale_;
                                }

								// 正の周波数だけにしている分の補正と、よくわからない補正。なぜかこれでゲインが合う
								s *= 2 * 2;
                                
                                // lowpass
								const Float filtered = config_.filter == kFilterTimeVarying ?
									scale_lowpass_filters_[track][ch][band].Clock(s) :
									scale_lowpass_fir_filters_[track][ch][band].Clock(s);
								sp[gain_frame * samples_per_gain] = delay.Clock(sp[gain_frame * samples_per_gain]) * filtered;

                                // upsample
                                const int ed = (gain_frame + 1) * samples_per_gain;
                                for (int sample_frame = gain_frame * samples_per_gain + 1; sample_frame < ed; sample_frame++) {
									const Float filtered = config_.filter == kFilterTimeVarying ?
										scale_lowpass_filters_[track][ch][band].Clock(s) :
										scale_lowpass_fir_filters_[track][ch][band].Clock(s);
									sp[sample_frame] = delay.Clock(sp[sample_frame]) * filtered;
                                }
                            }
#endif
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

						for (int frame = 0; frame < analysis_size_; frame++) {
							output_buffer_[ch][frame] += track_scales_[track] * (temp_synthesized_[frame]
#ifdef BA_CLEAR_MIXER_FILTER3_TRUE_DRY_ENABLED
								+ dry_scale_ * dry_delay_filters_[track][ch].Clock(input_buffer_[track][ch][frame])
#endif
								);
                        }
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
		std::vector<std::vector<std::vector<DelayFilter<std::complex<Float>>>>> splitted_delay_filters_; // [track][ch][band]
		std::vector<std::vector<DelayFilter<Float>>> dry_delay_filters_; // [track][ch]

		// kFilterTimeVarying用
        std::vector<std::vector<std::vector<TimeVaryingLowpassFilter<Float>>>> energy_lowpass_filters_; // [track][ch][band]
        std::vector<std::vector<std::vector<TimeVaryingLowpassFilter<Float>>>> scale_lowpass_filters_; // [track][ch][band]

		// kFilterFir用
		std::vector<std::vector<std::vector<FirFilter3<Float>>>> energy_lowpass_fir_filters_; // [track][ch][band]
		std::vector<std::vector<std::vector<FirFilter3<Float>>>> scale_lowpass_fir_filters_; // [track][ch][band]
        
        // noise reduction learn
        std::vector<std::vector<bakuage::AlignedPodVector<Float>>> noise_reduction_fixed_spectrum_learn_; // [ch][band][index]
        
        typename FilterBank::Config filter_bank_config_;
        int min_energy_delay_samples_;
        int min_scale_delay_samples_;
    };
}

#endif


