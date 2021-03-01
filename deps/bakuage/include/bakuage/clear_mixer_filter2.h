#ifndef BAKUAGE_BAKUAGE_CLEAR_MIXER_FILTER2_H_
#define BAKUAGE_BAKUAGE_CLEAR_MIXER_FILTER2_H_

#include <algorithm>
#include <limits>
#include <numeric>
#include <vector>
#include "bakuage/dft.h"
#include "bakuage/memory.h"
#include "bakuage/utils.h"
#include "bakuage/window_func.h"

namespace bakuage {

#define BA_CLEAR_MIXER_FILTER2_SAME_WINDOW_ENABLED
    
    // Delphiから移植
    // overlap add方式
    template <
    typename Float
    >
    class ClearMixerFilter2 {
    public:
        enum {
            kAlgorithmSparse = 1,
            kAlgorithmExpander = 2,
        };
		enum {
			kWindowHanning = 1,
			kWindowBlackmanHarris = 2,
		};
        
        class Config {
        public:
            Config(): num_tracks(0), num_channels(0), sample_rate(0), window_samples(0), overlap(0), algorithm(0), output_window(0) {}
            int num_tracks;
            int num_channels;
            int sample_rate;
            int window_samples; // analysis window
            int overlap; // analysis window / output window
            int algorithm;
			int output_window;
		};

		struct AnalysisResult {
			std::vector<AlignedPodVector<Float>> *pre_scales; // [ch][band][track] sqrt(energy)
			std::vector<AlignedPodVector<Float>> *post_scales; // [ch][band][track] sqrt(energy)
			AlignedPodVector<int> *primary_tracks; // [ch][band]
		};
        
        ClearMixerFilter2(const Config &config):
        config_(config),
        wet_scale_(1),
		dry_scale_(0),
		ratio_(1),
		primary_scale_(1),
		eps_(0),
		track_scales_(config.num_tracks, 1.0),
        bands_(CreateBandsByErb(config.sample_rate, 1)),
		temp_scale_(config.num_channels, std::vector<AlignedPodVector<Float>>(bands_.size(), AlignedPodVector<Float>(config.num_tracks))),
		temp_original_scale_(config.num_channels, std::vector<AlignedPodVector<Float>>(bands_.size(), AlignedPodVector<Float>(config.num_tracks))),
		temp_primary_track_(config.num_channels, AlignedPodVector<int>(bands_.size())),
		temp_z_(config.num_tracks),
		temp_is_silent_(config.num_tracks, AlignedPodVector<bool>(config.num_channels)),
        analysis_size_(config.window_samples),
        shift_size_(config.window_samples / config.overlap),
        analysis_spec_size_(analysis_size_ / 2 + 1),
#ifndef BA_CLEAR_MIXER_FILTER2_SAME_WINDOW_ENABLED
        analysis_spec_(config.num_tracks, AlignedPodVector<std::complex<Float>>(analysis_spec_size_)),
#endif
        output_spec_(config.num_tracks, AlignedPodVector<std::complex<Float>>(analysis_spec_size_)),
        input_buffer_(config.num_tracks, std::vector<AlignedPodVector<Float>>(config.num_channels, AlignedPodVector<Float>(analysis_size_))),
        output_buffer_(config.num_channels, AlignedPodVector<Float>(analysis_size_)),
        analysis_window_(analysis_size_),
        output_window_(analysis_size_),
        temp_windowed_(analysis_size_),
        buffer_pos_(analysis_size_ - shift_size_),
        dft_(analysis_size_)
        {
#ifndef BA_CLEAR_MIXER_FILTER2_SAME_WINDOW_ENABLED
            bakuage::CopyHanning(analysis_size_, analysis_window_.data(), 1.0 / std::sqrt(analysis_size_));
#else
			analysis_to_output_energy_ratio_ = bakuage::Sqr((1.0 / std::sqrt(analysis_size_)) / (1.0 / (0.5 * config.overlap) / analysis_size_));
#endif
			if (config.output_window == kWindowBlackmanHarris) {
				bakuage::CopyBlackmanHarris(analysis_size_, output_window_.data(), 1.0 / (0.35875 * config.overlap) / analysis_size_);
			}
			else {
				bakuage::CopyHanning(analysis_size_, output_window_.data(), 1.0 / (0.5 * config.overlap) / analysis_size_); // hanning窓の直流成分は0.5なので
			}
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
                
                // 入力 + 出力
                for (int ch = 0; ch < config_.num_channels; ch++) {
                    for (int track = 0; track < config_.num_tracks; track++) {
						if (input[track][ch]) {
							temp_is_silent_[track][ch] = false;
							std::memcpy(input_buffer_[track][ch].data() + buffer_pos_, input[track][ch] + base_frame, sizeof(Float) * fill_size);
						}
						else {
							if (!temp_is_silent_[track][ch]) {
								std::memset(input_buffer_[track][ch].data() + buffer_pos_, 0, sizeof(Float) * fill_size);
							}
						}
                    }
                    std::memcpy(output[ch] + base_frame, output_buffer_[ch].data() + buffer_pos_ - (analysis_size_ - shift_size_), sizeof(Float) * fill_size);
                }
                base_frame += fill_size;
                buffer_pos_ += fill_size;
                if (buffer_pos_ != analysis_size_) break;
                // input_bufferがたまり、output_bufferの削除可能領域ができたので処理
                
                for (int ch = 0; ch < config_.num_channels; ch++) {
					// shift and clear output buffer
					std::memmove(output_buffer_[ch].data(), output_buffer_[ch].data() + shift_size_, sizeof(Float) * (analysis_size_ - shift_size_));
					std::memset(output_buffer_[ch].data() + (analysis_size_ - shift_size_), 0, sizeof(Float) * shift_size_);

                    // FFT
                    for (int track = 0; track < config_.num_tracks; track++) {
						// 無音の場合は高速に処理
						temp_is_silent_[track][ch] = temp_is_silent_[track][ch] || std::all_of(
							input_buffer_[track][ch].data(), 
							input_buffer_[track][ch].data() + analysis_size_,
							[this](Float x) { return std::abs(x) < eps_; }
						);
						if (temp_is_silent_[track][ch]) {
#if 0
#ifndef BA_CLEAR_MIXER_FILTER2_SAME_WINDOW_ENABLED
							std::memset(analysis_spec_[track].data(), 0, sizeof(std::complex<Float>) * analysis_spec_size_);
#endif
							std::memset(output_spec_[track].data(), 0, sizeof(std::complex<Float>) * analysis_spec_size_);
#endif
						}
						else {
#ifndef BA_CLEAR_MIXER_FILTER2_SAME_WINDOW_ENABLED
							for (int frame = 0; frame < analysis_size_; frame++) {
								temp_windowed_[frame] = analysis_window_[frame] * input_buffer_[track][ch][frame];
							}
							dft_.Forward(temp_windowed_.data(), (Float *)analysis_spec_[track].data());
#endif
							for (int frame = 0; frame < analysis_size_; frame++) {
								temp_windowed_[frame] = output_window_[frame] * input_buffer_[track][ch][frame];
							}
							dft_.Forward(temp_windowed_.data(), (Float *)output_spec_[track].data());
						}
                    }
                    
                    for (int band = 0; band < num_bands(); band++) {
                        const int bg_bin = bands_[band].bg_bin;
                        const int ed_bin = bands_[band].ed_bin;
                        
                        // エネルギー計算
                        for (int track = 0; track < config_.num_tracks; track++) {
							if (temp_is_silent_[track][ch]) {
								temp_original_scale_[ch][band][track] = 0;
							}
							else {
								double energy = 0;
								for (int bin = bg_bin; bin < ed_bin; bin++) {
									// 実数FFT用の補正
									double compensation = 2;
									if (bin == 0) compensation = 1;
									if (analysis_size_ % 2 == 0 && bin == analysis_spec_size_ - 1) compensation = 1;
#ifndef BA_CLEAR_MIXER_FILTER2_SAME_WINDOW_ENABLED
									energy += compensation * std::norm(analysis_spec_[track][bin]);
#else
									energy += compensation * std::norm(output_spec_[track][bin]);
#endif
								}
#ifdef BA_CLEAR_MIXER_FILTER2_SAME_WINDOW_ENABLED
								energy *= analysis_to_output_energy_ratio_;
#endif
								temp_original_scale_[ch][band][track] = std::sqrt(std::max<double>(0, energy));
							}
							temp_scale_[ch][band][track] = temp_original_scale_[ch][band][track];
                        }

						temp_primary_track_[ch][band] = -1;
                    
                        // スパース化
                        if (config_.algorithm == kAlgorithmExpander) {
                            // expander方式
                            double l2 = 0;
                            double ma = -1;
							int mi = 0;
                            for (int track = 0; track < config_.num_tracks; track++) {
                                l2 += bakuage::Sqr(temp_scale_[ch][band][track]);
								if (ma < temp_scale_[ch][band][track]) {
									ma = temp_scale_[ch][band][track];
									mi = track;
								}
                            }
							temp_primary_track_[ch][band] = mi;
                            double l2_2 = 0;
                            for (int track = 0; track < config_.num_tracks; track++) {
                                temp_scale_[ch][band][track] *= std::pow(temp_scale_[ch][band][track] / (1e-37 + ma), ratio_ - 1);
#if 0
								// パラレルコンプみたいにやったらどうなるか実験。プチプチは消えなかった
								temp_scale_[ch][band][track] =
									temp_scale_[ch][band][track]
									+ 0.5 * temp_original_scale_[ch][band][track];
#endif
								l2_2 += bakuage::Sqr(temp_scale_[ch][band][track]);
                            }
                            double normalize_scale = std::sqrt(l2 / (1e-37 + l2_2));
                            for (int track = 0; track < config_.num_tracks; track++) {
                                temp_scale_[ch][band][track] *= normalize_scale;
                            }
                        } else if (config_.algorithm == kAlgorithmSparse) {
                            // l1指定方式
                            double l2 = 0;
                            double l1 = 0;
                            for (int track = 0; track < config_.num_tracks; track++) {
                                l2 += bakuage::Sqr(temp_scale_[ch][band][track]);
                                l1 += temp_scale_[ch][band][track];
                            }
                            l2 = std::sqrt(l2);
                            const double effective_tracks = std::max<double>(1, l1 / (1e-37 + l2));
                            const double target_effective_tracks = std::pow(effective_tracks, 1.0 / ratio_);
                            double target_l1 = std::min<double>(l1, l2 * target_effective_tracks);
                            Prox(temp_scale_[ch][band].data(), config_.num_tracks, target_l1, l2);
                        }
       
#if 0
                        double after_l2 = 0;
                        double after_l1 = 0;
                        for (int track = 0; track < config_.num_tracks; track++) {
                            after_l2 += bakuage::Sqr(temp_scale_[ch][band][track]);
                            after_l1 += temp_scale_[ch][band][track];
                        }
                        after_l2 = std::sqrt(after_l2);
                        std::cerr << l1 << " " << l2 << " " << after_l1 << " " << after_l2 << std::endl;
#endif
                        
                        // output_specにgainを掛ける
						for (int track = 0; track < config_.num_tracks; track++) {
							if (temp_is_silent_[track][ch]) continue;

							double s = wet_scale_ * (temp_scale_[ch][band][track] / (1e-37 + temp_original_scale_[ch][band][track])) + dry_scale_;
							if (track == temp_primary_track_[ch][band]) {
								s *= primary_scale_;
							}
							s *= track_scales_[track];
							for (int bin = bg_bin; bin < ed_bin; bin++) {
								output_spec_[track][bin] *= s;
							}
						}
                    }
                    
                    // IFFT + 出力バッファに足しこむ
                    for (int track = 0; track < config_.num_tracks; track++) {
						if (temp_is_silent_[track][ch]) continue;

                        dft_.Backward((Float *)output_spec_[track].data(), temp_windowed_.data());
                        for (int frame = 0; frame < analysis_size_; frame++) {
                            output_buffer_[ch][frame] += temp_windowed_[frame];
                        }
                    }

					// shift input buffer
					for (int track = 0; track < config_.num_tracks; track++) {
						if (temp_is_silent_[track][ch]) continue;
						std::memmove(input_buffer_[track][ch].data(), input_buffer_[track][ch].data() + shift_size_, sizeof(Float) * (analysis_size_ - shift_size_));
					}
                }

				AnalysisResult analysis_result = { 0 };
				analysis_result.pre_scales = temp_original_scale_.data();
				analysis_result.post_scales = temp_scale_.data();
				analysis_result.primary_tracks = temp_primary_track_.data();
				analysis_callback(analysis_result);
                
                buffer_pos_ = analysis_size_ - shift_size_;
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
        int delay_samples() const { return analysis_size_; }
        int num_bands() const { return bands_.size(); }
		int shift_samples() const { return shift_size_; }
		Float band_freqs(int i) const {
			return i < bands_.size() - 1 ? bands_[i].low_freq : bands_[bands_.size() - 1].high_freq;
		}
		int band_bins(int i) const {
			return i < bands_.size() - 1 ? bands_[i].bg_bin : bands_[bands_.size() - 1].ed_bin;
		}
    private:
        class Band {
        public:
            Float low_freq;
            Float high_freq;
            int bg_bin;
			int ed_bin;
        };
        
        // xを指定したL1, L2を持つ最近某点に移動させる
        // http://www.jmlr.org/papers/volume5/hoyer04a/hoyer04a.pdf
        void Prox(Float *x, int size, Float l1, Float l2) {
            // 目標のL1を持つ平面に射影
            const double c = (l1 - std::accumulate(x, x + size, 0.0)) / size;
            for (int i = 0; i < size; i++) {
                x[i] += c;
                temp_z_[i] = 1;
            }
            
            while (1) {
                // L1平面上のL2最小点を求める(m, m, m, ...)
                const int c2 = std::accumulate(temp_z_.data(), temp_z_.data() + size, 0);
                const auto m = l1 / c2;
                
                // xとL2最小点を結ぶ直線上で、目標L2を持つ点を見つける
                // 2つ見つかるのでxに近い方を選ぶ(alpha > 0)
                double k2 = 1e-37; // これがあると、L2とL1の両方を満たせない場合はL1を優先する
                // double k1 = 0;
                // k1は常にゼロになる
                double k0 = -bakuage::Sqr(l2);
                for (int i = 0; i < size; i++) {
                    k2 += bakuage::Sqr(x[i] - m * temp_z_[i]);
                    // k1 += 2 * (m * temp_z_[i] - x[i]) * (m * temp_z_[i]);
                    k0 += bakuage::Sqr(m * temp_z_[i]);
                }
                // 解けないケース
                if (k0 >= 0) break;
                
                const double alpha = std::sqrt(-k0 / k2);
                // std::cerr << "alpha " << alpha << std::endl;
                int flag = 0;
                for (int i = 0; i < size; i++) {
                    x[i] = m * temp_z_[i] + alpha * (x[i] - m * temp_z_[i]);
                    if (x[i] < 0) {
                        x[i] = 0;
                        temp_z_[i] = 0;
                        flag = 1;
                    }
                }
                if (flag == 0) break;
                
                const int c3 = std::accumulate(temp_z_.data(), temp_z_.data() + size, 0);
                if (c3 == 0) break;
                // L1平面へ射影
                const double c4 = (l1 - std::accumulate(x, x + size, 0.0)) / c3;
                for (int i = 0; i < size; i++) {
                    x[i] -= c4 * temp_z_[i];
                }
            }
        }
        
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
                band.bg_bin = prev_freq / sample_rate * config_.window_samples;
                band.ed_bin = next_freq / sample_rate * config_.window_samples;
                bands.push_back(band);
                
                if (next_freq >= sample_rate / 2) {
                    break;
                }
                prev_freq = next_freq;
            }
            
            bands.back().high_freq = sample_rate / 2;
            bands.back().ed_bin = config_.window_samples / 2;
            
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
		AlignedPodVector<int> temp_z_; // for Prox
		std::vector<AlignedPodVector<bool>> temp_is_silent_; // [track][ch] 無音最適化用、クロックをまたいで記憶される
        
        const int analysis_size_;
        const int shift_size_;
        const int analysis_spec_size_;
#ifndef BA_CLEAR_MIXER_FILTER2_SAME_WINDOW_ENABLED
        std::vector<AlignedPodVector<std::complex<Float>>> analysis_spec_; // [track][bin] analysis_spec_size_
#else
		Float analysis_to_output_energy_ratio_;
#endif
        std::vector<AlignedPodVector<std::complex<Float>>> output_spec_; // [track][bin]  analysis_spec_size_
        std::vector<std::vector<AlignedPodVector<Float>>> input_buffer_; // [track][ch][frame] analysis_size_
        std::vector<AlignedPodVector<Float>> output_buffer_; // [ch] analysis_size_
        AlignedPodVector<Float> analysis_window_;
        AlignedPodVector<Float> output_window_;
        AlignedPodVector<Float> temp_windowed_;
        int buffer_pos_;
        RealDft<Float> dft_;
    };
}

#endif


