#ifndef BAKUAGE_BAKUAGE_CLEAR_MIXER_FILTER_H_
#define BAKUAGE_BAKUAGE_CLEAR_MIXER_FILTER_H_

#include <algorithm>
#include <limits>
#include "bakuage/delay_filter.h"
#include "bakuage/fir_design.h"
#include "bakuage/fir_filter2.h"
#include "bakuage/loudness_filter.h"
#include "bakuage/time_varying_lowpass_filter.h"

namespace bakuage {

    // Delphiから移植
    template <typename Float>
    class ClearMixerFilter {
    public:
        enum {
            kMaxFrames = 4096,
        };

        class Config {
        public:
            Config(): num_tracks(0), num_channels(0), sample_rate(0), window_sec(0), mean_sec(0) {}
            int num_tracks;
            int num_channels;
            int sample_rate;
            float window_sec;
            float mean_sec;
        };

        ClearMixerFilter(const Config &config):
        config_(config), wet_(1), temp_filtered_(config.num_tracks, std::vector<Float>(kMaxFrames)), temp_scale_(config.num_tracks), temp_z_(config.num_tracks) {
            // design energy lowpass filter
            const int lowpass_filter_order = 2;
            const Float peak = std::min<Float>(1.0, 1.0 / (config_.sample_rate * config_.mean_sec
                                                     + 1e-30));
            const Float a = TimeVaryingLowpassFilter<Float>::CalculateAFromPeak(
                                                                          lowpass_filter_order, peak, &lowpass_delay_samples_);

            fir_delay_samples_ = config.window_sec / 2 * config.sample_rate;

            // 窓関数法でfirを作る
            const std::vector<Band> bands = CreateBandsByErb(config.sample_rate, 1);
            std::vector<std::vector<Float>> firs;
            for (const auto &band: bands) {
                const int n = 2 * fir_delay_samples_ + 1;
                const Float freq1 = std::min<Float>(0.5, band.low_freq / config.sample_rate);
                const Float freq2 = std::min<Float>(0.5, band.high_freq / config.sample_rate);
                firs.emplace_back(CalculateBandPassFir<Float>(freq1, freq2, n, 4));
            }

            // create filters
            for (int i = 0; i < config_.num_tracks; i++) {
                std::vector<std::vector<FirFilter2<Float>>> fir_filters_of_track;
                std::vector<std::vector<TimeVaryingLowpassFilter<Float>>> lowpass_filters_of_track;
                std::vector<std::vector<DelayFilter<Float>>> delay_filters_of_track;
                for (int j = 0; j < config_.num_channels; j++) {
                    std::vector<FirFilter2<Float>> fir_filters_of_channel;
                    std::vector<TimeVaryingLowpassFilter<Float>> lowpass_filters_of_channel;
                    std::vector<DelayFilter<Float>> delay_filters_of_channel;
                    for (int k = 0; k < bands.size(); k++) {
                        fir_filters_of_channel.emplace_back(firs[k].begin(), firs[k].end());
                        lowpass_filters_of_channel.emplace_back(lowpass_filter_order, a);
                        delay_filters_of_channel.emplace_back(lowpass_delay_samples_);
                    }
                    fir_filters_of_track.emplace_back(fir_filters_of_channel);
                    lowpass_filters_of_track.emplace_back(lowpass_filters_of_channel);
                    delay_filters_of_track.emplace_back(delay_filters_of_channel);
                }
                fir_filters_.emplace_back(fir_filters_of_track);
                lowpass_filters_.emplace_back(lowpass_filters_of_track);
                delay_filters_.emplace_back(delay_filters_of_track);
            }
        }

        // input[track][channel][frame], output[channel][frame]
        void Clock(Float ***input, int frames, Float **output, int frame_shift = 0) {
            if (frames > kMaxFrames) {
                for (int frame = 0; frame < frames; frame += kMaxFrames) {
                    Clock(input, std::min<int>(frames - frame, kMaxFrames), output, frame);
                }
                return;
            }

            for (int ch = 0; ch < config_.num_channels; ch++) {
                // fill 0 output
                for (int frame = 0; frame < frames; frame++) {
                    output[ch][frame + frame_shift] = 0;
                }

                for (int band = 0; band < num_bands(); band++) {
                    // FIR filter (それほど重くなぁE
                    for (int track = 0; track < config_.num_tracks; track++) {
                        fir_filters_[track][ch][band].Clock(input[track][ch] + frame_shift, input[track][ch] + frames + frame_shift, temp_filtered_[track].data());
                    }

#if 1
                    // こ繝ォ繝シ繝怜・縺ッ縺ゥ縺薙′驥阪＞縺ィ縺・≧繧上¢縺ァ縺ッ縺ェ縺・′縲∝・菴鍋噪縺ォ驥阪＞
                    for (int frame = 0; frame < frames; frame++) {
                        // sqr + lowpass
                        for (int track = 0; track < config_.num_tracks; track++) {
                            lowpass_filters_[track][ch][band].Clock(bakuage::Sqr(temp_filtered_[track][frame]));
                        }

#if 1
                        // calculate gain
                        double l2 = 0;
                        double l1 = 0;
                        for (int track = 0; track < config_.num_tracks; track++) {
                            temp_scale_[track] = std::sqrt(std::max<double>(0, lowpass_filters_[track][ch][band].output()));
                            l2 += bakuage::Sqr(temp_scale_[track]);
                            l1 += temp_scale_[track];
                        }
                        l2 = std::sqrt(l2);
                        l1 = std::min<double>(l1, l2 * 0.5);
#endif

#if 0
                        // 蟆代＠驥阪＞
                        Prox(temp_scale_.data(), config_.num_tracks, l1, l2);
#endif

#if 1
                        // apply gain + output
                        for (int track = 0; track < config_.num_tracks; track++) {
                            const Float s = temp_scale_[track] / (1e-37 + std::sqrt(std::max<double>(0, lowpass_filters_[track][ch][band].output())));
                            const Float s2 = wet_ * s + (1 - wet_);
                            output[ch][frame + frame_shift] += delay_filters_[track][ch][band].Clock(temp_filtered_[track][frame]) * s2;
                        }
#endif
                    }
#endif
                }
            }
        };

        void set_wet(Float value) { wet_ = value; }
        Float wet() const { return wet_; }
        int delay_samples() const { return fir_delay_samples_ + lowpass_delay_samples_; }
        int num_bands() const { return fir_filters_[0][0].size(); }
    private:
        class Band {
        public:
            Float low_freq;
            Float high_freq;
        };

#if 0
        // x繧呈欠螳壹＠縺櫚1, L2繧呈戟縺、譛某点に移動させる
        void Prox(Float *x, int size, Float l1, Float l2) {
            const double c = (l1 - std::accumulate(x, x + size, 0)) / size;
            for (int i = 0; i < size; i++) {
                x[i] += c;
                temp_z_[i] = 1;
            }

            while (1) {
                const int c2 = std::accumulate(temp_z_.data(), temp_z_.data() + size, 0);
                if (c2 == 1) {
                    for (int i = 0; i < size; i++) {
                        x[i] = temp_z_[i] * l1;
                    }
                    break;
                }
                const auto m = l1 / c2;

                double k2 = 1e-37;
                double k1 = 0;
                double k0 = -bakuage::Sqr(l2);
                for (int i = 0; i < size; i++) {
                    k2 += bakuage::Sqr(x[i] - m * temp_z_[i]);
                    k1 += 2 * (x[i] - m * temp_z_[i]) * (m * temp_z_[i]);
                    k0 += bakuage::Sqr(m * temp_z_[i]);
                }

                if (k0 >= 0) break;

                const double alpha = std::sqrt(-k0 / k2);
                int flag = 0;
                for (int i = 0; i < size; i++) {
                    x[i] = alpha * x[i] + (1 - alpha) * m * temp_z_[i];
                    if (x[i] < 0) {
                        temp_z_[i] = 0;
                        flag = 1;
                    }
                }
                if (flag == 0) break;
                for (int i = 0; i < size; i++) {
                    x[i] *= temp_z_[i];
                }

                const int c3 = std::accumulate(temp_z_.data(), temp_z_.data() + size, 0);
                if (c3 == 0) break;
                const double c4 = (std::accumulate(x, x + size, 0) - l1) / c3;
                for (int i = 0; i < size; i++) {
                    x[i] -= c4 * temp_z_[i];
                }
            }
        }
#endif

        std::vector<Band> CreateBandsByErb(int sample_rate, Float erb_scale) {
            std::vector<Band> bands;

            Float prev_freq = 0;
            while (1) {
                Float next_freq = prev_freq + erb_scale * bakuage::GlasbergErb(prev_freq);

                // 最後が短ぁE→縺阪・繧ケ繧ュ繝・・
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
        int lowpass_delay_samples_;
        int fir_delay_samples_;
        Float wet_;
        std::vector<std::vector<std::vector<FirFilter2<Float>>>> fir_filters_; // [track][ch][band]
        std::vector<std::vector<std::vector<TimeVaryingLowpassFilter<Float>>>> lowpass_filters_; // [track][ch][band]
        std::vector<std::vector<std::vector<DelayFilter<Float>>>> delay_filters_; // [track][ch][band]
        std::vector<std::vector<Float>> temp_filtered_; // [track][frame]
        std::vector<Float> temp_scale_; // [track]
        std::vector<Float> temp_z_; // for Prox
    };
}

#endif

