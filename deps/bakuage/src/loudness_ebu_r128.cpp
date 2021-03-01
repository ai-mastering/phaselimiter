#include "bakuage/loudness_ebu_r128.h"

#include <algorithm>
#include <vector>
#include <functional>
#include <complex>

#include "bakuage/loudness_filter.h"
#include "bakuage/ms_compressor_filter.h"
#include "bakuage/memory.h"
#include "bakuage/utils.h"
#include "bakuage/dft.h"
#include "bakuage/loudness_contours.h"
#include "bakuage/vector_math.h"

namespace bakuage {
    namespace loudness_ebu_r128 {
        
        // ラウドネス規格: BS.1770
        // Loudness Range http://www.abma-bvam.be/PDF/EBU_PLOUD/EBU_tech3342.pdf
        // http://jp.music-group.com/TCE/Tech/LRA.pdf
        // block_sec = 3, shift_sec = 2, relative_threshold_db = -20
        
        template <typename Float>
        void CalculateLoudnessCore(const Float *input, const int channels, const int samples, const int sample_freq,
                                   const Float block_sec, const Float shift_sec, const Float absolute_threshold_db, const Float relative_threshold_db,
                                   Float *loudness, Float *loudness_range, std::vector<int> *histogram,
                                   std::vector<Float> *loudness_time_series,
                                   int *block_samples, bool use_youtube_weighting, Float *max_loudness) {
            using namespace bakuage;
            
            bakuage::AlignedPodVector<Float> filtered(channels * samples);
            if (use_youtube_weighting) {
                const int fft_len = bakuage::CeilPowerOf2(samples);
                bakuage::RealDft<Float> dft(fft_len);
                bakuage::AlignedPodVector<Float> split(fft_len);
                bakuage::AlignedPodVector<std::complex<Float>> spec(fft_len / 2 + 1);
                
                bakuage::AlignedPodVector<Float> weights(spec.size());
                const auto scale = 1.0 / fft_len;
                for (int j = 0; j < spec.size(); j++) {
                    const auto hz = 1.0 * j * sample_freq / fft_len;
                    weights[j] = std::pow(10, bakuage::loudness_contours::HzToYoutubeWeighting(hz) / 20) * scale;
                }
                
                for (int i = 0; i < channels; i++) {
                    for (int j = 0; j < samples; j++) {
                        int k = channels * j + i;
                        split[j] = input[k];
                    }
                    bakuage::TypedFillZero(split.data() + samples, fft_len - samples);
                    
                    // fft
                    dft.Forward(split.data(), (Float *)spec.data());
                    
                    // filter
                    bakuage::VectorMulInplace(weights.data(), spec.data(), spec.size());
                    
                    // ifft
                    dft.Backward((Float *)spec.data(), split.data());
                    
                    for (int j = 0; j < samples; j++) {
                        int k = channels * j + i;
                        filtered[k] = split[j];
                    }
                }
            } else {
                std::vector<LoudnessFilter<double>> filters;
                for (int i = 0; i < channels; i++) {
                    LoudnessFilter<double> filter(sample_freq);
                    for (int j = 0; j < samples; j++) {
                        int k = channels * j + i;
                        filtered[k] = filter.Clock(input[k]);
                    }
                }
            }
            
            std::vector<int> &histo = *histogram;
            
            histo.clear();
            histo.resize(140);
            
            std::vector<Float> blocks;
            
            if (max_loudness) *max_loudness = -1e37;
            
            int pos = 0;
            // 400ms block
            int width = (int)(sample_freq * block_sec); // nearest samples
            int shift = (int)(sample_freq * shift_sec);
            // 規格では最後のブロックは使わないけど、
            // 使ったほうが実用的なので使う
            while (pos < samples) {
                double sum = 0;
                int end = std::min<int>(pos + width, samples);
                int len = end - pos;
                const Float *filtered_pos = filtered.data() + channels * pos;
                for (int i = pos; i < end; i++) {
                    for (int j = 0; j < channels; j++) {
                        sum += bakuage::Sqr(*filtered_pos);
                        filtered_pos++;
                    }
                }
                
                double z = 10 * std::log10(1e-37 + sum / len);
                if (!use_youtube_weighting) z -= 0.691; // ステレオの1kHz正弦波で0になるように補正。
                blocks.push_back(z);
                if (max_loudness) *max_loudness = std::max<Float>(*max_loudness, z);
                
                // -70 <-> [-70, -69)
                int index = std::floor(z) + 70;
                if (0 <= index && index < histo.size()) {
                    histo[index]++;
                }
                
                // 75% overlap
                pos += shift;
            }
            
            double threshold = absolute_threshold_db;
            for (int k = 0; k < 2; k++) {
                double count = 0;
                double sum = 0;
                for (double z : blocks) {
                    const bool valid = z >= threshold;
                    count += valid;
                    sum += valid * z;
                }
                
                double mean = sum / (1e-37 + count);
                if (k == 0) {
                    threshold = mean + relative_threshold_db;
                }
                else if (k == 1) {
                    if (loudness) {
                        *loudness = mean;
                    }
                }
            }
            
            if (loudness_range) {
                std::vector<Float> sorted_blocks;
                for (double z : blocks) {
                    if (z < threshold) continue;
                    sorted_blocks.push_back(z);
                }
                std::sort(sorted_blocks.begin(), sorted_blocks.end());
                
                double q10 = 0;
                for (int i = 0; i < sorted_blocks.size(); i++) {
                    if (10 * sorted_blocks.size() <= 100 * i) {
                        q10 = sorted_blocks[i];
                        break;
                    }
                }
                double q95 = 0;
                for (int i = 0; i < sorted_blocks.size(); i++) {
                    if (95 * sorted_blocks.size() <= 100 * i) {
                        q95 = sorted_blocks[i];
                        break;
                    }
                }
                *loudness_range = q95 - q10;
            }
            
            if (loudness_time_series) {
                *loudness_time_series = blocks;
            }
            if (block_samples) {
                *block_samples = width;
            }
        }
        template void CalculateLoudnessCore<float>(const float *input, const int channels, const int samples, const int sample_freq,
                                   const float block_sec, const float shift_sec, const float absolute_threshold_db, const float relative_threshold_db,
                                   float *loudness, float *loudness_range, std::vector<int> *histogram,
                                   std::vector<float> *loudness_time_series,
                                                   int *block_samples, bool use_youtube_weighting, float *max_loudness);
        template void CalculateLoudnessCore<double>(const double *input, const int channels, const int samples, const int sample_freq,
                                                   const double block_sec, const double shift_sec, double absolute_threshold_db, const double relative_threshold_db,
                                                   double *loudness, double *loudness_range, std::vector<int> *histogram,
                                                   std::vector<double> *loudness_time_series,
                                                   int *block_samples, bool use_youtube_weighting, double *max_loudness);
        
        template <typename Float>
        void CalculateHistogram(const Float *input, int channels, int samples, int sample_freq, Float mean_sec,
                                std::vector<Float> *histogram, std::vector<Float> *mid_to_side_histogram) {
            using namespace bakuage;
            typedef MsCompressorFilter<Float, std::function<Float(Float)>, std::function<Float(Float)>> Compressor;
            
            typename Compressor::Config config;
            config.num_channels = channels;
            config.sample_rate = sample_freq;
            config.max_mean_sec = mean_sec;
            config.loudness_mapping_func = [](Float x) { return x; };
            config.ms_loudness_mapping_func = [](Float x) { return x; };
            
            Compressor filter(config);
            
            std::vector<Float> &histo = *histogram;
            
            histo.clear();
            histo.resize(140);
            if (mid_to_side_histogram) {
                mid_to_side_histogram->clear();
                mid_to_side_histogram->resize(140);
            }
            
            bakuage::AlignedPodVector<Float> temp_input(channels);
            int len = samples + filter.delay_samples();
            for (int i = 0; i < len; i++) {
                for (int j = 0; j < channels; j++) {
                    if (i >= samples) {
                        temp_input[j] = 0;
                    }
                    else {
                        temp_input[j] = input[channels * i + j];
                    }
                }
                Float loudness, mid_loudness, side_loudness;
                filter.Analyze(&temp_input[0], &loudness, &mid_loudness, &side_loudness);
                if (i >= filter.delay_samples()) {
                    int index = std::max(0, std::min<int>(histo.size() - 1,
                                                          std::floor(loudness) + 70));
                    histo[index] += 1;
                    
                    if (mid_to_side_histogram) {
                        index = std::max(0, std::min<int>(mid_to_side_histogram->size() - 1,
                                                          std::floor(side_loudness - mid_loudness) + 70));
                        (*mid_to_side_histogram)[index] += 1;
                    }
                }
            }
        }
        
        template
        void CalculateHistogram<float>(const float *input, int channels, int samples, int sample_freq, float mean_sec,
                                       std::vector<float> *histogram, std::vector<float> *mid_to_side_histogram);
        template
        void CalculateHistogram<double>(const double *input, int channels, int samples, int sample_freq, double mean_sec,
                                       std::vector<double> *histogram, std::vector<double> *mid_to_side_histogram);
        
    }
}
