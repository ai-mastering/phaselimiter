#include "phase_limiter/pre_compression.h"

#include <cmath>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <chrono>
#include <string>
#include <random>
#include <stdexcept>
#include <algorithm>
#include <fstream>
#include <streambuf>
#include "gflags/gflags.h"

#include "audio_analyzer/peak.h"
#include "bakuage/loudness_ebu_r128.h"
#include "bakuage/sndfile_wrapper.h"
#include "bakuage/compressor_filter.h"
#include "bakuage/file_utils.h"
#include "bakuage/utils.h"
#include "bakuage/ffmpeg.h"

DECLARE_double(pre_compression_threshold);
DECLARE_double(pre_compression_mean_sec);

typedef float Float;
using namespace bakuage;

namespace {
class LimitingLoudnessMapping {
public:
    LimitingLoudnessMapping(Float threshold = 0): threshold_(threshold) {}

    Float operator () (Float x) {
        return std::min<Float>(threshold_, x);
    }

    Float threshold() const { return threshold_; }
private:
    Float threshold_;
};
typedef CompressorFilter<Float, LimitingLoudnessMapping> Compressor;
}

namespace phase_limiter {

void PreCompress(std::vector<float> *_wave, int sample_rate) {
    std::vector<float> &wave = *_wave;

    Float loudness;
    std::vector<int> histogram;
    bakuage::loudness_ebu_r128::CalculateLoudness(wave.data(), 
        2, wave.size() / 2, sample_rate,
        &loudness, &histogram);

    LimitingLoudnessMapping loudness_mapping(loudness + FLAGS_pre_compression_threshold);
    Compressor::Config compressor_config;
    compressor_config.loudness_mapping_func = loudness_mapping;
    compressor_config.mean_sec = FLAGS_pre_compression_mean_sec;
    compressor_config.num_channels = 2;
    compressor_config.sample_rate = sample_rate;
    Compressor compressor(compressor_config);

    Float max_loudness = 0;
    for (int i = histogram.size() - 1; i >= 0; i--) {
        if (histogram[i] > 0) {
            max_loudness = i - 70;
            break;
        }
    }

    std::cerr << "Pre-compression: " << std::endl;
    std::cerr << "  loudness: " << loudness << std::endl;
    std::cerr << "  threshold: " << loudness_mapping.threshold() << std::endl;
    std::cerr << "  max: " << max_loudness << std::endl;

    int len = wave.size() / 2;
    int len2 = len + compressor.delay_samples(); 
    Float zero[2] = {0, 0};
    for (int i = 0; i < len2; i++) {
        Float temp[2];
        if (i < len) {
            compressor.Clock(&wave[2 * i], temp);
        }
        else {
            compressor.Clock(zero, temp);
        }
        int j = i - compressor.delay_samples();
        if (j >= 0) {
            wave[2 * j + 0] = temp[0];
            wave[2 * j + 1] = temp[1];
        }
    }
}

}
