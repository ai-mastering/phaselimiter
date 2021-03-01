#include <iostream>

#include "gflags/gflags.h"
#include "sndfile.h"
#include "tbb/tbb.h"
#include "tbb/pipeline.h"
#include "tbb/scalable_allocator.h"
#include "tbb/cache_aligned_allocator.h"
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include "bakuage/sndfile_wrapper.h"
#include "bakuage/dft.h"
#include "bakuage/memory.h"
#include "bakuage/utils.h"
#include "bakuage/window_func.h"
#include "bakuage/transient_filter.h"
#include "bakuage/transient_filter2.h"

DEFINE_string(input, "", "comma separated input wav file paths");
DEFINE_string(output, "", "output wav file path");
DEFINE_double(window_sec, 0.02, "dft window size in sec");
DEFINE_double(mean_sec, 0.02, "energy lowpass mean time in sec");
DEFINE_double(ratio, 2, "ratio");
DEFINE_double(noise_reduction_threshold, -20, "noise reduction threshold in dB");
DEFINE_bool(normalize, false, "whether if normalize is done before output");
DEFINE_string(mode, "transient", "mode (transient)");

namespace {
typedef float Float;
    
template <class Effect>
void CalculateEffect(Effect *effect, const std::vector<std::vector<Float>> &input, int channels, int samples, int sample_freq, Float *output) {

    std::cerr << "calculate start" << std::endl;
    bakuage::StopWatch stop_watch;
    stop_watch.Start();
    Float temp[2] = { 0 };
    Float zero[2] = { 0 };
    for (int i = 0; i < samples + effect->delay_samples(); i++) {
        if (i < samples) {
            for (int j = 0; j < channels; j++) {
                temp[j] = input[0][channels * i + j];
            }
            effect->Clock(temp, temp);
        } else {
            effect->Clock(zero, temp);
        }
        
        int output_i = i - effect->delay_samples();
        if (0 <= output_i) {
            for (int j = 0; j < channels; j++) {
                output[channels * output_i + j] = temp[j];
            }
        }
    }
    
    std::cerr << "calculate finish " << stop_watch.time() << std::endl;
}

template <class Float>
void SaveFloatWave(const std::vector<Float> &wave, const std::string &filename) {
    bakuage::SndfileWrapper snd_file;
    SF_INFO sfinfo = { 0 };
    std::memset(&sfinfo, 0, sizeof(sfinfo));
    
    sfinfo.channels = 2;
    sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;
    int frames = wave.size() / sfinfo.channels;
    sfinfo.frames = frames;
    sfinfo.samplerate = 44100;
    
    if ((snd_file.set(sf_open(filename.c_str(), SFM_WRITE, &sfinfo))) == NULL) {
        std::stringstream message;
        message << "Not able to open output file " << filename << ", "
        << sf_strerror(NULL);
        throw std::logic_error(message.str());
    }
    
    sf_count_t size;
    if (sizeof(Float) == 4) {
        size = sf_writef_float(snd_file.get(), (float *)wave.data(), frames);
    } else {
        size = sf_writef_double(snd_file.get(), (double *)wave.data(), frames);
    }
    if (size != frames) {
        std::stringstream message;
        message << "sf_writef_float error: " << size;
        throw std::logic_error(message.str());
    }
}
}

// wavを受け取って、標準出力にrawvideoを出力
int main(int argc, char* argv[]) {
    gflags::SetVersionString("1.0.0");
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    bakuage::SndfileWrapper infile;
    SF_INFO sfinfo = { 0 };
    int frames = 1 << 30;
    
    // カンマ区切りを分解して、load wave (全てが同じフォーマットである必要がある)
    std::vector<std::vector<float>> buffers;
    std::vector<std::string> input_paths;
    boost::algorithm::split(input_paths, FLAGS_input, boost::is_any_of(","));
    for (const auto &input_file_path: input_paths) {
        if ((infile.set(sf_open (input_file_path.c_str(), SFM_READ, &sfinfo))) == NULL) {
            fprintf(stderr, "Not able to open input file %s.\n", input_file_path.c_str());
            fprintf(stderr, "%s\n", sf_strerror(NULL));
            return 1;
        }
        
        // check format
        fprintf(stderr, "sfinfo.format 0x%08x.\n", sfinfo.format);
        switch (sfinfo.format & SF_FORMAT_TYPEMASK) {
            case SF_FORMAT_WAV:
            case SF_FORMAT_WAVEX:
                break;
            default:
                fprintf(stderr, "Not supported sfinfo.format 0x%08x.\n", sfinfo.format);
                return 2;
        }
        
        std::vector<float> buffer(sfinfo.channels * sfinfo.frames);
        int read_size = sf_readf_float(infile.get(), buffer.data(), sfinfo.frames);
        fprintf(stderr, "%d samples read.\n", read_size);
        if (read_size != sfinfo.frames) {
            fprintf(stderr, "sf_readf_float error: %d %d\n", read_size, (int)sfinfo.frames);
            return 3;
        }
        buffers.emplace_back(buffer);
        
        frames = std::min<int>(frames, sfinfo.frames);
    }
    
    // calculate spectrogram (energy)
    std::vector<float> output(frames * sfinfo.channels);
    
    if (FLAGS_mode == "transient") {
        bakuage::TransientFilter<Float, std::function<Float (Float, Float)>>::Config config;
        config.num_channels = sfinfo.channels;
        config.sample_rate = sfinfo.samplerate;
        config.long_mean_sec = 0.2;
        config.short_mean_sec = 0.02;
        config.gain_func = [](Float long_loundess, Float short_loudness) {
            return (short_loudness - long_loundess) * (FLAGS_ratio - 1);
        };
        bakuage::TransientFilter<Float, std::function<Float (Float, Float)>> transient_filter(config);
        CalculateEffect(&transient_filter, buffers, sfinfo.channels, frames, sfinfo.samplerate, output.data());
    } else if (FLAGS_mode == "transient2") {
        bakuage::TransientFilter2<Float, std::function<Float (Float, Float)>>::Config config;
        config.num_channels = sfinfo.channels;
        config.sample_rate = sfinfo.samplerate;
        config.long_mean_sec = 0.2;
        config.short_mean_sec = 0.02;
        config.gain_func = [](Float long_loundess, Float short_loudness) {
            return (short_loudness - long_loundess) * (FLAGS_ratio - 1);
        };
        bakuage::TransientFilter2<Float, std::function<Float (Float, Float)>> transient_filter(config);
        CalculateEffect(&transient_filter, buffers, sfinfo.channels, frames, sfinfo.samplerate, output.data());
    }
    
    // normalize
    if (FLAGS_normalize) {
        double peak = 1e-37;
        for (int i = 0; i < output.size(); i++) {
            peak = std::max<double>(peak, std::abs(output[i]));
        }
        for (int i = 0; i < output.size(); i++) {
            output[i] /= peak;
        }
    }
    
    // save output
    SaveFloatWave(output, FLAGS_output);
}

