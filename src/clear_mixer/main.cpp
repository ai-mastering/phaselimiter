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
#include "bakuage/clear_mixer_filter.h"
#include "bakuage/clear_mixer_filter2.h"
#include "bakuage/clear_mixer_filter3.h"
#include "bakuage/clear_mixer_filter4.h"

DEFINE_string(input, "", "comma separated input wav file paths");
DEFINE_string(output, "", "output wav file path");
DEFINE_double(window_sec, 0.02, "dft window size in sec");
DEFINE_double(mean_sec, 0.02, "energy lowpass mean time in sec");
DEFINE_double(ratio, 2, "ratio");
DEFINE_double(noise_reduction_threshold, -20, "noise reduction threshold in dB");
DEFINE_bool(normalize, false, "whether if normalize is done before output");
DEFINE_string(mode, "3", "clear mixer mode (1/2sparse/2expander/3/gen_test_signal)");

namespace {
typedef float Float;

Float **Alloc2DFloat(int a, int b) {
    Float **result = new Float*[a];
    for (int i = 0; i < a; i++) {
        result[i] = new Float[b];
    }
    return result;
}

Float ***Alloc3DFloat(int a, int b, int c) {
    Float ***result = new Float**[a];
    for (int i = 0; i < a; i++) {
        result[i] = Alloc2DFloat(b, c);
    }
    return result;
}

// 動画の長さが音源の長さ以上になるようにする
void CalculateClearMixer(const std::vector<std::vector<Float>> &input, int channels, int samples, int sample_freq, Float *output) {
    
    bakuage::ClearMixerFilter<Float>::Config config;
    config.num_tracks = input.size();
    config.num_channels = channels;
    config.sample_rate = sample_freq;
    config.window_sec = FLAGS_window_sec;
    config.mean_sec = FLAGS_window_sec;
    bakuage::ClearMixerFilter<Float> clear_mixer_filter(config);
    
    bakuage::ClearMixerFilter2<Float>::Config config2;
    config2.num_tracks = input.size();
    config2.num_channels = channels;
    config2.sample_rate = sample_freq;
    config2.overlap = 2;
    config2.window_samples = config2.overlap * (int)((FLAGS_window_sec * sample_freq) / config2.overlap);
    config2.algorithm = FLAGS_mode == "2sparse" ? bakuage::ClearMixerFilter2<Float>::kAlgorithmSparse : bakuage::ClearMixerFilter2<Float>::kAlgorithmExpander;
    bakuage::ClearMixerFilter2<Float> clear_mixer_filter2(config2);
    clear_mixer_filter2.set_ratio(FLAGS_ratio);
    
    bakuage::ClearMixerFilter3<Float>::Config config3;
    config3.num_tracks = input.size();
    config3.num_channels = channels;
    config3.sample_rate = sample_freq;
    config3.fir_samples = 2 * (int)(0.020 * sample_freq / 2) + 1;
    config3.gain_decimation = bakuage::CeilPowerOf2(sample_freq / 1500);
    config3.energy_mean_sec = 0.020;
    config3.scale_mean_sec = 0.020;
    config3.filter = bakuage::ClearMixerFilter3<Float>::kFilterFir;
    config3.noise_reduction = bakuage::ClearMixerFilter3<Float>::kNoiseReductionFixedSpectrumLearn;
    config3.noise_reduction_threshold = std::pow(10, FLAGS_noise_reduction_threshold / 10.0);
    bakuage::ClearMixerFilter3<Float> clear_mixer_filter3_learn(config3);
    clear_mixer_filter3_learn.set_ratio(FLAGS_ratio);
    
    bakuage::ClearMixerFilter4<Float>::Config config4;
    config4.num_tracks = input.size();
    config4.num_channels = channels;
    config4.sample_rate = sample_freq;
    config4.fir_samples = 2 * (int)(0.020 * sample_freq / 2) + 1;
    config4.gain_decimation = bakuage::CeilPowerOf2(sample_freq / 1500);
    config4.energy_mean_sec = 0.020;
    config4.scale_mean_sec = 0.020;
    config4.filter = bakuage::ClearMixerFilter3<Float>::kFilterFir;
    config4.noise_reduction = bakuage::ClearMixerFilter3<Float>::kNoiseReductionFixedSpectrumLearn;
    config4.noise_reduction_threshold = std::pow(10, FLAGS_noise_reduction_threshold / 10.0);
    bakuage::ClearMixerFilter4<Float> clear_mixer_filter4_learn(config4);
    clear_mixer_filter4_learn.set_ratio(FLAGS_ratio);
    
    constexpr int process_size = 4096; //44100 * 0.004;
    Float ***input_buf = Alloc3DFloat(input.size(), channels, process_size);
    Float **output_buf = Alloc2DFloat(channels, process_size);
    
    // learn noise
    for (int base_frame = 0; base_frame < samples + clear_mixer_filter.delay_samples(); base_frame += process_size) {
        const int this_frames = std::min<int>(samples + clear_mixer_filter.delay_samples() - base_frame, process_size);
        
#if 1
        // input
        for (int track = 0; track < input.size(); track++) {
            for (int ch = 0; ch < channels; ch++) {
                for (int frame = 0; frame < this_frames; frame++) {
                    const int f = frame + base_frame;
                    input_buf[track][ch][frame] = f < samples ? input[track][channels * f + ch] : 0;
                }
            }
        }
#endif
        
        // process
#if 1
        if (FLAGS_mode == "1") {
        } else if (FLAGS_mode == "2sparse" || FLAGS_mode == "2expander") {
        } else if (FLAGS_mode == "3") {
            clear_mixer_filter3_learn.Clock(input_buf, this_frames, output_buf);
        } else if (FLAGS_mode == "4") {
            clear_mixer_filter4_learn.Clock(input_buf, this_frames, output_buf);
        } else {
            std::stringstream ss;
            ss << "unknown mode " << FLAGS_mode;
            throw std::logic_error(ss.str());
        }
#endif
    }
    config3.noise_reduction = bakuage::ClearMixerFilter3<Float>::kNoiseReductionFixedSpectrum;
    config3.noise_reduction_fixed_spectrum_profile = clear_mixer_filter3_learn.CalculateNoiseReductionFixedSpectrumProfile();
    bakuage::ClearMixerFilter3<Float> clear_mixer_filter3(config3);
    clear_mixer_filter3.set_ratio(FLAGS_ratio);
    
    config4.noise_reduction = bakuage::ClearMixerFilter4<Float>::kNoiseReductionFixedSpectrum;
    config4.noise_reduction_fixed_spectrum_profile = clear_mixer_filter4_learn.CalculateNoiseReductionFixedSpectrumProfile();
    bakuage::ClearMixerFilter4<Float> clear_mixer_filter4(config4);
    clear_mixer_filter4.set_ratio(FLAGS_ratio);
    
    bakuage::StopWatch stop_watch;
    for (int base_frame = 0; base_frame < samples + clear_mixer_filter.delay_samples(); base_frame += process_size) {
        // 最初の一回は飛ばす
        if (base_frame == process_size) stop_watch.Start();
        
        const int this_frames = std::min<int>(samples + clear_mixer_filter.delay_samples() - base_frame, process_size);
        
#if 1
        // input
        for (int track = 0; track < input.size(); track++) {
            for (int ch = 0; ch < channels; ch++) {
                for (int frame = 0; frame < this_frames; frame++) {
                    const int f = frame + base_frame;
                    input_buf[track][ch][frame] = f < samples ? input[track][channels * f + ch] : 0;
                }
            }
        }
#endif
        
        // process
#if 1
        if (FLAGS_mode == "1") {
            clear_mixer_filter.Clock(input_buf, this_frames, output_buf);
        } else if (FLAGS_mode == "2sparse" || FLAGS_mode == "2expander") {
            clear_mixer_filter2.Clock(input_buf, this_frames, output_buf);
        } else if (FLAGS_mode == "3") {
            clear_mixer_filter3.Clock(input_buf, this_frames, output_buf);
        } else if (FLAGS_mode == "4") {
            clear_mixer_filter4.Clock(input_buf, this_frames, output_buf);
        } else {
            std::stringstream ss;
            ss << "unknown mode " << FLAGS_mode;
            throw std::logic_error(ss.str());
        }
#endif
        
        // output
#if 1
        for (int ch = 0; ch < channels; ch++) {
            for (int frame = 0; frame < this_frames; frame++) {
                const int f = frame + base_frame - clear_mixer_filter.delay_samples();
                if (f >= 0) {
                    output[channels * f + ch] = output_buf[ch][frame];
                }
            }
        }
#endif
    }
    
    std::cerr << "clear mixer: " << stop_watch.time() << std::endl;
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
    
    if (FLAGS_mode == "gen_test_signal") {
        for (int shifts = 1; shifts <= 16; shifts++) {
            {
                std::vector<Float> wave(44100);
                for (int i = 0; i < 22050; i++) {
                    double x = 0;
                    for (int j = 0; j < shifts; j++) {
                        const double t = 50.0 * i / 44100 - 1.0 * j / 16;
                        const double theta = 2 * M_PI * t;
                        x += (0 <= t && t < 1) ? 0.5 - 0.5 * std::cos(theta) : 0;
                    }
                    wave[2 * i + 0] = wave[2 * i + 1] = x;
                }
                std::stringstream ss;
                ss << "/tmp/hanning50hz" << shifts << "shift.wav";
                SaveFloatWave(wave, ss.str());
            }
            {
                std::vector<Float> wave(44100);
                for (int i = 0; i < 22050; i++) {
                    double x = 0;
                    for (int j = 0; j < shifts; j++) {
                        const double t = 50.0 * i / 44100 - 1.0 * j / 16;
                        const double theta = 2 * M_PI * t;
                        x += (0 <= t && t < 1) ? 0.35875 - 0.48829 * std::cos(theta) + 0.14128 * std::cos(2 * theta) - 0.01168 * std::cos(3 * theta) : 0;
                    }
                    wave[2 * i + 0] = wave[2 * i + 1] = x;
                }
                std::stringstream ss;
                ss << "/tmp/blackman_harris50hz" << shifts << "shift.wav";
                SaveFloatWave(wave, ss.str());
            }
        }
        return 0;
    }
    
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
    CalculateClearMixer(buffers, sfinfo.channels, frames, sfinfo.samplerate, output.data());
    
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

