#include "gtest/gtest.h"
#include "bakuage/memory.h"
#include "bakuage/clear_mixer_filter3.h"

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
}

TEST(ClearMixerFilter3, TestRatio1) {
    const int num_inputs = 3;
    const int channels = 2;
    const int samples = 44100;
    const int sample_freq = 44100;
    
    std::vector<bakuage::AlignedPodVector<Float>> input(num_inputs, bakuage::AlignedPodVector<Float>(channels * samples));
    bakuage::AlignedPodVector<Float> output(channels * samples);
    for (int i = 0; i < num_inputs; i++) {
        for (int j = 0; j < samples; j++) {
            for (int ch = 0; ch < channels; ch++) {
                input[i][channels * j + ch] = 1 + i + j + ch;
            }
        }
    }
    
    bakuage::ClearMixerFilter3<Float>::Config config3;
    config3.num_tracks = input.size();
    config3.num_channels = channels;
    config3.sample_rate = sample_freq;
    config3.fir_samples = 2 * (int)(0.020 * sample_freq / 2) + 1;
    config3.gain_decimation = bakuage::CeilPowerOf2(sample_freq / 1500);
    config3.energy_mean_sec = 0.020;
    config3.scale_mean_sec = 0.020;
    config3.filter = bakuage::ClearMixerFilter3<Float>::kFilterFir;
    config3.noise_reduction = bakuage::ClearMixerFilter3<Float>::kNoiseReductionDisabled;
    bakuage::ClearMixerFilter3<Float> clear_mixer_filter3(config3);
    clear_mixer_filter3.set_ratio(1);
    
    constexpr int process_size = 4096; //44100 * 0.004;
    Float ***input_buf = Alloc3DFloat(input.size(), channels, process_size);
    Float **output_buf = Alloc2DFloat(channels, process_size);
    
    for (int base_frame = 0; base_frame < samples + clear_mixer_filter3.delay_samples(); base_frame += process_size) {
        const int this_frames = std::min<int>(samples + clear_mixer_filter3.delay_samples() - base_frame, process_size);
        
        // input
        for (int track = 0; track < input.size(); track++) {
            for (int ch = 0; ch < channels; ch++) {
                for (int frame = 0; frame < this_frames; frame++) {
                    const int f = frame + base_frame;
                    input_buf[track][ch][frame] = f < samples ? input[track][channels * f + ch] : 0;
                }
            }
        }
        
        // process
        clear_mixer_filter3.Clock(input_buf, this_frames, output_buf);
        
        // output
        for (int ch = 0; ch < channels; ch++) {
            for (int frame = 0; frame < this_frames; frame++) {
                const int f = frame + base_frame - clear_mixer_filter3.delay_samples();
                if (f >= 0) {
                    output[channels * f + ch] = output_buf[ch][frame];
                }
            }
        }
    }
    
    // verification
    for (int j = 0; j < samples; j++) {
        for (int ch = 0; ch < channels; ch++) {
            double expected = 0;
            for (int i = 0; i < num_inputs; i++) {
                expected += input[i][channels * j + ch];
            }
            EXPECT_NEAR(expected, output[channels * j + ch], 1e-2 * expected);
        }
    }
}
