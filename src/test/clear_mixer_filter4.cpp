#include "gtest/gtest.h"
#include <vector>
#include "bakuage/memory.h"
#include "bakuage/clear_mixer_filter3.h"
#include "bakuage/clear_mixer_filter4.h"

namespace {

template <class Float>
Float **Alloc2DFloat(int a, int b) {
    Float **result = new Float*[a];
    for (int i = 0; i < a; i++) {
        result[i] = new Float[b];
    }
    return result;
}

template <class Float>
Float ***Alloc3DFloat(int a, int b, int c) {
    Float ***result = new Float**[a];
    for (int i = 0; i < a; i++) {
        result[i] = Alloc2DFloat<Float>(b, c);
    }
    return result;
}
    
struct ClearMixerFilter4TestParam {
    double ratio;
    double wet_scale;
    double dry_scale;
    double primary_scale;
    int sample_freq;
    int process_size;
    int float_bits;
    
    friend std::ostream& operator<<(std::ostream& os, const ClearMixerFilter4TestParam& param) {
        os << "ClearMixerFilter4TestParam"
        << " ratio " << param.ratio
        << " wet_scale " << param.wet_scale
        << " dry_scale " << param.dry_scale
        << " primary_scale " << param.primary_scale
        << " sample_freq " << param.sample_freq
        << " process_size " << param.process_size
        << " float_bits " << param.float_bits
        ;
        return os;
    }
};
    
struct TestParamInitializer {
    TestParamInitializer() {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 3; k++) {
                    for (int m = 0; m < 2; m++) {
                        for (int n = 0; n < 4; n++) {
                            ClearMixerFilter4TestParam param;
                            param.ratio = i == 0 ? 1 : i == 1 ? 2 : 16;
                            param.sample_freq = j == 0 ? 44100 : 96000;
                            param.process_size = k == 0 ? 1 : k == 1 ? 100 : 10000;
                            param.float_bits = m == 0 ? 32 : 64;
                            switch (n) {
                                case 0:
                                    param.dry_scale = 1;
                                    param.wet_scale = 0;
                                    param.primary_scale = 1;
                                    break;
                                case 1:
                                    param.dry_scale = 0;
                                    param.wet_scale = 1;
                                    param.primary_scale = 1;
                                    break;
                                case 2:
                                    // wet - dryは桁落ちで精度が落ちてテストがfailするので、1 - 0.5にする
                                    param.dry_scale = -0.5;
                                    param.wet_scale = 1;
                                    param.primary_scale = 1;
                                    break;
                                case 3:
                                    // primary_scale == 0は桁落ちで精度が落ちてテストがfailするので、0.5にする
                                    param.dry_scale = 0;
                                    param.wet_scale = 1;
                                    param.primary_scale = 0.5;
                                    break;
                            }
                            test_params.push_back(param);
                            if (i == 0 && n < 2) {
                                test_ratio1_params.push_back(param);
                            }
                        }
                    }
                }
            }
        }
    }
    
    static TestParamInitializer &GetInstance() {
        static TestParamInitializer initializer;
        return initializer;
    }

    std::vector<ClearMixerFilter4TestParam> test_params;
    std::vector<ClearMixerFilter4TestParam> test_ratio1_params;
};

class ClearMixerFilter4Test : public ::testing::TestWithParam<ClearMixerFilter4TestParam> {};
class ClearMixerFilter4TestRatio1 : public ::testing::TestWithParam<ClearMixerFilter4TestParam> {};
    
template <class Float>
void TestDry(const ClearMixerFilter4TestParam &param) {
    const int num_inputs = 3;
    const int channels = 2;
    const int samples = 0.5 * param.sample_freq;
    const int sample_freq = param.sample_freq;
    
    std::vector<bakuage::AlignedPodVector<Float>> input(num_inputs, bakuage::AlignedPodVector<Float>(channels * samples));
    bakuage::AlignedPodVector<Float> output(channels * samples);
    for (int i = 0; i < num_inputs; i++) {
        for (int j = 0; j < samples; j++) {
            for (int ch = 0; ch < channels; ch++) {
                input[i][channels * j + ch] = 1 + i + j + ch;
            }
        }
    }
    
    typename bakuage::ClearMixerFilter4<Float>::Config config4;
    config4.num_tracks = input.size();
    config4.num_channels = channels;
    config4.sample_rate = sample_freq;
    config4.fir_samples = 2 * (int)(0.020 * sample_freq / 2) + 1;
    config4.gain_decimation = bakuage::CeilPowerOf2(sample_freq / 1500);
    config4.energy_mean_sec = 0.020;
    config4.scale_mean_sec = 0.020;
    config4.filter = bakuage::ClearMixerFilter4<Float>::kFilterFir;
    config4.noise_reduction = bakuage::ClearMixerFilter4<Float>::kNoiseReductionDisabled;
    bakuage::ClearMixerFilter4<Float> clear_mixer_filter4(config4);
    clear_mixer_filter4.set_ratio(param.ratio);
    clear_mixer_filter4.set_dry_scale(param.dry_scale);
    clear_mixer_filter4.set_wet_scale(param.wet_scale);
    clear_mixer_filter4.set_primary_scale(param.primary_scale);
    
    const int process_size = param.process_size; //44100 * 0.004;
    Float ***input_buf = Alloc3DFloat<Float>(input.size(), channels, process_size);
    Float **output_buf = Alloc2DFloat<Float>(channels, process_size);
    
    for (int base_frame = 0; base_frame < samples + clear_mixer_filter4.delay_samples(); base_frame += process_size) {
        const int this_frames = std::min<int>(samples + clear_mixer_filter4.delay_samples() - base_frame, process_size);
        
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
        clear_mixer_filter4.Clock(input_buf, this_frames, output_buf);
        
        // output
        for (int ch = 0; ch < channels; ch++) {
            for (int frame = 0; frame < this_frames; frame++) {
                const int f = frame + base_frame - clear_mixer_filter4.delay_samples();
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
            ASSERT_NEAR(expected, output[channels * j + ch], 1e-2 * expected);
        }
    }
}
    
template <class Float>
void TestEqualToClearMixer3(const ClearMixerFilter4TestParam &param) {
    const int num_inputs = 3;
    const int channels = 2;
    const int samples = param.sample_freq;
    const int sample_freq = param.sample_freq;
    
    std::vector<bakuage::AlignedPodVector<Float>> input(num_inputs, bakuage::AlignedPodVector<Float>(channels * samples));
    bakuage::AlignedPodVector<Float> output3(channels * samples);
    bakuage::AlignedPodVector<Float> output4(channels * samples);
    for (int i = 0; i < num_inputs; i++) {
        for (int j = 0; j < samples; j++) {
            for (int ch = 0; ch < channels; ch++) {
                input[i][channels * j + ch] = 1 + i + j + ch;
            }
        }
    }
    
    typename bakuage::ClearMixerFilter3<Float>::Config config3;
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
    clear_mixer_filter3.set_ratio(param.ratio);
    clear_mixer_filter3.set_dry_scale(param.dry_scale);
    clear_mixer_filter3.set_wet_scale(param.wet_scale);
    clear_mixer_filter3.set_primary_scale(param.primary_scale);
    clear_mixer_filter3.set_track_scale(0, 0.5);
    clear_mixer_filter3.set_track_scale(1, 1);
    clear_mixer_filter3.set_track_scale(2, 2);
    
    typename bakuage::ClearMixerFilter4<Float>::Config config4;
    config4.num_tracks = input.size();
    config4.num_channels = channels;
    config4.sample_rate = sample_freq;
    config4.fir_samples = 2 * (int)(0.020 * sample_freq / 2) + 1;
    config4.gain_decimation = bakuage::CeilPowerOf2(sample_freq / 1500);
    config4.energy_mean_sec = 0.020;
    config4.scale_mean_sec = 0.020;
    config4.filter = bakuage::ClearMixerFilter4<Float>::kFilterFir;
    config4.noise_reduction = bakuage::ClearMixerFilter4<Float>::kNoiseReductionDisabled;
    bakuage::ClearMixerFilter4<Float> clear_mixer_filter4(config4);
    clear_mixer_filter4.set_ratio(param.ratio);
    clear_mixer_filter4.set_dry_scale(param.dry_scale);
    clear_mixer_filter4.set_wet_scale(param.wet_scale);
    clear_mixer_filter4.set_primary_scale(param.primary_scale);
    clear_mixer_filter4.set_track_scale(0, 0.5);
    clear_mixer_filter4.set_track_scale(1, 1);
    clear_mixer_filter4.set_track_scale(2, 2);
    
    const int process_size = param.process_size; //44100 * 0.004;
    Float ***input_buf = Alloc3DFloat<Float>(input.size(), channels, process_size);
    Float **output_buf = Alloc2DFloat<Float>(channels, process_size);
    
    for (int base_frame = 0; base_frame < samples + clear_mixer_filter4.delay_samples(); base_frame += process_size) {
        const int this_frames = std::min<int>(samples + clear_mixer_filter4.delay_samples() - base_frame, process_size);
        
        // input
        for (int track = 0; track < input.size(); track++) {
            for (int ch = 0; ch < channels; ch++) {
                for (int frame = 0; frame < this_frames; frame++) {
                    const int f = frame + base_frame;
                    input_buf[track][ch][frame] = f < samples ? input[track][channels * f + ch] : 0;
                }
            }
        }
        
        // process 3
        clear_mixer_filter3.Clock(input_buf, this_frames, output_buf);
        
        // output 3
        for (int ch = 0; ch < channels; ch++) {
            for (int frame = 0; frame < this_frames; frame++) {
                const int f = frame + base_frame - clear_mixer_filter3.delay_samples();
                if (f >= 0) {
                    output3[channels * f + ch] = output_buf[ch][frame];
                }
            }
        }
        
        // process 4
        clear_mixer_filter4.Clock(input_buf, this_frames, output_buf);
        
        // output 4
        for (int ch = 0; ch < channels; ch++) {
            for (int frame = 0; frame < this_frames; frame++) {
                const int f = frame + base_frame - clear_mixer_filter4.delay_samples();
                if (f >= 0) {
                    output4[channels * f + ch] = output_buf[ch][frame];
                }
            }
        }
    }
    
    // verification
    for (int j = 0; j < samples * channels; j++) {
        const double max_relative_error = param.float_bits == 32 ? 1e-3 : 1e-7;
        ASSERT_NEAR(output3[j], output4[j], max_relative_error * output3[j]);
    }
}
}

TEST_P(ClearMixerFilter4TestRatio1, Dry) {
    const auto param = GetParam();
    
    if (param.float_bits == 32) {
        TestDry<float>(param);
    } else {
        TestDry<double>(param);
    }
}

TEST_P(ClearMixerFilter4Test, EqualToClearMixer3) {
    const auto param = GetParam();
    
    if (param.float_bits == 32) {
        TestEqualToClearMixer3<float>(param);
    } else {
        TestEqualToClearMixer3<double>(param);
    }
}

INSTANTIATE_TEST_CASE_P(ClearMixerFilter4TestInstance,
                        ClearMixerFilter4Test,
                        ::testing::ValuesIn(TestParamInitializer::GetInstance().test_params));

INSTANTIATE_TEST_CASE_P(ClearMixerFilter4TestRatio1Instance,
                        ClearMixerFilter4TestRatio1,
                        ::testing::ValuesIn(TestParamInitializer::GetInstance().test_ratio1_params));
