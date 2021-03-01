#include "gtest/gtest.h"

#include <iostream>
#include "bakuage/fir_resample_filter.h"

TEST(FirResampleFilter, Upsample) {
    std::vector<float> input(44100);
    for (int i = 0; i < input.size(); i++) {
        input[i] = std::sin(440 * 2 * M_PI * i / 44100);
    }
    
    std::vector<float> output(48000);
    
    bakuage::FirResampleFilter<float>::Config config;
    config.input_sample_rate = 44100;
    config.output_sample_rate = 48000;
    bakuage::FirResampleFilter<float> filter(config);
    
    std::cerr << "input_delay_samples " << filter.input_delay_samples() << std::endl;
    
    int output_i = 0;
    for (int i = 0; i < input.size(); i += filter.input_process_size()) {
        filter.Clock(input.data() + i, output.data() + output_i);
        output_i += filter.output_process_size();
    }
}

TEST(FirResampleFilter, Downsample) {
    std::vector<float> input(48000);
    for (int i = 0; i < input.size(); i++) {
        input[i] = std::sin(440 * 2 * M_PI * i / 48000);
    }
    
    std::vector<float> output(44100);
    
    bakuage::FirResampleFilter<float>::Config config;
    config.input_sample_rate = 48000;
    config.output_sample_rate = 44100;
    bakuage::FirResampleFilter<float> filter(config);
    
    std::cerr << "input_delay_samples " << filter.input_delay_samples() << std::endl;
    
    int output_i = 0;
    for (int i = 0; i < input.size(); i += filter.input_process_size()) {
        filter.Clock(input.data() + i, output.data() + output_i);
        output_i += filter.output_process_size();
    }
}
