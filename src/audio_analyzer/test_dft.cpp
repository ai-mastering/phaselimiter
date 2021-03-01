#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include "bakuage/memory.h"
#include "bakuage/file_utils.h"
#include "bakuage/dft.h"

void TestDft() {
    const int width = 12345;
    const int spec_len = width / 2 + 1;
    float *fft_input = (float *)bakuage::AlignedMalloc(sizeof(float) * width);
    std::complex<float> *fft_output = (std::complex<float> *)bakuage::AlignedMalloc(sizeof(std::complex<float>) * spec_len);
    bakuage::RealDft<float> dft(width);
    
    for (int i = 0; i < width; i++) {
        fft_input[i] = 0.1 * (i % 13);
    }
    for (int i = 0; i < spec_len; i++) {
        fft_output[i] = 1;
    }
    dft.Forward(fft_input, (float *)fft_output);
    
    if (std::abs(fft_output[0].imag()) > 1e-6) {
        std::cerr << "error fft_output[0] " << fft_output[0].imag() << std::endl;
    }
    if (width % 2 == 0 && std::abs(fft_output[spec_len - 1].imag()) > 1e-6) {
        std::cerr << "error fft_output[spec_len - 1] " << fft_output[spec_len - 1].imag() << std::endl;
    }
    
    dft.Backward((float *)fft_output, fft_input);
    
    for (int i = 0; i < width; i++) {
        const auto normalized = fft_input[i] / width;
        const auto error = normalized - 0.1 * (i % 13);
        if (std::abs(error) > 1e-6) {
            std::cerr << "error " << i << " " << normalized << " " << error << std::endl;
        }
    }
}







