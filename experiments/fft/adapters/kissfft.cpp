#include "plan.h"
#include "kiss_fft.h"
#include "kiss_fftr.h"
#include <cstdlib>
#include <stdexcept>

namespace fft_exp {
namespace {
struct KissPlan : Plan {
    KissPlan(int n, bool real) : Plan(n, real), optimized_real(real && n % 2 == 0), time(n), freq(n), real_time(n) {
        if (optimized_real) {
            rf = kiss_fftr_alloc(n, 0, nullptr, nullptr);
            rb = kiss_fftr_alloc(n, 1, nullptr, nullptr);
            if (!rf || !rb) { std::free(rf); std::free(rb); throw std::bad_alloc(); }
        } else {
            cf = kiss_fft_alloc(n, 0, nullptr, nullptr);
            cb = kiss_fft_alloc(n, 1, nullptr, nullptr);
            if (!cf || !cb) { std::free(cf); std::free(cb); throw std::bad_alloc(); }
        }
    }
    ~KissPlan() override { std::free(rf); std::free(rb); std::free(cf); std::free(cb); }
    void forward(const std::vector<Complex> &input, std::vector<Complex> &output) override {
        if (optimized_real) {
            for (int i = 0; i < n; ++i) real_time[i] = input[i].real();
            kiss_fftr(rf, real_time.data(), freq.data());
        } else {
            for (int i = 0; i < n; ++i) time[i] = {input[i].real(), input[i].imag()};
            kiss_fft(cf, time.data(), freq.data());
        }
        for (int k = 0; k < bins(); ++k) output[k] = Complex(freq[k].r, freq[k].i);
    }
    void backward(const std::vector<Complex> &input, std::vector<Complex> &output) override {
        for (int k = 0; k < bins(); ++k) freq[k] = {input[k].real(), input[k].imag()};
        if (optimized_real) {
            kiss_fftri(rb, freq.data(), real_time.data());
            for (int i = 0; i < n; ++i) output[i] = Complex(real_time[i], 0);
        } else {
            if (real) for (int k = 1; k <= (n - 1) / 2; ++k) freq[n - k] = {freq[k].r, -freq[k].i};
            kiss_fft(cb, freq.data(), time.data());
            for (int i = 0; i < n; ++i) output[i] = Complex(time[i].r, real ? Scalar(0) : time[i].i);
        }
    }
    bool optimized_real;
    kiss_fftr_cfg rf = nullptr, rb = nullptr;
    kiss_fft_cfg cf = nullptr, cb = nullptr;
    std::vector<kiss_fft_cpx> time, freq;
    std::vector<Scalar> real_time;
};
}
std::unique_ptr<Plan> make_plan(int n, bool real) { return std::make_unique<KissPlan>(n, real); }
std::string backend_details() { return "single thread; even real: fftr; odd real: same-library complex FFT"; }
}
