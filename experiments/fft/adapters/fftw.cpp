#include "plan.h"
#include <fftw3.h>

#ifdef FFT_DOUBLE
#define FFTW_FN(name) fftw_##name
#else
#define FFTW_FN(name) fftwf_##name
#endif

namespace fft_exp {
namespace {
struct FftwPlan : Plan {
    FftwPlan(int n, bool real) : Plan(n, real), time(2 * n), freq(2 * bins()) {
        const unsigned flags = std::string(FFT_FFTW_PLANNER) == "measure" ? FFTW_MEASURE : FFTW_ESTIMATE;
        auto *t = reinterpret_cast<FFTW_FN(complex) *>(time.data());
        auto *f = reinterpret_cast<FFTW_FN(complex) *>(freq.data());
        if (real) {
            forward_plan = FFTW_FN(plan_dft_r2c_1d)(n, time.data(), f, flags);
            backward_plan = FFTW_FN(plan_dft_c2r_1d)(n, f, time.data(), flags);
        } else {
            forward_plan = FFTW_FN(plan_dft_1d)(n, t, f, FFTW_FORWARD, flags);
            backward_plan = FFTW_FN(plan_dft_1d)(n, f, t, FFTW_BACKWARD, flags);
        }
        if (!forward_plan || !backward_plan) {
            if (forward_plan) FFTW_FN(destroy_plan)(forward_plan);
            if (backward_plan) FFTW_FN(destroy_plan)(backward_plan);
            throw std::bad_alloc();
        }
    }
    ~FftwPlan() override { FFTW_FN(destroy_plan)(forward_plan); FFTW_FN(destroy_plan)(backward_plan); }
    void forward(const std::vector<Complex> &input, std::vector<Complex> &output) override {
        for (int i = 0; i < n; ++i) {
            if (real) time[i] = input[i].real();
            else { time[2 * i] = input[i].real(); time[2 * i + 1] = input[i].imag(); }
        }
        FFTW_FN(execute)(forward_plan);
        for (int k = 0; k < bins(); ++k) output[k] = Complex(freq[2 * k], freq[2 * k + 1]);
    }
    void backward(const std::vector<Complex> &input, std::vector<Complex> &output) override {
        for (int k = 0; k < bins(); ++k) { freq[2 * k] = input[k].real(); freq[2 * k + 1] = input[k].imag(); }
        FFTW_FN(execute)(backward_plan);
        for (int i = 0; i < n; ++i)
            output[i] = real ? Complex(time[i], 0) : Complex(time[2 * i], time[2 * i + 1]);
    }
    std::vector<Scalar> time, freq;
    FFTW_FN(plan) forward_plan = nullptr, backward_plan = nullptr;
};
}
std::unique_ptr<Plan> make_plan(int n, bool real) { return std::make_unique<FftwPlan>(n, real); }
std::string backend_details() { return std::string(FFTW_FN(version)) + "; planner=" + FFT_FFTW_PLANNER + "; single thread"; }
}
