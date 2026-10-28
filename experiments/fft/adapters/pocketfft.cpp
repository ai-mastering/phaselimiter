#include "plan.h"
#include "pocketfft_hdronly.h"

namespace fft_exp {
namespace {
struct PocketPlan : Plan {
    PocketPlan(int n, bool real) : Plan(n, real), time(n), frequency(n), real_work(n) {
        if (real) rplan = std::make_unique<pocketfft::detail::pocketfft_r<Scalar>>(n);
        else cplan = std::make_unique<pocketfft::detail::pocketfft_c<Scalar>>(n);
    }
    void forward(const std::vector<Complex> &input, std::vector<Complex> &output) override {
        if (real) {
            for (int i = 0; i < n; ++i) real_work[i] = input[i].real();
            rplan->exec(real_work.data(), Scalar(1), true);
            output[0] = Complex(real_work[0], 0);
            for (int k = 1; k <= (n - 1) / 2; ++k)
                output[k] = Complex(real_work[2 * k - 1], real_work[2 * k]);
            if (n % 2 == 0) output[n / 2] = Complex(real_work[n - 1], 0);
        } else {
            for (int i = 0; i < n; ++i) time[i].Set(input[i].real(), input[i].imag());
            cplan->exec(time.data(), Scalar(1), true);
            for (int i = 0; i < n; ++i) output[i] = Complex(time[i].r, time[i].i);
        }
    }
    void backward(const std::vector<Complex> &input, std::vector<Complex> &output) override {
        if (real) {
            real_work[0] = input[0].real();
            for (int k = 1; k <= (n - 1) / 2; ++k) {
                real_work[2 * k - 1] = input[k].real();
                real_work[2 * k] = input[k].imag();
            }
            if (n % 2 == 0) real_work[n - 1] = input[n / 2].real();
            rplan->exec(real_work.data(), Scalar(1), false);
            for (int i = 0; i < n; ++i) output[i] = Complex(real_work[i], 0);
        } else {
            for (int i = 0; i < n; ++i) frequency[i].Set(input[i].real(), input[i].imag());
            cplan->exec(frequency.data(), Scalar(1), false);
            for (int i = 0; i < n; ++i) output[i] = Complex(frequency[i].r, frequency[i].i);
        }
    }
    std::unique_ptr<pocketfft::detail::pocketfft_r<Scalar>> rplan;
    std::unique_ptr<pocketfft::detail::pocketfft_c<Scalar>> cplan;
    std::vector<pocketfft::detail::cmplx<Scalar>> time, frequency;
    std::vector<Scalar> real_work;
};
}
std::unique_ptr<Plan> make_plan(int n, bool real) { return std::make_unique<PocketPlan>(n, real); }
std::string backend_details() { return "reusable 1D plans; single thread; pinned detail API"; }
}
