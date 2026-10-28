#include "plan.h"
#include <Accelerate/Accelerate.h>
#include <stdexcept>

namespace fft_exp {
namespace {
#ifdef FFT_DOUBLE
using Setup = FFTSetupD;
using DftSetup = vDSP_DFT_SetupD;
using Split = DSPDoubleSplitComplex;
#define VDSP(name) name##D
#else
using Setup = FFTSetup;
using DftSetup = vDSP_DFT_Setup;
using Split = DSPSplitComplex;
#define VDSP(name) name
#endif

struct AcceleratePlan : Plan {
    AcceleratePlan(int n, bool real) : Plan(n, real), re(n), im(n) {
        if (n >= 4 && (n & (n - 1)) == 0) {
            for (int i = n; i > 1; i >>= 1) ++order;
            fft = VDSP(vDSP_create_fftsetup)(order, kFFTRadix2);
            if (!fft) throw std::bad_alloc();
        } else if (real) {
            df = VDSP(vDSP_DFT_zrop_CreateSetup)(nullptr, n, vDSP_DFT_FORWARD);
            db = VDSP(vDSP_DFT_zrop_CreateSetup)(nullptr, n, vDSP_DFT_INVERSE);
        } else {
            df = VDSP(vDSP_DFT_zop_CreateSetup)(nullptr, n, vDSP_DFT_FORWARD);
            db = VDSP(vDSP_DFT_zop_CreateSetup)(nullptr, n, vDSP_DFT_INVERSE);
        }
        if (!fft && (!df || !db)) {
            if (df) VDSP(vDSP_DFT_DestroySetup)(df);
            if (db) VDSP(vDSP_DFT_DestroySetup)(db);
            throw std::runtime_error("Accelerate failed to create a plan for a documented supported length");
        }
    }
    ~AcceleratePlan() override {
        if (fft) VDSP(vDSP_destroy_fftsetup)(fft);
        if (df) VDSP(vDSP_DFT_DestroySetup)(df);
        if (db) VDSP(vDSP_DFT_DestroySetup)(db);
    }
    void execute(bool inverse) {
        if (fft) {
            Split split{re.data(), im.data()};
            if (real) VDSP(vDSP_fft_zrip)(fft, &split, 1, order, inverse ? FFT_INVERSE : FFT_FORWARD);
            else VDSP(vDSP_fft_zip)(fft, &split, 1, order, inverse ? FFT_INVERSE : FFT_FORWARD);
        } else {
            VDSP(vDSP_DFT_Execute)(inverse ? db : df, re.data(), im.data(), re.data(), im.data());
        }
    }
    void forward(const std::vector<Complex> &input, std::vector<Complex> &output) override {
        if (real) {
            // vDSP's real forward transform is twice the mathematical DFT.
            for (int i = 0; i < n / 2; ++i) {
                re[i] = input[2 * i].real() * Scalar(.5);
                im[i] = input[2 * i + 1].real() * Scalar(.5);
            }
            execute(false);
            output[0] = Complex(re[0], 0);
            output[n / 2] = Complex(im[0], 0);
            for (int k = 1; k < n / 2; ++k) output[k] = Complex(re[k], im[k]);
        } else {
            for (int i = 0; i < n; ++i) { re[i] = input[i].real(); im[i] = input[i].imag(); }
            execute(false);
            for (int k = 0; k < n; ++k) output[k] = Complex(re[k], im[k]);
        }
    }
    void backward(const std::vector<Complex> &input, std::vector<Complex> &output) override {
        if (real) {
            re[0] = input[0].real();
            im[0] = input[n / 2].real();
            for (int k = 1; k < n / 2; ++k) { re[k] = input[k].real(); im[k] = input[k].imag(); }
            execute(true);
            for (int i = 0; i < n / 2; ++i) {
                output[2 * i] = Complex(re[i], 0);
                output[2 * i + 1] = Complex(im[i], 0);
            }
        } else {
            for (int k = 0; k < n; ++k) { re[k] = input[k].real(); im[k] = input[k].imag(); }
            execute(true);
            for (int i = 0; i < n; ++i) output[i] = Complex(re[i], im[i]);
        }
    }
    int order = 0;
    Setup fft = nullptr;
    DftSetup df = nullptr, db = nullptr;
    std::vector<Scalar> re, im;
};
}
std::unique_ptr<Plan> make_plan(int n, bool real) {
    if (n >= 4 && (n & (n - 1)) == 0) return std::make_unique<AcceleratePlan>(n, real);
    // Documented split-DFT lengths. Do not confuse allocation/setup failure
    // for a supported length with an unsupported size and silently skip it.
    for (int factor : {3, 5, 15}) {
        if (n % factor) continue;
        const int power = n / factor;
        if (power >= (real ? 16 : 8) && (power & (power - 1)) == 0)
            return std::make_unique<AcceleratePlan>(n, real);
    }
    return nullptr;
}
std::string backend_details() { return "radix-2 FFT; otherwise native split DFT; unsupported lengths explicitly skipped; no fallback"; }
}
