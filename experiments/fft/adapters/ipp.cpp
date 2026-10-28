#include "plan.h"
#include <ipp.h>
#include <stdexcept>

#ifdef FFT_DOUBLE
using Native = Ipp64fc;
using RealSpec = IppsDFTSpec_R_64f;
using ComplexSpec = IppsDFTSpec_C_64fc;
#define IPP_REAL(name) name##_64f
#define IPP_COMPLEX(name) name##_64fc
#else
using Native = Ipp32fc;
using RealSpec = IppsDFTSpec_R_32f;
using ComplexSpec = IppsDFTSpec_C_32fc;
#define IPP_REAL(name) name##_32f
#define IPP_COMPLEX(name) name##_32fc
#endif

namespace fft_exp {
namespace {
void check(IppStatus status) {
    if (status != ippStsNoErr) throw std::runtime_error(ippGetStatusString(status));
}
struct IppDelete { void operator()(Ipp8u *p) const { if (p) ippsFree(p); } };
using IppBuffer = std::unique_ptr<Ipp8u, IppDelete>;
IppBuffer allocate(int bytes) {
    IppBuffer p(bytes ? ippsMalloc_8u(bytes) : nullptr);
    if (bytes && !p) throw std::bad_alloc();
    return p;
}
struct IppPlan : Plan {
    IppPlan(int n, bool real) : Plan(n, real), real_time(n), ccs(2 * bins()), time(n), freq(n) {
        int spec_bytes, init_bytes, work_bytes;
        if (real) check(IPP_REAL(ippsDFTGetSize_R)(n, IPP_FFT_NODIV_BY_ANY, ippAlgHintNone,
                                                &spec_bytes, &init_bytes, &work_bytes));
        else check(IPP_COMPLEX(ippsDFTGetSize_C)(n, IPP_FFT_NODIV_BY_ANY, ippAlgHintNone,
                                               &spec_bytes, &init_bytes, &work_bytes));
        spec = allocate(spec_bytes);
        work = allocate(work_bytes);
        auto init = allocate(init_bytes);
        if (real) check(IPP_REAL(ippsDFTInit_R)(n, IPP_FFT_NODIV_BY_ANY, ippAlgHintNone,
                                              real_spec(), init.get()));
        else check(IPP_COMPLEX(ippsDFTInit_C)(n, IPP_FFT_NODIV_BY_ANY, ippAlgHintNone,
                                             complex_spec(), init.get()));
    }
    RealSpec *real_spec() { return reinterpret_cast<RealSpec *>(spec.get()); }
    ComplexSpec *complex_spec() { return reinterpret_cast<ComplexSpec *>(spec.get()); }
    void forward(const std::vector<Complex> &input, std::vector<Complex> &output) override {
        if (real) {
            for (int i = 0; i < n; ++i) real_time[i] = input[i].real();
            check(IPP_REAL(ippsDFTFwd_RToCCS)(real_time.data(), ccs.data(), real_spec(), work.get()));
            for (int k = 0; k < bins(); ++k) output[k] = Complex(ccs[2 * k], ccs[2 * k + 1]);
        } else {
            for (int i = 0; i < n; ++i) time[i] = {input[i].real(), input[i].imag()};
            check(IPP_COMPLEX(ippsDFTFwd_CToC)(time.data(), freq.data(), complex_spec(), work.get()));
            for (int k = 0; k < n; ++k) output[k] = Complex(freq[k].re, freq[k].im);
        }
    }
    void backward(const std::vector<Complex> &input, std::vector<Complex> &output) override {
        if (real) {
            for (int k = 0; k < bins(); ++k) { ccs[2 * k] = input[k].real(); ccs[2 * k + 1] = input[k].imag(); }
            check(IPP_REAL(ippsDFTInv_CCSToR)(ccs.data(), real_time.data(), real_spec(), work.get()));
            for (int i = 0; i < n; ++i) output[i] = Complex(real_time[i], 0);
        } else {
            for (int k = 0; k < n; ++k) freq[k] = {input[k].real(), input[k].imag()};
            check(IPP_COMPLEX(ippsDFTInv_CToC)(freq.data(), time.data(), complex_spec(), work.get()));
            for (int i = 0; i < n; ++i) output[i] = Complex(time[i].re, time[i].im);
        }
    }
    IppBuffer spec, work;
    std::vector<Scalar> real_time, ccs;
    std::vector<Native> time, freq;
};
}
std::unique_ptr<Plan> make_plan(int n, bool real) { return std::make_unique<IppPlan>(n, real); }
std::string backend_details() {
    const auto status = ippInit();
    // IPP supports non-Intel x86 CPUs but reports a dispatch warning there.
    if (status != ippStsNonIntelCpu) check(status);
    const auto *version = ippGetLibVersion();
    return std::string(version->Name) + "; " + version->Version + "; DFT CCS; single thread; init="
        + ippGetStatusString(status);
}
}
