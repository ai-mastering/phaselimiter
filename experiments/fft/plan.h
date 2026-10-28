#pragma once
#include <complex>
#include <memory>
#include <string>
#include <vector>

namespace fft_exp {
#ifdef FFT_DOUBLE
using Scalar = double;
constexpr const char *precision = "double";
#else
using Scalar = float;
constexpr const char *precision = "float";
#endif
using Complex = std::complex<Scalar>;

// Unnormalized transforms. Real spectra contain floor(N/2)+1 bins.
// Canonical buffers belong to the caller; plans own all reusable workspaces.
// Real time-domain values use Complex with an exactly zero imaginary part.
struct Plan {
    Plan(int size, bool is_real) : n(size), real(is_real) {}
    virtual ~Plan() = default;
    virtual void forward(const std::vector<Complex> &input, std::vector<Complex> &spectrum) = 0;
    virtual void backward(const std::vector<Complex> &spectrum, std::vector<Complex> &output) = 0;
    int bins() const { return real ? n / 2 + 1 : n; }
    int n;
    bool real;
};

// A null plan is an explicitly unsupported size, never a fallback to another library.
std::unique_ptr<Plan> make_plan(int n, bool real);
std::string backend_details();
}
