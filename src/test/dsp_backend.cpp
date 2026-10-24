#include <gtest/gtest.h>
#include <cmath>
#include <complex>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>
#include "bakuage/dft.h"
#include "bakuage/fir_filter.h"
#include "bakuage/fir_filter4.h"
#include "bakuage/vector_math.h"
#include "phase_limiter/GradCalculator.h"

namespace {
template <class T> void CheckRealDft() {
    const double pi = std::acos(-1.0);
    const double tolerance = std::is_same<T, float>::value ? 3e-5 : 1e-11;
    for (int n: {5, 6, 7, 8, 15, 16, 32}) {
        SCOPED_TRACE(n);
        std::vector<T> input(n), ccs(2 * (n / 2 + 1)), restored(n), packed(n), perm(n);
        for (int i = 0; i < n; ++i) input[i] = .17 * i + std::sin(.7 * i);
        std::vector<std::complex<double>> expected(n / 2 + 1);
        for (int k = 0; k <= n / 2; ++k)
            for (int j = 0; j < n; ++j)
                expected[k] += double(input[j]) * std::polar(1.0, -2 * pi * j * k / n);
        bakuage::RealDft<T> fft(n, true);
        bakuage::FftMemoryBuffer work(fft.work_size());
        fft.Forward(input.data(), ccs.data(), work.data());
        for (int k = 0; k <= n / 2; ++k) {
            EXPECT_NEAR(expected[k].real(), ccs[2 * k], tolerance * n);
            EXPECT_NEAR(expected[k].imag(), ccs[2 * k + 1], tolerance * n);
        }
        fft.Backward(ccs.data(), restored.data(), work.data());
        for (int i = 0; i < n; ++i) EXPECT_NEAR(input[i], restored[i] / n, tolerance);
        fft.ForwardPack(input.data(), packed.data(), work.data());
        fft.ForwardPerm(input.data(), perm.data(), work.data());
        EXPECT_NEAR(ccs[0], packed[0], tolerance * n);
        EXPECT_NEAR(ccs[0], perm[0], tolerance * n);
        for (int k = 1; k <= (n - 1) / 2; ++k) {
            EXPECT_NEAR(ccs[2 * k], packed[2 * k - 1], tolerance * n);
            EXPECT_NEAR(ccs[2 * k + 1], packed[2 * k], tolerance * n);
            const int offset = n % 2 ? 2 * k - 1 : 2 * k;
            EXPECT_NEAR(ccs[2 * k], perm[offset], tolerance * n);
            EXPECT_NEAR(ccs[2 * k + 1], perm[offset + 1], tolerance * n);
        }
        if (n % 2 == 0) {
            EXPECT_NEAR(ccs[n], packed[n - 1], tolerance * n);
            EXPECT_NEAR(ccs[n], perm[1], tolerance * n);
        }
        fft.BackwardPack(packed.data(), restored.data(), work.data());
        for (int i = 0; i < n; ++i) EXPECT_NEAR(input[i], restored[i] / n, tolerance);
        fft.BackwardPerm(perm.data(), restored.data(), work.data());
        for (int i = 0; i < n; ++i) EXPECT_NEAR(input[i], restored[i] / n, tolerance);
        // The legacy in-place fast path supports power-of-two sizes.
        if ((n & (n - 1)) == 0) {
            auto inplace = input;
            fft.ForwardPerm(inplace.data(), inplace.data(), work.data());
            for (int i = 0; i < n; ++i) EXPECT_NEAR(perm[i], inplace[i], tolerance * n);
            fft.BackwardPerm(inplace.data(), inplace.data(), work.data());
            for (int i = 0; i < n; ++i) EXPECT_NEAR(input[i], inplace[i] / n, tolerance);
            inplace = input;
            fft.ForwardPack(inplace.data(), inplace.data(), work.data());
            for (int i = 0; i < n; ++i) EXPECT_NEAR(packed[i], inplace[i], tolerance * n);
            fft.BackwardPack(inplace.data(), inplace.data(), work.data());
            for (int i = 0; i < n; ++i) EXPECT_NEAR(input[i], inplace[i] / n, tolerance);
        }
    }
}

template <class T> void CheckComplexDft() {
    for (int n: {5, 8, 16}) {
        std::vector<std::complex<T>> input(n), output(n), restored(n);
        for (int i = 0; i < n; ++i) input[i] = {T(std::sin(.7 * i)), T(.2 * i)};
        bakuage::Dft<T> fft(n);
        fft.Forward(reinterpret_cast<const T *>(input.data()), reinterpret_cast<T *>(output.data()));
        for (int k = 0; k < n; ++k) {
            std::complex<double> expected = 0;
            for (int j = 0; j < n; ++j)
                expected += std::complex<double>(input[j]) * std::polar(1.0, -2 * std::acos(-1.0) * j * k / n);
            EXPECT_NEAR(0, std::abs(expected - std::complex<double>(output[k])), 2e-5 * n);
        }
        fft.Backward(reinterpret_cast<const T *>(output.data()), reinterpret_cast<T *>(restored.data()));
        for (int i = 0; i < n; ++i) EXPECT_NEAR(0, std::abs(restored[i] / T(n) - input[i]), 2e-5);
    }
}

template <class T> void CheckMultirateFir() {
    for (auto factors: {std::make_pair(1, 1), {2, 1}, {1, 3}, {3, 2}, {4, 3}}) {
        const int up = factors.first, down = factors.second;
        SCOPED_TRACE(::testing::Message() << up << '/' << down);
        std::vector<T> taps(17), input(36), expected(input.size() / down * up), output(expected.size()), split(expected.size());
        for (int k = 0; k < 17; ++k) taps[k] = .03 * (k + 1);
        for (int i = 0; i < 36; ++i) input[i] = std::sin(.37 * i) + .02 * i;
        // Reference: insert zeros, apply a causal FIR, then sample every down-th point.
        for (int j = 0; j < expected.size(); ++j) {
            const int time = j * down;
            for (int k = 0; k < taps.size(); ++k)
                if (time >= k && (time - k) % up == 0)
                    expected[j] += taps[k] * input[(time - k) / up];
        }
        bakuage::FirFilter4<T> whole(taps.begin(), taps.end(), up, down);
        bakuage::FirFilter4<T> blocks(taps.begin(), taps.end(), up, down);
        whole.Clock(input.data(), input.data() + input.size(), output.data());
        for (int i = 0; i < 36; i += 6)
            blocks.Clock(input.data() + i, input.data() + i + 6, split.data() + i / down * up);
        for (int i = 0; i < expected.size(); ++i) {
            EXPECT_NEAR(expected[i], output[i], 2e-6);
            EXPECT_NEAR(expected[i], split[i], 2e-6);
        }
    }
}
}

TEST(DspBackend, RealDftFloat) { CheckRealDft<float>(); }
TEST(DspBackend, RealDftDouble) { CheckRealDft<double>(); }
TEST(DspBackend, ComplexDftFloat) { CheckComplexDft<float>(); }
TEST(DspBackend, ComplexDftDouble) { CheckComplexDft<double>(); }
TEST(DspBackend, MultirateFirFloat) { CheckMultirateFir<float>(); }
TEST(DspBackend, MultirateFirDouble) { CheckMultirateFir<double>(); }

TEST(DspBackend, Dct2DOrthonormal) {
    const int width = 8, height = 4;
    std::vector<float> input(width * height), output(input.size());
    for (int i = 0; i < input.size(); ++i) input[i] = std::sin(.7 * i) + .01 * i;
    bakuage::Dct2D<float> dct(width, height);
    dct.Forward(input.data(), output.data());
    const double pi = std::acos(-1.0);
    for (int v = 0; v < height; ++v) for (int u = 0; u < width; ++u) {
        double expected = 0;
        for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x)
            expected += input[y * width + x] * std::cos(pi * (x + .5) * u / width) * std::cos(pi * (y + .5) * v / height);
        expected *= std::sqrt((u ? 2.0 : 1.0) / width) * std::sqrt((v ? 2.0 : 1.0) / height);
        EXPECT_NEAR(expected, output[v * width + u], 2e-5);
    }
}

TEST(DspBackend, Complex2DUnnormalizedRoundTrip) {
    const int width = 8, height = 4;
    std::vector<float> input(2 * width * height), spectrum(input.size()), output(input.size());
    for (int i = 0; i < input.size(); ++i) input[i] = std::sin(.7 * i);
    bakuage::Dft2D<float> fft(width, height);
    fft.Forward(input.data(), spectrum.data());
    fft.Backward(spectrum.data(), output.data());
    for (int i = 0; i < input.size(); ++i) EXPECT_NEAR(input[i], output[i] / (width * height), 2e-6);
}

TEST(DspBackend, ProxSpecialValues) {
    using Simd = simdpp::float32x8;
    alignas(64) float input[8] = {std::numeric_limits<float>::quiet_NaN(), INFINITY, -INFINITY, -2, 2, -.5, .5, 0};
    alignas(64) float output[8];
    const Simd x = simdpp::load(input);
    simdpp::store(output, phase_limiter::impl::ProxOperator(x));
    const float expected[8] = {1, 1, -1, -1, 1, -.5, .5, 0};
    for (int i = 0; i < 8; ++i) EXPECT_EQ(expected[i], output[i]);
}

TEST(DspBackend, ShortFirAndTail) {
    for (int n: {1, 3, 4, 7, 8, 17}) {
        std::vector<float> taps(n), history(n, 0);
        for (int i = 0; i < n; ++i) taps[i] = .1 * (i + 1);
        bakuage::FirFilter<float> filter(taps.begin(), taps.end());
        for (int i = 0; i < 40; ++i) {
            std::move_backward(history.begin(), history.end() - 1, history.end());
            history[0] = std::sin(.7 * i);
            double expected = 0;
            for (int j = 0; j < n; ++j) expected += history[j] * taps[j];
            EXPECT_NEAR(expected, filter.Clock(history[0]), 2e-5);
        }
    }
}

namespace {
template <class T> void CheckVectors() {
    const double tolerance = std::is_same<T, float>::value ? 2e-5 : 1e-11;
    for (int n: {1, 3, 8, 17, 129}) {
        SCOPED_TRACE(n);
        std::vector<T> a(n + 1), b(n + 1), output(n + 2, T(-99));
        T *x = a.data() + 1, *y = b.data() + 1, *dest = output.data() + 1;
        for (int i = 0; i < n; ++i) { x[i] = .7 + .01 * i; y[i] = -.3 + .007 * i; }
        const T scalar = T(.37);
        bakuage::VectorMul(x, y, dest, n);
        for (int i = 0; i < n; ++i) EXPECT_NEAR(x[i] * y[i], dest[i], tolerance);
        bakuage::VectorMadInplace(x, y, dest, n);
        for (int i = 0; i < n; ++i) EXPECT_NEAR(2 * x[i] * y[i], dest[i], tolerance);
        bakuage::VectorAdd(x, y, dest, n);
        bakuage::VectorMulConstantInplace(scalar, dest, n);
        bakuage::VectorMadConstantInplace(x, scalar, dest, n);
        for (int i = 0; i < n; ++i) EXPECT_NEAR(scalar * (2 * x[i] + y[i]), dest[i], tolerance);
        bakuage::VectorMulConstant(x, scalar, dest, n);
        bakuage::VectorMulInplace(y, dest, n);
        bakuage::VectorDivInplace(x, dest, n);
        for (int i = 0; i < n; ++i) EXPECT_NEAR(scalar * y[i], dest[i], tolerance);
        bakuage::VectorPowConstant(x, T(1.7), dest, n);
        for (int i = 0; i < n; ++i) EXPECT_NEAR(std::pow(x[i], T(1.7)), dest[i], tolerance);
        bakuage::VectorInvInplace(dest, n);
        bakuage::VectorSqrtInplace(dest, n);
        for (int i = 0; i < n; ++i) EXPECT_NEAR(std::pow(x[i], T(-.85)), dest[i], tolerance);
        double sum = 0, dot = 0, l2 = 0, l1diff = 0, l2diff = 0, infdiff = 0, inf = 0;
        for (int i = 0; i < n; ++i) {
            sum += x[i]; dot += double(x[i]) * y[i]; l2 += double(x[i]) * x[i];
            const double delta = double(x[i]) - y[i];
            l1diff += std::abs(delta); l2diff += delta * delta;
            infdiff = std::max(infdiff, std::abs(delta)); inf = std::max(inf, std::abs(double(x[i])));
        }
        EXPECT_NEAR(sum, bakuage::VectorSum(x, n), tolerance * n);
        EXPECT_NEAR(dot, bakuage::VectorDot(x, y, n), tolerance * n);
        EXPECT_NEAR(std::sqrt(l2), bakuage::VectorL2(x, n), tolerance * n);
        EXPECT_NEAR(inf, bakuage::VectorLInf(x, n), tolerance);
        EXPECT_NEAR(l1diff, bakuage::VectorNormDiffL1(x, y, n), tolerance * n);
        EXPECT_NEAR(std::sqrt(l2diff), bakuage::VectorNormDiffL2(x, y, n), tolerance * n);
        EXPECT_NEAR(infdiff, bakuage::VectorNormDiffInf(x, y, n), tolerance);
        EXPECT_EQ(T(-99), output.front()); EXPECT_EQ(T(-99), output.back());
    }
}
template <class T> void CheckComplexVectors() {
    using C = std::complex<T>;
    const int n = 17;
    const double tolerance = std::is_same<T, float>::value ? 2e-5 : 1e-11;
    std::vector<C> x(n), y(n), output(n), expected(n);
    std::vector<T> real(n), imag(n), norm(n);
    for (int i = 0; i < n; ++i) { real[i] = .03 * i; imag[i] = .2 - .05 * i; x[i] = {real[i], imag[i]}; y[i] = {T(.7), T(.01 * i)}; }
    bakuage::VectorRealToComplex(real.data(), imag.data(), output.data(), n);
    EXPECT_EQ(x, output);
    bakuage::VectorComplexToReal(x.data(), real.data(), imag.data(), n);
    for (int i = 0; i < n; ++i) { EXPECT_EQ(x[i].real(), real[i]); EXPECT_EQ(x[i].imag(), imag[i]); }
    bakuage::VectorNorm(x.data(), norm.data(), n);
    for (int i = 0; i < n; ++i) EXPECT_NEAR(std::norm(x[i]), norm[i], tolerance);
    bakuage::VectorMul(x.data(), y.data(), output.data(), n);
    for (int i = 0; i < n; ++i) EXPECT_NEAR(0, std::abs(x[i] * y[i] - output[i]), tolerance);
    output = y;
    bakuage::VectorMulInplace(x.data(), output.data(), n);
    for (int i = 0; i < n; ++i) EXPECT_NEAR(0, std::abs(x[i] * y[i] - output[i]), tolerance);
    bakuage::VectorMulConj(x.data(), y.data(), output.data(), n);
    for (int i = 0; i < n; ++i) EXPECT_NEAR(0, std::abs(x[i] * std::conj(y[i]) - output[i]), tolerance);
    output = y;
    bakuage::VectorMadInplace(x.data(), y.data(), output.data(), n);
    for (int i = 0; i < n; ++i) EXPECT_NEAR(0, std::abs(y[i] + x[i] * y[i] - output[i]), tolerance);
    output = y;
    const C scalar(T(.7), T(-.2));
    bakuage::VectorMulConstantInplace(scalar, output.data(), n);
    for (int i = 0; i < n; ++i) EXPECT_NEAR(0, std::abs(scalar * y[i] - output[i]), tolerance);
    bakuage::VectorMul(real.data(), y.data(), output.data(), n);
    for (int i = 0; i < n; ++i) EXPECT_NEAR(0, std::abs(real[i] * y[i] - output[i]), tolerance);
    output = y;
    bakuage::VectorMulInplace(real.data(), output.data(), n);
    for (int i = 0; i < n; ++i) EXPECT_NEAR(0, std::abs(real[i] * y[i] - output[i]), tolerance);
    output = y;
    bakuage::VectorMulPermInplace(x.data(), output.data(), n);
    EXPECT_NEAR(x[0].real() * y[0].real(), output[0].real(), tolerance);
    EXPECT_NEAR(x[0].imag() * y[0].imag(), output[0].imag(), tolerance);
    for (int i = 1; i < n; ++i) EXPECT_NEAR(0, std::abs(x[i] * y[i] - output[i]), tolerance);
}
template <class T> void CheckResamplingAndConvolution() {
    const T input[] = {1, 2, -3, 4, 5};
    std::vector<T> up(15), down(5), held(15);
    bakuage::VectorInterpolate(input, 5, up.data(), 3);
    bakuage::VectorDecimate(up.data(), up.size(), down.data(), 3);
    bakuage::VectorInterpolateHold(input, 5, held.data(), 3);
    for (int i = 0; i < 15; ++i) { EXPECT_EQ(i % 3 ? 0 : input[i / 3], up[i]); EXPECT_EQ(input[i / 3], held[i]); }
    for (int i = 0; i < 5; ++i) EXPECT_EQ(input[i], down[i]);
    const T taps[] = {T(.2), T(-.7), T(.3)};
    T convolution[7];
    bakuage::VectorConvolve(input, 5, taps, 3, convolution);
    for (int i = 0; i < 7; ++i) {
        double expected = 0;
        for (int j = 0; j < 3; ++j) if (i >= j && i - j < 5) expected += double(taps[j]) * input[i - j];
        EXPECT_NEAR(expected, convolution[i], 2e-6);
    }
}
}
TEST(DspBackend, VectorsFloat) { CheckVectors<float>(); }
TEST(DspBackend, VectorsDouble) { CheckVectors<double>(); }
TEST(DspBackend, ComplexVectorsFloat) { CheckComplexVectors<float>(); }
TEST(DspBackend, ComplexVectorsDouble) { CheckComplexVectors<double>(); }
TEST(DspBackend, ResamplingAndConvolutionFloat) { CheckResamplingAndConvolution<float>(); }
TEST(DspBackend, ResamplingAndConvolutionDouble) { CheckResamplingAndConvolution<double>(); }

TEST(DspBackend, HalfConversion) {
    const float input[] = {0, -0.0f, 1, -2, 65504, 0.00006103515625f, 0.000000059604644775390625f};
    const unsigned short bits[] = {0x0000, 0x8000, 0x3c00, 0xc000, 0x7bff, 0x0400, 0x0001};
    bakuage::Float16 half[7];
    float restored[7];
    bakuage::VectorConvert(input, half, 7);
    bakuage::VectorConvert(half, restored, 7);
    for (int i = 0; i < 7; ++i) {
        EXPECT_EQ(bits[i], static_cast<unsigned short>(half[i].as_int));
        EXPECT_EQ(input[i], restored[i]);
        EXPECT_EQ(std::signbit(input[i]), std::signbit(restored[i]));
    }
}
