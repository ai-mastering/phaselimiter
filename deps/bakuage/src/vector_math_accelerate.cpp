#include "bakuage/vector_math.h"
#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include "bakuage/memory.h"

namespace {
// Interleaved std::complex storage can be viewed as two strided vDSP planes.
// Accelerate's split-complex type has mutable fields even for const inputs.
DSPSplitComplex Split(const std::complex<float> *values) {
    auto *data = reinterpret_cast<float *>(const_cast<std::complex<float> *>(values));
    return {data, data + 1};
}
DSPDoubleSplitComplex Split(const std::complex<double> *values) {
    auto *data = reinterpret_cast<double *>(const_cast<std::complex<double> *>(values));
    return {data, data + 1};
}
}

namespace bakuage {

template <>
void VectorMulConstantInplace<float, float>(const float &c, float *output, int n) {
    if (n <= 0) return;
    const float scalar = static_cast<float>(c);
    vDSP_vsmul(output, 1, &scalar, output, 1, n);
}

template <>
void VectorMulConstantInplace<double, float>(const double &c, float *output, int n) {
    if (n <= 0) return;
    const float scalar = static_cast<float>(c);
    vDSP_vsmul(output, 1, &scalar, output, 1, n);
}

template <>
void VectorMulConstantInplace<double, double>(const double &c, double *output, int n) {
    if (n <= 0) return;
    const double scalar = static_cast<double>(c);
    vDSP_vsmulD(output, 1, &scalar, output, 1, n);
}

template <>
void VectorMulConstantInplace<float, double>(const float &c, double *output, int n) {
    if (n <= 0) return;
    const double scalar = static_cast<double>(c);
    vDSP_vsmulD(output, 1, &scalar, output, 1, n);
}

template <>
void VectorMulConstantInplace<float, std::complex<float>>(const float &c, std::complex<float> *output, int n) {
    if (n <= 0) return;
    VectorMulConstantInplace(static_cast<float>(c), reinterpret_cast<float *>(output), 2 * n);
}

template <>
void VectorMulConstantInplace<double, std::complex<float>>(const double &c, std::complex<float> *output, int n) {
    if (n <= 0) return;
    VectorMulConstantInplace(static_cast<float>(c), reinterpret_cast<float *>(output), 2 * n);
}

template <>
void VectorMulConstantInplace<double, std::complex<double>>(const double &c, std::complex<double> *output, int n) {
    if (n <= 0) return;
    VectorMulConstantInplace(static_cast<double>(c), reinterpret_cast<double *>(output), 2 * n);
}

template <>
void VectorMulConstantInplace<std::complex<float>, std::complex<float>>(const std::complex<float> &c, std::complex<float> *output, int n) {
    if (n <= 0) return;
    const auto scalar = Split(&c), dest = Split(output);
    vDSP_zvzsml(&dest, 2, &scalar, &dest, 2, n);
}

template <>
void VectorMulConstantInplace<std::complex<double>, std::complex<double>>(const std::complex<double> &c, std::complex<double> *output, int n) {
    if (n <= 0) return;
    const auto scalar = Split(&c), dest = Split(output);
    vDSP_zvzsmlD(&dest, 2, &scalar, &dest, 2, n);
}

template <>
void VectorMulConstant<float>(const float *x, const float &c, float *output, int n) {
    if (n <= 0) return;
    vDSP_vsmul(x, 1, &c, output, 1, n);
}

template <>
void VectorMulConstant<double>(const double *x, const double &c, double *output, int n) {
    if (n <= 0) return;
    vDSP_vsmulD(x, 1, &c, output, 1, n);
}

template <>
void VectorMulInplace<float, float>(const float *x, float *output, int n) {
    if (n <= 0) return;
    vDSP_vmul(x, 1, output, 1, output, 1, n);
}

template <>
void VectorMulInplace<double, double>(const double *x, double *output, int n) {
    if (n <= 0) return;
    vDSP_vmulD(x, 1, output, 1, output, 1, n);
}

template <>
void VectorMulInplace<float, std::complex<float>>(const float *x, std::complex<float> *output, int n) {
    if (n <= 0) return;
    const auto src = Split(output), dest = Split(output);
    vDSP_zrvmul(&src, 2, x, 1, &dest, 2, n);
}

template <>
void VectorMulInplace<double, std::complex<double>>(const double *x, std::complex<double> *output, int n) {
    if (n <= 0) return;
    const auto src = Split(output), dest = Split(output);
    vDSP_zrvmulD(&src, 2, x, 1, &dest, 2, n);
}

template <>
void VectorMulInplace<std::complex<float>, std::complex<float>>(const std::complex<float> *x, std::complex<float> *output, int n) {
    if (n <= 0) return;
    const auto a = Split(x);
    const auto b = Split(output);
    const auto dest = Split(output);
    vDSP_zvmul(&a, 2, &b, 2, &dest, 2, n, 1);
}

template <>
void VectorMulInplace<std::complex<double>, std::complex<double>>(const std::complex<double> *x, std::complex<double> *output, int n) {
    if (n <= 0) return;
    const auto a = Split(x);
    const auto b = Split(output);
    const auto dest = Split(output);
    vDSP_zvmulD(&a, 2, &b, 2, &dest, 2, n, 1);
}

template <>
void VectorMul<float, float>(const float *x, const float *y, float *output, int n) {
    if (n <= 0) return;
    vDSP_vmul(x, 1, y, 1, output, 1, n);
}

template <>
void VectorMul<double, double>(const double *x, const double *y, double *output, int n) {
    if (n <= 0) return;
    vDSP_vmulD(x, 1, y, 1, output, 1, n);
}

template <>
void VectorMul<float, std::complex<float>>(const float *x, const std::complex<float> *y, std::complex<float> *output, int n) {
    if (n <= 0) return;
    const auto src = Split(y), dest = Split(output);
    vDSP_zrvmul(&src, 2, x, 1, &dest, 2, n);
}

template <>
void VectorMul<std::complex<float>, std::complex<float>>(const std::complex<float> *x, const std::complex<float> *y, std::complex<float> *output, int n) {
    if (n <= 0) return;
    const auto a = Split(x);
    const auto b = Split(y);
    const auto dest = Split(output);
    vDSP_zvmul(&a, 2, &b, 2, &dest, 2, n, 1);
}

template <>
void VectorMul<double, std::complex<double>>(const double *x, const std::complex<double> *y, std::complex<double> *output, int n) {
    if (n <= 0) return;
    const auto src = Split(y), dest = Split(output);
    vDSP_zrvmulD(&src, 2, x, 1, &dest, 2, n);
}

template <>
void VectorMul<std::complex<double>, std::complex<double>>(const std::complex<double> *x, const std::complex<double> *y, std::complex<double> *output, int n) {
    if (n <= 0) return;
    const auto a = Split(x);
    const auto b = Split(y);
    const auto dest = Split(output);
    vDSP_zvmulD(&a, 2, &b, 2, &dest, 2, n, 1);
}

template <>
void VectorMulPermInplace<std::complex<float>>(const std::complex<float> *x, std::complex<float> *output, int n) {
    if (n <= 0) return;
    auto *dest = reinterpret_cast<float *>(output);
    const auto *src = reinterpret_cast<const float *>(x);
    const float dc = dest[0] * src[0], nyquist = dest[1] * src[1];
    for (int i = 1; i < n; ++i) output[i] *= x[i];
    dest[0] = dc;
    dest[1] = nyquist;
}

template <>
void VectorMulPermInplace<std::complex<double>>(const std::complex<double> *x, std::complex<double> *output, int n) {
    if (n <= 0) return;
    auto *dest = reinterpret_cast<double *>(output);
    const auto *src = reinterpret_cast<const double *>(x);
    const double dc = dest[0] * src[0], nyquist = dest[1] * src[1];
    for (int i = 1; i < n; ++i) output[i] *= x[i];
    dest[0] = dc;
    dest[1] = nyquist;
}

template <>
void VectorMulConj<std::complex<float>>(const std::complex<float> *x, const std::complex<float> *y, std::complex<float> *output, int n) {
    if (n <= 0) return;
    const auto a = Split(y), b = Split(x), dest = Split(output);
    vDSP_zvmul(&a, 2, &b, 2, &dest, 2, n, -1);
}

template <>
void VectorMulConj<std::complex<double>>(const std::complex<double> *x, const std::complex<double> *y, std::complex<double> *output, int n) {
    if (n <= 0) return;
    const auto a = Split(y), b = Split(x), dest = Split(output);
    vDSP_zvmulD(&a, 2, &b, 2, &dest, 2, n, -1);
}

template <>
void VectorAddConstantInplace<float>(const float &c, float *output, int n) {
    if (n <= 0) return;
    vDSP_vsadd(output, 1, &c, output, 1, n);
}

template <>
void VectorAddConstantInplace<double>(const double &c, double *output, int n) {
    if (n <= 0) return;
    vDSP_vsaddD(output, 1, &c, output, 1, n);
}

template <>
void VectorAddInplace<float>(const float *x, float *output, int n) {
    if (n <= 0) return;
    vDSP_vadd(reinterpret_cast<const float *>(x), 1, reinterpret_cast<const float *>(output), 1, reinterpret_cast<float *>(output), 1, n);
}

template <>
void VectorAddInplace<double>(const double *x, double *output, int n) {
    if (n <= 0) return;
    vDSP_vaddD(reinterpret_cast<const double *>(x), 1, reinterpret_cast<const double *>(output), 1, reinterpret_cast<double *>(output), 1, n);
}

template <>
void VectorAddInplace<std::complex<float>>(const std::complex<float> *x, std::complex<float> *output, int n) {
    if (n <= 0) return;
    vDSP_vadd(reinterpret_cast<const float *>(x), 1, reinterpret_cast<const float *>(output), 1, reinterpret_cast<float *>(output), 1, 2 * n);
}

template <>
void VectorAddInplace<std::complex<double>>(const std::complex<double> *x, std::complex<double> *output, int n) {
    if (n <= 0) return;
    vDSP_vaddD(reinterpret_cast<const double *>(x), 1, reinterpret_cast<const double *>(output), 1, reinterpret_cast<double *>(output), 1, 2 * n);
}

template <>
void VectorAdd<float>(const float *x, const float *y, float *output, int n) {
    if (n <= 0) return;
    vDSP_vadd(reinterpret_cast<const float *>(x), 1, reinterpret_cast<const float *>(y), 1, reinterpret_cast<float *>(output), 1, n);
}

template <>
void VectorAdd<double>(const double *x, const double *y, double *output, int n) {
    if (n <= 0) return;
    vDSP_vaddD(reinterpret_cast<const double *>(x), 1, reinterpret_cast<const double *>(y), 1, reinterpret_cast<double *>(output), 1, n);
}

template <>
void VectorAdd<std::complex<float>>(const std::complex<float> *x, const std::complex<float> *y, std::complex<float> *output, int n) {
    if (n <= 0) return;
    vDSP_vadd(reinterpret_cast<const float *>(x), 1, reinterpret_cast<const float *>(y), 1, reinterpret_cast<float *>(output), 1, 2 * n);
}

template <>
void VectorAdd<std::complex<double>>(const std::complex<double> *x, const std::complex<double> *y, std::complex<double> *output, int n) {
    if (n <= 0) return;
    vDSP_vaddD(reinterpret_cast<const double *>(x), 1, reinterpret_cast<const double *>(y), 1, reinterpret_cast<double *>(output), 1, 2 * n);
}

template <>
void VectorSubConstantRev<float>(const float *x, const float &c, float *output, int n) {
    if (n <= 0) return;
    for (int i = 0; i < n; ++i) { output[i] = c - x[i]; }
}

template <>
void VectorDivInplace<float>(const float *x, float *output, int n) {
    if (n <= 0) return;
    vDSP_vdiv(x, 1, output, 1, output, 1, n);
}

template <>
void VectorDivInplace<double>(const double *x, double *output, int n) {
    if (n <= 0) return;
    vDSP_vdivD(x, 1, output, 1, output, 1, n);
}

template <>
void VectorMadInplace<float>(const float *x, const float *y, float *output, int n) {
    if (n <= 0) return;
    vDSP_vma(x, 1, y, 1, output, 1, output, 1, n);
}

template <>
void VectorMadInplace<double>(const double *x, const double *y, double *output, int n) {
    if (n <= 0) return;
    vDSP_vmaD(x, 1, y, 1, output, 1, output, 1, n);
}

template <>
void VectorMadInplace<std::complex<float>>(const std::complex<float> *x, const std::complex<float> *y, std::complex<float> *output, int n) {
    if (n <= 0) return;
    const auto a = Split(x), b = Split(y), dest = Split(output);
    vDSP_zvma(&a, 2, &b, 2, &dest, 2, &dest, 2, n);
}

template <>
void VectorMadInplace<std::complex<double>>(const std::complex<double> *x, const std::complex<double> *y, std::complex<double> *output, int n) {
    if (n <= 0) return;
    const auto a = Split(x), b = Split(y), dest = Split(output);
    vDSP_zvmaD(&a, 2, &b, 2, &dest, 2, &dest, 2, n);
}

template <>
void VectorMadConstantInplace<float>(const float *x, const float &c, float *output, int n) {
    if (n <= 0) return;
    vDSP_vsma(x, 1, &c, output, 1, output, 1, n);
}

template <>
void VectorMadConstantInplace<double>(const double *x, const double &c, double *output, int n) {
    if (n <= 0) return;
    vDSP_vsmaD(x, 1, &c, output, 1, output, 1, n);
}

template <>
void VectorPowConstant<float>(const float *x, const float &c, float *output, int n) {
    if (n <= 0) return;
    vvpowsf(output, &c, x, &n);
}

template <>
void VectorPowConstant<double>(const double *x, const double &c, double *output, int n) {
    if (n <= 0) return;
    vvpows(output, &c, x, &n);
}

template <>
void VectorSqrtInplace<float>(float *output, int n) {
    if (n <= 0) return;
    vvsqrtf(output, output, &n);
}

template <>
void VectorSqrtInplace<double>(double *output, int n) {
    if (n <= 0) return;
    vvsqrt(output, output, &n);
}

template <>
void VectorNorm<std::complex<float>, float>(const std::complex<float> *x, float *output, int n) {
    if (n <= 0) return;
    const auto src = Split(x);
    vDSP_zvmags(&src, 2, output, 1, n);
}

template <>
void VectorNorm<std::complex<double>, double>(const std::complex<double> *x, double *output, int n) {
    if (n <= 0) return;
    const auto src = Split(x);
    vDSP_zvmagsD(&src, 2, output, 1, n);
}

template <>
float VectorNormDiffL1<float>(const float *x, const float *y, int n) {
    if (n <= 0) return 0;
    AlignedPodVector<float> delta(n);
    vDSP_vsub(y, 1, x, 1, delta.data(), 1, n);
    float result = 0;
    vDSP_svemg(delta.data(), 1, &result, n);
    return result;
}

template <>
double VectorNormDiffL1<double>(const double *x, const double *y, int n) {
    if (n <= 0) return 0;
    AlignedPodVector<double> delta(n);
    vDSP_vsubD(y, 1, x, 1, delta.data(), 1, n);
    double result = 0;
    vDSP_svemgD(delta.data(), 1, &result, n);
    return result;
}

template <>
float VectorNormDiffL2<float>(const float *x, const float *y, int n) {
    if (n <= 0) return 0;
    AlignedPodVector<float> delta(n);
    vDSP_vsub(y, 1, x, 1, delta.data(), 1, n);
    return cblas_snrm2(n, delta.data(), 1);
}

template <>
double VectorNormDiffL2<double>(const double *x, const double *y, int n) {
    if (n <= 0) return 0;
    AlignedPodVector<double> delta(n);
    vDSP_vsubD(y, 1, x, 1, delta.data(), 1, n);
    return cblas_dnrm2(n, delta.data(), 1);
}

template <>
float VectorNormDiffInf<float>(const float *x, const float *y, int n) {
    if (n <= 0) return 0;
    AlignedPodVector<float> delta(n);
    vDSP_vsub(y, 1, x, 1, delta.data(), 1, n);
    float result = 0;
    vDSP_maxmgv(delta.data(), 1, &result, n);
    return result;
}

template <>
double VectorNormDiffInf<double>(const double *x, const double *y, int n) {
    if (n <= 0) return 0;
    AlignedPodVector<double> delta(n);
    vDSP_vsubD(y, 1, x, 1, delta.data(), 1, n);
    double result = 0;
    vDSP_maxmgvD(delta.data(), 1, &result, n);
    return result;
}

template <>
float VectorLInf<float>(const float *x, int n) {
    if (n <= 0) return 0;
    float result = 0;
    vDSP_maxmgv(x, 1, &result, n);
    return result;
}

template <>
double VectorLInf<double>(const double *x, int n) {
    if (n <= 0) return 0;
    double result = 0;
    vDSP_maxmgvD(x, 1, &result, n);
    return result;
}

template <>
float VectorL2<float>(const float *x, int n) {
    if (n <= 0) return 0;
    return cblas_snrm2(n, x, 1);
}

template <>
float VectorL2<std::complex<float>>(const std::complex<float> *x, int n) {
    if (n <= 0) return 0;
    return VectorL2(reinterpret_cast<const float *>(x), 2 * n);
}

template <>
double VectorL2<double>(const double *x, int n) {
    if (n <= 0) return 0;
    return cblas_dnrm2(n, x, 1);
}

template <>
double VectorL2<std::complex<double>>(const std::complex<double> *x, int n) {
    if (n <= 0) return 0;
    return VectorL2(reinterpret_cast<const double *>(x), 2 * n);
}

template <>
float VectorL2Sqr<float>(const std::complex<float> *x, int n) {
    if (n <= 0) return 0;
    // Preserve the legacy specialization's semantics despite its name.
    return VectorL2(reinterpret_cast<const float *>(x), n);
}

template <>
float VectorSum<float>(const float *x, int n) {
    if (n <= 0) return 0;
    float result = 0;
    vDSP_sve(x, 1, &result, n);
    return result;
}

template <>
double VectorSum<double>(const double *x, int n) {
    if (n <= 0) return 0;
    double result = 0;
    vDSP_sveD(x, 1, &result, n);
    return result;
}

template <>
void VectorInvInplace<float>(float *output, int n) {
    if (n <= 0) return;
    vvrecf(output, output, &n);
}

template <>
void VectorInvInplace<double>(double *output, int n) {
    if (n <= 0) return;
    vvrec(output, output, &n);
}

template <>
void VectorSet<float>(const float &c, float *output, int n) {
    if (n <= 0) return;
    vDSP_vfill(&c, output, 1, n);
}

template <>
void VectorSet<double>(const double &c, double *output, int n) {
    if (n <= 0) return;
    vDSP_vfillD(&c, output, 1, n);
}

template <>
void VectorSet<int>(const int &c, int *output, int n) {
    if (n <= 0) return;
    for (int i = 0; i < n; ++i) { output[i] = c; }
}

template <>
void VectorDecimate<float>(const float *x, int src_n, float *output, int factor) {
    if (src_n <= 0) return;
    for (int i = 0, j = 0; i < src_n; i += factor, ++j) output[j] = x[i];
}

template <>
void VectorDecimate<double>(const double *x, int src_n, double *output, int factor) {
    if (src_n <= 0) return;
    for (int i = 0, j = 0; i < src_n; i += factor, ++j) output[j] = x[i];
}

template <>
void VectorInterpolate<float>(const float *x, int src_n, float *output, int factor) {
    if (src_n <= 0) return;
    std::fill(output, output + src_n * factor, 0);
    for (int i = 0; i < src_n; ++i) output[i * factor] = x[i];
}

template <>
void VectorInterpolate<double>(const double *x, int src_n, double *output, int factor) {
    if (src_n <= 0) return;
    std::fill(output, output + src_n * factor, 0);
    for (int i = 0; i < src_n; ++i) output[i * factor] = x[i];
}

template <>
void VectorInterpolateHold<float>(const float *x, int src_n, float *output, int factor) {
    if (src_n <= 0) return;
    for (int i = 0; i < src_n; ++i) std::fill(output + i * factor, output + (i + 1) * factor, x[i]);
}

template <>
void VectorInterpolateHold<double>(const double *x, int src_n, double *output, int factor) {
    if (src_n <= 0) return;
    for (int i = 0; i < src_n; ++i) std::fill(output + i * factor, output + (i + 1) * factor, x[i]);
}

template <>
void VectorReverseInplace<std::complex<float>>(std::complex<float> *output, int n) {
    if (n <= 0) return;
    std::reverse(output, output + n);
}

template <>
void VectorReverseInplace<std::complex<double>>(std::complex<double> *output, int n) {
    if (n <= 0) return;
    std::reverse(output, output + n);
}

template <>
void VectorReverse<float>(const float *x, float *output, int n) {
    if (n <= 0) return;
    std::reverse_copy(x, x + n, output);
}

template <>
void VectorReverse<double>(const double *x, double *output, int n) {
    if (n <= 0) return;
    std::reverse_copy(x, x + n, output);
}

template <>
void VectorReverse<std::complex<float>>(const std::complex<float> *x, std::complex<float> *output, int n) {
    if (n <= 0) return;
    std::reverse_copy(x, x + n, output);
}

template <>
void VectorReverse<std::complex<double>>(const std::complex<double> *x, std::complex<double> *output, int n) {
    if (n <= 0) return;
    std::reverse_copy(x, x + n, output);
}

template <>
void VectorConjInplace<std::complex<float>>(std::complex<float> *output, int n) {
    if (n <= 0) return;
    for (int i = 0; i < n; ++i) { output[i] = std::conj(output[i]); }
}

template <>
void VectorConjInplace<std::complex<double>>(std::complex<double> *output, int n) {
    if (n <= 0) return;
    for (int i = 0; i < n; ++i) { output[i] = std::conj(output[i]); }
}

template <>
void VectorMove<float>(const float *x, float *output, int n) {
    if (n <= 0) return;
    std::memmove(output, x, sizeof(*output) * n);
}

template <>
void VectorMove<double>(const double *x, double *output, int n) {
    if (n <= 0) return;
    std::memmove(output, x, sizeof(*output) * n);
}

template <>
void VectorMove<std::complex<float>>(const std::complex<float> *x, std::complex<float> *output, int n) {
    if (n <= 0) return;
    std::memmove(output, x, sizeof(*output) * n);
}

template <>
void VectorZero<float>(float *output, int n) {
    if (n <= 0) return;
    std::fill(output, output + n, 0);
}

template <>
void VectorZero<std::complex<float>>(std::complex<float> *output, int n) {
    if (n <= 0) return;
    std::fill(output, output + n, 0);
}

template <>
float VectorDot<float>(const float *x, const float *y, int n) {
    if (n <= 0) return 0;
    float result = 0;
    vDSP_dotpr(x, 1, y, 1, &result, n);
    return result;
}

template <>
double VectorDot<double>(const double *x, const double *y, int n) {
    if (n <= 0) return 0;
    double result = 0;
    vDSP_dotprD(x, 1, y, 1, &result, n);
    return result;
}

template <>
void VectorReplaceNanInplace<float>(const float &c, float *x, int n) {
    if (n <= 0) return;
    for (int i = 0; i < n; ++i) { if (std::isnan(x[i])) x[i] = c; }
}

template <>
void VectorReplaceNanInplace<double>(const double &c, double *x, int n) {
    if (n <= 0) return;
    for (int i = 0; i < n; ++i) { if (std::isnan(x[i])) x[i] = c; }
}

template <>
void VectorBothThresholdInplace<double>(const double &c, double *x, int n) {
    if (n <= 0) return;
    for (int i = 0; i < n; ++i) { if (x[i] < -c) x[i] = -c; else if (x[i] > c) x[i] = c; }
}

template <>
void VectorEnsureNonnegativeInplace<float>(float *x, int n) {
    if (n <= 0) return;
    for (int i = 0; i < n; ++i) { if (x[i] < 0) x[i] = 0; }
}

template <>
void VectorEnsureNonnegativeInplace<double>(double *x, int n) {
    if (n <= 0) return;
    for (int i = 0; i < n; ++i) { if (x[i] < 0) x[i] = 0; }
}

template <>
void VectorBothThresholdInplace<float>(const float &c, float *x, int n) {
    if (n <= 0) return;
    for (int i = 0; i < n; ++i) { if (x[i] < -c) x[i] = -c; else if (x[i] > c) x[i] = c; }
}

template <>
void VectorConvert<float, float>(const float *x, float *output, int n) {
    if (n <= 0) return;
    if (x != output) VectorMove(x, output, n);
}

template <>
void VectorConvert<double, double>(const double *x, double *output, int n) {
    if (n <= 0) return;
    if (x != output) VectorMove(x, output, n);
}

template <>
void VectorConvert<float, Float16>(const float *x, Float16 *output, int n) {
    if (n <= 0) return;
    vImage_Buffer src = {const_cast<void *>(static_cast<const void *>(x)), 1, static_cast<vImagePixelCount>(n), sizeof(*x) * static_cast<std::size_t>(n)};
    vImage_Buffer dest = {output, 1, static_cast<vImagePixelCount>(n), sizeof(*output) * static_cast<std::size_t>(n)};
    const auto status = vImageConvert_PlanarFtoPlanar16F(&src, &dest, kvImageNoFlags);
    if (status != kvImageNoError) throw std::runtime_error("Accelerate half conversion failed");
}

template <>
void VectorConvert<Float16, float>(const Float16 *x, float *output, int n) {
    if (n <= 0) return;
    vImage_Buffer src = {const_cast<void *>(static_cast<const void *>(x)), 1, static_cast<vImagePixelCount>(n), sizeof(*x) * static_cast<std::size_t>(n)};
    vImage_Buffer dest = {output, 1, static_cast<vImagePixelCount>(n), sizeof(*output) * static_cast<std::size_t>(n)};
    const auto status = vImageConvert_Planar16FtoPlanarF(&src, &dest, kvImageNoFlags);
    if (status != kvImageNoError) throw std::runtime_error("Accelerate half conversion failed");
}

template <>
void VectorConvert<std::complex<float>, ComplexFloat16>(const std::complex<float> *x, ComplexFloat16 *output, int n) {
    if (n <= 0) return;
    VectorConvert(reinterpret_cast<const float *>(x), reinterpret_cast<Float16 *>(output), 2 * n);
}

template <>
void VectorConvert<ComplexFloat16, std::complex<float>>(const ComplexFloat16 *x, std::complex<float> *output, int n) {
    if (n <= 0) return;
    VectorConvert(reinterpret_cast<const Float16 *>(x), reinterpret_cast<float *>(output), 2 * n);
}

template <>
void VectorRealToComplex<float>(const float *x, const float *y, std::complex<float> *output, int n) {
    if (n <= 0) return;
    for (int i = 0; i < n; ++i) { output[i] = {x ? x[i] : 0, y ? y[i] : 0}; }
}

template <>
void VectorRealToComplex<double>(const double *x, const double *y, std::complex<double> *output, int n) {
    if (n <= 0) return;
    for (int i = 0; i < n; ++i) { output[i] = {x ? x[i] : 0, y ? y[i] : 0}; }
}

template <>
void VectorComplexToReal<float>(const std::complex<float> *x, float *output_real, float *output_imag, int n) {
    if (n <= 0) return;
    for (int i = 0; i < n; ++i) { if (output_real) output_real[i] = x[i].real(); if (output_imag) output_imag[i] = x[i].imag(); }
}

template <>
void VectorComplexToReal<double>(const std::complex<double> *x, double *output_real, double *output_imag, int n) {
    if (n <= 0) return;
    for (int i = 0; i < n; ++i) { if (output_real) output_real[i] = x[i].real(); if (output_imag) output_imag[i] = x[i].imag(); }
}

template <>
void VectorConvolve<float>(const float *x, int nx, const float *y, int ny, float *output) {
    if (nx <= 0 || ny <= 0) return;
    AlignedPodVector<float> padded(nx + 2 * (ny - 1));
    std::fill(padded.begin(), padded.end(), 0);
    std::copy(x, x + nx, padded.data() + ny - 1);
    vDSP_conv(padded.data(), 1, y + ny - 1, -1, output, 1, nx + ny - 1, ny);
}

template <>
void VectorConvolve<double>(const double *x, int nx, const double *y, int ny, double *output) {
    if (nx <= 0 || ny <= 0) return;
    AlignedPodVector<double> padded(nx + 2 * (ny - 1));
    std::fill(padded.begin(), padded.end(), 0);
    std::copy(x, x + nx, padded.data() + ny - 1);
    vDSP_convD(padded.data(), 1, y + ny - 1, -1, output, 1, nx + ny - 1, ny);
}

}
