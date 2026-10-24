#include "bakuage/fir_filter4.h"
#include <Accelerate/Accelerate.h>
#include <stdexcept>
#include <cstdint>
#include <vector>

namespace {
float Dot(const float *a, const float *b, int n) {
    float result = 0;
    if (n) vDSP_dotpr(a, 1, b, 1, &result, n);
    return result;
}
double Dot(const double *a, const double *b, int n) {
    double result = 0;
    if (n) vDSP_dotprD(a, 1, b, 1, &result, n);
    return result;
}
template <class T> struct MultirateFir: bakuage::FirFilter4ImplBase {
    MultirateFir(const T *first, const T *last, int up, int down): up(up), down(down) {
        if (first == last || up <= 0 || down <= 0) throw std::invalid_argument("Invalid multirate FIR parameters");
        phases.resize(up);
        for (int p = 0; p < up; ++p) {
            for (int k = p; k < last - first; k += up) phases[p].push_back(first[k]);
            std::reverse(phases[p].begin(), phases[p].end());
        }
        history.resize((last - first + up - 1) / up - 1);
        std::fill(history.begin(), history.end(), 0);
    }
    void Clock(const T *first, const T *last, T *dest) {
        const int n = last - first;
        if (n % down) throw std::invalid_argument("FIR input length must be divisible by down factor");
        if (!n) return;
        buffer.resize(history.size() + n);
        std::copy(history.begin(), history.end(), buffer.begin());
        std::copy(first, last, buffer.begin() + history.size());
        for (int j = 0; j < n / down * up; ++j) {
            const auto time = static_cast<std::int64_t>(j) * down;
            const auto &coefficients = phases[time % up];
            const auto end = history.size() + time / up + 1;
            dest[j] = coefficients.empty() ? T(0) : Dot(coefficients.data(), buffer.data() + end - coefficients.size(), coefficients.size());
        }
        std::copy(buffer.end() - history.size(), buffer.end(), history.begin());
    }
    int up, down;
    std::vector<std::vector<T>> phases;
    bakuage::AlignedPodVector<T> history, buffer;
};
}

namespace bakuage {
template <> void FirFilter4<float>::PrepareFir(const float *first, const float *last, int up, int down) {
    impl_.reset(new MultirateFir<float>(first, last, up, down));
}
template <> void FirFilter4<double>::PrepareFir(const double *first, const double *last, int up, int down) {
    impl_.reset(new MultirateFir<double>(first, last, up, down));
}
template <> void FirFilter4<float>::Clock(const float *first, const float *last, float *dest) {
    static_cast<MultirateFir<float> *>(impl_.get())->Clock(first, last, dest);
}
template <> void FirFilter4<double>::Clock(const double *first, const double *last, double *dest) {
    static_cast<MultirateFir<double> *>(impl_.get())->Clock(first, last, dest);
}
}
