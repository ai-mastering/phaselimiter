#include "bakuage/dft.h"
#include <Accelerate/Accelerate.h>
#include <unsupported/Eigen/FFT>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <vector>
#include "bakuage/memory.h"

namespace {
template <class T> struct AccelerateFft;
template <> struct AccelerateFft<float> {
    using Setup = FFTSetup;
    static Setup Create(int order) { return vDSP_create_fftsetup(order, kFFTRadix2); }
    static void Destroy(Setup setup) { vDSP_destroy_fftsetup(setup); }
    static void Execute(Setup setup, float *data, int order, bool inverse) {
        DSPSplitComplex split = {data, data + 1};
        vDSP_fft_zip(setup, &split, 2, order, inverse ? FFT_INVERSE : FFT_FORWARD);
    }
    // Packed split-complex real FFT: re[0] = DC, im[0] = Nyquist, re/im[k] = bin k.
    static void ExecuteReal(Setup setup, float *re, float *im, int order, bool inverse) {
        DSPSplitComplex split = {re, im};
        vDSP_fft_zrip(setup, &split, 1, order, inverse ? FFT_INVERSE : FFT_FORWARD);
    }
};
template <> struct AccelerateFft<double> {
    using Setup = FFTSetupD;
    static Setup Create(int order) { return vDSP_create_fftsetupD(order, kFFTRadix2); }
    static void Destroy(Setup setup) { vDSP_destroy_fftsetupD(setup); }
    static void Execute(Setup setup, double *data, int order, bool inverse) {
        DSPDoubleSplitComplex split = {data, data + 1};
        vDSP_fft_zipD(setup, &split, 2, order, inverse ? FFT_INVERSE : FFT_FORWARD);
    }
    static void ExecuteReal(Setup setup, double *re, double *im, int order, bool inverse) {
        DSPDoubleSplitComplex split = {re, im};
        vDSP_fft_zripD(setup, &split, 1, order, inverse ? FFT_INVERSE : FFT_FORWARD);
    }
};

template <class T> class Plan {
public:
    explicit Plan(int length): length(length), order(0), setup(nullptr) {
        if (length <= 0 || length > std::numeric_limits<int>::max() / (2 * sizeof(T)))
            throw std::invalid_argument("Invalid FFT length");
        if (length >= 4 && (length & (length - 1)) == 0) {
            for (int n = length; n > 1; n >>= 1) ++order;
            setup = AccelerateFft<T>::Create(order);
            if (!setup) throw std::bad_alloc();
        }
    }
    ~Plan() { if (setup) AccelerateFft<T>::Destroy(setup); }
    Plan(const Plan &) = delete;
    Plan &operator=(const Plan &) = delete;

    void Execute(const T *src, T *dest, bool inverse) const {
        if (length == 1) {
            if (src != dest) std::memmove(dest, src, 2 * sizeof(T));
        } else if (length == 2) {
            const T r0 = src[0], i0 = src[1], r1 = src[2], i1 = src[3];
            dest[0] = r0 + r1; dest[1] = i0 + i1;
            dest[2] = r0 - r1; dest[3] = i0 - i1;
        } else if (setup) {
            if (src != dest) std::memmove(dest, src, 2 * length * sizeof(T));
            AccelerateFft<T>::Execute(setup, dest, order, inverse);
        } else {
            // vDSP radix-2 does not cover arbitrary/odd lengths. Eigen's bundled
            // KissFFT preserves that public API without another dependency.
            struct Fallback {
                explicit Fallback(int n): input(n), output(n) { fft.SetFlag(Eigen::FFT<T>::Unscaled); }
                Eigen::FFT<T> fft;
                std::vector<std::complex<T>> input, output;
            };
            static thread_local std::map<int, std::unique_ptr<Fallback>> fallbacks;
            auto &entry = fallbacks[length];
            if (!entry) entry.reset(new Fallback(length));
            auto &state = *entry;
            std::memcpy(state.input.data(), src, 2 * length * sizeof(T));
            if (inverse) state.fft.inv(state.output.data(), state.input.data(), length);
            else state.fft.fwd(state.output.data(), state.input.data(), length);
            std::memcpy(dest, state.output.data(), 2 * length * sizeof(T));
        }
    }
    const int length;

    // vDSP's real FFT (zrip) computes the same transform as the complex path
    // at roughly half the work. It is only available for radix-2 lengths.
    bool supportsRealFft() const { return setup != nullptr; }
    void ExecuteReal(T *re, T *im, bool inverse) const {
        AccelerateFft<T>::ExecuteReal(setup, re, im, order, inverse);
    }

private:
    int order;
    typename AccelerateFft<T>::Setup setup;
};

template <class T> const Plan<T> *GetPlan(int length) {
    static std::mutex mutex;
    static std::map<int, std::unique_ptr<Plan<T>>> plans;
    std::lock_guard<std::mutex> lock(mutex);
    auto &plan = plans[length];
    if (!plan) plan.reset(new Plan<T>(length));
    return plan.get();
}

enum class Format { Ccs, Perm, Pack };

template <class T> void RealForward(const Plan<T> &plan, const T *src, T *dest, void *work, Format format) {
    const int n = plan.length;
    if (!work) throw std::invalid_argument("FFT workspace is required");
    if (plan.supportsRealFft()) {
        const int h = n / 2;
        auto *re = static_cast<T *>(work);
        auto *im = re + h;
        // vDSP's real FFT returns twice the unnormalised DFT; fold the
        // factor into the input copy so the public output matches IPP.
        for (int i = 0; i < h; ++i) {
            re[i] = src[2 * i] * T(0.5);
            im[i] = src[2 * i + 1] * T(0.5);
        }
        plan.ExecuteReal(re, im, false);
        if (format == Format::Ccs) {
            dest[0] = re[0];
            dest[1] = 0;
            for (int k = 1; k < h; ++k) {
                dest[2 * k] = re[k];
                dest[2 * k + 1] = im[k];
            }
            dest[n] = im[0];
            dest[n + 1] = 0;
        } else if (format == Format::Perm && n % 2 == 0) {
            dest[0] = re[0];
            dest[1] = im[0];
            for (int k = 1; k < h; ++k) {
                dest[2 * k] = re[k];
                dest[2 * k + 1] = im[k];
            }
        } else {
            dest[0] = re[0];
            for (int k = 1; k < h; ++k) {
                dest[2 * k - 1] = re[k];
                dest[2 * k] = im[k];
            }
            dest[n - 1] = im[0];
        }
        return;
    }
    auto *spectrum = static_cast<T *>(work);
    for (int i = 0; i < n; ++i) {
        spectrum[2 * i] = src[i];
        spectrum[2 * i + 1] = 0;
    }
    plan.Execute(spectrum, spectrum, false);
    if (format == Format::Ccs) {
        std::copy(spectrum, spectrum + 2 * (n / 2 + 1), dest);
        dest[1] = 0;
        if (n % 2 == 0) dest[n + 1] = 0;
    } else if (format == Format::Perm && n % 2 == 0) {
        dest[0] = spectrum[0];
        dest[1] = spectrum[n];
        std::copy(spectrum + 2, spectrum + n, dest + 2);
    } else {
        dest[0] = spectrum[0];
        std::copy(spectrum + 2, spectrum + n + 1, dest + 1);
    }
}

template <class T> void RealBackward(const Plan<T> &plan, const T *src, T *dest, void *work, Format format) {
    const int n = plan.length;
    if (!work) throw std::invalid_argument("FFT workspace is required");
    if (plan.supportsRealFft()) {
        const int h = n / 2;
        auto *re = static_cast<T *>(work);
        auto *im = re + h;
        if (format == Format::Ccs) {
            re[0] = src[0];
            im[0] = src[n];
            for (int k = 1; k < h; ++k) {
                re[k] = src[2 * k];
                im[k] = src[2 * k + 1];
            }
        } else if (format == Format::Perm && n % 2 == 0) {
            re[0] = src[0];
            im[0] = src[1];
            for (int k = 1; k < h; ++k) {
                re[k] = src[2 * k];
                im[k] = src[2 * k + 1];
            }
        } else {
            re[0] = src[0];
            im[0] = src[n - 1];
            for (int k = 1; k < h; ++k) {
                re[k] = src[2 * k - 1];
                im[k] = src[2 * k];
            }
        }
        plan.ExecuteReal(re, im, true);
        for (int i = 0; i < h; ++i) {
            dest[2 * i] = re[i];
            dest[2 * i + 1] = im[i];
        }
        return;
    }
    auto *spectrum = static_cast<T *>(work);
    if (format == Format::Ccs) {
        std::copy(src, src + 2 * (n / 2 + 1), spectrum);
    } else if (format == Format::Perm && n % 2 == 0) {
        spectrum[0] = src[0];
        spectrum[n] = src[1];
        std::copy(src + 2, src + n, spectrum + 2);
    } else {
        spectrum[0] = src[0];
        std::copy(src + 1, src + n, spectrum + 2);
    }
    spectrum[1] = 0;
    if (n % 2 == 0) spectrum[n + 1] = 0;
    for (int k = 1; k <= (n - 1) / 2; ++k) {
        spectrum[2 * (n - k)] = spectrum[2 * k];
        spectrum[2 * (n - k) + 1] = -spectrum[2 * k + 1];
    }
    plan.Execute(spectrum, spectrum, true);
    for (int i = 0; i < n; ++i) dest[i] = spectrum[2 * i];
}

struct Plan2D {
    Plan2D(int width, int height): width(width), height(height), row(GetPlan<float>(width)), column(GetPlan<float>(height)) {}
    const int width, height;
    const Plan<float> *row, *column;
};
const Plan2D *GetPlan2D(int width, int height) {
    static std::mutex mutex;
    static std::map<std::pair<int, int>, std::unique_ptr<Plan2D>> plans;
    std::lock_guard<std::mutex> lock(mutex);
    auto &plan = plans[{width, height}];
    if (!plan) plan.reset(new Plan2D(width, height));
    return plan.get();
}
void Transform2D(const Plan2D &plan, const float *src, float *dest, float *work, bool inverse) {
    for (int y = 0; y < plan.height; ++y)
        plan.row->Execute(src + 2 * y * plan.width, dest + 2 * y * plan.width, inverse);
    for (int x = 0; x < plan.width; ++x) {
        for (int y = 0; y < plan.height; ++y) {
            work[2 * y] = dest[2 * (y * plan.width + x)];
            work[2 * y + 1] = dest[2 * (y * plan.width + x) + 1];
        }
        plan.column->Execute(work, work, inverse);
        for (int y = 0; y < plan.height; ++y) {
            dest[2 * (y * plan.width + x)] = work[2 * y];
            dest[2 * (y * plan.width + x) + 1] = work[2 * y + 1];
        }
    }
}

// An orthonormal DCT-II, matching IPP's image DCT. Use an even extension
// and a complex FFT, including for non-power-of-two dimensions.
void Dct(const float *src, int stride, float *dest, int out_stride, int n, float *work) {
    for (int i = 0; i < n; ++i) {
        work[2 * i] = work[2 * (2 * n - 1 - i)] = src[i * stride];
        work[2 * i + 1] = work[2 * (2 * n - 1 - i) + 1] = 0;
    }
    GetPlan<float>(2 * n)->Execute(work, work, false);
    const double pi = std::acos(-1.0);
    for (int k = 0; k < n; ++k) {
        const double phase = pi * k / (2 * n);
        dest[k * out_stride] = .5 * (work[2 * k] * std::cos(phase) + work[2 * k + 1] * std::sin(phase))
            * std::sqrt((k == 0 ? 1.0 : 2.0) / n);
    }
}
}

namespace bakuage {
FftMemoryBuffer::FftMemoryBuffer(int size): size_(size), data_(nullptr) {
    if (size < 0) throw std::invalid_argument("FFT buffer size must be nonnegative");
    if (size) { data_ = AlignedMalloc(size, 64); std::memset(data_, 0, size); }
}
FftMemoryBuffer::~FftMemoryBuffer() { AlignedFree(data_); }
FftMemoryBuffer::FftMemoryBuffer(const FftMemoryBuffer &other): FftMemoryBuffer(other.size_) {
    if (size_) std::memcpy(data_, other.data_, size_);
}
FftMemoryBuffer::FftMemoryBuffer(FftMemoryBuffer &&other): size_(other.size_), data_(other.data_) {
    other.size_ = 0; other.data_ = nullptr;
}
FftMemoryBuffer &FftMemoryBuffer::operator=(const FftMemoryBuffer &other) {
    if (this != &other) { FftMemoryBuffer copy(other); *this = std::move(copy); }
    return *this;
}
FftMemoryBuffer &FftMemoryBuffer::operator=(FftMemoryBuffer &&other) {
    if (this != &other) {
        AlignedFree(data_); size_ = other.size_; data_ = other.data_;
        other.size_ = 0; other.data_ = nullptr;
    }
    return *this;
}

Dft<float>::Dft(int len): dft_ptr_(GetPlan<float>(len)) {}
void Dft<float>::Forward(const Float *src, Float *dest) {
    static_cast<const Plan<Float> *>(dft_ptr_)->Execute(src, dest, false);
}
void Dft<float>::Backward(const Float *src, Float *dest) {
    static_cast<const Plan<Float> *>(dft_ptr_)->Execute(src, dest, true);
}
RealDft<float>::RealDft(int len, bool no_internal_work): dft_ptr_(GetPlan<float>(len)) {
    if (!no_internal_work) work_ = FftMemoryBuffer(work_size());
}
int RealDft<float>::work_size() const {
    return 2 * static_cast<const Plan<Float> *>(dft_ptr_)->length * sizeof(Float);
}
void RealDft<float>::Forward(const Float *src, Float *dest, void *work) const {
    RealForward(*static_cast<const Plan<Float> *>(dft_ptr_), src, dest, work, Format::Ccs);
}
void RealDft<float>::ForwardPerm(const Float *src, Float *dest, void *work) const {
    RealForward(*static_cast<const Plan<Float> *>(dft_ptr_), src, dest, work, Format::Perm);
}
void RealDft<float>::ForwardPack(const Float *src, Float *dest, void *work) const {
    RealForward(*static_cast<const Plan<Float> *>(dft_ptr_), src, dest, work, Format::Pack);
}
void RealDft<float>::Backward(const Float *src, Float *dest, void *work) const {
    RealBackward(*static_cast<const Plan<Float> *>(dft_ptr_), src, dest, work, Format::Ccs);
}
void RealDft<float>::BackwardPerm(const Float *src, Float *dest, void *work) const {
    RealBackward(*static_cast<const Plan<Float> *>(dft_ptr_), src, dest, work, Format::Perm);
}
void RealDft<float>::BackwardPack(const Float *src, Float *dest, void *work) const {
    RealBackward(*static_cast<const Plan<Float> *>(dft_ptr_), src, dest, work, Format::Pack);
}

Dft<double>::Dft(int len): dft_ptr_(GetPlan<double>(len)) {}
void Dft<double>::Forward(const Float *src, Float *dest) {
    static_cast<const Plan<Float> *>(dft_ptr_)->Execute(src, dest, false);
}
void Dft<double>::Backward(const Float *src, Float *dest) {
    static_cast<const Plan<Float> *>(dft_ptr_)->Execute(src, dest, true);
}
RealDft<double>::RealDft(int len, bool no_internal_work): dft_ptr_(GetPlan<double>(len)) {
    if (!no_internal_work) work_ = FftMemoryBuffer(work_size());
}
int RealDft<double>::work_size() const {
    return 2 * static_cast<const Plan<Float> *>(dft_ptr_)->length * sizeof(Float);
}
void RealDft<double>::Forward(const Float *src, Float *dest, void *work) const {
    RealForward(*static_cast<const Plan<Float> *>(dft_ptr_), src, dest, work, Format::Ccs);
}
void RealDft<double>::ForwardPerm(const Float *src, Float *dest, void *work) const {
    RealForward(*static_cast<const Plan<Float> *>(dft_ptr_), src, dest, work, Format::Perm);
}
void RealDft<double>::ForwardPack(const Float *src, Float *dest, void *work) const {
    RealForward(*static_cast<const Plan<Float> *>(dft_ptr_), src, dest, work, Format::Pack);
}
void RealDft<double>::Backward(const Float *src, Float *dest, void *work) const {
    RealBackward(*static_cast<const Plan<Float> *>(dft_ptr_), src, dest, work, Format::Ccs);
}
void RealDft<double>::BackwardPerm(const Float *src, Float *dest, void *work) const {
    RealBackward(*static_cast<const Plan<Float> *>(dft_ptr_), src, dest, work, Format::Perm);
}
void RealDft<double>::BackwardPack(const Float *src, Float *dest, void *work) const {
    RealBackward(*static_cast<const Plan<Float> *>(dft_ptr_), src, dest, work, Format::Pack);
}

Dft2D<float>::Dft2D(int width, int height): dft_ptr_(GetPlan2D(width, height)), work_(2 * height * sizeof(Float)) {}
void Dft2D<float>::Forward(const Float *src, Float *dest) {
    Transform2D(*static_cast<const Plan2D *>(dft_ptr_), src, dest, static_cast<Float *>(work_.data()), false);
}
void Dft2D<float>::Backward(const Float *src, Float *dest) {
    Transform2D(*static_cast<const Plan2D *>(dft_ptr_), src, dest, static_cast<Float *>(work_.data()), true);
}
Dct2D<float>::Dct2D(int width, int height): dct_ptr_(GetPlan2D(width, height)), work_(4 * std::max(width, height) * sizeof(Float)) {}
void Dct2D<float>::Forward(const Float *src, Float *dest) {
    const auto &plan = *static_cast<const Plan2D *>(dct_ptr_);
    auto *work = static_cast<Float *>(work_.data());
    for (int y = 0; y < plan.height; ++y)
        Dct(src + y * plan.width, 1, dest + y * plan.width, 1, plan.width, work);
    for (int x = 0; x < plan.width; ++x)
        Dct(dest + x, plan.width, dest + x, plan.width, plan.height, work);
}
}
