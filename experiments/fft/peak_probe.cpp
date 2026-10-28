#include "plan.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>

double dbesi0(double);

namespace {
using namespace fft_exp;
using Double = std::complex<double>;
using Wide = std::complex<long double>;
constexpr int fft_size = 16384, fir_size = 4097, bins = fft_size / 2 + 1;
constexpr int samples = 44100, oversample = 4, original_chunk = 16384;
constexpr int output_size = ((samples + fir_size + 4095) / 4096) * original_chunk;
constexpr double pi = 3.141592653589793238462643383279502884;

double read_double(std::istream &stream) {
    std::uint64_t bits = 0;
    for (int i = 0; i < 8; ++i) {
        const int byte = stream.get();
        if (byte == EOF) throw std::runtime_error("Truncated probe data");
        bits |= std::uint64_t(byte) << (i * 8);
    }
    double value;
    std::memcpy(&value, &bits, 8);
    return value;
}
std::vector<Double> read_values(const std::filesystem::path &path, int count) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot read " + path.string());
    std::vector<Double> values(count);
    for (auto &value : values) {
        const double re = read_double(stream), im = read_double(stream);
        value = {re, im};
    }
    if (stream.peek() != EOF) throw std::runtime_error("Unexpected probe data length");
    return values;
}
template <class T> void write_values(const std::filesystem::path &path, const std::vector<T> &values) {
    std::ofstream stream(path, std::ios::binary);
    stream.exceptions(std::ios::badbit | std::ios::failbit);
    for (const auto &value : values) for (double part : {double(value.real()), double(value.imag())}) {
        std::uint64_t bits;
        std::memcpy(&bits, &part, 8);
        for (int i = 0; i < 8; ++i) stream.put(char((bits >> (i * 8)) & 255));
    }
}

std::vector<Double> native_fir(const std::string &filter) {
    std::vector<Double> fir(fir_size);
    if (filter == "endpoints") { fir[0] = .75; fir[1] = .25; return fir; }
    // Frozen scalar formula from CalculateBandPassFir/Keiser at 14b5027.
    // Reproduces the original I0 approximation and local libm independently
    // of the production build. The mathematical reference uses Decimal I0.
    const int center = (fir_size - 1) / 2;
    const double b = 1 / pi;
    fir[center] = b * pi * .25;
    for (int i = 1; i <= center; ++i) {
        const double x = double(i) / center;
        const double window = dbesi0(pi * 7 * std::sqrt(1 - x * x)) / dbesi0(pi * 7);
        const double value = b * std::sin(i * pi * .25) / i * window;
        fir[center - i] = fir[center + i] = value;
    }
    return fir;
}

struct Transform {
    explicit Transform(bool real) : plan(make_plan(fft_size, real)), native(real ? bins : fft_size) {
        if (!plan) throw std::runtime_error("16384-point FFT is unsupported");
    }
    void forward(const std::vector<Complex> &input, std::vector<Complex> &half) {
        plan->forward(input, native);
        std::copy_n(native.begin(), bins, half.begin());
        half.front().imag(0);
        half.back().imag(0);
    }
    void backward(const std::vector<Complex> &half, std::vector<Complex> &output) {
        std::copy(half.begin(), half.end(), native.begin());
        if (!plan->real) for (int k = 1; k < fft_size / 2; ++k) native[fft_size - k] = std::conj(half[k]);
        plan->backward(native, output);
    }
    std::unique_ptr<Plan> plan;
    std::vector<Complex> native;
};

// Keep identical scalar expression/code generation for the interior bins of
// both layouts, even when contraction is allowed. This comparison isolates
// packing rather than two differently optimized multiplication loops.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
void multiply_bins(const Complex *kernel, Complex *spectrum, int count) {
    for (int k = 0; k < count; ++k) spectrum[k] *= kernel[k];
}

void multiply(const std::vector<Complex> &kernel, std::vector<Complex> &spectrum,
              bool packed, const std::string &fault) {
    if (packed) {
        // Frozen Perm layout: the first complex slot stores two REAL bins.
        // This deliberately reproduces VectorMulPermInplace's special case.
        static_assert(sizeof(Complex) == 2 * sizeof(Scalar), "Perm storage");
        std::vector<Complex> packed_input(fft_size / 2), packed_kernel(fft_size / 2);
        packed_input[0] = Complex(spectrum.front().real(), spectrum.back().real());
        packed_kernel[0] = Complex(kernel.front().real(), kernel.back().real());
        std::copy(spectrum.begin() + 1, spectrum.end() - 1, packed_input.begin() + 1);
        std::copy(kernel.begin() + 1, kernel.end() - 1, packed_kernel.begin() + 1);
        auto *dest = reinterpret_cast<Scalar *>(packed_input.data());
        const auto *src = reinterpret_cast<const Scalar *>(packed_kernel.data());
        const Scalar dc = dest[0] * src[0], nyquist = dest[1] * src[1];
        multiply_bins(packed_kernel.data() + 1, packed_input.data() + 1, bins - 2);
        if (fault == "mix_dc_nyquist") packed_input[0] *= packed_kernel[0];
        else { dest[0] = dc; dest[1] = nyquist; }
        spectrum.front() = Complex(packed_input[0].real(), 0);
        spectrum.back() = Complex(packed_input[0].imag(), 0);
        std::copy(packed_input.begin() + 1, packed_input.end(), spectrum.begin() + 1);
    } else {
        spectrum.front() = Complex(spectrum.front().real() * kernel.front().real(), 0);
        spectrum.back() = Complex(spectrum.back().real() * kernel.back().real(), 0);
        multiply_bins(kernel.data() + 1, spectrum.data() + 1, bins - 2);
    }
    if (fault == "drop_dc") spectrum[0] = 0;
    if (fault == "drop_nyquist") spectrum.back() = 0;
}

struct Stage {
    double input_error = 0, input_dc_error = 0, input_nyquist_error = 0;
    double product_error = 0, product_dc_error = 0, product_nyquist_error = 0;
    double kernel_cast_at_peak = 0, forward_at_peak = 0, multiply_at_peak = 0, inverse_at_peak = 0;
    double direct_inverse_at_peak = 0, fft_inverse_at_peak = 0;
    double dc_contribution = 0, nyquist_contribution = 0;
};

struct Result {
    std::vector<Complex> wave;
    Stage stage;
    double scatter_error = 0;
};

Result stream_impulse(Transform &transform, const std::vector<Complex> &kernel,
                      const std::vector<Double> &reference_kernel, const std::vector<Double> &fir,
                      const std::vector<Double> &roots, int position, int chunk, bool packed,
                      const std::string &fault) {
    Result result;
    result.wave.resize(output_size);
    std::vector<Scalar> history(2 * fft_size), scatter(output_size);
    std::vector<Complex> input(fft_size), spectrum(bins), transformed(fft_size);
    int history_position = 0;
    const int impulse = position * oversample;
    const int fir_peak = int(std::max_element(fir.begin(), fir.end(), [](Double a, Double b) {
        return std::abs(a) < std::abs(b);
    }) - fir.begin());
    for (int outer = 0; outer < output_size; outer += chunk) {
        const int outer_end = std::min(output_size, outer + chunk);
        for (int begin = outer; begin < outer_end;) {
            const int length = std::min(fft_size - fir_size + 1, outer_end - begin);
            std::fill(input.begin(), input.end(), Complex(0));
            const bool contains_impulse = begin <= impulse && impulse < begin + length;
            if (contains_impulse) input[impulse - begin] = oversample;
            transform.forward(input, spectrum);
            const auto before_product = contains_impulse ? spectrum : std::vector<Complex>();
            multiply(kernel, spectrum, packed, fault);
            if (contains_impulse) {
                const int local = impulse - begin, peak = local + fir_peak;
                Wide sum_kernel = 0, sum_input = 0, sum_product = 0;
                for (int k = 0; k < bins; ++k) {
                    const Wide ideal_input = static_cast<long double>(oversample)
                        * std::conj(Wide(roots[(std::int64_t(k) * local) % fft_size]));
                    const Wide x(before_product[k]), h(kernel[k]);
                    const Wide product(spectrum[k]);
                    const Wide reference = Wide(reference_kernel[k]) * ideal_input;
                    const double x_error = double(std::abs(x - ideal_input));
                    const double product_error = double(std::abs(product - reference));
                    result.stage.input_error = std::max(result.stage.input_error, x_error);
                    result.stage.product_error = std::max(result.stage.product_error, product_error);
                    const Wide phase(roots[(std::int64_t(k) * peak) % fft_size]);
                    const int weight = k == 0 || k == fft_size / 2 ? 1 : 2;
                    sum_kernel += Wide(weight * (h * ideal_input * phase).real(), 0);
                    sum_input += Wide(weight * (h * x * phase).real(), 0);
                    sum_product += Wide(weight * (product * phase).real(), 0);
                    if (k == 0) {
                        result.stage.input_dc_error = x_error;
                        result.stage.product_dc_error = product_error;
                        result.stage.dc_contribution = double((product * phase).real());
                    }
                    if (k == fft_size / 2) {
                        result.stage.input_nyquist_error = x_error;
                        result.stage.product_nyquist_error = product_error;
                        result.stage.nyquist_contribution = double((product * phase).real());
                    }
                }
                result.stage.kernel_cast_at_peak = double(sum_kernel.real() - oversample * fir[fir_peak].real());
                result.stage.forward_at_peak = double(sum_input.real() - sum_kernel.real());
                result.stage.multiply_at_peak = double(sum_product.real() - sum_input.real());
                result.stage.direct_inverse_at_peak = double(sum_product.real());
                transform.backward(spectrum, transformed);
                result.stage.fft_inverse_at_peak = transformed[peak].real();
                result.stage.inverse_at_peak = double(static_cast<long double>(transformed[peak].real()) - sum_product.real());
            } else transform.backward(spectrum, transformed);

            const int convolved = length + fir_size - 1;
            // Frozen FirFilter2 overlap-add buffer update; independent scatter
            // accumulation below detects a shift/flush error without another FFT.
            if (history_position + convolved > int(history.size())) {
                const int remaining = int(history.size()) - history_position;
                std::memmove(history.data(), history.data() + history_position, remaining * sizeof(Scalar));
                std::fill(history.begin() + remaining, history.end(), Scalar(0));
                history_position = 0;
            }
            for (int i = 0; i < length; ++i)
                result.wave[begin + i] = transformed[i].real() + history[history_position + i];
            for (int i = length; i < convolved; ++i)
                history[history_position + i] += transformed[i].real();
            for (int i = 0; i < convolved && begin + i < output_size; ++i)
                scatter[begin + i] += transformed[i].real();
            history_position += length;
            begin += length;
        }
    }
    for (int i = 0; i < output_size; ++i)
        result.scatter_error = std::max(result.scatter_error, std::abs(double(result.wave[i].real()) - scatter[i]));
    return result;
}

void field(std::ostream &out, const char *name, double value) {
    if (!std::isfinite(value)) throw std::runtime_error(std::string("Non-finite diagnostic: ") + name);
    out << ",\"" << name << "\":" << value;
}
}

int main(int argc, char **argv) {
    try {
        std::map<std::string, std::string> args;
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 == argc) throw std::runtime_error("Probe arguments require values");
            args[argv[i]] = argv[i + 1];
        }
        const std::filesystem::path output = args.at("--output"), reference = args.at("--reference");
        const bool real = args.at("--domain") == "real";
        if (!real && args.at("--domain") != "complex") throw std::runtime_error("Invalid domain");
        std::filesystem::create_directories(output);
        const auto details = backend_details();
        Transform transform(real);
        if (args.count("--prepare")) {
#ifndef FFT_DOUBLE
            throw std::runtime_error("Kernel preparation must use the double executable");
#endif
            for (const std::string filter : {"lowpass", "endpoints"}) {
                const auto fir = native_fir(filter);
                std::vector<Complex> input(fft_size), half(bins);
                std::copy(fir.begin(), fir.end(), input.begin());
                transform.forward(input, half);
                for (auto &value : half) value /= Scalar(fft_size);
                write_values(output / ("fir_" + filter + ".bin"), fir);
                write_values(output / ("kernel_" + filter + ".bin"), half);
            }
            return 0;
        }
        const std::filesystem::path kernels = args.at("--kernels");
        const auto roots = read_values(reference / "roots.bin", fft_size);
        const auto ideal = read_values(reference / "fir_ideal.bin", fir_size);
        std::ofstream report(output / "peak.jsonl");
        report.exceptions(std::ios::badbit | std::ios::failbit);
        report << std::setprecision(17);
        report << "{\"type\":\"metadata\",\"backend\":\"" << FFT_BACKEND_NAME
               << "\",\"precision\":\"" << precision << "\",\"domain\":\"" << (real ? "real" : "complex")
               << "\",\"version\":" << std::quoted(FFT_BACKEND_VERSION) << ",\"compiler\":" << std::quoted(FFT_COMPILER)
               << ",\"target\":" << std::quoted(FFT_TARGET)
               << ",\"caller_fp_contract\":" << std::quoted(PEAK_FP_CONTRACT)
               << ",\"fft_size\":16384,\"fir_size\":4097,\"samples\":44100,\"oversample\":4,"
               << "\"original_position\":22050,\"original_chunk\":16384,\"reference_digits\":80,\"details\":"
               << std::quoted(details) << ",\"direct_sum_mantissa_bits\":" << std::numeric_limits<long double>::digits << "}\n";
        int cases = 0, structural_failures = 0;
        for (const std::string filter : {"lowpass", "endpoints"}) {
            const auto fir = read_values(kernels / ("fir_" + filter + ".bin"), fir_size);
            const auto native = read_values(kernels / ("kernel_" + filter + ".bin"), bins);
            const auto exact = read_values(kernels / ("reference_" + filter + ".bin"), bins);
            for (const std::string mode : {"native_double", "reference80", "time_scalar"}) {
                std::vector<Complex> kernel(bins);
                if (mode == "time_scalar") {
                    std::vector<Complex> input(fft_size);
                    std::copy(fir.begin(), fir.end(), input.begin());
                    transform.forward(input, kernel);
                    for (auto &value : kernel) value /= Scalar(fft_size);
                } else {
                    const auto &source = mode == "native_double" ? native : exact;
                    std::copy(source.begin(), source.end(), kernel.begin());
                }
                double kernel_error = 0, coefficient_error = 0;
                for (int k = 0; k < bins; ++k)
                    kernel_error = std::max(kernel_error, std::abs(Double(kernel[k]) - exact[k]));
                if (filter == "lowpass") for (int k = 0; k < fir_size; ++k)
                    coefficient_error = std::max(coefficient_error, std::abs(fir[k] - ideal[k]));
                struct Case { int position, chunk; bool packed; std::string fault; };
                std::vector<Case> tests;
                if (mode == "native_double") {
                    for (int position : {0, 1, 3071, 3072, 3073, 4095, 4096, 4097, 22050, 44099})
                        tests.push_back({position, original_chunk, true, "none"});
                    for (int chunk : {4096, 12288}) tests.push_back({22050, chunk, true, "none"});
                    tests.push_back({22050, original_chunk, false, "none"});
                    for (const std::string fault : {"drop_dc", "drop_nyquist", "mix_dc_nyquist"})
                        tests.push_back({22050, original_chunk, true, fault});
                } else tests.push_back({22050, original_chunk, true, "none"});
                std::vector<Complex> original_perm_wave;
                for (const auto &test : tests) {
                    auto result = stream_impulse(transform, kernel, exact, fir, roots, test.position,
                                                 test.chunk, test.packed, test.fault);
                    double max_error = 0, error_energy = 0, energy = 0, peak = 0;
                    int max_index = 0, peak_index = 0;
                    for (int i = 0; i < output_size; ++i) {
                        const int k = i - test.position * oversample;
                        const double expected = 0 <= k && k < fir_size ? oversample * fir[k].real() : 0;
                        const double value = result.wave[i].real(), error = value - expected;
                        if (std::abs(value) > peak) { peak = std::abs(value); peak_index = i; }
                        if (std::abs(error) > max_error) { max_error = std::abs(error); max_index = i; }
                        error_energy += error * error;
                        energy += expected * expected;
                    }
                    const Scalar db = Scalar(20 * std::log10(peak + 1e-37));
                    const bool original = filter == "lowpass" && mode == "native_double" && test.position == 22050
                        && test.chunk == original_chunk && test.packed && test.fault == "none";
                    report << "{\"type\":\"case\",\"filter\":\"" << filter << "\",\"kernel_mode\":\"" << mode
                           << "\",\"position\":" << test.position << ",\"chunk\":" << test.chunk
                           << ",\"layout\":\"" << (test.packed ? "perm" : "canonical") << "\",\"fault\":\"" << test.fault
                           << "\",\"original_case\":" << (original ? "true" : "false")
                           << ",\"peak_index\":" << peak_index << ",\"max_error_index\":" << max_index;
                    field(report, "peak_amplitude", peak);
                    field(report, "peak_db", db);
                    field(report, "wave_max_abs_error", max_error);
                    field(report, "wave_relative_l2_error", std::sqrt(error_energy / energy));
                    field(report, "scatter_max_error", result.scatter_error);
                    field(report, "coefficient_max_abs_error", coefficient_error);
                    field(report, "kernel_spectrum_max_abs_error", kernel_error);
                    field(report, "kernel_dc_error", std::abs(Double(kernel.front()) - exact.front()));
                    field(report, "kernel_nyquist_error", std::abs(Double(kernel.back()) - exact.back()));
                    const auto &s = result.stage;
                    field(report, "input_spectrum_max_abs_error", s.input_error);
                    field(report, "input_dc_error", s.input_dc_error);
                    field(report, "input_nyquist_error", s.input_nyquist_error);
                    field(report, "product_spectrum_max_abs_error", s.product_error);
                    field(report, "product_dc_error", s.product_dc_error);
                    field(report, "product_nyquist_error", s.product_nyquist_error);
                    field(report, "kernel_cast_error_at_peak", s.kernel_cast_at_peak);
                    field(report, "forward_error_at_peak", s.forward_at_peak);
                    field(report, "multiply_error_at_peak", s.multiply_at_peak);
                    field(report, "inverse_error_at_peak", s.inverse_at_peak);
                    field(report, "direct_inverse_at_peak", s.direct_inverse_at_peak);
                    field(report, "fft_inverse_at_peak", s.fft_inverse_at_peak);
                    field(report, "dc_contribution_at_peak", s.dc_contribution);
                    field(report, "nyquist_contribution_at_peak", s.nyquist_contribution);
                    if (!test.packed && !original_perm_wave.empty()) {
                        double difference = 0;
                        for (int i = 0; i < output_size; ++i)
                            difference = std::max(difference, std::abs(Double(result.wave[i]) - Double(original_perm_wave[i])));
                        field(report, "perm_comparison_max_error", difference);
                        if (difference != 0) ++structural_failures;
                    }
                    report << ",\"passes_original_db_tolerance\":" << (std::abs(db) <= 1e-7 ? "true" : "false") << "}\n";
                    if (original) write_values(output / "original_wave.bin", result.wave);
                    if (mode == "native_double" && test.position == 22050 && test.chunk == original_chunk
                            && test.packed && test.fault == "none") original_perm_wave = result.wave;
                    if (result.scatter_error != 0) ++structural_failures;
                    ++cases;
                }
            }
        }
        report << "{\"type\":\"summary\",\"cases\":" << cases << ",\"structural_failures\":" << structural_failures << "}\n";
        std::cout << FFT_BACKEND_NAME << ' ' << precision << ' ' << (real ? "real" : "complex") << ": "
                  << cases << " peak diagnostic cases, " << structural_failures << " structural failures\n";
        return structural_failures ? 1 : 0;
    } catch (const std::exception &error) {
        std::cerr << "Peak probe: " << error.what() << '\n';
        return 2;
    }
}
