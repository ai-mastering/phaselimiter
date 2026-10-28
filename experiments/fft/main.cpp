#include "plan.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace {
using namespace fft_exp;
using Clock = std::chrono::steady_clock;

std::string quote(const std::string &value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
        else out << c;
    }
    out << '"';
    return out.str();
}

double read_double(std::istream &stream) {
    std::uint64_t bits = 0;
    for (int i = 0; i < 8; ++i) {
        const int c = stream.get();
        if (c == EOF) throw std::runtime_error("Truncated fixture data");
        bits |= std::uint64_t(c) << (8 * i);
    }
    double value;
    std::memcpy(&value, &bits, 8);
    return value;
}

void write_double(std::ostream &stream, double value) {
    std::uint64_t bits;
    std::memcpy(&bits, &value, 8);
    for (int i = 0; i < 8; ++i) stream.put(char((bits >> (8 * i)) & 255));
}

std::vector<std::complex<double>> read_values(std::ifstream &file, std::uint64_t offset, int count) {
    file.clear();
    file.seekg(offset * 16);
    std::vector<std::complex<double>> values(count);
    for (auto &value : values) {
        const double re = read_double(file), im = read_double(file);
        if (!std::isfinite(re) || !std::isfinite(im)) throw std::runtime_error("Non-finite fixture");
        value = {re, im};
    }
    return values;
}

struct Metrics {
    double max_abs = 0, relative_l2 = 0, reference_max = 0;
    int max_index = 0;
    bool finite = true;
    bool relative_defined = true;
};
Metrics metrics(const std::vector<Complex> &actual, const std::vector<std::complex<double>> &expected) {
    if (actual.size() != expected.size()) throw std::runtime_error("Output size mismatch");
    Metrics result;
    long double squared_error = 0, energy = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const std::complex<double> value(actual[i].real(), actual[i].imag());
        if (!std::isfinite(value.real()) || !std::isfinite(value.imag())) { result.finite = false; continue; }
        const double error = std::abs(value - expected[i]);
        if (error > result.max_abs) { result.max_abs = error; result.max_index = int(i); }
        result.reference_max = std::max(result.reference_max, std::abs(expected[i]));
        squared_error += static_cast<long double>(error) * error;
        energy += static_cast<long double>(expected[i].real()) * expected[i].real()
            + static_cast<long double>(expected[i].imag()) * expected[i].imag();
    }
    result.relative_l2 = energy ? double(std::sqrt(squared_error / energy)) : 0;
    result.relative_defined = energy > 0 || squared_error == 0;
    return result;
}

// Timings include the adapter's copies/packing, with warmed reusable plans.
// A common canonical API prevents an omitted layout conversion looking faster.
void timing(std::ostream &report, const std::string &operation, int n, bool real,
            const std::function<void()> &execute, double milliseconds, int trials) {
    for (int i = 0; i < 3; ++i) execute();
    int iterations = 1;
    double elapsed;
    do {
        const auto start = Clock::now();
        for (int i = 0; i < iterations; ++i) execute();
        elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        if (elapsed >= milliseconds || iterations >= 1 << 20) break;
        iterations *= 2;
    } while (true);
    std::vector<double> samples;
    for (int trial = 0; trial < trials; ++trial) {
        const auto start = Clock::now();
        for (int i = 0; i < iterations; ++i) execute();
        samples.push_back(std::chrono::duration<double, std::nano>(Clock::now() - start).count() / iterations);
    }
    auto sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    report << "{\"type\":\"timing\",\"operation\":" << quote(operation)
           << ",\"domain\":" << quote(real ? "real" : "complex") << ",\"n\":" << n
           << ",\"iterations\":" << iterations << ",\"min_ns\":" << sorted.front()
           << ",\"median_ns\":" << sorted[sorted.size() / 2] << ",\"samples_ns\":[";
    for (std::size_t i = 0; i < samples.size(); ++i) report << (i ? "," : "") << samples[i];
    report << "]}\n";
}
}

int main(int argc, char **argv) {
    try {
        static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559, "Fixtures require IEEE binary64");
        std::filesystem::path fixtures, destination;
        double timing_ms = 10;
        int trials = 7;
        std::set<int> timing_sizes{256, 1000, 1024, 4096, 4097, 8192, 16384};
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (i + 1 == argc) throw std::runtime_error("Missing value for " + arg);
            const std::string value = argv[++i];
            if (arg == "--fixtures") fixtures = value;
            else if (arg == "--output") destination = value;
            else if (arg == "--timing-ms") timing_ms = std::stod(value);
            else if (arg == "--trials") trials = std::stoi(value);
            else if (arg == "--timing-sizes") {
                timing_sizes.clear();
                std::istringstream sizes(value);
                std::string item;
                while (std::getline(sizes, item, ',')) timing_sizes.insert(std::stoi(item));
            } else throw std::runtime_error("Unknown argument: " + arg);
        }
        if (fixtures.empty() || destination.empty() || timing_ms < 0 || !std::isfinite(timing_ms)
                || trials < 1 || trials > 100) throw std::runtime_error("Invalid experiment arguments");
        std::filesystem::create_directories(destination);
        std::ifstream manifest(fixtures / "cases.tsv"), inputs(fixtures / "inputs.bin", std::ios::binary),
            references(fixtures / "references.bin", std::ios::binary);
        std::ofstream report(destination / "results.jsonl"), snapshots(destination / "outputs.bin", std::ios::binary);
        if (!manifest || !inputs || !references || !report || !snapshots) throw std::runtime_error("Cannot open experiment files");
        report.exceptions(std::ios::badbit | std::ios::failbit);
        snapshots.exceptions(std::ios::badbit | std::ios::failbit);
        report << std::setprecision(17);
        const auto details = backend_details();
        report << "{\"type\":\"metadata\",\"schema\":1,\"backend\":" << quote(FFT_BACKEND_NAME)
               << ",\"version\":" << quote(FFT_BACKEND_VERSION) << ",\"precision\":" << quote(precision)
               << ",\"compiler\":" << quote(FFT_COMPILER) << ",\"target\":" << quote(FFT_TARGET)
               << ",\"details\":" << quote(details)
               << ",\"timing_scope\":\"adapter including packing/copies; reusable plans; unnormalized transforms\"}\n";
        std::map<std::pair<int, bool>, std::unique_ptr<Plan>> plans;
        std::string header;
        std::getline(manifest, header);
        int failures = 0, completed = 0, skipped = 0;
        std::string name, domain, operation, reference_kind;
        int n, input_count, output_count;
        std::uint64_t input_offset, reference_offset;
        while (manifest >> name >> domain >> n >> operation >> input_offset >> input_count
                >> reference_offset >> output_count >> reference_kind) {
            const bool real = domain == "real";
            if ((domain != "real" && domain != "complex") || n < 1 || n > 1 << 20
                    || input_count < 1 || input_count > 2 * n || output_count < 1 || output_count > n)
                throw std::runtime_error("Invalid fixture manifest row");
            const auto key = std::make_pair(n, real);
            if (!plans.count(key)) {
                const auto start = Clock::now();
                auto plan = make_plan(n, real);
                const double plan_ns = std::chrono::duration<double, std::nano>(Clock::now() - start).count();
                report << "{\"type\":\"plan\",\"n\":" << n << ",\"domain\":" << quote(domain)
                       << ",\"supported\":" << (plan ? "true" : "false") << ",\"creation_ns\":" << plan_ns << "}\n";
                plans.emplace(key, std::move(plan));
            }
            auto &plan = plans.at(key);
            const std::string identity = "\"case\":" + quote(name) + ",\"domain\":" + quote(domain)
                + ",\"operation\":" + quote(operation) + ",\"n\":" + std::to_string(n);
            if (!plan) {
                report << "{\"type\":\"skip\"," << identity << ",\"reason\":\"native API does not support this length; no fallback\"}\n";
                ++skipped;
                continue;
            }
            const int bins = plan->bins();
            const int required_input = operation == "inverse" ? bins : operation == "convolution" ? 2 * n : n;
            const int required_output = operation == "forward" ? bins : n;
            if (input_count != required_input || output_count != required_output) throw std::runtime_error("Fixture dimensions disagree");
            auto source = read_values(inputs, input_offset, input_count);
            auto expected = read_values(references, reference_offset, output_count);
            std::vector<Complex> input(input_count), output(output_count), spectrum(bins), kernel_spectrum(bins), kernel;
            for (int i = 0; i < input_count; ++i) input[i] = Complex(Scalar(source[i].real()), Scalar(source[i].imag()));
            if (operation == "convolution") {
                kernel.assign(input.begin() + n, input.end());
                input.resize(n);
            }
            auto execute = [&]() {
                if (operation == "forward") plan->forward(input, output);
                else if (operation == "inverse") {
                    plan->backward(input, output);
                    for (auto &value : output) value /= Scalar(n);
                } else if (operation == "roundtrip" || operation == "convolution") {
                    plan->forward(input, spectrum);
                    if (operation == "convolution") {
                        plan->forward(kernel, kernel_spectrum);
                        for (int k = 0; k < bins; ++k) spectrum[k] *= kernel_spectrum[k];
                    }
                    plan->backward(spectrum, output);
                    for (auto &value : output) value /= Scalar(n);
                } else throw std::runtime_error("Unknown fixture operation");
            };
            execute();
            const auto first = output;
            execute();
            const bool repeatable = first == output;
            const auto error = metrics(output, expected);
            // A coarse sanity bound only, not a claimed accuracy guarantee.
            // Always retain the measured errors, including failures.
            const double bound = 64 * std::numeric_limits<Scalar>::epsilon() * std::max(1.0, std::log2(n))
                * std::max(1.0, error.reference_max);
            const bool passed = error.finite && repeatable && error.max_abs <= bound;
            if (!passed) ++failures;
            ++completed;
            const auto offset = std::uint64_t(snapshots.tellp()) / 16;
            for (const auto &value : output) { write_double(snapshots, value.real()); write_double(snapshots, value.imag()); }
            report << "{\"type\":\"accuracy\"," << identity << ",\"reference\":" << quote(reference_kind)
                   << ",\"finite\":" << (error.finite ? "true" : "false")
                   << ",\"repeatable\":" << (repeatable ? "true" : "false")
                   << ",\"sanity_pass\":" << (passed ? "true" : "false") << ",\"sanity_bound\":" << bound
                   << ",\"max_abs_error\":" << error.max_abs << ",\"max_error_index\":" << error.max_index
                   << ",\"relative_l2_error\":";
            if (error.relative_defined) report << error.relative_l2;
            else report << "null";
            report << ",\"reference_max\":" << error.reference_max
                   << ",\"output_offset\":" << offset << ",\"output_count\":" << output_count;
            if (operation == "forward" && error.finite) {
                report << ",\"dc_abs_error\":" << std::abs(std::complex<double>(output[0]) - expected[0]);
                if (n % 2 == 0) report << ",\"nyquist_abs_error\":"
                    << std::abs(std::complex<double>(output[n / 2]) - expected[n / 2]);
            }
            report << "}\n";
            if (timing_ms > 0 && name == "mixed" && operation == "forward" && timing_sizes.count(n)) {
                timing(report, "forward", n, real, execute, timing_ms, trials);
                // Capture a stable spectrum: inverse implementations may destroy native work buffers.
                const auto inverse_input = output;
                std::vector<Complex> restored(n);
                timing(report, "inverse", n, real, [&]() { plan->backward(inverse_input, restored); }, timing_ms, trials);
            }
        }
        if (!manifest.eof()) throw std::runtime_error("Malformed fixture manifest");
        if (!completed) throw std::runtime_error("No accuracy cases executed");
        report << "{\"type\":\"summary\",\"completed\":" << completed << ",\"skipped\":" << skipped
               << ",\"failures\":" << failures << "}\n";
        std::cout << FFT_BACKEND_NAME << ' ' << precision << ": " << completed << " cases, " << skipped
                  << " unsupported, " << failures << " failed\n";
        return failures ? 1 : 0;
    } catch (const std::exception &error) {
        std::cerr << "FFT experiment: " << error.what() << '\n';
        return 2;
    }
}
