#include "bakuage/file_utils.h"

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <chrono>
#include <complex>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>
#include <sstream>
#include <random>
#include <mutex>
#include "boost/filesystem.hpp"

namespace bakuage {
    void PublishFile(const std::string &source, const std::string &destination) {
        namespace fs = boost::filesystem;
        const fs::path target = fs::absolute(destination);
        // Reserve a private directory so another process cannot claim the temporary file.
        fs::path staging;
        do {
            staging = target.parent_path() / fs::unique_path(".phaselimiter-%%%%-%%%%-%%%%");
        } while (!fs::create_directory(staging));
        const fs::path temporary = staging / "output";
        try {
            // Stream the copy: older Boost versions may fail across filesystems
            // when copy_file_range is available but returns EXDEV.
            std::ifstream input(source, std::ios::binary);
            std::ofstream output(temporary.string(), std::ios::binary);
            if (!input || !output) throw std::runtime_error("Cannot stage output: " + destination);
            char buffer[64 * 1024];
            while (input.read(buffer, sizeof(buffer)) || input.gcount()) {
                output.write(buffer, input.gcount());
            }
            if (input.bad() || !input.eof()) throw std::runtime_error("Cannot read staged output source: " + source);
            output.close();
            if (!output) throw std::runtime_error("Cannot write staged output: " + destination);
            // boost::filesystem::rename replaces existing files on POSIX and Windows.
            fs::rename(temporary, target);
        } catch (...) {
            boost::system::error_code ignored;
            fs::remove_all(staging, ignored);
            throw;
        }
        boost::system::error_code ignored;
        fs::remove_all(staging, ignored);
    }

    TemporaryFiles::TemporaryFiles(const std::string &_directory):
    directory_(_directory) {
        boost::filesystem::create_directories(_directory);
    }
    
    TemporaryFiles::~TemporaryFiles() {
        for (const auto temporary: temporaries_) {
            boost::filesystem::remove(temporary);
        }
    }
    
    std::string TemporaryFiles::UniquePath(const std::string &extension) {
        std::stringstream result;
        result << directory_ << "/";
        for (int i = 0; i < 32; i++) {
            result << RandomChar();
        }
        result << extension;
        temporaries_.push_back(result.str());
        return result.str();
    }
    
    int TemporaryFiles::Seed() {
        auto now = std::chrono::system_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    }
    
    char TemporaryFiles::RandomChar() {
        static std::mutex m;
        static std::mt19937 random(Seed());
        static const std::string list("0123456789"
                                      "abcdefghijklmnopqrstuvwxyz"
                                      "ABCDEFGHIJKLMNOPQRSTUVWXYZ");
        std::lock_guard<std::mutex> lock(m);
        return list[random() % list.size()];
    }
}
