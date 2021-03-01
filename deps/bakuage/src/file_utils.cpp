#include "bakuage/file_utils.h"

#include <cmath>
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
