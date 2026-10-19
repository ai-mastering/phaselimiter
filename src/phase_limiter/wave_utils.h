#ifndef wave_utils_h
#define wave_utils_h

#include <algorithm>
#include <cstring>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <type_traits>
#include "sndfile.h"
#include "bakuage/sndfile_wrapper.h"
#include "bakuage/vector_math.h"

namespace phase_limiter {
    template <class Float>
    void SaveFloatWave(const std::vector<Float> &wave, const std::string &filename, int channels = 2, int sample_rate = 44100) {
        if (channels <= 0 || sample_rate <= 0) {
            throw std::invalid_argument("channels and sample rate must be positive");
        }
        if (wave.size() % channels) {
            throw std::invalid_argument("input wave length must be multiple of channels");
        }
        const size_t frame_count = wave.size() / channels;
        if (static_cast<uintmax_t>(frame_count) > static_cast<uintmax_t>(std::numeric_limits<sf_count_t>::max())) {
            throw std::length_error("wave exceeds audio file frame limit");
        }
        static_assert(std::is_same<Float, float>::value || std::is_same<Float, double>::value,
                      "wave sample type must be float or double");
        bakuage::SndfileWrapper snd_file;
        SF_INFO sfinfo = { 0 };
        std::memset(&sfinfo, 0, sizeof(sfinfo));
        
        sfinfo.channels = channels;
        sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;
        const sf_count_t frames = static_cast<sf_count_t>(frame_count);
        sfinfo.frames = frames;
        sfinfo.samplerate = sample_rate;
        
        if ((snd_file.set(sf_open(filename.c_str(), SFM_WRITE, &sfinfo))) == NULL) {
            std::stringstream message;
            message << "Not able to open output file " << filename << ", "
            << sf_strerror(NULL);
            throw std::logic_error(message.str());
        }
        
        std::vector<Float> buffer = wave;
        // VectorSanitizeInplace accepts int lengths; process long files in chunks.
        for (size_t offset = 0; offset < buffer.size();) {
            const int count = static_cast<int>(std::min<size_t>(
                buffer.size() - offset, std::numeric_limits<int>::max()));
            bakuage::VectorSanitizeInplace<Float>(1e7, buffer.data() + offset, count);
            offset += count;
        }
        
        sf_count_t size;
        if (sizeof(Float) == 4) {
            size = sf_writef_float(snd_file.get(), (float *)buffer.data(), frames);
        } else {
            size = sf_writef_double(snd_file.get(), (double *)buffer.data(), frames);
        }
        if (size != frames) {
            std::stringstream message;
            message << "sf_writef_float error: " << size;
            throw std::logic_error(message.str());
        }
        snd_file.CloseChecked();
    }
    
    template <class Float>
    std::vector<Float> LoadFloatWave(const std::string &filename, int expected_channels = 0, int expected_sample_rate = 0) {
        static_assert(std::is_same<Float, float>::value || std::is_same<Float, double>::value,
                      "wave sample type must be float or double");
        bakuage::SndfileWrapper infile;
        SF_INFO sfinfo = { 0 };
        
        if ((infile.set(sf_open (filename.c_str(), SFM_READ, &sfinfo))) == NULL) {
            std::stringstream message;
            message << "Not able to open input file " << filename << ", "
            << sf_strerror(NULL);
            throw std::logic_error(message.str());
        }
        
        if ((expected_channels && sfinfo.channels != expected_channels) ||
            (expected_sample_rate && sfinfo.samplerate != expected_sample_rate)) {
            std::ostringstream message;
            message << "Input audio must have " << expected_channels << " channels at "
                    << expected_sample_rate << " Hz; got " << sfinfo.channels
                    << " channels at " << sfinfo.samplerate
                    << " Hz. Enable input encoding to convert the format.";
            throw std::logic_error(message.str());
        }

        // check format
        fprintf(stderr, "sfinfo.format 0x%08x.\n", sfinfo.format);
        switch (sfinfo.format & SF_FORMAT_TYPEMASK) {
            case SF_FORMAT_WAV:
            case SF_FORMAT_WAVEX:
                break;
            default:
                std::stringstream message;
                message << "Not supported sfinfo.format " << sfinfo.format;
                throw std::logic_error(message.str());
        }
        
        std::vector<Float> buffer;
        if (sfinfo.channels <= 0 || sfinfo.frames < 0) {
            throw std::logic_error("invalid audio file dimensions");
        }
        if (static_cast<uintmax_t>(sfinfo.frames) > buffer.max_size() / static_cast<size_t>(sfinfo.channels)) {
            throw std::length_error("audio file is too large");
        }
        buffer.resize(static_cast<size_t>(sfinfo.frames) * static_cast<size_t>(sfinfo.channels));
        sf_count_t read_size;
        if (sizeof(Float) == 4) {
            read_size = sf_readf_float(infile.get(), (float *)buffer.data(), sfinfo.frames);
        } else {
            read_size = sf_readf_double(infile.get(), (double *)buffer.data(), sfinfo.frames);
        }
        fprintf(stderr, "%lld frames read.\n", static_cast<long long>(read_size));
        if (read_size != sfinfo.frames) {
            std::stringstream message;
            message << "sf_readf_float error: " << read_size;
            throw std::logic_error(message.str());
        }
        
        return buffer;
    }
}

#endif /* wave_utils_h */
