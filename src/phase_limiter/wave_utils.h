#ifndef wave_utils_h
#define wave_utils_h

#include <cstring>
#include <sstream>
#include <stdexcept>
#include <vector>
#include "sndfile.h"
#include "bakuage/sndfile_wrapper.h"
#include "bakuage/vector_math.h"

namespace phase_limiter {
    template <class Float>
    void SaveFloatWave(const std::vector<Float> &wave, const std::string &filename, int channels = 2, int sample_rate = 44100) {
        bakuage::SndfileWrapper snd_file;
        SF_INFO sfinfo = { 0 };
        std::memset(&sfinfo, 0, sizeof(sfinfo));
        
        sfinfo.channels = channels;
        sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;
        int frames = wave.size() / sfinfo.channels;
        sfinfo.frames = frames;
        sfinfo.samplerate = sample_rate;
        
        if ((snd_file.set(sf_open(filename.c_str(), SFM_WRITE, &sfinfo))) == NULL) {
            std::stringstream message;
            message << "Not able to open output file " << filename << ", "
            << sf_strerror(NULL);
            throw std::logic_error(message.str());
        }
        
        std::vector<Float> buffer = wave;
        bakuage::VectorSanitizeInplace<Float>(1e7, buffer.data(), buffer.size());
        
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
    }
    
    template <class Float>
    std::vector<Float> LoadFloatWave(const std::string &filename) {
        bakuage::SndfileWrapper infile;
        SF_INFO sfinfo = { 0 };
        
        if ((infile.set(sf_open (filename.c_str(), SFM_READ, &sfinfo))) == NULL) {
            std::stringstream message;
            message << "Not able to open input file " << filename << ", "
            << sf_strerror(NULL);
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
        
        std::vector<float> buffer(sfinfo.channels * sfinfo.frames);
        sf_count_t read_size;
        if (sizeof(Float) == 4) {
            read_size = sf_readf_float(infile.get(), (float *)buffer.data(), sfinfo.frames);
        } else {
            read_size = sf_readf_double(infile.get(), (double *)buffer.data(), sfinfo.frames);
        }
        fprintf(stderr, "%d samples read.\n", (int)read_size);
        if (read_size != sfinfo.frames) {
            std::stringstream message;
            message << "sf_readf_float error: " << read_size;
            throw std::logic_error(message.str());
        }
        
        return buffer;
    }
}

#endif /* wave_utils_h */
