#include <iostream>

#include "gflags/gflags.h"
#include "picojson.h"
#include "sndfile.h"
#include "bakuage/sndfile_wrapper.h"
#include "bakuage/dft.h"
#include "bakuage/memory.h"
#include "bakuage/utils.h"
#include "bakuage/window_func.h"

DEFINE_string(input, "", "input wav file path");
DEFINE_int32(min_hz, 50, "min freq in hz");
DEFINE_int32(max_hz, 8000, "max freq in hz");
DEFINE_int32(octave_div, 12, "octave division count");
DEFINE_double(a4_hz, 440, "a4 freq in hz");
DEFINE_double(shift_sec, 0.1, "dft shift in sec");
DEFINE_double(window_sec, 0.4, "dft window size in sec");

typedef float Float;

int FreqToGroup(double freq) {
    return static_cast<int>(std::floor(std::log2(1e-37 + freq / FLAGS_a4_hz) * FLAGS_octave_div + 0.5));
};

void CalculateSpectrogram(Float *input, int channels, int samples, int sample_freq, std::vector<std::vector<float>> *output) {
    using namespace bakuage;
    
    const int width = 2 * (sample_freq * FLAGS_window_sec / 2); // even
    const int shift = sample_freq * FLAGS_shift_sec;
    const int spec_len = width / 2 + 1;
    std::vector<float> window(width);
    bakuage::CopyHanning(width, window.begin());
    float *fft_input = (float *)bakuage::AlignedMalloc(sizeof(float) * width);
    std::complex<float> *fft_output = (std::complex<float> *)bakuage::AlignedMalloc(sizeof(std::complex<float>) * spec_len);
    bakuage::RealDft<float> dft(width);
    
    std::vector<std::vector<float>> spectrogram;
    
    const int min_group = FreqToGroup(FLAGS_min_hz);
    const int max_group = FreqToGroup(FLAGS_max_hz);
    
    int pos = -width + shift;
    while (pos < samples) {
        std::vector<std::complex<float>> complex_spec_mid(spec_len);
        std::vector<std::complex<float>> complex_spec_side(spec_len);
        
        // window and fft
        for (int i = 0; i < channels; i++) {
            for (int j = 0; j < width; j++) {
                int k = pos + j;
                fft_input[j] = (0 <= k && k < samples) ? input[channels * k + i] * window[j] : 0;
            }
            dft.Forward(fft_input, (float *)fft_output);
            for (int j = 0; j < spec_len; j++) {
                const auto spec = fft_output[j];
                complex_spec_mid[j] += spec;
                complex_spec_side[j] += spec * (2.0f * i - 1);
            }
        }
        
        // 3dB/oct スロープ補正 + エネルギー正規化
        // mean モードを使うので、ピンクノイズは-3dB/octになることに注意
        for (int j = 0; j < spec_len; j++) {
            const auto freq = 1.0 * j / width * sample_freq;
            const auto slope_scale = std::sqrt(freq); // linear空間なのでsqrt
            const auto normalize_scale = 1.0 / std::sqrt(width);
            const auto scale = slope_scale * normalize_scale;
            complex_spec_mid[j] *= scale;
            complex_spec_side[j] *= scale;
        }
        
        std::vector<float> row(max_group - min_group);
        std::vector<float> row_count(max_group - min_group);
        for (int j = 0; j < spec_len; j++) {
            const double compensation = (j == 0 || j == spec_len - 1) ? 1 : 2;
            const double freq = 1.0 * j / width * sample_freq;
            const int group = FreqToGroup(freq);
            const int k = group - min_group;
            if (0 <= k && k < row.size()) {
                row[k] += std::norm(complex_spec_mid[j]) * compensation;
                row_count[k] += 1;
            }
        }
        for (int j = 0; j < row.size(); j++) {
            row[j] /= 1e-37 + row_count[j];
        }
        spectrogram.emplace_back(std::move(row));
        
        pos += shift;
    }
    
    *output = std::move(spectrogram);
    
    bakuage::AlignedFree(fft_input);
    bakuage::AlignedFree(fft_output);
}

// wavを受け取って、spectrogram(json)に変換して、出力
int main(int argc, char* argv[]) {
    gflags::SetVersionString("1.0.0");
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    bakuage::SndfileWrapper infile;
    SF_INFO sfinfo = { 0 };
    
    const auto input_file_path = FLAGS_input;
    
    if ((infile.set(sf_open (input_file_path.c_str(), SFM_READ, &sfinfo))) == NULL) {
        fprintf(stderr, "Not able to open input file %s.\n", input_file_path.c_str());
        fprintf(stderr, "%s\n", sf_strerror(NULL));
        return 1;
    }
    
    // check format
    fprintf(stderr, "sfinfo.format 0x%08x.\n", sfinfo.format);
    switch (sfinfo.format & SF_FORMAT_TYPEMASK) {
        case SF_FORMAT_WAV:
        case SF_FORMAT_WAVEX:
            break;
        default:
            fprintf(stderr, "Not supported sfinfo.format 0x%08x.\n", sfinfo.format);
            return 2;
    }
    
    std::vector<float> buffer(sfinfo.channels * sfinfo.frames);
    int read_size = sf_readf_float(infile.get(), buffer.data(), sfinfo.frames);
    fprintf(stderr, "%d samples read.\n", read_size);
    if (read_size != sfinfo.frames) {
        fprintf(stderr, "sf_readf_float error: %d %d\n", read_size, (int)sfinfo.frames);
        return 3;
    }
    
    std::vector<std::vector<float>> spectrogram;
    CalculateSpectrogram(buffer.data(), sfinfo.channels, sfinfo.frames, sfinfo.samplerate, &spectrogram);
    
    picojson::object output_json;
    // format
    output_json.insert(std::make_pair("channels", picojson::value((double)sfinfo.channels)));
    output_json.insert(std::make_pair("format", picojson::value((double)sfinfo.format)));
    output_json.insert(std::make_pair("frames", picojson::value((double)sfinfo.frames)));
    output_json.insert(std::make_pair("sample_rate", picojson::value((double)sfinfo.samplerate)));
    output_json.insert(std::make_pair("sections", picojson::value((double)sfinfo.sections)));
    output_json.insert(std::make_pair("seekable", picojson::value((double)sfinfo.seekable)));
    
    // settings
    output_json.insert(std::make_pair("min_hz", picojson::value((double)FLAGS_min_hz)));
    output_json.insert(std::make_pair("max_hz", picojson::value((double)FLAGS_max_hz)));
    output_json.insert(std::make_pair("octave_div", picojson::value((double)FLAGS_octave_div)));
    output_json.insert(std::make_pair("a4_hz", picojson::value((double)FLAGS_a4_hz)));
    output_json.insert(std::make_pair("shift_sec", picojson::value((double)FLAGS_shift_sec)));
    output_json.insert(std::make_pair("window_sec", picojson::value((double)FLAGS_window_sec)));
    
    picojson::array spectrogram_json;
    for (const auto &row: spectrogram) {
        picojson::array row_json;
        for (const auto v: row) {
            row_json.push_back(picojson::value(v));
        }
        spectrogram_json.push_back(picojson::value(row_json));
    }
    output_json.insert(std::make_pair("spectrogram", picojson::value(spectrogram_json)));
    
    std::cout << picojson::value(output_json).serialize(true);
}

