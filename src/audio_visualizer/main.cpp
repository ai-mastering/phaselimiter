#include <oneapi/tbb/parallel_pipeline.h>
#include <oneapi/tbb/global_control.h>
#include <oneapi/tbb/info.h>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#include "gflags/gflags.h"
#include "sndfile.h"
#include "CImg.h"
#include "bakuage/dsp.h"
#include <oneapi/tbb.h>
#include <oneapi/tbb/scalable_allocator.h>
#include <oneapi/tbb/cache_aligned_allocator.h>
#include <boost/filesystem.hpp>
#include "bakuage/sndfile_wrapper.h"
#include "bakuage/dft.h"
#include "bakuage/memory.h"
#include "bakuage/utils.h"
#include "bakuage/window_func.h"
#include "frame_renderer.h"

DEFINE_bool(quick_exit, true, "quick exit");
DEFINE_int32(worker_count, 0, "thread count");
DEFINE_string(input, "", "input wav file path");
DEFINE_string(output, "", "output directory; omitted or - writes an image stream to stdout");
DEFINE_string(output_format, "png", "output image format: png or ppm");
DEFINE_string(background, "", "background image path");
DEFINE_string(foreground, "", "foreground image path");
DEFINE_int32(min_hz, 50, "min freq in hz");
DEFINE_int32(max_hz, 8000, "max freq in hz");
DEFINE_int32(band_count, 30, "band count");
DEFINE_int32(width, 640, "width of output video");
DEFINE_int32(height, 480, "height of output video");
DEFINE_double(fps, 30, "frame per sec");
DEFINE_double(window_sec, 0.04, "dft window size in sec");

namespace {

typedef float Float;
typedef float PixelType;
    typedef std::shared_ptr<audio_visualizer::FrameRenderer::Buffer> BufferPtr;

    void PrintMemoryUsage() {
        std::cerr << "Peak RSS(MB)\t" << bakuage::GetPeakRss() / (1024 * 1024)
        << "\tCurrent RSS(MB)\t" << bakuage::GetCurrentRss() / (1024 * 1024)
        << std::endl;
    }

// 動画の長さが音源の長さ以上になるようにする
void CalculateSpectrogram(Float *input, int channels, int samples, int sample_freq, std::vector<bakuage::AlignedPodVector<float>> *output) {
    using namespace bakuage;

    const double min_mel = bakuage::HzToMel(FLAGS_min_hz);
    const double max_mel = bakuage::HzToMel(FLAGS_max_hz);

    const int width = 2 * (sample_freq * FLAGS_window_sec / 2); // even
    const int spec_len = width / 2 + 1;
    bakuage::AlignedPodVector<float> window(width);
    bakuage::CopyHanning(width, window.begin());
    bakuage::AlignedPodVector<float> fft_input(width);
    bakuage::AlignedPodVector<std::complex<float>> fft_output(spec_len);
    bakuage::RealDft<float> dft(width);

    std::vector<bakuage::AlignedPodVector<float>> spectrogram;

    for (int pos_i = 0; pos_i < samples * FLAGS_fps / sample_freq; pos_i++) {
        const int pos = std::floor((double)pos_i / FLAGS_fps * sample_freq) - width / 2;

        bakuage::AlignedPodVector<std::complex<float>> complex_spec_mid(spec_len);
        bakuage::AlignedPodVector<std::complex<float>> complex_spec_side(spec_len);

        // window and fft
        for (int i = 0; i < channels; i++) {
            for (int j = 0; j < width; j++) {
                int k = pos + j;
                fft_input[j] = (0 <= k && k < samples) ? input[channels * k + i] * window[j] : 0;
            }
            dft.Forward(fft_input.data(), (float *)fft_output.data());
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

        bakuage::AlignedPodVector<float> row(FLAGS_band_count);
        bakuage::AlignedPodVector<float> row_count(FLAGS_band_count);
        for (int j = 0; j < spec_len; j++) {
            const double compensation = (j == 0 || j == spec_len - 1) ? 1 : 2;
            const double freq = 1.0 * j / width * sample_freq;
            const int k = FLAGS_band_count * (bakuage::HzToMel(freq) - min_mel) / (max_mel - min_mel);
            if (0 <= k && k < row.size()) {
                row[k] += std::norm(complex_spec_mid[j]) * compensation;
                row_count[k] += 1;
            }
        }
        for (int j = 0; j < row.size(); j++) {
            row[j] /= 1e-37 + row_count[j];
        }
        spectrogram.emplace_back(std::move(row));
    }

    *output = std::move(spectrogram);
}

}

// WAVを受け取って、標準出力に画像ストリーム、またはディレクトリに連番画像を出力
int main(int argc, char* argv[]) {
    gflags::SetVersionString("1.0.0-oss");
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    if (FLAGS_worker_count < 0) {
        std::cerr << "worker_count must be nonnegative" << std::endl;
        return 1;
    }

    if (FLAGS_output_format != "png" && FLAGS_output_format != "ppm") {
        std::cerr << "output_format must be png or ppm" << std::endl;
        return 1;
    }
    const bool output_stdout = FLAGS_output.empty() || FLAGS_output == "-";
    const bool output_ppm = FLAGS_output_format == "ppm";

#ifdef _WIN32
    if (output_stdout && _setmode(_fileno(stdout), _O_BINARY) == -1) {
        std::cerr << "Failed to set stdout to binary mode" << std::endl;
        return 1;
    }
#endif

    bakuage::InitializeDsp();
    PrintMemoryUsage();

    tbb::global_control tbb_limit(tbb::global_control::max_allowed_parallelism,
            FLAGS_worker_count ? FLAGS_worker_count : tbb::info::default_concurrency());
    std::cerr << "TBB default_num_threads:" << tbb::info::default_concurrency() << std::endl;
    PrintMemoryUsage();

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

    bakuage::AlignedPodVector<float> buffer(sfinfo.channels * sfinfo.frames);
    int read_size = sf_readf_float(infile.get(), buffer.data(), sfinfo.frames);
    fprintf(stderr, "%d samples read.\n", read_size);
    if (read_size != sfinfo.frames) {
        fprintf(stderr, "sf_readf_float error: %d %d\n", read_size, (int)sfinfo.frames);
        return 3;
    }

    // calculate spectrogram (energy)
    std::vector<bakuage::AlignedPodVector<float>> spectrogram;
    CalculateSpectrogram(buffer.data(), sfinfo.channels, sfinfo.frames, sfinfo.samplerate, &spectrogram);

    // log and normalize spectrogram
    {
        double min_x = 1e100;
        double max_x = -1e100;
        for (auto &spectrum: spectrogram) {
            for (auto &x: spectrum) {
                x = std::log(1e-7 + x);
                min_x = std::min<double>(min_x, x);
                max_x = std::max<double>(max_x, x);
            }
        }
        for (auto &spectrum: spectrogram) {
            for (auto &x: spectrum) {
                x = (x - min_x) / (max_x - min_x);
            }
        }
    }

    cimg_library::CImg<PixelType> background(FLAGS_width, FLAGS_height, 1, 3, 0);
    if (!FLAGS_background.empty()) {
        cimg_library::CImg<PixelType> tmp(FLAGS_background.c_str());
        tmp.resize(FLAGS_width, FLAGS_height);
        for (int i = 0; i < FLAGS_width; i++) {
            for (int j = 0; j < FLAGS_height; j++) {
                background(i, j, 0, 0) = tmp(i, j, 0, 0);
                background(i, j, 0, 1) = tmp(i, j, 0, 1);
                background(i, j, 0, 2) = tmp(i, j, 0, 2);
            }
        }
    }

    cimg_library::CImg<PixelType> foreground(FLAGS_width, FLAGS_height, 1, 4, 0);
    if (!FLAGS_foreground.empty()) {
        cimg_library::CImg<PixelType> tmp(FLAGS_foreground.c_str());
        tmp.resize(FLAGS_width, FLAGS_height);
        for (int i = 0; i < FLAGS_width; i++) {
            for (int j = 0; j < FLAGS_height; j++) {
                foreground(i, j, 0, 0) = tmp(i, j, 0, 0);
                foreground(i, j, 0, 1) = tmp(i, j, 0, 1);
                foreground(i, j, 0, 2) = tmp(i, j, 0, 2);
                foreground(i, j, 0, 3) = tmp.spectrum() < 4 ? 255 : tmp(i, j, 0, 3);
            }
        }
    }

    const audio_visualizer::FrameRenderer renderer(background, foreground, FLAGS_band_count);

    auto spectrogram_it = spectrogram.begin();
    const auto spectrogram_end = spectrogram.end();
    const auto filter1_func = [&spectrogram_it, spectrogram_end](tbb::flow_control& fc) -> bakuage::AlignedPodVector<float> * {
        if (spectrogram_it != spectrogram_end) {
            return &(*(spectrogram_it++));
        } else {
            fc.stop();
            return nullptr;
        }
    };
    const auto filter2_func = [&renderer, output_ppm](bakuage::AlignedPodVector<float> *spectrum){
        if (output_ppm) {
            return std::make_shared<audio_visualizer::FrameRenderer::Buffer>(renderer.RenderPpm(*spectrum));
        }
        const auto img = renderer.RenderImage(*spectrum);

        const auto temp_path = bakuage::NormalizeToString((boost::filesystem::temp_directory_path() / boost::filesystem::unique_path()).native());
        // img.save_bmp(output_path.str().c_str()); // slow
        img.save_png(temp_path.c_str());
        BufferPtr buffer;
        bakuage::LoadDataFromFile(temp_path.c_str(), [&buffer](const char *data, size_t size) {
            buffer = std::make_shared<audio_visualizer::FrameRenderer::Buffer>(data, data + size);
        });
        boost::filesystem::remove(temp_path);
        return buffer;
    };
    int spectrogram_i = 0;
    const auto filter3_func = [&spectrogram_i, output_stdout](BufferPtr buffer) {
        if (output_stdout) {
            const size_t n = std::fwrite(buffer->data(), 1, buffer->size(), stdout);
            if (n != buffer->size()) {
                throw std::logic_error("failed to write stdout");
            }
        } else {
            std::stringstream output_path;
            output_path << FLAGS_output << "/" << spectrogram_i++ << "." << FLAGS_output_format;

            FILE* pf = std::fopen(output_path.str().c_str(), "wb");
            if (!pf) throw std::runtime_error("failed to open " + output_path.str());
            const size_t n = std::fwrite(buffer->data(), 1, buffer->size(), pf);
            const int close_result = std::fclose(pf);
            if (n != buffer->size() || close_result != 0) {
                throw std::runtime_error("failed to write " + output_path.str());
            }
        }
    };
    tbb::parallel_pipeline(256,
                           tbb::make_filter<void, bakuage::AlignedPodVector<float> *>(tbb::filter_mode::serial_in_order, filter1_func)
                           & tbb::make_filter<bakuage::AlignedPodVector<float> *,BufferPtr>(tbb::filter_mode::parallel, filter2_func)
                           & tbb::make_filter<BufferPtr, void>(tbb::filter_mode::serial_in_order, filter3_func)
                           );
    if (std::fflush(stdout) != 0) {
        throw std::runtime_error("failed to flush stdout");
    }

    PrintMemoryUsage();

    if (FLAGS_quick_exit) {
        // 普通に終了するとクラッシュする。多分thread_localとかのデストラクタ周り
        // CircleCI上で再現したので要注意
        // クラッシュ回避ついでに効率的に終了できるので使うが、根本的にバグも直したい
        // https://stackoverflow.com/questions/24821265/exiting-a-c-app-immediately
        std::cerr << "quick exiting: " << 0 << std::endl;
        std::_Exit(0);
    } else {
        return 0;
    }
}

