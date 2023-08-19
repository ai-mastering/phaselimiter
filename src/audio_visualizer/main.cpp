#include <cstdio>
#include <iostream>

#include "gflags/gflags.h"
#include "sndfile.h"
#include "CImg.h"
#include "ipp.h"
#include "tbb/tbb.h"
#include "tbb/pipeline.h"
#include "tbb/scalable_allocator.h"
#include "tbb/cache_aligned_allocator.h"
#include <boost/filesystem.hpp>
#include "bakuage/sndfile_wrapper.h"
#include "bakuage/dft.h"
#include "bakuage/memory.h"
#include "bakuage/utils.h"
#include "bakuage/window_func.h"
#include "bakuage/vector_math.h"

DEFINE_bool(quick_exit, true, "quick exit");
DEFINE_int32(worker_count, 0, "thread count");
DEFINE_string(input, "", "input wav file path");
DEFINE_string(output, "", "output directory or - for stdout");
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
    typedef std::shared_ptr<bakuage::AlignedPodVector<char>> BufferPtr;

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

void overlay_4ch_on_3ch(const cimg_library::CImg<PixelType> &src, cimg_library::CImg<PixelType> *dest) {
    // CImgのメモリ配置
    /*
     T& operator()(const unsigned int x, const unsigned int y, const unsigned int z, const unsigned int c) {
     return _data[x + y*(ulongT)_width + z*(ulongT)_width*_height + c*(ulongT)_width*_height*_depth];
     }
     */

    bakuage::AlignedPodVector<float> one_minus_alpha(src.width());

    const double scale = 1.0 / 255.0;
    for (int c = 0; c < 3; c++) {
        for (int j = 0; j < src.height(); j++) {
#if 1
            const PixelType *src_row = &src(0, j, 0, c);
            const PixelType *src_alpha_row = &src(0, j, 0, 3);
            PixelType *dest_row = &(*dest)(0, j, 0, c);

            bakuage::VectorSubConstantRev(src_alpha_row, 255.0f, one_minus_alpha.data(), src.width());
            bakuage::VectorMulInplace(one_minus_alpha.data(), dest_row, src.width());
            bakuage::VectorMadInplace(src_alpha_row, src_row, dest_row, src.width());
            bakuage::VectorMulConstantInplace(scale, dest_row, src.width());
#else
#if 1
            // for integer
            for (int i = 0; i < src.width(); i++) {
                const PixelType alpha = src(i, j, 0, 3);
                (*dest)(i, j, 0, c) = ((*dest)(i, j, 0, c) * (255 - alpha) + src(i, j, 0, c) * alpha) >> 8;
            }
#else
            for (int i = 0; i < src.width(); i++) {
                const double alpha = src(i, j, 0, 3) * scale;
                (*dest)(i, j, 0, c) = (*dest)(i, j, 0, c) * (1 - alpha) + src(i, j, 0, c) * alpha;
            }
#endif
#endif
        }
    }
}

}

// wavを受け取って、標準出力にrawvideoを出力
int main(int argc, char* argv[]) {
    gflags::SetVersionString("1.0.0-oss");
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    ippInit();
    const IppLibraryVersion *lib = ippGetLibVersion();
    std::cerr << "Ipp initialized " << lib->Name << " " << lib->Version << std::endl;
    PrintMemoryUsage();

    // TBBの初期化とか (ここで初期化しておくと、毎回初期化しなくても良いらしい)
    // https://www.xlsoft.com/jp/products/intel/perflib/tbb/41/tbb_userguide_lnx/reference/task_scheduler/task_scheduler_init_cls.htm
    tbb::task_scheduler_init tbb_init(FLAGS_worker_count ? FLAGS_worker_count : tbb::task_scheduler_init::default_num_threads());
    std::cerr << "TBB default_num_threads:" << tbb::task_scheduler_init::default_num_threads() << std::endl;
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
    const auto filter2_func = [&foreground, &background](bakuage::AlignedPodVector<float> *spectrum){
        cimg_library::CImg<PixelType> img(FLAGS_width, FLAGS_height, 1, 3, 0);

        // draw background image
        img = background;

        // draw spectrum
        cimg_library::CImg<PixelType> spec_img(FLAGS_width, FLAGS_height, 1, 4, 0);
        spec_img.fill((PixelType)0, 0, 0, 0);
        for (int i = 0; i < spectrum->size(); i++) {
            const int block_div = 24;
            const double center_x = FLAGS_width / 2;
            const double center_y = FLAGS_height * 0.7;
            const double spectrum_width = FLAGS_width * 0.5;
            const double spectrum_height = FLAGS_height * 0.3;
            const double spectrum_left = center_x - spectrum_width / 2;
            const double spectrum_bottom = center_y + spectrum_height / 2;
            const double x1 = spectrum_left + spectrum_width * i / spectrum->size();
            const double x2 = spectrum_left + spectrum_width * (i + 1) / spectrum->size();
            const double space_x = (x2 - x1) * 0.5;
            const int block_count = std::floor((*spectrum)[i] * block_div + 0.5);
            for (int j = 0; j < block_count; j++) {
                const double y1 = spectrum_bottom - spectrum_height * (j + 1) / block_div;
                const double y2 = spectrum_bottom - spectrum_height * j / block_div;
                const double space_y = (y2 - y1) * 0.5;
                const PixelType color[4] = { 255, 255, 255, 200 };
                spec_img.draw_rectangle(x1 + space_x / 2, y1 + space_y / 2, 0, x2 - space_x / 2, y2 - space_y / 2, 1, color);
            }
        }
        overlay_4ch_on_3ch(spec_img, &img);

        // alpha blend foreground image
        overlay_4ch_on_3ch(foreground, &img);

        const auto temp_path = bakuage::NormalizeToString((boost::filesystem::temp_directory_path() / boost::filesystem::unique_path()).native());
        // img.save_bmp(output_path.str().c_str()); // slow
        img.save_png(temp_path.c_str());
        BufferPtr buffer;
        bakuage::LoadDataFromFile(temp_path.c_str(), [&buffer](const char *data, size_t size) {
            buffer = std::make_shared<bakuage::AlignedPodVector<char>>(data, data + size);
        });
        boost::filesystem::remove(temp_path);
        return buffer;
    };
    int spectrogram_i = 0;
    const auto filter3_func = [&spectrogram_i](BufferPtr buffer) {
        if (FLAGS_output == "-") {
            const size_t n = std::fwrite(buffer->data(), 1, buffer->size(), stdout);
            if (n != buffer->size()) {
                throw std::logic_error("failed to write stdout");
            }
        } else {
            std::stringstream output_path;
            output_path << FLAGS_output << "/" << spectrogram_i++ << ".png";

            FILE* pf = std::fopen(output_path.str().c_str(), "wb");
            std::fwrite(buffer->data(), 1, buffer->size(), pf);
            std::fclose(pf);
        }
    };
    tbb::parallel_pipeline(256,
                           tbb::make_filter<void, bakuage::AlignedPodVector<float> *>(tbb::filter::serial, filter1_func)
                           & tbb::make_filter<bakuage::AlignedPodVector<float> *,BufferPtr>(tbb::filter::parallel, filter2_func)
                           & tbb::make_filter<BufferPtr, void>(tbb::filter::serial, filter3_func)
                           );
    std::fflush(stdout);

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

