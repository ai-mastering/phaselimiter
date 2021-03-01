#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <sstream>
#include <chrono>
#include <string>
#include <fstream>
#include <stdexcept>

#include "ipp.h"
#include "gflags/gflags.h"
#include "picojson.h"

#include "audio_analyzer/single_mode.h"

DEFINE_bool(quick_exit, true, "Enable quick exit (this is not compatible with profiler. not output gmon.out)");

DEFINE_bool(analysis_for_visualization, true, "Enable analysis for visualuzation.");
DEFINE_bool(freq_pan_to_db, true, "Enable freq_pan_to_db.");
DEFINE_bool(drr, true, "Enable direct_reverb_ratio.");
DEFINE_bool(sound_quality, false, "Enable sound_quality (currently sound_quality is calculated in bakuage_api_server. so, this is not needed).");
DEFINE_bool(sound_quality2, true, "Enable sound_quality2");
DEFINE_string(ffmpeg, "ffmpeg", "ffmpeg executable path.");
DEFINE_string(input, "", "Input wave file path");
DEFINE_string(spectrogram_output, "", "spectrogram output png path");
DEFINE_string(rhythm_spectrogram_output, "", "rhythm spectrogram output png path");
DEFINE_string(nmf_spectrogram_output, "", "nmf spectrogram output png path");
DEFINE_string(spectrum_distribution_output, "", "spectrum distribution output png path");
DEFINE_string(stereo_distribution_output, "", "stereo distribution output png path");
DEFINE_string(analysis_data_dir, "resource/analysis_data", "analysis data dir path");
DEFINE_string(sound_quality2_cache, "resource/sound_quality2_cache", "sound quality2 cache path.");
DEFINE_string(sound_quality2_cache_archiver, "binary", "sound quality2 cache archiver type. binary/text");
DEFINE_int32(mastering3_acoustic_entropy_band_count, 40, "band count of mel bands used by mastering3 acoustic entropy");
DEFINE_int32(true_peak_oversample, 4, "true peak oversample");

DEFINE_int32(worker_count, 0, "0: auto detect");

DEFINE_double(youtube_loudness_window_sec, 3.0, "youtube loudness window sec");
DEFINE_double(youtube_loudness_shift_sec, 0.1, "youtube loudness shift sec");
DEFINE_double(youtube_loudness_absolute_threshold, -70, "youtube loudness absolute threshold");
DEFINE_double(youtube_loudness_relative_threshold, -10, "youtube loudness relative threshold");

DEFINE_string(mode, "default", "default / sound_quality2_preparation / sound_quality2_find_nn / sound_quality_test / dft_test");

#ifdef _MSC_VER
DEFINE_string(tmp, "tmp", "Temporary file directory.");
#else
DEFINE_string(tmp, "/tmp/phase_limiter", "Temporary file directory.");
#endif

namespace {

typedef float Float;
using namespace audio_analyzer;
using namespace bakuage;
using std::fprintf;

    void PrintMemoryUsage() {
        std::cerr << "Peak RSS(MB)\t" << bakuage::GetPeakRss() / (1024 * 1024)
        << "\tCurrent RSS(MB)\t" << bakuage::GetCurrentRss() / (1024 * 1024)
        << std::endl;
    }
}

void PrepareSoundQuality2();
void SoundQuality2FindNn();
void TestSoundQuality();
void TestDft();

int main(int argc, char* argv[]) {
    int exit_status = 0;
	try {
		gflags::SetVersionString("1.0.0-oss");
		gflags::ParseCommandLineFlags(&argc, &argv, true);

		ippInit();
		const IppLibraryVersion *lib = ippGetLibVersion();
		std::cerr << "Ipp initialized " << lib->Name << " " << lib->Version << std::endl;
		PrintMemoryUsage();

		tbb::task_scheduler_init tbb_init(FLAGS_worker_count ? FLAGS_worker_count : tbb::task_scheduler_init::default_num_threads());
		std::cerr << "TBB default_num_threads:" << tbb::task_scheduler_init::default_num_threads() << std::endl;


		if (FLAGS_mode == "default") {
			exit_status = single_mode();
		}
		else if (FLAGS_mode == "sound_quality2_preparation") {
			PrepareSoundQuality2();
		}
		else if (FLAGS_mode == "sound_quality2_find_nn") {
			SoundQuality2FindNn();
		}
		else if (FLAGS_mode == "sound_quality_test") {
			TestSoundQuality();
		}
		else if (FLAGS_mode == "dft_test") {
			TestDft();
		}
		else {
			throw std::logic_error("Unknown mode");
		}
	}
	catch (const std::exception& ex) {
		// ensure flush by std::endl
		std::cerr << "exception (std::exception): " << ex.what() << std::endl;
		exit_status = 1;
	}
	catch (const std::string& ex) {
		std::cerr << "exception (std::string): " << ex << std::endl;
		exit_status = 2;
	}
	catch (...) {
		std::cerr << "exception (unknown)" << std::endl;
		exit_status = 3;
	}

    PrintMemoryUsage();

    if (FLAGS_quick_exit) {
        std::cerr << "quick exiting: " << exit_status << std::endl;
        std::_Exit(exit_status);
    }

    return exit_status;
}
