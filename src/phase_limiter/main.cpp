#define BOOST_LIB_DIAGNOSTIC

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <chrono>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <fstream>
#include <streambuf>

#include "boost/filesystem.hpp"
#include "boost/serialization/vector.hpp"
#include "gflags/gflags.h"
#include "picojson.h"
#include "ipp.h"

#include "audio_analyzer/peak.h"
#include "bakuage/loudness_ebu_r128.h"
#include "bakuage/compressor_filter.h"
#include "bakuage/file_utils.h"
#include "bakuage/utils.h"
#include "bakuage/memory.h"
#include "bakuage/ffmpeg.h"
#include "phase_limiter/GradCalculator.h"
#include "phase_limiter/pre_compression.h"
#include "phase_limiter/auto_mastering.h"
#include "phase_limiter/equalization.h"
#include "phase_limiter/resampling.h"
#include "phase_limiter/enhancement.h"
#include "phase_limiter/freq_expander.h"
#include "phase_limiter/config.h"
#include "phase_limiter/wave_utils.h"

DEFINE_bool(quick_exit, true, "quick exit");

DEFINE_bool(enhancement, false, "Enhancement before mastering.");

DEFINE_bool(freq_expansion, false, "Freq expansion before mastering.");
DEFINE_double(freq_expansion_ratio, 1.5, "Freq expansion ratio(>1).");

DEFINE_bool(mastering, false, "Mastering before pre-compression.");
DEFINE_bool(mastering_parallel_compression, true, "Mastering parallel compression enable.");
DEFINE_bool(mastering_reverb, false, "Mastering reverb enable");
DEFINE_double(mastering_reverb_gain, -10, "Mastering reverb gain in db.");
DEFINE_double(mastering_reverb_drr_range, 20, "Mastering reverb drr change range in db.");
DEFINE_double(mastering_reverb_predelay, 0.02, "Mastering reverb gain in sec.");
DEFINE_double(mastering_reverb_target_drr, -12, "Mastering target drr in db (unused).");
DEFINE_bool(mastering_reverb_ensure_monotone, true, "Ensure that the loudness mapping of reverb compressor is monotone.");
DEFINE_string(mastering_reverb_ir, "", "Mastering reverb IR file path.");
DEFINE_string(mastering_reverb_ir_left, "", "Mastering reverb IR left wav file path.");
DEFINE_string(mastering_reverb_ir_right, "", "Mastering reverb IR right wav file path.");
DEFINE_string(mastering_reference_file, "./mastering_reference.json", "Mastering reference json.");
DEFINE_double(mastering_matching_level, 1, "degree of loudness matching in 0-1.");
DEFINE_double(mastering_ms_matching_level, 1, "degree of stereo matching in 0-1 (independent from mastering_matching_level).");
DEFINE_string(mastering_mode, "classic", "classic / mastering2 / mastering3 / mastering5 (improve of classic)");
DEFINE_string(mastering2_config_file, "", "Mastering 2 config file path.");
DEFINE_int32(mastering3_iteration, 1000, "Mastering 3 optimization iteration count.");
DEFINE_double(mastering3_target_sn, 12, "Target S/N in dB used for Acoustic entropy calculation.");
DEFINE_string(sound_quality2_cache, "./sound_quality2_cache", "sound quality2 cache path.");
DEFINE_string(mastering5_optimization_algorithm, "de_prmm", "de / nm / pso / de_prmm / pso_dv");
DEFINE_int32(mastering5_optimization_max_eval_count, 40000, "Mastering5 optimization max eval count.");
DEFINE_double(mastering5_mastering_level, 0.5, "Mastering5 mastering level.");
DEFINE_string(mastering5_mastering_reference_file, "", "Mastering reference json path.");

DEFINE_int32(worker_count, 0, "worker count (0: auto detect)");

DEFINE_string(reference_mode, "loudness", "reference mode (loudness, youtube_loudness, rms, peak, zero)");
DEFINE_double(reference, -6.0, "reference level in db.");
DEFINE_bool(disable_input_encode, false, "disable input encode before load wave.");
DEFINE_string(input, "", "Input wave file path.");
DEFINE_string(output, "", "Output wave file path.");
DEFINE_string(output_after_pre_compression, "", "Path of output wave file just after pre compression.");
DEFINE_bool(pre_compression, true, "Pre-compression before limiting.");
DEFINE_double(pre_compression_threshold, 6.0, "Pre-compression threshold relative to loudness.");
DEFINE_double(pre_compression_mean_sec, 0.2, "Pre-compression mean sec.");
DEFINE_double(start_at, 0, "Output wave start time.");
DEFINE_double(end_at, -1, "Output wave end time (set negative value to disable).");
DEFINE_double(low_cut_freq, 20, "Low cut frequency (set 0 to disable).");
DEFINE_double(high_cut_freq, 20000, "High cut frequency (set 0 to disable).");
DEFINE_bool(histogram, false, "Histogram mode.");
DEFINE_bool(erb_eval_func_weighting, false, "Enable eval function weighting by ERB");
DEFINE_string(limiting_mode, "phase", "limiting mode (phase, simple)");
DEFINE_string(ceiling_mode, "true_peak", "peak, true_peak, or lowpass_true_peak");
DEFINE_double(ceiling, 0, "ceiling level in db.");
DEFINE_double(lowpass_true_peak_cut_freq, 15000, "lowpass true peak cut off frequency");
DEFINE_int32(true_peak_oversample, 4, "true peak oversample");

DEFINE_string(output_format, "wav", "output format (wav/mp3/aac)");
DEFINE_int32(bit_depth, 16, "bit depth (16 or 24)");
DEFINE_int32(sample_rate, 44100, "Output sample rate (not processing sample rate)");

DEFINE_string(grad_output, "", "grad output path");
DEFINE_string(limiting_error_spectrogram_output, "", "limiting error spectrogram output png path");
DEFINE_int32(limiting_error_spectrogram_width, 640, "limiting error spectrogram width");
DEFINE_int32(limiting_error_spectrogram_height, 480, "limiting error spectrogram height");
DEFINE_double(limiting_error_spectrogram_gain, 60, "gain in dB");

DEFINE_int32(limiter_external_oversample, 1, "limiter oversample out of phase_limiter algorithm (slow, consume memory)");
DEFINE_int32(limiter_internal_oversample, 1, "limiter oversample in phase_limiter algorithm (fast, memory efficient)");

DEFINE_string(max_available_freq_mode, "disabled", "disabled / detect");

DEFINE_string(test_mode, "", "empty / grad / grad_calculator / perfect_hash_power_of_2");

DEFINE_string(noise_update_mode, "linear", "linear / adaptive");
DEFINE_double(noise_update_min_noise, 1e-6, "min noise");
DEFINE_double(noise_update_initial_noise, 1, "initial noise");
DEFINE_double(noise_update_fista_enable_ratio, 1.0, "the ratio of fista enable iteration. fista is enabled only if log(noise_update_initial_noise / noise) <= noise_update_fista_enable_ratio * log(noise_update_initial_noise / noise_update_min_noise).");
DEFINE_double(absolute_min_noise, 1e-6, "absolute min noise (independent from noise weighting)");
DEFINE_int32(max_iter1, 100, "max optimization iteration count (outer loop)");
DEFINE_int32(max_iter2, 400, "max optimization iteration count (inner loop)");

DEFINE_bool(perf_src_cache, true, "use IFFT of src wave cache (performance option)");

DEFINE_string(ffmpeg, "ffmpeg", "ffmpeg executable path.");

#ifdef _MSC_VER
DEFINE_string(tmp, "tmp", "Temporary file directory.");
#else
DEFINE_string(tmp, "/tmp/phase_limiter", "Temporary file directory.");
#endif

#undef max

using namespace bakuage;

namespace {

class Exception: public std::exception {
public:
    Exception(const std::string &message): std::exception(), what_(message) {}
    virtual const char *what() const throw () { return what_.c_str(); }
private:
    std::string what_;
};

void InitializeGFlags(int *argc, char **argv[]) {
    gflags::SetVersionString("1.0.0-oss");

    std::stringstream usage;
    usage << "Currently, output format is only (44.1kHz, stereo, 16bit, wave).\n"
        << "So, pre-encode is not supported.";
    gflags::SetUsageMessage(usage.str());

    gflags::ParseCommandLineFlags(argc, argv, true);
}

void PrintMemoryUsage() {
    std::cerr << "Peak RSS(MB)\t" << bakuage::GetPeakRss() / (1024 * 1024)
        << "\tCurrent RSS(MB)\t" << bakuage::GetCurrentRss() / (1024 * 1024)
    << std::endl;
}

template <class Float>
void Normalize(std::vector<Float> *wave, float l2_normalization = false) {
    Float scale = 0;
    if (l2_normalization) {
        scale = 1 / (1e-37 + bakuage::VectorL2(wave->data(), wave->size()));
    } else {
        scale = 1 / (1e-37 + bakuage::VectorLInf(wave->data(), wave->size()));
    }
    bakuage::VectorMulConstantInplace(scale, wave->data(), wave->size());
}

void OutputProgression(double progression) {
	std::cout << "progression: " << progression << std::endl;
}

// in-place for memory efficiency
template <class Float>
void PhaseLimitInplace(std::vector<Float> *wave) {
    auto original_wave = *wave;
    const int base_sample_rate = 44100;
    const int limiter_sample_rate = base_sample_rate * FLAGS_limiter_external_oversample * FLAGS_limiter_internal_oversample;

    float max_avilable_normalized_freq = 0.5;
    if (FLAGS_max_available_freq_mode == "detect") {
        phase_limiter::CalcMaxAvailableNormalizedFreq(wave, 2, &max_avilable_normalized_freq);
        std::cerr << "max_avilable_freq(Hz) " << max_avilable_normalized_freq * base_sample_rate << std::endl;
        PrintMemoryUsage();
    }

    phase_limiter::Upsample(wave, 2, FLAGS_limiter_external_oversample * FLAGS_limiter_internal_oversample);
    std::cerr << "upsampled" << std::endl;
    PrintMemoryUsage();

    {
        phase_limiter::GradCalculator<phase_limiter::DefaultSimdType> calculator(wave->size() / 2, limiter_sample_rate, base_sample_rate * max_avilable_normalized_freq, FLAGS_worker_count, FLAGS_noise_update_mode.c_str(), FLAGS_noise_update_min_noise, FLAGS_noise_update_initial_noise, FLAGS_noise_update_fista_enable_ratio, FLAGS_max_iter1, FLAGS_max_iter2, FLAGS_limiter_internal_oversample);
        PrintMemoryUsage();
        if (FLAGS_histogram) {
            calculator.histogram = new std::vector<int>();
        }
        // max_available_freqを使ったときのnormalized evalはあてにならないが、
        // なるべく当てになる計算方法を使って計算する
        Float *ptr_array[2];
        ptr_array[0] = &(*wave)[0];
        ptr_array[1] = &(*wave)[1];
        calculator.copyWaveSrcFrom(ptr_array, 2);
        const auto unit_eval = calculator.outputUnitEval("src_with_cut"); // waveSrcとwaveOutを汚染
        calculator.copyWaveSrcFrom(ptr_array, 2);

        calculator.optimizeWithProgressCallback([&calculator](double progress) {
            OutputProgression(0.3 + 0.7 * progress);

            if (FLAGS_histogram && calculator.histogram->size() > 0) {
                std::vector<int> &h = *calculator.histogram;
                for (int i = 0; i < h.size(); i++) {
                    std::cout << i - 100 << ": " << h[i] << std::endl;
                }
                int a;
                std::cin >> a;
            }
        }, unit_eval);

        calculator.copyWaveProxTo(ptr_array, 2);
        PrintMemoryUsage();
    }

    phase_limiter::Downsample(wave, 2, FLAGS_limiter_external_oversample * FLAGS_limiter_internal_oversample);
    std::cerr << "downsampled" << std::endl;
    PrintMemoryUsage();

    // max_available_freqを使ったときのnormalized evalはあてにならないのと、
    // oversampleによって計算がずれるのも分かりづらいので、ここで計算
    {
        phase_limiter::GradCoreSettings::GetInstance().set_src_cache(false);
        std::cerr << "calculate correct normalized eval" << std::endl;
        // max_available_freqは十分大きくして、normalized_evalを正確に計算できるようにする
        phase_limiter::GradCalculator<phase_limiter::DefaultSimdType> calculator(wave->size() / 2, base_sample_rate, 2 * base_sample_rate, FLAGS_worker_count, FLAGS_noise_update_mode.c_str(), FLAGS_noise_update_min_noise, FLAGS_noise_update_initial_noise, FLAGS_noise_update_fista_enable_ratio, FLAGS_max_iter1, FLAGS_max_iter2, 1);
        const auto unit_eval = calculator.outputUnitEval("noise");
        Float *ptr_array[2];
        ptr_array[0] = &original_wave[0];
        ptr_array[1] = &original_wave[1];
        calculator.copyWaveSrcFrom(ptr_array, 2);
        ptr_array[0] = &(*wave)[0];
        ptr_array[1] = &(*wave)[1];
        calculator.copyWaveProxFrom(ptr_array, 2);
        const auto eval = calculator.CalcEvalGradFromProx(FLAGS_noise_update_min_noise, unit_eval);
        const auto normalized_eval = eval / (1e-37 + unit_eval);
        const auto limiting_error = 10 * std::log10(1.0 + normalized_eval * (std::pow(10, 0.1) - 1.0));
        std::cerr << "normalized_eval:" << normalized_eval <<
        "\tlimiting_error:" << limiting_error << std::endl;
        phase_limiter::GradCoreSettings::GetInstance().set_src_cache(FLAGS_perf_src_cache);

        if (!FLAGS_grad_output.empty() || !FLAGS_limiting_error_spectrogram_output.empty()) {
            // output limiting error spectrogram (一旦は簡易的にgradのspectrogramをffmpegで生成する)
            auto grad = *wave;
            ptr_array[0] = &grad[0];
            ptr_array[1] = &grad[1];
            calculator.copyGradTo(ptr_array, 2);
            Normalize(&grad);

            if (!FLAGS_grad_output.empty()) {
                phase_limiter::SaveFloatWave(grad, FLAGS_grad_output);
            }

            if (!FLAGS_limiting_error_spectrogram_output.empty()) {
                TemporaryFiles temporary_files(FLAGS_tmp);
                std::string float_wav_filename = temporary_files.UniquePath(".wav");
                bakuage::VectorMulConstantInplace(std::pow(10, FLAGS_limiting_error_spectrogram_gain / 20), grad.data(), grad.size());
                phase_limiter::SaveFloatWave(grad, float_wav_filename);

                std::stringstream ss;
                ss << "-lavfi showspectrumpic=scale=log:s=" << FLAGS_limiting_error_spectrogram_width << "x" << FLAGS_limiting_error_spectrogram_height;
                boost::filesystem::remove(FLAGS_limiting_error_spectrogram_output);
                FFMpeg::Execute(FLAGS_ffmpeg, float_wav_filename, FLAGS_limiting_error_spectrogram_output, ss.str());
            }
        }
    }
}

// シンプルなリミッター。あまり音質がよくない
template <class Float>
void SimpleLimitInplace(std::vector<Float> *wave) {
	const int frames = wave->size() / 2;
	std::vector<Float> gains(frames, 1.0);
	OutputProgression(0.3 + 0.7 * 0);
	const int peak_half_window = (int)(44100 * 0.02);
	std::vector<Float> weights(peak_half_window + 1);
	for (int i = 0; i <= peak_half_window; i++) {
		weights[i] = 0.5 + 0.5 * std::cos(M_PI * i / peak_half_window);
	}
	for (int i = 0; i < frames; i++) {
		const Float peak = std::max(std::abs((*wave)[2 * i + 0]), std::abs((*wave)[2 * i + 1]));
		const Float gain = 1.0 / std::max<Float>(1, peak);
		for (int j = std::max<int>(0, i - peak_half_window); j < std::min<int>(frames, i + peak_half_window); j++) {
			const Float weight = weights[std::abs(i - j)];
			gains[j] = std::min<Float>(gains[j], gain * weight + 1 * (1 - weight));
		}
	}
	OutputProgression(0.3 + 0.7 * 0.5);
	Float min_gain = 1;
	for (int i = 0; i < frames; i++) {
		(*wave)[2 * i + 0] *= gains[i];
		(*wave)[2 * i + 1] *= gains[i];
		min_gain = std::min<Float>(min_gain, gains[i]);
	}
	std::cerr << "simple limiting min gain " << min_gain << std::endl;
	OutputProgression(0.3 + 0.7 * 1);
}

template <typename T>
std::string FormatMetadata(const std::string &key, T value) {
    std::stringstream s;
    s << " -metadata " << key << "=\"" << value << "\" ";
    return s.str();
}

template <class Float>
Float CalculateCeilingPeak(const std::vector<Float> &wave, int channels, int sample_rate) {
    if (FLAGS_ceiling_mode == "peak") {
        Float peak;
        audio_analyzer::CalculatePeakAndRMS<Float>(wave.data(),
                                                   channels, wave.size() / channels,
                                                   &peak, nullptr, 0, nullptr);
        return peak;
    } else if (FLAGS_ceiling_mode == "true_peak") {
        Float true_peak;
        audio_analyzer::CalculatePeakAndRMS<Float>(wave.data(),
                                                   channels, wave.size() / channels,
                                                   nullptr, nullptr, FLAGS_true_peak_oversample, &true_peak);
        return true_peak;
    } else if (FLAGS_ceiling_mode == "lowpass_true_peak") {
        Float true_peak;
        audio_analyzer::CalculatePeakAndRMS<Float>(wave.data(),
                                                   channels, wave.size() / channels,
                                                   nullptr, nullptr, FLAGS_true_peak_oversample, &true_peak);
        Float lowpass_true_peak;
        audio_analyzer::CalculateLowpassTruePeak<Float>(wave.data(),
                                                   channels, wave.size() / channels,
                                                   sample_rate, FLAGS_lowpass_true_peak_cut_freq, FLAGS_true_peak_oversample, &lowpass_true_peak);
        return std::max<Float>(true_peak, lowpass_true_peak);
    } else {
        std::stringstream er;
        er << "unknown ceiling mode " << FLAGS_ceiling_mode;
        throw std::logic_error(er.str());
    }
}

std::string FFMpegOutputFormatOptions(const std::string &output_format, int bit_depth, int channels, int sample_rate) {
	std::stringstream ss;
	if (output_format == "wav") {
		if (bit_depth == 16) {
			ss << " -acodec pcm_s16le -f wav ";
		}
		else if (bit_depth == 24) {
			ss << " -acodec pcm_s24le -f wav ";
		}
		else if (bit_depth == 32) {
			ss << " -acodec pcm_f32le -f wav ";
		}
		else {
			std::stringstream er;
			er << "unknown bit_depth " << bit_depth;
			throw std::logic_error(er.str());
		}
	}
	else if (output_format == "mp3") {
		ss << " -acodec libmp3lame -ab 320k -f mp3 ";
	}
    else if (output_format == "aac") {
        ss << " -acodec aac -b:a 256k -f adts ";
    }
	else {
		std::stringstream er;
		er << "unknown output_format " << output_format;
		throw std::logic_error(er.str());
	}
	ss << " -ac " << channels << " -ar " << sample_rate << " ";
	return ss.str();
}

std::string OutputFileExtension(const std::string &output_format) {
	std::stringstream ss;
	ss << "." << output_format;
	return ss.str();
}

template <class Float>
void EncodeAvoidingClipping(const std::string &input, const std::string &output, const std::string &temp, const std::string &output_format_options, std::vector<Float> *encoded_wave) {
    const Float log2Threshold = std::log2(std::pow(10, FLAGS_ceiling / 20.0));
    const Float log2Resolution = std::log2(std::pow(10, 0.5 / 20.0));
    const int max_iter = 3;
    Float log2NewScale = log2Threshold - 0.5 * log2Resolution;

    for (int iter = 0; iter < max_iter; iter++) {
		std::stringstream ss;
		ss << output_format_options << "-filter:a \"volume = " << std::pow(2, log2NewScale) << "\"";
		boost::filesystem::remove(output);
		FFMpeg::Execute(FLAGS_ffmpeg, input, output, ss.str());

		// クリップ検知
		boost::filesystem::remove(temp);
		FFMpeg::Execute(FLAGS_ffmpeg, output, temp, "-acodec pcm_f32le -ac 2 -f wav"); // not convert sample rate
        *encoded_wave = phase_limiter::LoadFloatWave<Float>(temp);
        const auto ceiling_peak = std::pow(10, CalculateCeilingPeak(*encoded_wave, 2, FLAGS_sample_rate) / 20.0);
        const auto log2Peak = std::log2(ceiling_peak + 1e-37);

        if (log2Peak < log2Threshold - log2Resolution) {
            // 小さすぎる
            std::cerr << "clip not detected but too small" << std::endl;
            log2NewScale += log2Threshold - 0.5 * log2Resolution - log2Peak;
        } else if (log2Peak <= log2Threshold) {
            // ちょうど良い
            std::cerr << "clip not detected" << std::endl;
            break;
        } else {
            // 大きすぎる
            std::cerr << "clip detected " << ceiling_peak << " shrinking wave " << std::pow(2, log2NewScale) << std::endl;
            log2NewScale += log2Threshold - 0.5 * log2Resolution - log2Peak;
        }
	}
}

void MainFunc() {
    typedef float Float;

    if (FLAGS_limiter_external_oversample != bakuage::CeilPowerOf2(FLAGS_limiter_external_oversample)) {
        throw std::logic_error("limiter_external_oversample must be power of 2");
    }
    if (FLAGS_limiter_internal_oversample != bakuage::CeilPowerOf2(FLAGS_limiter_internal_oversample)) {
        throw std::logic_error("limiter_internal_oversample must be power of 2");
    }

	StopWatch stop_watch;
	stop_watch.Start();

    TemporaryFiles temporary_files(FLAGS_tmp);
	std::string encoded_filename = temporary_files.UniquePath(OutputFileExtension(FLAGS_output_format));
	std::string float_wav_filename = temporary_files.UniquePath(".wav");
	std::string float_wav_filename2 = temporary_files.UniquePath(".wav");

	// 44100で処理
    std::vector<Float> wave;
    if (FLAGS_disable_input_encode) {
        // load wave in float
        wave = phase_limiter::LoadFloatWave<Float>(FLAGS_input);
        std::cerr << "load wave in float lap: " << stop_watch.time() << std::endl;
        PrintMemoryUsage();
    } else {
        // create normalized format wave
        FFMpeg::Execute(FLAGS_ffmpeg, FLAGS_input, float_wav_filename, "-acodec pcm_f32le -ac 2 -ar 44100 -f wav");
        std::cerr << "create normalized format wave lap: " << stop_watch.time() << std::endl;
        PrintMemoryUsage();

        // load wave in float
        wave = phase_limiter::LoadFloatWave<Float>(float_wav_filename);
        std::cerr << "load wave in float lap: " << stop_watch.time() << std::endl;
        PrintMemoryUsage();
    }

	// cut wave with margin
	const int start_frame = std::max<int>(0, std::min<int>(wave.size() / 2, std::floor(44100 * FLAGS_start_at)));
	const int end_frame = std::max<int>(start_frame, std::min<int>(wave.size() / 2,
		FLAGS_end_at < 0 ? wave.size() / 2 : std::floor(44100 * FLAGS_end_at)));
	const int start_frame_with_margin = std::max<int>(start_frame - 0.5 * 44100, 0);
	const int end_frame_with_margin = std::min<int>(end_frame + 0.5 * 44100, wave.size() / 2);
	const int start_frame_in_margin = start_frame - start_frame_with_margin;
	const int end_frame_in_margin = end_frame - start_frame_with_margin;
	for (int i = start_frame_with_margin; i < end_frame_with_margin; i++) {
		for (int j = 0; j < 2; j++) {
			wave[2 * (i - start_frame_with_margin) + j] = wave[2 * i + j];
		}
	}
	wave.resize(2 * (end_frame_with_margin - start_frame_with_margin));

	// 整える
	phase_limiter::CutLowAndHighFreq(&wave, 2, FLAGS_low_cut_freq / 44100, FLAGS_high_cut_freq / 44100);
	std::cerr << "CutLowAndHighFreq lap: " << stop_watch.time() << std::endl;
    PrintMemoryUsage();

	if (FLAGS_enhancement) {
		std::cerr << "Enhance" << std::endl;
		phase_limiter::Enhance(&wave, 2);
		std::cerr << "Enhance lap: " << stop_watch.time() << std::endl;
        PrintMemoryUsage();
	}

	if (FLAGS_freq_expansion) {
		std::cerr << "Freq Expansion" << std::endl;
		phase_limiter::FreqExpand(&wave, 2, 44100, FLAGS_freq_expansion_ratio);
		std::cerr << "Freq Expansion lap: " << stop_watch.time() << std::endl;
        PrintMemoryUsage();
	}

	// normalize
	Normalize(&wave);
	std::cerr << "Normalize lap: " << stop_watch.time() << std::endl;
    PrintMemoryUsage();

    // auto mastering
    if (FLAGS_mastering) {
		std::cerr << "AutoMastering " << FLAGS_mastering_mode << std::endl;
		if (FLAGS_mastering_mode == "classic") {
			std::vector<float> ir_left(2, 1);
			std::vector<float> ir_right(2, 1);
			if (FLAGS_mastering_reverb) {
				if (!FLAGS_mastering_reverb_ir.empty()) {
					boost::filesystem::remove(float_wav_filename);
					FFMpeg::Execute(FLAGS_ffmpeg, FLAGS_mastering_reverb_ir, float_wav_filename, "-acodec pcm_f32le -ac 2 -ar 44100 -f wav");
					std::vector<float> ir_mono_to_stereo = phase_limiter::LoadFloatWave<Float>(float_wav_filename);
					ir_left.resize(ir_mono_to_stereo.size());
					ir_right.resize(ir_mono_to_stereo.size());
					for (int i = 0; i < ir_mono_to_stereo.size() / 2; i++) {
						ir_left[2 * i + 0] = ir_mono_to_stereo[2 * i + 0];
						ir_right[2 * i + 1] = ir_mono_to_stereo[2 * i + 1];
					}
				}
				else {
					boost::filesystem::remove(float_wav_filename);
					FFMpeg::Execute(FLAGS_ffmpeg, FLAGS_mastering_reverb_ir_left, float_wav_filename, "-acodec pcm_f32le -ac 2 -ar 44100 -f wav");
					ir_left = phase_limiter::LoadFloatWave<Float>(float_wav_filename);
					boost::filesystem::remove(float_wav_filename);
					FFMpeg::Execute(FLAGS_ffmpeg, FLAGS_mastering_reverb_ir_right, float_wav_filename, "-acodec pcm_f32le -ac 2 -ar 44100 -f wav");
					ir_right = phase_limiter::LoadFloatWave<Float>(float_wav_filename);
				}
			}
			const float *irs[2] = { ir_left.data(), ir_right.data() };
			const int ir_lens[2] = { (int)ir_left.size() / 2, (int)ir_right.size() / 2 };
			phase_limiter::AutoMastering(&wave, irs, ir_lens, 44100, [](float p) {
				OutputProgression(0.3 * p);
			});
		}
		else if (FLAGS_mastering_mode == "mastering2") {
			phase_limiter::AutoMastering2(&wave, 44100, [](float p) {
				OutputProgression(0.3 * p);
			});
		}
		else if (FLAGS_mastering_mode == "mastering3") {
			phase_limiter::AutoMastering3(&wave, 44100, [](float p) {
				OutputProgression(0.3 * p);
			});
		}
        else if (FLAGS_mastering_mode == "mastering5") {
            phase_limiter::AutoMastering5(&wave, 44100, [](float p) {
                OutputProgression(0.3 * p);
            });
        }
		else {
            throw std::logic_error(std::string("Unknown mastering mode: ") + FLAGS_mastering_mode);
		}

		std::cerr << "AutoMastering lap: " << stop_watch.time() << std::endl;
        PrintMemoryUsage();
    }

	// 整える
	phase_limiter::CutLowAndHighFreq(&wave, 2, FLAGS_low_cut_freq / 44100, FLAGS_high_cut_freq / 44100);
	std::cerr << "CutLowAndHighFreq lap: " << stop_watch.time() << std::endl;
    PrintMemoryUsage();

	// normalize
	Normalize(&wave);
	std::cerr << "Normalize lap: " << stop_watch.time() << std::endl;
    PrintMemoryUsage();

	// pre-encode (なぜここか？フォーマットによって最適なマスタリングは変わると考えるから、pre-compressionの前。だが、auto-masteringの前だとmatching EQで元に戻されるから都合が悪い)
	if (FLAGS_output_format != "wav") {
		// 事前にエンコードして無駄な高周波を落としておくことで、phase_limit後のエンコードでピークが大きく飛び出るのを防ぐ
        if (FLAGS_output_format == "aac") { // remove priming
            wave.erase(wave.begin(), wave.begin() + std::min<int>(1024 * 2, wave.size() - 2));
        }
		phase_limiter::SaveFloatWave(wave, float_wav_filename);
		EncodeAvoidingClipping(float_wav_filename, encoded_filename, float_wav_filename2, FFMpegOutputFormatOptions(
			FLAGS_output_format,
			FLAGS_bit_depth,
			2,
			44100
		), &wave);
		std::cerr << "pre-encode lap: " << stop_watch.time() << std::endl;
        PrintMemoryUsage();
	}

    // pre-compression
    if (FLAGS_pre_compression) {
        phase_limiter::PreCompress(&wave, 44100);
		std::cerr << "pre-compression lap: " << stop_watch.time() << std::endl;
        PrintMemoryUsage();
    }

	// 整える
	phase_limiter::CutLowAndHighFreq(&wave, 2, FLAGS_low_cut_freq / 44100, FLAGS_high_cut_freq / 44100);
	std::cerr << "pre-CutLowAndHighFreq lap: " << stop_watch.time() << std::endl;
    PrintMemoryUsage();

	// normalize
	Normalize(&wave);
	std::cerr << "Normalize lap: " << stop_watch.time() << std::endl;
    PrintMemoryUsage();

	// save just after pre-compression
	if (!FLAGS_output_after_pre_compression.empty()) {
		const bool clip_detect = FLAGS_output_format != "wav" || FLAGS_sample_rate != 44100;
		phase_limiter::SaveFloatWave(wave, float_wav_filename);
		std::stringstream options;
		options << FFMpegOutputFormatOptions(
			FLAGS_output_format,
			FLAGS_bit_depth,
			2,
			44100
		);
		if (clip_detect) {
			std::cerr << "clip detect enabled" << std::endl;
			EncodeAvoidingClipping(float_wav_filename, encoded_filename, float_wav_filename2, options.str(), &wave);
		}
		else {
			std::cerr << "clip detect disabled" << std::endl;
			boost::filesystem::remove(encoded_filename);
			FFMpeg::Execute(FLAGS_ffmpeg, float_wav_filename, encoded_filename, options.str());
		}
		std::cerr << "save lap: " << stop_watch.time() << std::endl;

		boost::filesystem::remove(FLAGS_output_after_pre_compression);
		boost::filesystem::rename(encoded_filename, FLAGS_output_after_pre_compression);
		std::cerr << "rename lap: " << stop_watch.time() << std::endl;

		std::cout << "output_after_pre_compression" << std::endl;
	}

	// 44100で処理
	// phase_limiter::Downsample(&wave, 2, 2);

    // calculate gain
    Float gain = 0;
    if (FLAGS_reference_mode == "loudness") {
        // calculate loudness
        Float loudness;
        std::vector<int> histogram;
        bakuage::loudness_ebu_r128::CalculateLoudness(wave.data(),
            2, wave.size() / 2, 44100,
            &loudness, &histogram);
        gain = FLAGS_reference - loudness;
    }
    else if (FLAGS_reference_mode == "youtube_loudness") {
        // calculate loudness
        Float loudness;
        std::vector<int> histogram;
        bakuage::loudness_ebu_r128::CalculateLoudnessCore<Float>(wave.data(), 2, wave.size() / 2, 44100,
                                                                 3, 0.1, -70, -10, nullptr, nullptr, &histogram, nullptr, nullptr, true, &loudness);
        gain = FLAGS_reference - loudness;
    }
    else if (FLAGS_reference_mode == "rms") {
        // calculate rms peak
        Float peak, rms;
        audio_analyzer::CalculatePeakAndRMS<Float>(wave.data(),
            2, wave.size() / 2,
            &peak, &rms, 0, nullptr);
        gain = FLAGS_reference - rms;
    }
	else if (FLAGS_reference_mode == "peak") {
		// calculate rms peak
		Float peak, rms;
		audio_analyzer::CalculatePeakAndRMS<Float>(wave.data(),
			2, wave.size() / 2,
			&peak, &rms, 0, nullptr);
		gain = FLAGS_reference - peak;
	}
    else if (FLAGS_reference_mode == "zero") {
        gain = 0;
    }
    else {
        std::stringstream message;
        message << "Not supported reference_mode: " << FLAGS_reference_mode;
        throw Exception(message.str());
    }
    std::cerr << "gain: " << gain << std::endl;
	std::cerr << "calculate gain lap: " << stop_watch.time() << std::endl;
    PrintMemoryUsage();

    // Apply gain (ceiling調整も含む)
	bool need_limiting = false;
    const Float r = std::pow(10.0, (gain - FLAGS_ceiling) / 20);
    for (int i = 0; i < wave.size(); i++) {
        wave[i] *= r;
		if (std::abs(wave[i]) >= 1 + 0.5 / 65536) {
			need_limiting = true;
		}
    }
	std::cerr << "Apply gain lap: " << stop_watch.time() << std::endl;
    PrintMemoryUsage();

    // phase limiting
	if (need_limiting) {
		if (FLAGS_limiting_mode == "phase") {
			PhaseLimitInplace(&wave);
		}
		else if (FLAGS_limiting_mode == "simple") {
			SimpleLimitInplace(&wave);
		}
		else {
			throw std::logic_error("unknown limiting mode: " + FLAGS_limiting_mode);
		}

		std::cerr << "phase limiting lap: " << stop_watch.time() << std::endl;
        PrintMemoryUsage();
	}

    // post ceiling調整
    bakuage::VectorMulConstantInplace(std::pow(10.0, FLAGS_ceiling / 20), wave.data(), wave.size());

	// trim margin
	for (int i = start_frame_in_margin; i < end_frame_in_margin; i++) {
		for (int j = 0; j < 2; j++) {
			wave[2 * (i - start_frame_in_margin) + j] = wave[2 * i + j];
		}
	}
	wave.resize(2 * (end_frame_in_margin - start_frame_in_margin));

    // 強制的にceilingに収める
    {
        const auto ceiling_peak_db = CalculateCeilingPeak(wave, 2, 44100);
        if (ceiling_peak_db > FLAGS_ceiling) {
            bakuage::VectorMulConstantInplace(std::pow(10, (FLAGS_ceiling - ceiling_peak_db) / 20.0), wave.data(), wave.size());
        }
    }

    // save
	{
		const bool clip_detect = FLAGS_output_format != "wav" || FLAGS_sample_rate != 44100;
        if (FLAGS_output_format == "aac") { // remove priming
            wave.erase(wave.begin(), wave.begin() + std::min<int>(1024 * 2, wave.size() - 2));
        }
		phase_limiter::SaveFloatWave(wave, float_wav_filename);
		std::stringstream options;
		options << FFMpegOutputFormatOptions(
			FLAGS_output_format,
			FLAGS_bit_depth,
			2,
			FLAGS_sample_rate
		);
		// 無視されるっぽい
		/* << FormatMetadata("bakuage_version", Version())
		<< FormatMetadata("bakuage_reference_db", FLAGS_reference)
		<< FormatMetadata("bakuage_reference_mode", FLAGS_reference_mode)*/;
		if (clip_detect) {
			std::cerr << "clip detect enabled" << std::endl;
			EncodeAvoidingClipping(float_wav_filename, encoded_filename, float_wav_filename2, options.str(), &wave);
		}
		else {
			std::cerr << "clip detect disabled" << std::endl;
			boost::filesystem::remove(encoded_filename);
			FFMpeg::Execute(FLAGS_ffmpeg, float_wav_filename, encoded_filename, options.str());
		}
		std::cerr << "save lap: " << stop_watch.time() << std::endl;
        PrintMemoryUsage();

		boost::filesystem::remove(FLAGS_output);
		boost::filesystem::rename(encoded_filename, FLAGS_output);
		std::cerr << "rename lap: " << stop_watch.time() << std::endl;
        PrintMemoryUsage();
	}
}
} // namespace

void TestGrad();
void TestGradCalculator();
void TestPerfectHashPowerOf2();

int main(int argc, char* argv[]) {
    int exit_status = 0;

    try {
        InitializeGFlags(&argc, &argv);

        PrintMemoryUsage();

        ippInit();
        const IppLibraryVersion *lib = ippGetLibVersion();
        std::cerr << "Ipp initialized " << lib->Name << " " << lib->Version << std::endl;
        PrintMemoryUsage();

        // TBBの初期化とか (ここで初期化しておくと、毎回初期化しなくても良いらしい)
        // https://www.xlsoft.com/jp/products/intel/perflib/tbb/41/tbb_userguide_lnx/reference/task_scheduler/task_scheduler_init_cls.htm
        tbb::task_scheduler_init tbb_init(FLAGS_worker_count ? FLAGS_worker_count : tbb::task_scheduler_init::default_num_threads());
        std::cerr << "TBB default_num_threads:" << tbb::task_scheduler_init::default_num_threads() << std::endl;
        PrintMemoryUsage();

        phase_limiter::GradCoreSettings::GetInstance().set_erb_eval_func_weighting(FLAGS_erb_eval_func_weighting);
        phase_limiter::GradCoreSettings::GetInstance().set_src_cache(FLAGS_perf_src_cache);
        phase_limiter::GradCoreSettings::GetInstance().set_absolute_min_noise(FLAGS_absolute_min_noise);

        if (FLAGS_test_mode == "grad") {
            TestGrad();
        } else if (FLAGS_test_mode == "grad_calculator") {
            TestGradCalculator();
        } else if (FLAGS_test_mode == "perfect_hash_power_of_2") {
            TestPerfectHashPowerOf2();
        } else {
            MainFunc();
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
    catch (const std::exception& ex) {
        // ensure flush by std::endl
        std::cerr << "exception (std::exception): " << ex.what() << std::endl;
        exit_status = 1;
    } catch (const std::string& ex) {
        std::cerr << "exception (std::string): " << ex << std::endl;
        exit_status = 2;
    } catch (...) {
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
