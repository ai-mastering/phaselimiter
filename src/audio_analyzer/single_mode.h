#ifndef BAKUAGE_AUDIO_ANALYZER_SINGLE_MODE_H_
#define BAKUAGE_AUDIO_ANALYZER_SINGLE_MODE_H_

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <vector>
#include <sstream>
#include <chrono>
#include <string>
#include <fstream>
#include <mutex>

#include "boost/filesystem.hpp"
#include "sndfile.h"
#include <Eigen/Dense>
#include "tbb/tbb.h"
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include "audio_analyzer/peak.h"
#include "audio_analyzer/reverb.h"
#include "bakuage/loudness_ebu_r128.h"
#include "audio_analyzer/multiband_histogram.h"
#include "audio_analyzer/acoustic_entropy.h"
#include "audio_analyzer/spectrum.h"
#include "audio_analyzer/waveform.h"
#include "audio_analyzer/dynamics.h"
#include "audio_analyzer/stereo.h"
#include "audio_analyzer/spectrogram.h"
#include "audio_analyzer/rhythm_spectrogram.h"
#include "audio_analyzer/nmf_spectrogram.h"
#include "audio_analyzer/spectrum_distribution.h"
#include "bakuage/dissonance.h"
#include "bakuage/hardness.h"
#include "bakuage/sndfile_wrapper.h"
#include "bakuage/file_utils.h"
#include "bakuage/window_func.h"
#include "bakuage/dct.h"
#include "bakuage/mastering3_score.h"
#include "bakuage/sound_quality.h"
#include "bakuage/sound_quality2.h"
#include "bakuage/dft.h"
#include "bakuage/memory.h"

DECLARE_bool(analysis_for_visualization);
DECLARE_bool(freq_pan_to_db);
DECLARE_bool(drr);
DECLARE_bool(sound_quality);
DECLARE_bool(sound_quality2);
DECLARE_string(input);
DECLARE_string(spectrogram_output);
DECLARE_string(rhythm_spectrogram_output);
DECLARE_string(nmf_spectrogram_output);
DECLARE_string(spectrum_distribution_output);
DECLARE_string(stereo_distribution_output);
DECLARE_string(analysis_data_dir);
DECLARE_string(sound_quality2_cache);
DECLARE_string(sound_quality2_cache_archiver);
DECLARE_int32(mastering3_acoustic_entropy_band_count);
DECLARE_int32(true_peak_oversample);

DECLARE_double(youtube_loudness_window_sec);
DECLARE_double(youtube_loudness_shift_sec);
DECLARE_double(youtube_loudness_absolute_threshold);
DECLARE_double(youtube_loudness_relative_threshold);

namespace audio_analyzer {

    struct Task {
        Task(const char *_name = "", const std::function<void()> &_func = [](){}): name(_name), func(_func) {}

        void operator () () {
            {
                std::stringstream ss;
                ss << "Task start " << name << std::endl;
                std::cerr << ss.str() << std::flush;
            }
            started_at = std::chrono::steady_clock::now();
            func();
            finished_at = std::chrono::steady_clock::now();
            {
                std::stringstream ss;
                ss << "Task finish " << name << std::endl;
                std::cerr << ss.str() << std::flush;
            }
        }

        double elapsed_sec() const {
            std::chrono::duration<double> dur(finished_at - started_at);
            return dur.count();
        }

        std::string name;
        std::function<void ()> func;
        std::chrono::time_point<std::chrono::steady_clock> started_at, finished_at;
    };

int single_mode() {
    typedef float Float;
    using namespace audio_analyzer;
    using namespace bakuage;
    using std::fprintf;
    using boost::filesystem::recursive_directory_iterator;

	fprintf(stderr, "single mode\n");

    StopWatch stop_watch;

    SndfileWrapper infile;
	SF_INFO sfinfo = { 0 };

    if ((infile.set(sf_open (FLAGS_input.c_str(), SFM_READ, &sfinfo))) == NULL) {
        fprintf(stderr, "Not able to open input file %s.\n", FLAGS_input.c_str());
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

	fprintf(stderr, "lap time %.3f sec\n", stop_watch.time());

	std::mutex task_mtx;
    std::deque<Task> tasks;

    Float peak, rms, true_peak;
	tasks.emplace_back("CalculatePeakAndRMS", [&buffer, sfinfo, &peak, &rms, &true_peak]() {
		CalculatePeakAndRMS(buffer.data(), sfinfo.channels, sfinfo.frames, &peak, &rms, FLAGS_true_peak_oversample, &true_peak);
	});

    Float lowpass_true_peak_15khz;
    tasks.emplace_back("CalculateLowpassTruePeak", [&buffer, sfinfo, &lowpass_true_peak_15khz]() {
        CalculateLowpassTruePeak<Float>(buffer.data(), sfinfo.channels, sfinfo.frames, sfinfo.samplerate, 15000, FLAGS_true_peak_oversample, &lowpass_true_peak_15khz);
    });

    Float loudness;
    std::vector<int> histogram;
	std::vector<Float> loudness_time_series;
	int loudness_block_samples;
	tasks.emplace_back("bakuage::loudness_ebu_r128::CalculateLoudness", [&buffer, sfinfo, &loudness, &histogram, &loudness_time_series, &loudness_block_samples]() {
		bakuage::loudness_ebu_r128::CalculateLoudness(buffer.data(), sfinfo.channels, sfinfo.frames, sfinfo.samplerate,
			&loudness, &histogram, &loudness_time_series, &loudness_block_samples);
	});
	Float loudness_range;
	tasks.emplace_back("bakuage::loudness_ebu_r128::CalculateLoudnessRange", [&buffer, sfinfo, &loudness_range]() {
		bakuage::loudness_ebu_r128::CalculateLoudnessRange(buffer.data(), sfinfo.channels, sfinfo.frames, sfinfo.samplerate,
			&loudness_range);
	});
	Float loudness_range_short;
	tasks.emplace_back("bakuage::loudness_ebu_r128::CalculateLoudnessRangeShort", [&buffer, sfinfo, &loudness_range_short]() {
		bakuage::loudness_ebu_r128::CalculateLoudnessRangeShort(buffer.data(), sfinfo.channels, sfinfo.frames, sfinfo.samplerate,
			&loudness_range_short);
	});
    std::vector<Float> youtube_loudness(24);
    tasks.emplace_back("youtube_loudness", [&buffer, sfinfo, &youtube_loudness]() {
        std::vector<int> histogram;
        Float loudness_range;
#if 1
        Float loudness;
        bakuage::loudness_ebu_r128::CalculateLoudnessCore<Float>(buffer.data(), sfinfo.channels, sfinfo.frames, sfinfo.samplerate,
                                                          FLAGS_youtube_loudness_window_sec, FLAGS_youtube_loudness_shift_sec, FLAGS_youtube_loudness_absolute_threshold, FLAGS_youtube_loudness_relative_threshold, &loudness, &loudness_range, &histogram, nullptr, nullptr, true, &youtube_loudness[0]);
#else
        int idx = 0;
        for (int m = 0; m < 2; m++) {
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 2; k++) {
                        Float max_loudness;
                        bakuage::loudness_ebu_r128::CalculateLoudnessCore<Float>(buffer.data(), sfinfo.channels, sfinfo.frames, sfinfo.samplerate,
                                                                                 k == 0 ? 0.4 : 3, k == 0 ? 0.1 : 0.1, i == 0 ? -360 : -70, j == 0 ? -360 : (j == 1 ? -10 : -20), &youtube_loudness[idx], &loudness_range, &histogram, nullptr, nullptr, true, &max_loudness);
                        if (m == 1) {
                            youtube_loudness[idx] = max_loudness;
                        }
                        idx++;
                    }
                }
            }
        }
#endif
    });

    std::vector<Float> waveform;
	tasks.emplace_back("CalculateWaveform", [&buffer, sfinfo, &waveform]() {
		if (FLAGS_analysis_for_visualization) {
			CalculateWaveform(buffer.data(), sfinfo.channels, sfinfo.frames, sfinfo.samplerate,
				512, &waveform);
		}
	});

    Float dynamics, sharpness, space;
	tasks.emplace_back("CalculateDyanmics", [&buffer, sfinfo, &dynamics, &sharpness, &space]() {
		CalculateDyanmics(buffer.data(), sfinfo.channels, sfinfo.frames, sfinfo.samplerate,
			&dynamics, &sharpness, &space);
	});

    std::vector<Float> spectrum;
    std::vector<std::vector<Float>> freq_pan_to_db;
    if (FLAGS_analysis_for_visualization) {
        tasks.emplace_back("CalculateSpectrum", [&buffer, sfinfo, &spectrum]() {
            CalculateSpectrum(buffer.data(), sfinfo.channels, sfinfo.frames, sfinfo.samplerate,
                              4096, &spectrum);
        });

        if (FLAGS_freq_pan_to_db) {
            tasks.emplace_back("CalculateStereo", [&buffer, sfinfo, &freq_pan_to_db]() {
                CalculateStereo(buffer.data(), sfinfo.channels, sfinfo.frames, sfinfo.samplerate,
                                &freq_pan_to_db);
            });
        }
    }

    if (!FLAGS_spectrogram_output.empty()) {
        tasks.emplace_back("WriteSpectrogramPng", [&buffer, sfinfo]() {
            WriteSpectrogramPng(buffer.data(), sfinfo.channels, sfinfo.frames, sfinfo.samplerate,
                                200, FLAGS_spectrogram_output.c_str());
        });
    }

    if (!FLAGS_rhythm_spectrogram_output.empty()) {
        tasks.emplace_back("WriteRhythmSpectrogramPng", [&buffer, sfinfo]() {
            WriteRhythmSpectrogramPng(buffer.data(), sfinfo.channels, sfinfo.frames, sfinfo.samplerate,
                                40, FLAGS_rhythm_spectrogram_output.c_str());
        });
    }

    if (!FLAGS_nmf_spectrogram_output.empty()) {
        tasks.emplace_back("WriteNmfSpectrogramPng", [&buffer, sfinfo]() {
            WriteNmfSpectrogramPng(buffer.data(), sfinfo.channels, sfinfo.frames, sfinfo.samplerate,
                                      200, FLAGS_nmf_spectrogram_output.c_str());
        });
    }

    if (!FLAGS_spectrum_distribution_output.empty()) {
        tasks.emplace_back("WriteSpectrumDistributionPng", [&buffer, sfinfo]() {
            WriteSpectrumDistributionPng(buffer.data(), sfinfo.channels, sfinfo.frames, sfinfo.samplerate,
                                         FLAGS_spectrum_distribution_output.c_str());
        });
    }

    if (!FLAGS_stereo_distribution_output.empty()) {
        tasks.emplace_back("WriteStereoDistributionPng", [&buffer, sfinfo]() {
            WriteStereoDistributionPng(buffer.data(), sfinfo.channels, sfinfo.frames, sfinfo.samplerate,
                                       400, 400, FLAGS_stereo_distribution_output.c_str());
        });
    }

    Float direct_reverb_ratio;
    if (FLAGS_drr) {
        tasks.emplace_back("CalculateDirectReverbEnergyRatio", [&buffer, sfinfo, &direct_reverb_ratio]() {
            CalculateDirectReverbEnergyRatio(buffer.data(), sfinfo.channels, sfinfo.frames, sfinfo.samplerate, &direct_reverb_ratio);
        });
    }

    std::vector<Band<Float>> bands = CreateBandsByErb<Float>(44100, 6);
    fprintf(stderr, "band count %d\n", (int)bands.size());
    std::vector<std::vector<Float>> covariance;
    double sound_quality = 0;
    float sound_quality2 = 0;
    tasks.emplace_back("band, covariance, sound_quality", [&buffer, sfinfo, &bands, &covariance, &sound_quality, &sound_quality2]() {
        CalculateMultibandLoudness<Float>(buffer.data(), sfinfo.channels, sfinfo.frames, sfinfo.samplerate,
                                          0.4, 0.1, -20, bands.data(), bands.size(), nullptr);
        CalculateMultibandLoudness2<Float>(buffer.data(), sfinfo.channels, sfinfo.frames, sfinfo.samplerate,
                                           0.4, 0.1, -20, bands.data(), bands.size(), &covariance, nullptr);

        if (FLAGS_sound_quality) {
            // Sound Quality
            bakuage::SoundQualityCalculator calculator;
            recursive_directory_iterator last;
            int reference_count = 0;
            for (recursive_directory_iterator itr(FLAGS_analysis_data_dir); itr != last; ++itr) {
                const std::string path = itr->path().string();
                if (!bakuage::StrEndsWith(path, ".json")) continue;

                const std::string json_str = bakuage::LoadStrFromFile(path.c_str());
                calculator.AddReference(json_str.c_str());
                reference_count++;
            }
            std::cerr << "SoundQualityCalculator reference count " << reference_count << std::endl;
            calculator.Prepare();

            Eigen::VectorXd mean_vec(2 * bands.size());
            Eigen::MatrixXd covariance_mat(2 * bands.size(), 2 * bands.size());
            for (int i = 0; i < bands.size(); i++) {
                mean_vec(2 * i + 0) = bands[i].mid_mean;
                mean_vec(2 * i + 1) = bands[i].side_mean;
            }
            for (int i = 0; i < 2 * bands.size(); i++) {
                for (int j = 0; j < 2 * bands.size(); j++) {
                    covariance_mat(i, j) = covariance[i][j];
                }
            }
            calculator.CalculateSoundQuality(mean_vec, covariance_mat, &sound_quality, nullptr);
        }

        if (FLAGS_sound_quality2) {
            bakuage::SoundQuality2Calculator calculator;
            {
                std::ifstream ifs(FLAGS_sound_quality2_cache);
                if (FLAGS_sound_quality2_cache_archiver == "binary") {
                    boost::archive::binary_iarchive ia(ifs);
                    ia >> calculator;
                } else if (FLAGS_sound_quality2_cache_archiver == "text") {
                    boost::archive::text_iarchive ia(ifs);
                    ia >> calculator;
                } else {
                    throw std::logic_error("unknown archive type " + FLAGS_sound_quality2_cache_archiver);
                }
            }
            Eigen::VectorXd mean_vec(2 * bands.size());
            Eigen::MatrixXd covariance_mat(2 * bands.size(), 2 * bands.size());
            for (int i = 0; i < bands.size(); i++) {
                mean_vec(2 * i + 0) = bands[i].mid_mean;
                mean_vec(2 * i + 1) = bands[i].side_mean;
            }
            for (int i = 0; i < 2 * bands.size(); i++) {
                for (int j = 0; j < 2 * bands.size(); j++) {
                    covariance_mat(i, j) = covariance[i][j];
                }
            }
            calculator.CalculateSoundQuality(mean_vec, covariance_mat, &sound_quality2, nullptr);
        }
    });

    std::vector<Band<Float>> bands_short = bands;
    std::vector<std::vector<Float>> covariance_short;
    tasks.emplace_back("bands_short, covariance_short", [&buffer, sfinfo, &bands_short, &covariance_short]() {
        CalculateMultibandLoudness<Float>(buffer.data(), sfinfo.channels, sfinfo.frames, sfinfo.samplerate,
                                          0.04, 0.01, -20, bands_short.data(), bands_short.size(), nullptr);
        CalculateMultibandLoudness2<Float>(buffer.data(), sfinfo.channels, sfinfo.frames, sfinfo.samplerate,
                                           0.04, 0.01, -20, bands_short.data(), bands_short.size(), &covariance_short, nullptr);
    });

    Float dissonance;
    tasks.emplace_back("CalculateDissonance", [&buffer, sfinfo, &dissonance]() {
        bakuage::CalculateDissonance<Float>(buffer.data(), sfinfo.channels, sfinfo.frames, sfinfo.samplerate,
                                        &dissonance, true);
    });

    Float bandwidth, diff_bandwidth;
    tasks.emplace_back("CalculateBandwidth", [&buffer, sfinfo, &bandwidth, &diff_bandwidth]() {
        bakuage::CalculateBandwidth<Float>(buffer.data(), sfinfo.channels, sfinfo.frames, sfinfo.samplerate,
                                            &bandwidth, &diff_bandwidth);
    });

    Float acoustic_entropy, damage;
    tasks.emplace_back("CalculateAcousticEntropy", [&buffer, sfinfo, &acoustic_entropy, &damage]() {
        CalculateAcousticEntropy<Float>(buffer.data(), sfinfo.channels, sfinfo.frames, sfinfo.samplerate,
                                        nullptr, &acoustic_entropy, &damage);
    });

	float mastering3_loudness, mastering3_ear_damage, mastering3_acoustic_entropy_mfcc, mastering3_acoustic_entropy_eigen, mastering3_diff_acoustic_entropy_eigen;
	const int num_filters = FLAGS_mastering3_acoustic_entropy_band_count;
	bakuage::MfccCalculator<float> mfcc_calculator(sfinfo.samplerate, 0, 22000, num_filters);
	std::vector<int> mastering3_sns = { 6, 12, 24, 72 };
	std::vector<float> mastering3_acoustic_entropies(mastering3_sns.size());
    std::vector<float> mastering3_diff_acoustic_entropies(mastering3_sns.size());
	std::vector<float> mastering3_mono_acoustic_entropies(mastering3_sns.size());
	std::vector<std::vector<float>> mastering3_band_eliminated_acoustic_entropies(mastering3_sns.size(), std::vector<float>(num_filters));
	std::vector<std::vector<float>> mastering3_stereo_band_eliminated_acoustic_entropies(mastering3_sns.size(), std::vector<float>(num_filters));

	tasks.emplace_back("Mastering3", [&buffer, sfinfo,
		&mastering3_loudness, &mastering3_ear_damage, &mastering3_acoustic_entropy_mfcc, &mastering3_acoustic_entropy_eigen, &mastering3_diff_acoustic_entropy_eigen,
		&mfcc_calculator, num_filters, &mastering3_sns, &mastering3_acoustic_entropies,
		&mastering3_band_eliminated_acoustic_entropies, &mastering3_stereo_band_eliminated_acoustic_entropies,
		&mastering3_mono_acoustic_entropies, &mastering3_diff_acoustic_entropies]() {

		// calculate mfcc
		int shift_resolution = 2;
		int output_shift_resolution = 2;
		int width = output_shift_resolution * ((16384 * sfinfo.samplerate / 44100) / output_shift_resolution); // 0.372 sec, 4x only
		int shift = width / shift_resolution;
		int output_shift = width / output_shift_resolution;
		int mfcc_len = 13;
		int pos = -width + shift;
		int spec_len = width / 2 + 1;
		std::vector<std::complex<float>> complex_spec_mid(spec_len);
		std::vector<std::complex<float>> complex_spec_side(spec_len);
		std::vector<float> src_mid_mel_bands;
		std::vector<float> src_side_mel_bands;
		std::vector<float> window(width);
		bakuage::CopyHanning(width, window.begin());
		bakuage::Dct dct(num_filters);
        float *fft_input = (float *)bakuage::AlignedMalloc(sizeof(float) * width);
        std::complex<float> *fft_output = (std::complex<float> *)bakuage::AlignedMalloc(sizeof(std::complex<float>) * spec_len);
        bakuage::RealDft<float> dft(width);
		while (pos < sfinfo.frames) {
			// window and fft
			std::fill_n(complex_spec_mid.data(), spec_len, 0);
			std::fill_n(complex_spec_side.data(), spec_len, 0);
			for (int i = 0; i < sfinfo.channels; i++) {
				for (int j = 0; j < width; j++) {
					int k = pos + j;
					fft_input[j] = (0 <= k && k < sfinfo.frames) ? buffer[sfinfo.channels * k + i] * window[j] : 0;
				}
                dft.Forward(fft_input, (float *)fft_output);
				for (int j = 0; j < spec_len; j++) {
                    auto spec = fft_output[j];
					complex_spec_mid[j] += spec;
					complex_spec_side[j] += spec * (2.0f * i - 1);
				}
			}

			// calculate mel band (energy sum mode)
			src_mid_mel_bands.resize(src_mid_mel_bands.size() + num_filters);
			src_side_mel_bands.resize(src_side_mel_bands.size() + num_filters);
			mfcc_calculator.calculateMelSpectrumFromDFT((float *)complex_spec_mid.data(),
				width, false, true, &src_mid_mel_bands[src_mid_mel_bands.size() - num_filters]);
			mfcc_calculator.calculateMelSpectrumFromDFT((float *)complex_spec_side.data(),
				width, false, true, &src_side_mel_bands[src_side_mel_bands.size() - num_filters]);

			pos += shift;
		}
		// calculate noise melband (実際のノイズではなく、エネルギー平均)
		std::vector<std::complex<float>> complex_spec_noise(spec_len);
		std::vector<float> noise_mel_bands(num_filters);
		for (int i = 0; i < spec_len; i++) {
			double cutoff_freq = 20;
			double freq = 1.0 * (i + 0.5) / width * sfinfo.samplerate;
			double noise_energy = std::pow(10, -4.5 * std::log2(cutoff_freq + freq) * 0.1);
			complex_spec_noise[i] = std::sqrt(noise_energy);
		}
		mfcc_calculator.calculateMelSpectrumFromDFT((float *)complex_spec_noise.data(),
			width, false, true, noise_mel_bands.data());

		const int count = src_mid_mel_bands.size() / num_filters;
		auto speaker_compensation_flat = [](double hz) { return 0; };
		auto speaker_compensation_typical = [](double hz) {
			if (hz < 200) {
				return 10 * std::log2(hz / 200);
			}
			else {
				return 0.0;
			}
		};
		CalculateMastering3Score(src_mid_mel_bands.data(), src_side_mel_bands.data(), noise_mel_bands.data(),
			count, mfcc_calculator, 12, speaker_compensation_flat, speaker_compensation_flat, speaker_compensation_flat,
			&mastering3_loudness, &mastering3_ear_damage,
			&mastering3_acoustic_entropy_mfcc, &mastering3_acoustic_entropy_eigen, &mastering3_diff_acoustic_entropy_eigen);

		// different sn
        tbb::parallel_for(0, (int)mastering3_sns.size(), [&mastering3_sns, &src_mid_mel_bands, &src_side_mel_bands, &noise_mel_bands, count, &mfcc_calculator, &speaker_compensation_flat, &mastering3_acoustic_entropies, &mastering3_diff_acoustic_entropies, &mastering3_mono_acoustic_entropies, num_filters, &mastering3_band_eliminated_acoustic_entropies, &mastering3_stereo_band_eliminated_acoustic_entropies](int sn_idx) {
			const auto sn = mastering3_sns[sn_idx];

			float loudness, ear_damage, acoustic_entropy_mfcc, acoustic_entropy_eigen, diff_acoustic_entropy_eigen;
			CalculateMastering3Score(src_mid_mel_bands.data(), src_side_mel_bands.data(), noise_mel_bands.data(),
				count, mfcc_calculator, sn, speaker_compensation_flat, speaker_compensation_flat, speaker_compensation_flat, &loudness, &ear_damage,
				&acoustic_entropy_mfcc, &acoustic_entropy_eigen, &diff_acoustic_entropy_eigen);
			mastering3_acoustic_entropies[sn_idx] = acoustic_entropy_eigen;
            mastering3_diff_acoustic_entropies[sn_idx] = diff_acoustic_entropy_eigen;

			CalculateMastering3Score(src_mid_mel_bands.data(), src_side_mel_bands.data(), noise_mel_bands.data(),
				count, mfcc_calculator, sn, speaker_compensation_flat, speaker_compensation_flat, [](double hz) { return -1000; }, &loudness, &ear_damage,
				&acoustic_entropy_mfcc, &acoustic_entropy_eigen, &diff_acoustic_entropy_eigen);
			mastering3_mono_acoustic_entropies[sn_idx] = acoustic_entropy_eigen;

            tbb::parallel_for(0, num_filters, [sn_idx, sn, &src_mid_mel_bands, &src_side_mel_bands, &noise_mel_bands, count, &mfcc_calculator, &speaker_compensation_flat, &mastering3_band_eliminated_acoustic_entropies, &mastering3_stereo_band_eliminated_acoustic_entropies](int band_idx) {
				const auto eliminated_freq = mfcc_calculator.center_freq(band_idx);
				const auto speaker_compensation_elimination = [eliminated_freq](double hz) {
					return std::abs(hz - eliminated_freq) < 1 ? -1000 : 0;
				};
                float loudness, ear_damage, acoustic_entropy_mfcc, acoustic_entropy_eigen, diff_acoustic_entropy_eigen;
				CalculateMastering3Score(src_mid_mel_bands.data(), src_side_mel_bands.data(), noise_mel_bands.data(),
					count, mfcc_calculator, sn, speaker_compensation_flat, speaker_compensation_elimination, speaker_compensation_elimination, &loudness, &ear_damage,
					&acoustic_entropy_mfcc, &acoustic_entropy_eigen, &diff_acoustic_entropy_eigen);
				mastering3_band_eliminated_acoustic_entropies[sn_idx][band_idx] = acoustic_entropy_eigen;

				CalculateMastering3Score(src_mid_mel_bands.data(), src_side_mel_bands.data(), noise_mel_bands.data(),
					count, mfcc_calculator, sn, speaker_compensation_flat, speaker_compensation_flat, speaker_compensation_elimination, &loudness, &ear_damage,
					&acoustic_entropy_mfcc, &acoustic_entropy_eigen, &diff_acoustic_entropy_eigen);
				mastering3_stereo_band_eliminated_acoustic_entropies[sn_idx][band_idx] = acoustic_entropy_eigen;
            });
        });
	});

#if 0
    // single thread mode
    for (auto &task: tasks) {
        task();
    }
#else
    tbb::parallel_for(0, (int)tasks.size(),
                      [&tasks](int i) {
                          tasks[i]();
                      }
                      );
#endif
	fprintf(stderr, "total time %.3f sec\n", stop_watch.time());

    std::stable_sort(tasks.begin(), tasks.end(), [](const Task &a, const Task &b) {
        return b.elapsed_sec() < a.elapsed_sec();
    });
    for (const auto &task: tasks) {
        std::cerr << "name:" << task.name << "\telapsed_sec:" << task.elapsed_sec() << std::endl;
    }

    picojson::object output_json;

    // version
    output_json.insert(std::make_pair("version", "oss"));

	// フォーマット系
    output_json.insert(std::make_pair("channels", picojson::value((double)sfinfo.channels)));
    output_json.insert(std::make_pair("format", picojson::value((double)sfinfo.format)));
    output_json.insert(std::make_pair("frames", picojson::value((double)sfinfo.frames)));
    output_json.insert(std::make_pair("sample_rate", picojson::value((double)sfinfo.samplerate)));
    output_json.insert(std::make_pair("sections", picojson::value((double)sfinfo.sections)));
    output_json.insert(std::make_pair("seekable", picojson::value((double)sfinfo.seekable)));

	// 特徴量系
    output_json.insert(std::make_pair("peak", picojson::value(peak)));
    output_json.insert(std::make_pair("true_peak", picojson::value(true_peak)));
    output_json.insert(std::make_pair("lowpass_true_peak_15khz", picojson::value(lowpass_true_peak_15khz)));
    output_json.insert(std::make_pair("rms", picojson::value(rms)));
    output_json.insert(std::make_pair("loudness", picojson::value(loudness)));
	output_json.insert(std::make_pair("loudness_range", picojson::value(loudness_range)));
	output_json.insert(std::make_pair("loudness_range_short", picojson::value(loudness_range_short)));
    output_json.insert(std::make_pair("dynamics", picojson::value(dynamics)));
    output_json.insert(std::make_pair("sharpness", picojson::value(sharpness)));
	output_json.insert(std::make_pair("space", picojson::value(space)));
	output_json.insert(std::make_pair("acoustic_entropy", picojson::value(acoustic_entropy)));
	output_json.insert(std::make_pair("damage", picojson::value(damage)));
	output_json.insert(std::make_pair("mastering3_loudness", picojson::value(mastering3_loudness)));
	output_json.insert(std::make_pair("mastering3_ear_damage", picojson::value(mastering3_ear_damage)));
    if (FLAGS_sound_quality) {
        output_json.insert(std::make_pair("sound_quality", picojson::value(sound_quality)));
    }
    if (FLAGS_sound_quality2) {
        output_json.insert(std::make_pair("sound_quality2", picojson::value(sound_quality2)));
    }
    output_json.insert(std::make_pair("dissonance", picojson::value(10 * std::log10(1e-37 + dissonance))));
    output_json.insert(std::make_pair("bandwidth", picojson::value(bandwidth)));
    output_json.insert(std::make_pair("diff_bandwidth", picojson::value(diff_bandwidth)));
#if 1
    output_json.insert(std::make_pair("youtube_loudness", picojson::value(youtube_loudness[0])));
#else
    for (int i = 0; i < 24; i++) {
        std::stringstream ss;
        ss << "youtube_loudness" << i;
    output_json.insert(std::make_pair(ss.str(), picojson::value(youtube_loudness[i])));
    }
#endif
	// deprecated
	// output_json.insert(std::make_pair("mastering3_acoustic_entropy_mfcc", picojson::value(mastering3_acoustic_entropy_mfcc)));
	// output_json.insert(std::make_pair("mastering3_acoustic_entropy_eigen", picojson::value(mastering3_acoustic_entropy_eigen)));
	picojson::array acoustic_entropies;
	for (int sn_idx = 0; sn_idx < mastering3_sns.size(); sn_idx++) {
		picojson::object acoustic_entropy_obj;
		auto sn = mastering3_sns[sn_idx];
		acoustic_entropy_obj.insert(std::make_pair("sn", picojson::value((double)sn)));
		acoustic_entropy_obj.insert(std::make_pair("acoustic_entropy", picojson::value(mastering3_acoustic_entropies[sn_idx])));
        acoustic_entropy_obj.insert(std::make_pair("diff_acoustic_entropy", picojson::value(mastering3_diff_acoustic_entropies[sn_idx])));
		acoustic_entropy_obj.insert(std::make_pair("acoustic_entropy_mono", picojson::value(mastering3_mono_acoustic_entropies[sn_idx])));

		picojson::array band_eliminated_acoustic_entropies;
		for (int band_idx = 0; band_idx < num_filters; band_idx++) {
			picojson::object band_eliminated_acoustic_entropy;
			auto eliminated_freq = mfcc_calculator.center_freq(band_idx);
			band_eliminated_acoustic_entropy.insert(std::make_pair("hz", picojson::value(eliminated_freq)));
			band_eliminated_acoustic_entropy.insert(std::make_pair("acoustic_entropy", picojson::value(mastering3_band_eliminated_acoustic_entropies[sn_idx][band_idx])));
			band_eliminated_acoustic_entropy.insert(std::make_pair("acoustic_entropy_side", picojson::value(mastering3_stereo_band_eliminated_acoustic_entropies[sn_idx][band_idx])));
			band_eliminated_acoustic_entropies.push_back(picojson::value(band_eliminated_acoustic_entropy));
		}
		acoustic_entropy_obj.insert(std::make_pair("band_eliminated_acoustic_entropies", picojson::value(band_eliminated_acoustic_entropies)));
		acoustic_entropies.push_back(picojson::value(acoustic_entropy_obj));
	}
	output_json.insert(std::make_pair("mastering3_acoustic_entropies", picojson::value(acoustic_entropies)));

	if (FLAGS_drr) {
		output_json.insert(std::make_pair("drr", picojson::value(direct_reverb_ratio)));
	}
	for (int is_short = 0; is_short < 2; is_short++) {
		const auto &bands_select = is_short ? bands_short : bands;
		picojson::array bands_json;
		for (int i = 0; i < bands_select.size(); i++) {
			picojson::object band_json;

			// ver 1
			band_json.insert(std::make_pair("loudness", picojson::value(bands_select[i].loudness)));
			band_json.insert(std::make_pair("loudness_range", picojson::value(bands_select[i].loudness_range)));
			band_json.insert(std::make_pair("mid_to_side_loudness", picojson::value(bands_select[i].mid_to_side_loudness)));
			band_json.insert(std::make_pair("mid_to_side_loudness_range", picojson::value(bands_select[i].mid_to_side_loudness_range)));

			// ver 2
			band_json.insert(std::make_pair("mid_mean", picojson::value(bands_select[i].mid_mean)));
			band_json.insert(std::make_pair("side_mean", picojson::value(bands_select[i].side_mean)));

			if (i > 0) {
				band_json.insert(std::make_pair("low_freq", picojson::value(bands_select[i].low_freq)));
			}
			if (i < bands_select.size() - 1) {
				band_json.insert(std::make_pair("high_freq", picojson::value(bands_select[i].high_freq)));
			}
			bands_json.push_back(picojson::value(band_json));
		}
		output_json.insert(std::make_pair(is_short ? "bands_short" : "bands", picojson::value(bands_json)));

		const auto &covariance_select = is_short ? covariance_short : covariance;
		picojson::array covariance_json;
		fprintf(stderr, "covariance size %d\n", (int)covariance_select.size());
		fflush(stderr);
		for (int i = 0; i < covariance_select.size(); i++) {
			picojson::array row_json;
			for (int j = 0; j < covariance_select.size(); j++) {
				row_json.push_back(picojson::value(covariance_select[i][j]));
			}
			covariance_json.push_back(picojson::value(row_json));
		}
		output_json.insert(std::make_pair(is_short ? "covariance_short" : "covariance", picojson::value(covariance_json)));
	}

	// 主に表示用 (計算の定義とかを適当に決めているので。自動マスタリングなどによる再利用を想定していない)
	if (FLAGS_analysis_for_visualization) {
		{
			picojson::array spectrum_json;
			for (int i = 0; i < spectrum.size() / 2; i++) {
				picojson::object point;
				point.insert(std::make_pair("hz", picojson::value(spectrum[2 * i + 0])));
				point.insert(std::make_pair("db", picojson::value(spectrum[2 * i + 1])));
				spectrum_json.push_back(picojson::value(point));
			}
			output_json.insert(std::make_pair("spectrum", picojson::value(spectrum_json)));
		}

		{
			picojson::array waveform_json;
			for (int i = 0; i < waveform.size(); i++) {
				waveform_json.push_back(picojson::value(waveform[i]));
			}
			output_json.insert(std::make_pair("waveform", picojson::value(waveform_json)));
		}

		{
			picojson::array histogram_json;
			for (int i = 0; i < histogram.size(); i++) {
				picojson::object point;
				point.insert(std::make_pair("db", picojson::value((double)(i - 70))));
				point.insert(std::make_pair("frequency", picojson::value((double)histogram[i])));
				histogram_json.push_back(picojson::value(point));
			}
			output_json.insert(std::make_pair("histogram", picojson::value(histogram_json)));
		}

		{
			picojson::array data;
			for (int i = 0; i < loudness_time_series.size(); i++) {
				picojson::object point;
				point.insert(std::make_pair("sec", picojson::value(1.0 * i * (loudness_block_samples / 4) / sfinfo.samplerate)));
				point.insert(std::make_pair("db", picojson::value(loudness_time_series[i])));
				data.push_back(picojson::value(point));
			}
			output_json.insert(std::make_pair("loudness_time_series", picojson::value(data)));
		}

		if (FLAGS_freq_pan_to_db) {
			picojson::array freq_pan_to_db_json;
			for (int i = 0; i < freq_pan_to_db.size(); i++) {
				picojson::array ar;
				for (int j = 0; j < freq_pan_to_db[i].size(); j++) {
					double a = freq_pan_to_db[i][j];
					ar.push_back(picojson::value(a));
				}
				freq_pan_to_db_json.push_back(picojson::value(ar));
			}
			output_json.insert(std::make_pair("freq_pan_to_db", picojson::value(freq_pan_to_db_json)));
		}
	}

    std::cout << picojson::value(output_json).serialize(true);

    return 0;
}

}

#endif
