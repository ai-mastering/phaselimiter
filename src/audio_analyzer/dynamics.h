#ifndef BAKUAGE_AUDIO_ANALYZER_DYNAMICS_H_
#define BAKUAGE_AUDIO_ANALYZER_DYNAMICS_H_

#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <exception>
#include "bakuage/delay_filter.h"
#include "bakuage/loudness_filter.h"
#include "bakuage/time_varying_lowpass_filter.h"
#include "bakuage/utils.h"
#include "audio_analyzer/statistics.h"

namespace audio_analyzer {

template <typename Float>
void CalculateDyanmics(Float *input, int channels, int samples, int sample_freq, 
                       Float *dynamics_range, Float *sharpness, Float *space) {  
    using namespace bakuage;

    if (channels != 2) {
        throw std::logic_error("CalculateDyanmics: channels must be 2");
    }

    const double short_mean_sec = 0.025;
    const double long_mean_sec = 0.2;
    const int lowpass_filter_order = 2;
    const double loudness_bias = -0.691;
    const double sqrt_0_5 = std::sqrt(0.5);
    const double initial_threshold_db = -70;
	const double second_threshold_relative_to_mean_db = -10;

    int short_delay_samples, long_delay_samples;
    Float short_a, long_a;
    {
        Float peak = std::min<Float>(1.0, 1.0 / (sample_freq * short_mean_sec + 1e-30));
        short_a = TimeVaryingLowpassFilter<Float>::CalculateAFromPeak(
            lowpass_filter_order, peak, &short_delay_samples);
    }
    {
        Float peak = std::min<Float>(1.0, 1.0 / (sample_freq * long_mean_sec + 1e-30));
        long_a = TimeVaryingLowpassFilter<Float>::CalculateAFromPeak(
            lowpass_filter_order, peak, &long_delay_samples);
    }
    const int max_delay_samples = (std::max)(short_delay_samples, long_delay_samples);

	double long_rms_threshold_db = initial_threshold_db;
	double short_rms_threshold_db = initial_threshold_db;
	double mid_rms_threshold_db = initial_threshold_db;
	double side_rms_threshold_db = initial_threshold_db;
	for (int pass = 0; pass < 2; pass++) {
		TimeVaryingLowpassFilter<double> short_lowpass_filter(lowpass_filter_order, short_a);
		TimeVaryingLowpassFilter<double> long_lowpass_filter(lowpass_filter_order, long_a);
		TimeVaryingLowpassFilter<double> mid_lowpass_filter(lowpass_filter_order, short_a);
		TimeVaryingLowpassFilter<double> side_lowpass_filter(lowpass_filter_order, short_a);
		std::vector<LoudnessFilter<double>> loudness_filters;
		for (int i = 0; i < channels; i++) {
			loudness_filters.emplace_back(sample_freq);
		}

		DelayFilter<double> short_delay_filter(max_delay_samples - short_delay_samples);
		DelayFilter<double> long_delay_filter(max_delay_samples - long_delay_samples);
		DelayFilter<double> mid_delay_filter(max_delay_samples - short_delay_samples);
		DelayFilter<double> side_delay_filter(max_delay_samples - short_delay_samples);

		Statistics long_rms_db_stats;
		Statistics short_rms_db_stats;
		Statistics mid_rms_db_stats;
		Statistics side_rms_db_stats;
		Statistics long_to_short_db_stats;
		Statistics mid_to_side_db_stats;

		for (int i = 0; i < samples + max_delay_samples; i++) {
			double left, right;
			if (i < samples) {
				left = loudness_filters[0].Clock(input[2 * i + 0]);
				right = loudness_filters[1].Clock(input[2 * i + 1]);
			}
			else {
				left = loudness_filters[0].Clock(0);
				right = loudness_filters[1].Clock(0);
			}
			const double mid = sqrt_0_5 * (left + right);
			const double side = sqrt_0_5 * (left - right);

			const double squared = Sqr(left) + Sqr(right);
			const double short_rms = short_delay_filter.Clock(short_lowpass_filter.Clock(squared));
			const double long_rms = long_delay_filter.Clock(long_lowpass_filter.Clock(squared));
			const double mid_rms = mid_delay_filter.Clock(mid_lowpass_filter.Clock(Sqr(mid)));
			const double side_rms = side_delay_filter.Clock(side_lowpass_filter.Clock(Sqr(side)));

			double short_rms_db = loudness_bias + bakuage::ToDb(short_rms);
			const double long_rms_db = loudness_bias + bakuage::ToDb(long_rms);
			double mid_rms_db = loudness_bias + bakuage::ToDb(mid_rms);
			double side_rms_db = loudness_bias + bakuage::ToDb(side_rms);

			if (long_rms_threshold_db <= long_rms_db) {
				short_rms_db = (std::max)(short_rms_db, short_rms_threshold_db);
				mid_rms_db = (std::max)(mid_rms_db, mid_rms_threshold_db);
				side_rms_db = (std::max)(side_rms_db, side_rms_threshold_db);

				long_rms_db_stats.Add(long_rms_db);
				short_rms_db_stats.Add(short_rms_db);
				mid_rms_db_stats.Add(mid_rms_db);
				side_rms_db_stats.Add(side_rms_db);
				long_to_short_db_stats.Add(short_rms_db - long_rms_db);
				mid_to_side_db_stats.Add(side_rms_db - mid_rms_db);
			}
		}

		if (pass == 0) {
			long_rms_threshold_db = long_rms_db_stats.mean() + second_threshold_relative_to_mean_db;
			short_rms_threshold_db = short_rms_db_stats.mean() + second_threshold_relative_to_mean_db;
			mid_rms_threshold_db = mid_rms_db_stats.mean() + second_threshold_relative_to_mean_db;
			side_rms_threshold_db = side_rms_db_stats.mean() + second_threshold_relative_to_mean_db;
		}
		else {
			*dynamics_range = long_rms_db_stats.stddev();
			*sharpness = long_to_short_db_stats.stddev();
			*space = mid_to_side_db_stats.mean();
		}
	}
}

}

#endif 