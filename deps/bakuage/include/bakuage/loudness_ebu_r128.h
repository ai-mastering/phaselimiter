#ifndef BAKUAGE_LOUDNESS_EBU_R128_H_
#define BAKUAGE_LOUDNESS_EBU_R128_H_

#include <vector>

namespace bakuage {
namespace loudness_ebu_r128 {

// ラウドネス規格: BS.1770
// Loudness Range http://www.abma-bvam.be/PDF/EBU_PLOUD/EBU_tech3342.pdf
// http://jp.music-group.com/TCE/Tech/LRA.pdf
// block_sec = 3, shift_sec = 2, relative_threshold_db = -20

template <typename Float>
void CalculateLoudnessCore(const Float *input, const int channels, const int samples, const int sample_freq,
	const Float block_sec, const Float shift_sec, const Float absolute_threshold_db, const Float relative_threshold_db,
	Float *loudness, Float *loudness_range, std::vector<int> *histogram,
	std::vector<Float> *loudness_time_series,
                           int *block_samples, bool use_youtube_weighting = false, Float *max_loudness = nullptr);

template <typename Float>
void CalculateLoudness(const Float *input, const int channels, const int samples, const int sample_freq,
	Float *loudness, std::vector<int> *histogram,
	std::vector<Float> *loudness_time_series = nullptr,
	int *block_samples = nullptr) {
	CalculateLoudnessCore<Float>(input, channels, samples, sample_freq,
		0.4, 0.1, -70, -10, loudness, nullptr, histogram, loudness_time_series, block_samples);
}

template <typename Float>
void CalculateLoudnessRange(const Float *input, const int channels, const int samples, const int sample_freq,
	Float *loudness_range) {
	std::vector<int> histo;
	CalculateLoudnessCore<Float>(input, channels, samples, sample_freq,
		0.4, 0.1, -70, -20, nullptr, loudness_range, &histo, nullptr, nullptr);
}

template <typename Float>
void CalculateLoudnessRangeShort(const Float *input, const int channels, const int samples, const int sample_freq,
	Float *loudness_range) {
	std::vector<int> histo;
	CalculateLoudnessCore<Float>(input, channels, samples, sample_freq,
		0.04, 0.01, -70, -20, nullptr, loudness_range, &histo, nullptr, nullptr);
}

template <typename Float>
void CalculateHistogram(const Float *input, int channels, int samples, int sample_freq, Float mean_sec,
                        std::vector<Float> *histogram, std::vector<Float> *mid_to_side_histogram = NULL);

}
}

#endif
