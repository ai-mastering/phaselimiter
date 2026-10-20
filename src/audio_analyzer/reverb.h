#ifndef BAKUAGE_AUDIO_ANALYZER_REVERB_H_
#define BAKUAGE_AUDIO_ANALYZER_REVERB_H_

#include <algorithm>
#include <cmath>
#include <vector>
#include <random>
#include <tbb/parallel_for.h>

#include "bakuage/memory.h"
#include "bakuage/dft.h"
#include "bakuage/loudness_filter.h"
#include "bakuage/utils.h"
#include "bakuage/convolution.h"

namespace audio_analyzer {

namespace impl {

// LPC
// https://gist.github.com/hecomi/7398975
//template <typename Float>
//std::vector<Float> CalculateLPCReconstruction(const Float *input, const int N, int order) {
//
//}

// ESTIMATION OF REVERBERATION TIME USING LPC FILTER AND MAXIMUM LIKELIHOOD ESTIMATOR AND DRR
// https://www.ijarcce.com/upload/january/30_ESTIMATION%20OF%20REVERBERATION.pdf
// calc_irがtrueのとき、得られた自己相関が、IRの自己相関だとして、IRを計算する。
// IRは一意に定まらないので、原点が最大になるように選ぶ。

#if 0
template <typename Float>
std::vector<Float> CalculateAutoCorrelation(const Float *input, int samples, bool calc_ir = false) {
	// prepare FFT
	double *fft_input = (double *)fftw_malloc(sizeof(double) * samples);
	std::memset(fft_input, 0, sizeof(double) * samples);
	fftw_complex *fft_output = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (samples / 2 + 1));
	std::memset(fft_output, 0, sizeof(fftw_complex) * (samples / 2 + 1));
	fftw_plan plan = fftw_plan_dft_r2c_1d(samples, fft_input, fft_output, FFTW_ESTIMATE);
	fftw_plan plan_inv = fftw_plan_dft_c2r_1d(samples, fft_output, fft_input, FFTW_ESTIMATE);

	// auto correlation
	double scale = 1.0 / std::sqrt(samples);
	for (int i = 0; i < samples; i++) {
		fft_input[i] = input[i] * scale;
	}
	fftw_execute(plan);
	for (int i = 0; i < samples / 2 + 1; i++) {
		fft_output[i][0] = fft_output[i][0] * fft_output[i][0] + fft_output[i][1] * fft_output[i][1];
		if (calc_ir) {
			fft_output[i][0] = std::sqrt(fft_output[i][0]);
		}
		fft_output[i][1] = 0;
	}

	fftw_execute(plan_inv);
	std::vector<Float> correlation(samples);
	for (int i = 0; i < samples; i++) {
		correlation[i] = fft_input[i] * scale;
	}

	// free FFT
	fftw_destroy_plan(plan);
	fftw_destroy_plan(plan_inv);
	fftw_free(fft_input);
	fftw_free(fft_output);

	return correlation;
}
#endif

/*
LPCはうまくない。なぜかというと、スペクトルがホワイトノイズから遠いときにエネルギーが減るから、
ホワイトノイズに近いか遠いかで重みがばらついてしまう。

スペクトルを均す方法は良いが、下手をするとリバーブ成分をdirectとして扱ってしまうことで、
逆にDRRが下がるみたいなことが起きる。
direct_lengthは重要。
*/
// Kahan compensation is also needed for the small increments in the rotation.
struct ReverbCompensatedSum {
    double value = 0;
    double error = 0;

    void Add(double x) {
        const double y = x - error;
        const double next = value + y;
        error = (next - value) - y;
        value = next;
    }
};

// Circular convolution with w[j] = (1 + cos(pi*j/L)) / (2*L), |j| < L.
// Maintain the rectangular sum A and complex cosine sum B over [-L, L).
// Advancing one bin gives A' = A - old + next and
// B' = exp(-i*pi/L) * (B + old - next), so smoothing takes O(samples).
inline std::vector<double> ReverbSmoothPsd(const double *psd, int samples, int window_length) {
    const int bins = samples / 2 + 1;
    const double theta = M_PI / window_length;
    const double sine = std::sin(theta);
    const double half_sine = std::sin(theta / 2);
    // Avoid cancellation in cos(theta) - 1 for long windows.
    const double cosine_minus_one = -2 * half_sine * half_sine;
    auto at = [&](int i) {
        if (i < 0) i = -i;
        if (i >= samples) i %= samples;
        if (i > samples / 2) i = samples - i;
        return psd[i];
    };
    ReverbCompensatedSum sum, real, imag;
    for (int j = -window_length; j < window_length; ++j) {
        const double value = at(j);
        sum.Add(value);
        real.Add(value * std::cos(theta * j));
        imag.Add(value * std::sin(theta * j));
    }
    std::vector<double> smoothed(bins);
    for (int k = 0; k < bins; ++k) {
        // Nonnegative weights and w[0] = 1/L imply S[k] >= P[k]/L.
        // Enforce this bound to prevent roundoff from amplifying near-zero bins.
        smoothed[k] = std::max(psd[k] / window_length,
                              (sum.value + real.value) / (2.0 * window_length));
        if (k + 1 == bins) break;
        const double old = at(k - window_length);
        const double next = at(k + window_length);
        sum.Add(-old);
        sum.Add(next);
        real.Add(old);
        real.Add(-next);
        const double re = real.value, im = imag.value;
        real.Add(std::fma(cosine_minus_one, re, sine * im));
        imag.Add(std::fma(cosine_minus_one, im, -sine * re));
    }
    return smoothed;
}

template <typename Float>
void CalculateDirectReverbCore(const Float *input, int samples, int sample_rate, Float *direct, Float *reverb) {
	const int max_length = 0.25 * sample_rate;
	const int direct_length = 0.001 * sample_rate;
	const int dr_transition_length = 0.0001 * sample_rate;
	const int spec_length = samples / 2 + 1;
	const Float scale = 1.0 / std::sqrt(samples);

	// prepare FFT
    float *fft_input = (float *)bakuage::AlignedMalloc(sizeof(float) * samples);
    std::complex<float> *fft_output = (std::complex<float> *)bakuage::AlignedMalloc(sizeof(std::complex<float>) * spec_length);
    bakuage::RealDft<float> dft(samples);

	// whitening by lpc
//	std::vector<Float> lpc_reconstruction = CalculateLPCReconstruction(input, samples, 64);

	// calculate psd
	std::vector<double> psd(spec_length);
	double energy = 0;
	for (int i = 0; i < samples; i++) {
		energy += input[i] * input[i];
	}
	// const double scale2 = 1.0 / (1e-37 + std::sqrt(energy));
	const int seed = 1;
	std::mt19937 mt(seed);
	std::normal_distribution<> dist(0.0, std::pow(10, -140 / 20.0) / std::sqrt(samples)); // noise db
	for (int i = 0; i < samples; i++) {
		// fprintf(stderr, "%d, %e, %e\n", i, input[i], lpc_reconstruction[i]);
		fft_input[i] = ((input[i] /*- 0 * lpc_reconstruction[i]*/)/* * scale2 + dist(mt)*/) * scale;
	}
    dft.Forward(fft_input, (float *)fft_output);
	for (int i = 0; i < spec_length; i++) {
        psd[i] = std::norm(fft_output[i]);
	}

    const int smooth_window_size = std::max<int>(1, 1.0 * samples / sample_rate * 125);
    const auto smooth_psd = ReverbSmoothPsd(psd.data(), samples, smooth_window_size);

	// calculate whitened IR and decompose direct, reverb
	for (int i = 0; i < spec_length; i++) {
		// fprintf(stderr, "%e, %e\n", psd[i], smooth_psd[i]);
        fft_output[i] = std::complex<float>(std::sqrt(psd[i] / std::max(1e-37, smooth_psd[i])) * scale, 0);
	}
	dft.Backward((float *)fft_output, fft_input);
    // Only the head and tail can contribute; the omitted middle is zero by construction.
    double sum_direct = 0, sum_reverb = 0;
    auto accumulate = [&](int i) {
        const int x = i <= samples / 2 ? i : samples - i;
        Float direct_value = 0, reverb_value = 0;
        if (x < direct_length) {
            direct_value = fft_input[i];
        } else if (x < direct_length + dr_transition_length) {
            const double t = 1.0 * (x - direct_length) / dr_transition_length;
            const double s = 0.5 + 0.5 * std::cos(M_PI * t);
            direct_value = fft_input[i] * s;
            reverb_value = fft_input[i] * (1 - s);
        } else if (x < max_length) {
            reverb_value = fft_input[i];
        }
        sum_direct += direct_value * direct_value;
        sum_reverb += reverb_value * reverb_value;
    };
    const int head_end = std::min(samples, max_length);
    for (int i = 0; i < head_end; ++i) accumulate(i);
    // Avoid visiting overlapping head/tail ranges twice for short inputs.
    const int tail_start = std::max(head_end, samples - max_length + 1);
    for (int i = tail_start; i < samples; ++i) accumulate(i);
    *direct = sum_direct;
    *reverb = sum_reverb;

    bakuage::AlignedFree(fft_input);
    bakuage::AlignedFree(fft_output);
}

}

template <typename Float>
void CalculateDirectReverbEnergyRatio(const Float *input, int channels, int samples, int sample_rate, Float *drr) {
	using namespace impl;

	std::vector<double> ratios(channels);
	tbb::parallel_for(0, channels, [&](int ch) {
		std::vector<double> temp(samples);
		// main
		for (int i = 0; i < samples; i++) {
			temp[i] = input[i * channels + ch];
		}

		double direct, reverb;
		CalculateDirectReverbCore<double>(temp.data(), samples, sample_rate, &direct, &reverb);

		ratios[ch] = reverb / (1e-37 + direct);
	});
	// Preserve the original channel reduction order regardless of scheduling.
	double sum_reverb = 0;
	for (int ch = 0; ch < channels; ++ch) sum_reverb += ratios[ch];

	*drr = -10 * std::log10(1e-37 + sum_reverb / channels);
	fprintf(stderr, "drr %f\n", *drr);
}
}

#endif
