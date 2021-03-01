#ifndef BAKUAGE_BAKUAGE_GAMMATONE_FILTER_H_
#define BAKUAGE_BAKUAGE_GAMMATONE_FILTER_H_

#include <cstring>
#include <cmath>
#include <complex>
#include "bakuage/utils.h"

namespace bakuage {

	// もとになっている論文は忘れたけど、たぶんこれ https://engineering.purdue.edu/~malcolm/apple/tr35/PattersonsEar.pdf
    template <typename Float = double>
    class GammatoneFilter {
    public:
        GammatoneFilter() {}

        GammatoneFilter(Float center_freq, Float band_width, Float sample_rate): 
            center_freq_(center_freq),
            band_width_(band_width),
            sample_rate_(sample_rate)
        {
            ClearState();

            double t = 1.0 / sample_rate;
            double b = 2 * M_PI * 1.019 * band_width;
            double cf = center_freq;

            double theta = 2 * cf * M_PI * t;
            double s = std::sin(theta);
            double c = std::cos(theta);
            double y = -2;

            double sign[2] = {1, -1};

            std::complex<double> temp = 1;
            for (int i = 0; i < 4; i++) {
                temp *= (-2.0 * std::exp(std::complex<double>(0, 4 * cf * M_PI * t)) * t
                        + 2.0 * std::exp(std::complex<double>(-b * t, 2 * cf * M_PI * t)) * t
                            * (c + sign[i % 2] * std::sqrt(3 + sign[i / 2] * std::pow(2, 3 / 2)) * s
                            )
                        )
                    / (-2.0 / std::exp(2 * b * t) - 2.0 * std::exp(std::complex<double>(0, 4 * cf * M_PI * t))
                        + 2.0 * (1.0 + std::exp(std::complex<double>(0, 4 * cf * M_PI * t))) / std::exp(b * t));
            }

            const double gain = std::pow(std::abs(temp), 1.0 / 4);

            for (int i = 0; i < 4; i++) {
                // coef_x2[i] = 0;
                coef_x1[i] = 2 * t * c / std::exp(b * t) / y / gain + sign[i % 2]
                    * 2 * std::sqrt(3 + sign[i / 2] * std::pow(2, 3 / 2)) * t * s
                    / std::exp(b * t) / y / gain;
                coef_x[i] = -2 * t / y / gain;

                coef_y2[i] = 2 / std::exp(2 * b * t) / y;
                coef_y1[i] = -4 * c / std::exp(b * t) / y;
            }
        }

        Float Clock(const Float &_x) {
            Float x = _x;
			Float y;
            for (int i = 0; i < 4; i++) {
                // 足し算を並列実行
#if 0
				y = (x * coef_x[i] + prev_x1[i] * coef_x1[i])
                    + (prev_y1[i] * coef_y1[i] + prev_y2[i] * coef_y2[i]);
#else
                y = std::fma(x, coef_x[i], std::fma(prev_x1[i], coef_x1[i], std::fma(prev_y1[i], coef_y1[i], prev_y2[i] * coef_y2[i])));
#endif
                prev_x1[i] = x;
                prev_y2[i] = prev_y1[i];
                prev_y1[i] = y;
                x = y;
            }
            return y;
        }

        void ClearState() {
            memset(prev_y2, 0, sizeof(prev_y2));
            memset(prev_y1, 0, sizeof(prev_y1));
            memset(prev_x2, 0, sizeof(prev_x2));
            memset(prev_x1, 0, sizeof(prev_x1));
        }
    private:
        

        Float center_freq_;
        Float band_width_;
        Float sample_rate_;
        Float coef_y2[4];
        Float coef_y1[4];
        Float coef_x1[4];
        Float coef_x[4];
        Float prev_y2[4];
        Float prev_y1[4];
        Float prev_x2[4];
        Float prev_x1[4];
    };
}

#endif 
