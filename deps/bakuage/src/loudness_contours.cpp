#include "bakuage/loudness_contours.h"

#include <cmath>
#include <vector>
#include "bakuage/linear_interpolator.h"

namespace bakuage {
namespace loudness_contours {

namespace {
static const std::vector<float> hz_to_spl_at_60_phon = {
	20, 108,
	30,  100,
	100, 78,
	300, 66,
	700, 60,
	1000, 60,
	1400, 63,
	2000, 60,
	3000, 56,
	4000, 58,
	5000, 62,
	6000, 67,
	8000, 72,
	9000, 73,
	10000, 73,
	12000, 72,
	14000, 69,
	15000, 68
};

static const std::vector<float> hz_to_spl_at_40_phon = {
	20, 101,
	30,  89,
	100, 62,
	300, 46,
	700, 41,
	1000, 40,
	1400, 43,
	2000, 39,
	3000, 36,
	4000, 47,
	5000, 40,
	6000, 45,
	8000, 52,
	9000, 54,
	10000, 55,
	12000, 54,
	14000, 52,
	15000, 51
};
    
    // https://docs.google.com/spreadsheets/d/1gt6tgzKKBd__nSdlXtwbtwGLkoDhbQeWhMq2TLp748I/edit#gid=264445328
    static const std::vector<double> hz_to_youtube_weighting = {
        44, -20.2147481,
        50, -17.81939869,
        54, -16.71462936,
        75, -11.31462984,
        100, -7.014690876,
        165, -1.392479324,
        200, 0.08062324524,
        400, 0.5856996536,
        668, 0.3854717255,
        800, 0.1945802689,
        1000, 0,
        2000, 2.251997185,
        3000, 2.244224739,
        3500, 7.58545742,
        4000, 7.420843315,
        5000, 4.463936424,
        6000, -0.1869785309,
        7500, -6.488230515,
        8000, -7.376457977,
        9000, -7.429050064,
        10000, -(7.357099342 + 7.153114796) / 2,
        11000, -10.07579699,
        11500, -12.7019001,
        12000, -15.39600229,
        13000, -(19.51467896 + 19.45669737) / 2,
        14000, -(23.41446676 + 23.35474176) / 2,
        15000, -(29.11408577 + 29.10816641) / 2,
    };

struct Initializer {
	static Initializer &GetInstance() {
		static Initializer initializer;
		return initializer;
	}

	Initializer() :
		hz_to_spl_at_40_phon_interpolator(convert(hz_to_spl_at_40_phon)),
    hz_to_spl_at_60_phon_interpolator(convert(hz_to_spl_at_60_phon)),
    hz_to_youtube_weighting_interpolator(convert(hz_to_youtube_weighting, true)) {}

	const LinearInterpolator<float> hz_to_spl_at_40_phon_interpolator;
	const LinearInterpolator<float> hz_to_spl_at_60_phon_interpolator;
    const LinearInterpolator<float> hz_to_youtube_weighting_interpolator;

private:
	template <class Float>
	static std::vector<float> convert(const std::vector<Float> &hz_to_spl, bool extrapolation = false) {
		std::vector<float> result;
		for (int i = 0; i < hz_to_spl.size() / 2; i++) {
			result.push_back(std::log10(hz_to_spl[2 * i + 0]));
			result.push_back(hz_to_spl[2 * i + 1]);
		}
        if (extrapolation) {
            {
                const auto x1 = result[0];
                const auto y1 = result[1];
                const auto x2 = result[2];
                const auto y2 = result[3];
                const auto x = std::log10(1e-37);
                result.insert(result.begin(), (x - x1) / (x2 - x1) * (y2 - y1) + y1);
                result.insert(result.begin(), x);
            }
            {
                const auto x1 = result[result.size() - 4];
                const auto y1 = result[result.size() - 3];
                const auto x2 = result[result.size() - 2];
                const auto y2 = result[result.size() - 1];
                const auto x = std::log10(96000 * 4); // 十分大きい数
                result.push_back(x);
                result.push_back((x - x1) / (x2 - x1) * (y2 - y1) + y1);
            }
        }
		return result;
	}
};

}

double SplToPhon(double log10_hz, double spl) {
	auto spl_at_40_phon = Initializer::GetInstance().hz_to_spl_at_40_phon_interpolator.Get(log10_hz);
	auto spl_at_60_phon = Initializer::GetInstance().hz_to_spl_at_60_phon_interpolator.Get(log10_hz);

	return (spl - spl_at_40_phon) / (spl_at_60_phon - spl_at_40_phon) * 20 + 40;
}

double HzToSplAt60Phon(double hz) {
	return Initializer::GetInstance().hz_to_spl_at_60_phon_interpolator.Get(std::log10(hz));
}
    
    double HzToYoutubeWeighting(double hz) {
        return Initializer::GetInstance().hz_to_youtube_weighting_interpolator.Get(std::log10(1e-37 + hz));
    }

}
}
