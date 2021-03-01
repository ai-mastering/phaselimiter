#ifndef BAKUAGE_BAKUAGE_UTILS_H_
#define BAKUAGE_BAKUAGE_UTILS_H_

#include <cmath>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <complex>
#include <functional>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

// dont abuse (使いすぎると気になることが増えて効率が落ちる)
// 基本ライブラリでは使わずにまったんのコードで使う。
// 効果が出た試しがないので、使わないほうが良いと思う。
// 使ったがまったく効果がなかった例 (さいごのhyperfine参照)
// fast mathあり https://circleci.com/bb/sh1/bakuage_native/779
// fast mathなし https://circleci.com/bb/sh1/bakuage_native/778
#if __GNUC__
#define BA_FAST_MATH __attribute__((optimize("-ffast-math")))
#else
#define BA_FAST_MATH
#endif

namespace bakuage {

class StopWatch {
public:
	StopWatch() : is_paused_(true) { Start(); }

	double Pause() {
		time_ = SystemClockTime();
		is_paused_ = true;
		return time();
	}

	double StopAndStart() {
		double ret = Stop();
		Start();
		return ret;
	}

	double Stop() {
		double ret = time();
		is_paused_ = true;
		time_ = std::chrono::system_clock::duration(0);
		return ret;
	}

	double Start() {
		time_ = std::chrono::system_clock::duration(0);
		is_paused_ = false;
		started_at_ = Now();
		return time();
	}

	double time() {
		auto d = SystemClockTime();
		return 1e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(d).count();
	}
private:
	std::chrono::system_clock::duration SystemClockTime() {
		auto d = time_;
		if (!is_paused_) {
			d += Now() - started_at_;
		}
		return d;
	}

	std::chrono::system_clock::time_point Now() {
		return std::chrono::system_clock::now();
	}

	std::chrono::system_clock::time_point started_at_;
	std::chrono::system_clock::duration time_;
	bool is_paused_;
};

template <typename Float>
inline Float Sqr(const Float &x) {
    static_assert(std::is_floating_point<Float>::value == true, "Sqr Float must be floating point");
	return x * x;
}

    template <typename Float>
    inline Float SignedSqr(const Float &x) {
        static_assert(std::is_floating_point<Float>::value == true, "SignedSqr Float must be floating point");
        return std::abs(x) * x;
    }

    template <typename Float>
    inline Float SignedSqrt(const Float &x) {
        static_assert(std::is_floating_point<Float>::value == true, "SignedSqrt Float must be floating point");
        return (2 * (x > 0) - 1) * std::sqrt(std::abs(x));
    }

template <typename Int>
inline Int CeilInt(Int x, Int unit) {
	typedef typename std::make_unsigned<Int>::type UInt;
	UInt y = static_cast<UInt>(x + unit - 1);
	return (Int)((y / static_cast<UInt>(unit)) * static_cast<UInt>(unit));
}

// https://stackoverflow.com/questions/466204/rounding-up-to-next-power-of-2
inline unsigned int CeilPowerOf2(unsigned int v) {
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v++;
	return v;
}

inline void SleepMs(int ms) {
	std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

#ifdef _MSC_VER
std::string WStringToString(const std::wstring &w);
std::wstring StringToWString(const std::string &s);
//std::string UTF8toSjis(std::string srcUTF8);
#endif


inline double Erb(double hz, double ear_q, double min_band_width, double order) {
	return std::pow(std::pow(hz / ear_q, order) + std::pow(min_band_width, order), 1 / order);
}

inline double GlasbergErb(double hz) {
	// return Erb(f, 9.26449, 24.7, 1);
	return hz * (1.0 / 9.26449) + 24.7;
}

inline double GlasbergErbScale(double hz) {
    return 21.4 * std::log10(1 + 0.00437 * hz);
}

inline double ToDb(double energy) {
	return 10 * std::log10(energy + 1e-37);
}

// これのsによる
// http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/doc/voicebox/frq2bark.html
// 計算式: grep bark in https://www.researchgate.net/publication/228940314_Speech_enhancement_using_temporal_masking_and_fractional_bark_gammatone_filters

inline double HzToBark(double hz) {
	return 7 * std::asinh((1.0 / 650) * hz);
}

inline double BarkToHz(double bark) {
	return 650 * std::sinh((1.0 / 7) * bark);
}

// baseが固定の場合はこっちのほうが速い
inline double Pow(double base, double e) {
	return std::exp(std::log(base) * e);
}

inline double Pow10(double e) {
	return std::exp(2.30258509299404568401799145468436 * e);
}

inline double Log(double base, double x) {
	return std::log(x) * (1.0 / std::log(base));
}

inline double DbToLinearScale(double db) {
	return std::exp((std::log(10) / 20) * db);
}

inline double DbToEnergy(double db) {
	return std::exp((std::log(10) / 10) * db);
}

inline double HzToMel(double hz) {
	return 1127 * std::log1p(hz * (1.0 / 700));
}

inline double MelToHz(double mel) {
	return 700 * (std::expm1(mel * (1.0 / 1127)));
}

// https://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
template <typename T>
int Sign(T val) {
	return (T(0) < val) - (val < T(0));
}

// x = 無限で x - SoftShrink(x) = sqrt(sqr_threshold)
// Fig7 (b) in https://www.researchgate.net/publication/51798904_Robust_Multichannel_Blind_Deconvolution_via_Fast_Alternating_Minimization?_sg=0nKxBjaFE2ARzTIHHyqgvZUy9h0nXbf-9WnDUFRWo0_2l0fZs4w3Fq5lZUQMLrsnUtLntlvaSQ
inline double SoftShrink(double x, double threshold) {
	if (x < 0) return SoftShrink(-x, threshold);

	double alpha = 1.0 / (2.0 * threshold);

	if (x < 1.0 / alpha) {
		return alpha / 2.0 * x * x;
	}
	else {
		return x - 1.0 / (2.0 * alpha);
	}
}

    inline int IntLog2(int x) {
		// https://stackoverflow.com/questions/11376288/fast-computing-of-log2-for-64-bit-integers
#ifdef _MSC_VER
		static const uint8_t tab32[32] = {
			0,  9,  1, 10, 13, 21,  2, 29,
			11, 14, 16, 18, 22, 25,  3, 30,
			8, 12, 20, 28, 15, 17, 24,  7,
			19, 27, 23,  6, 26,  5,  4, 31
		};
		x |= x >> 1;
		x |= x >> 2;
		x |= x >> 4;
		x |= x >> 8;
		x |= x >> 16;
		return tab32[(uint32_t)(x * 0x07C4ACDD) >> 27];
#else
        return 32 - __builtin_clz(x) - 1;
#endif
    }

    inline bool StrEndsWith(const std::string &a, const std::string &b) {
        return a.size() >= b.size() && a.find(b, a.size() - b.size()) != std::string::npos;
    }

    template <class T>
    struct IsTrivial {
        static constexpr bool value = std::is_trivial<T>::value;
    };
    template <>
    struct IsTrivial<std::complex<float>> {
        static_assert(sizeof(std::complex<float>) == 2 * sizeof(float), "std::complex<float> must be packed");
        static constexpr bool value = true;
    };
    template <>
    struct IsTrivial<std::complex<double>> {
        static_assert(sizeof(std::complex<double>) == 2 * sizeof(double), "std::complex<double> must be packed");
        static constexpr bool value = true;
    };

    // 型チェック付きのmemcpy
    // T must be pod
    template <class T>
    void TypedMemcpy(T *dest, const T *src, int len) {
        static_assert(IsTrivial<T>::value, "TypedMemcpy T must be trivial");
        if (len > 0) {
            std::memcpy(reinterpret_cast<void *>(dest), reinterpret_cast<const void *>(src), sizeof(T) * len);
        }
    }
    template <class T>
    void TypedMemmove(T *dest, const T *src, int len) {
        static_assert(IsTrivial<T>::value, "TypedMemmove T must be trivial");
        if (len > 0) {
            std::memmove(reinterpret_cast<void *>(dest), reinterpret_cast<const void *>(src), sizeof(T) * len);
        }
    }
    template <class T>
    void TypedFillZero(T *dest, int len) {
        static_assert(IsTrivial<T>::value == true, "TypedFillZero T must be trivial");
        if (len > 0) {
            std::memset(reinterpret_cast<void *>(dest), 0, sizeof(T) * len);
        }
    }

    template <class T>
    T Sinc(const T &x) {
        if (x == 0) return 1;
        return std::sin(x) / x;
    }

	// CreateMultidimensionalVector
	template <class T>
	std::vector<T> Create1DVector(int size) {
		return std::vector<T>(size);
	}
	template <class T>
	std::vector<T> Create1DVector(int size, const T &value) {
		return std::vector<T>(size, value);
	}
	template <class T>
	std::vector<std::vector<T>> Create2DVector(int size, int size2) {
		return std::vector<std::vector<T>>(size, Create1DVector<T>(size2));
	}
	template <class T>
	std::vector<std::vector<T>> Create2DVector(int size, int size2, const T &value) {
		return std::vector<std::vector<T>>(size, Create1DVector(size2, value));
	}
	template <class T>
	std::vector<std::vector<std::vector<T>>> Create3DVector(int size, int size2, int size3) {
		return std::vector<std::vector<std::vector<T>>>(size, Create2DVector<T>(size2, size3));
	}
	template <class T>
	std::vector<std::vector<std::vector<T>>> Create3DVector(int size, int size2, int size3, const T &value) {
		return std::vector<std::vector<std::vector<T>>>(size, Create2DVector(size2, size3, value));
	}

	template <class T>
	T SanitizeFloat(const T &x, const T &threshold) {
		if (std::isnan(x)) {
			return 0;
		}
		else {
			return std::max<T>(-threshold, std::min<T>(threshold, x));
		}
	}

    // Area under curve (sample and hold)
    template <class It1, class It2>
    double IntegrateAlongX(It1 x_bg, It1 x_ed, It2 y_bg, It2 y_ed) {
        if (x_bg == x_ed) return 0;
        double sum = 0;
        auto x = x_bg;
        auto y = y_bg;
        while (true) {
            auto x_next = x + 1;
            auto y_next = y + 1;
            if (x_next == x_ed) break;

            auto delta_x = *x_next - *x;
            sum += delta_x * (*y);

            x = x_next;
            y = y_next;
        }
        return sum;
    }

    // うまい名前が思いつかないが、
    // 指標に対して、指標 < 任意のスレッショルドで不正解判定したときの
    // ROCのAUC
    // 不正解データ、正解データの順で与える
    // http://www.randpy.tokyo/entry/roc_auc
    template <class It1, class It2>
    double CalcAUC(It1 bg1, It1 ed1, It2 bg2, It2 ed2) {
        std::vector<double> sorted1(bg1, ed1);
        std::sort(sorted1.begin(), sorted1.end());

        std::vector<double> sorted2(bg2, ed2);
        std::sort(sorted2.begin(), sorted2.end());

        std::vector<double> sorted_all(bg1, ed1);
        sorted_all.insert(sorted_all.end(), bg2, ed2);
        std::sort(sorted_all.begin(), sorted_all.end());

        auto lb1 = sorted1.begin();
        auto lb2 = sorted2.begin();
        std::vector<double> xs;
        std::vector<double> ys;
        xs.push_back(0);
        ys.push_back(0);
        for (const auto &value: sorted_all) {
            lb1 = std::lower_bound(lb1, sorted1.end(), value);
            lb2 = std::lower_bound(lb2, sorted2.end(), value);
            ys.push_back(1.0 * std::distance(sorted1.begin(), lb1) / sorted1.size()); // true positive
            xs.push_back(1.0 * std::distance(sorted2.begin(), lb2) / sorted2.size()); // false positive
        }
        xs.push_back(1);
        ys.push_back(1);

        return IntegrateAlongX(xs.begin(), xs.end(), ys.begin(), ys.end());
    }

    // ガウス分布のKL divergence
    template <class V, class M>
    inline double KLDivergence(const V &mean1, const M &covariance1, const V &mean2, const M &inv_covariance2) {
#if 1
        // trace公式: https://mathtrain.jp/trace
        // covarianceは対称行列なので転置不要
        // performance data: https://stackoverflow.com/questions/27030554/column-wise-dot-product-in-eigen-c
        // eigenはexpression templateなのでなるべくまとめて書くべき
        const auto t2 = mean2 - mean1;
        return 0.5 * (inv_covariance2.cwiseProduct(covariance1).sum() + t2.dot(inv_covariance2 * t2) - mean1.size());
#else
        // before optimization
        const auto t1 = (inv_covariance2 * covariance1).trace();
        const auto t2 = mean2 - mean1;
        return 0.5 * (t1 + t2.dot(inv_covariance2 * t2) - mean1.size());
#endif
    }

    // ガウス分布のJensen Shanon Distance
    template <class V, class M>
    inline double JensenShannonDistance(const V &mean1, const M &covariance1, const M &inv_covariance1, const V &mean2, const M &covariance2, const M &inv_covariance2) {
        return std::sqrt(std::max<double>(0.0, 0.5 * (KLDivergence(mean1, covariance1, mean2, inv_covariance2) +
                                                      KLDivergence(mean2, covariance2, mean1, inv_covariance1))));
    }

    std::string LoadStrFromFile(const char *path);
    void LoadDataFromFile(const char *path, const std::function<void (const char *, size_t)> &write);

	template <class T>
	std::string NormalizeToString(const T &input);

    inline int64_t gcd(int64_t a, int64_t b) {
        if (b == 0) return a;
        return gcd(b, a % b);
    }

    inline int64_t lcm(int64_t a , int64_t b) {
        return a * b / gcd(a, b);
    }
}

#endif
