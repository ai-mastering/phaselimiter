#ifndef BAKUAGE_AUDIO_ANALYZER_STATISTICS_H_
#define BAKUAGE_AUDIO_ANALYZER_STATISTICS_H_

#include <cmath>
#include <algorithm>

namespace audio_analyzer {

class Statistics {
public:
    Statistics(): count_(0), sum_(0), sum2_(0) {}

    void Add(double value, double count = 1) {
        count_ += count;
        sum_ += value * count;
        sum2_ += value * value * count;
    }

    double count() const { return count_; }
    double mean() const { return sum_ / (1e-300 + count_); }
    double sum() const { return sum_; }
    double sum2() const { return sum2_; }
    double variance() const { 
        double m = mean();
        return std::max<double>(0.0, sum2_ / (1e-300 + count_) - m * m);
    }
    double stddev() const {
        return std::sqrt(variance());
    }

private:
    double count_;
    double sum_;
    double sum2_;
};

}

#endif 