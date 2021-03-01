#pragma once



#include <algorithm>
#include <vector>

namespace bakuage {

template <class Float = double>
class LinearInterpolator {
public:
	LinearInterpolator(const std::vector<Float> &v) {
		// assert(v.size() % 2 == 0);
		for (int i = 0; i < v.size(); i += 2) {
			data_.emplace_back(v[i], v[i + 1]);
		}
	}
	Float Get(Float x) const {
		int i = 0;
		while (i < data_.size() && data_[i].first < x) {
			i++;
		}
		if (i == 0) {
			return data_[0].second;
		}
		else if (i == data_.size()) {
			return data_[data_.size() - 1].second;
		}
		else {
			Float t = (x - data_[i - 1].first) / (data_[i].first - data_[i - 1].first);
			return data_[i - 1].second * (1 - t) + data_[i].second * t;
		}
	}
private:
	typedef std::pair<Float, Float> Point;
	std::vector<Point> data_;
};

/*
TEST(LinearInterpolator, Basic) {
LinearInterpolator<float> interpolator(std::vector<float>({ 0, 0, 1, 100 }));
EXPECT_EQ(0, (int)interpolator.get(-1));
EXPECT_EQ(0, (int)interpolator.get(0));
EXPECT_EQ(10, (int)interpolator.get(0.1));
EXPECT_EQ(90, (int)interpolator.get(0.9));
EXPECT_EQ(100, (int)interpolator.get(1));
EXPECT_EQ(100, (int)interpolator.get(2));
}
*/

}