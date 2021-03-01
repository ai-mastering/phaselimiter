#ifndef BAKUAGE_MEMOIZE_FILTER_H_
#define BAKUAGE_MEMOIZE_FILTER_H_

#include <limits>

namespace bakuage {

// パラメータにLPFをかけたときにパラメータが細かく変わるたびに
// 音声処理に必要な値を計算しなおすコストを減らすために、
// 前回と同じ値なら再計算をはしょるフィルター。
// float, doubleどちらにしても100HzのLPFなら44.1kHzで10000 sampleくらいで安定化するので、
// epsとか考えなくてもよさそう。
template <class Float, class Func>
class MemoizeFilter {
public:
	MemoizeFilter() : input_(std::numeric_limits<Float>::quiet_NaN()), output_(0) {}
	Float Clock(Float x) {
		if (x != input_) {
			input_ = x;
			output_ = func_(x);
		}
		return output_;
	}
	Float output() const { return output_; }
private:
	Func func_;
	Float input_;
	Float output_;
};

}

#endif 
