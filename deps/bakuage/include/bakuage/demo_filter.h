#ifndef BAKUAGE_BAKUAGE_DEMO_FILTER_H_
#define BAKUAGE_BAKUAGE_DEMO_FILTER_H_

#include "bakuage/memory.h"

namespace bakuage {

// VSTのデモ版で使うフィルタ、一定時間ごとに無音になる
// 無音の理由は、解除が難しいから (小さいノイズを足すとかだと、十分大きい音ならかき消せるし)
// あと、複雑にすると汎用性がなくなるし、明確にデモということがわからないと、
// プラグインの性能の問題かどうかも切り分けられなくなるし

template <typename Float = double>
class DemoFilter {
public:
	struct Config {
		int sample_rate;
		int num_channels;
	};

	DemoFilter(const Config &config):
		config_(config), 
		frame_(0),
		bypass_frames_(config.sample_rate * kBypassSec()),
		silent_frames_(config.sample_rate * kSilentSec()),
		transition_frames_(config.sample_rate * kTransitionSec()),
		temp_(config.num_channels),
		transition_window_(transition_frames_)
	{
		for (int i = 0; i < transition_frames_; i++) {
			const double t = 1.0 * i / transition_frames_;
			transition_window_[i] = 0.5 - 0.5 * std::cos(M_PI * t);
		}
	}

	// input: [ch]
	// output: [ch]
	// まったく同じであれば、同じメモリ領域を指していても良い
	void Clock(const Float *input, Float *output) {
		const auto w = wet();
		for (int ch = 0; ch < config_.num_channels; ch++) {
			output[ch] = input[ch] * w;
		}

		frame_++;
		if (frame_ >= bypass_frames_ + transition_frames_ + silent_frames_ + transition_frames_) {
			frame_ = 0;
		}
	}

	// input: [ch][frame]
	// output: [ch][frame]
	// まったく同じであれば、同じメモリ領域を指していても良い
	void Clock(Float **input, int frames, Float **output) {
		for (int i = 0; i < frames; i++) {
			for (int ch = 0; ch < config_.num_channels; ch++) {
				temp_[ch] = input[ch][i];
			}
			Clock(temp_.data(), temp_.data());
			for (int ch = 0; ch < config_.num_channels; ch++) {
				output[ch][i] = temp_[ch];
			}
		}
	}

	Float wet() const {
		Float w = 0;
		if (frame_ < bypass_frames_) {
			// バイパス
			w = 1;
		}
		else if (frame_ < bypass_frames_ + transition_frames_) {
			// 無音へ移行
			w = 1 - transition_window_[frame_ - bypass_frames_];
		}
		else if (frame_ < bypass_frames_ + transition_frames_ + silent_frames_) {
			// 無音
			w = 0;
		}
		else {
			// バイパスへ移行
			w = transition_window_[frame_ - (bypass_frames_ + transition_frames_ + silent_frames_)];
		}
		return w;
	}

private:
	constexpr double kSilentSec() { return 0.5; }
	constexpr double kBypassSec() { return 15; }
	constexpr double kTransitionSec() { return 0.5; }

	Config config_;
	int frame_;
	int bypass_frames_;
	int silent_frames_;
	int transition_frames_;
	std::vector<Float> temp_;
	std::vector<Float> transition_window_;
};

}

#endif 
