#ifndef PHASE_LIMITER_CLI_AUDIO_H_
#define PHASE_LIMITER_CLI_AUDIO_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace phase_limiter {
// The CLI processes stereo audio at 44100 Hz. Its DSP modules use int lengths.
inline int CheckedOversamplingFactor(int external, int internal) {
    for (int factor : {external, internal}) {
        if (factor <= 0 || (factor & (factor - 1)) != 0) {
            throw std::invalid_argument("limiter oversampling factors must be positive powers of 2");
        }
    }
    const int limit = std::numeric_limits<int>::max() / 44100;
    if (external > limit / internal) {
        throw std::length_error("limiter sample rate exceeds DSP size limit");
    }
    return external * internal;
}

inline void CheckDspSampleCount(size_t samples, int factor = 1) {
    if (factor <= 0 || samples > static_cast<size_t>(std::numeric_limits<int>::max()) / factor) {
        throw std::length_error("audio exceeds DSP sample limit");
    }
}

struct AudioFrameRange {
    size_t start;
    size_t end;
    size_t margin_start;
    size_t margin_end;
};

inline AudioFrameRange CalculateAudioFrameRange(size_t frames, double start, double end) {
    if (!std::isfinite(start) || !std::isfinite(end)) {
        throw std::invalid_argument("start_at and end_at must be finite");
    }
    const auto to_frame = [frames](double seconds) -> size_t {
        if (seconds <= 0) return 0;
        // Clamp before multiplication/conversion, including extremely large times.
        if (seconds >= static_cast<double>(frames) / 44100) return frames;
        return std::min(frames, static_cast<size_t>(std::floor(seconds * 44100)));
    };
    const size_t first = to_frame(start);
    const size_t last = std::max(first, end < 0 ? frames : to_frame(end));
    const size_t margin = 22050;
    return {first, last, first - std::min(first, margin), last + std::min(frames - last, margin)};
}
}

#endif
