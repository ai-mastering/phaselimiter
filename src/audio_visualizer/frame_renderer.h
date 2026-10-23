#ifndef AUDIO_VISUALIZER_FRAME_RENDERER_H_
#define AUDIO_VISUALIZER_FRAME_RENDERER_H_

#include <array>
#include <vector>

#include "CImg.h"
#include "bakuage/memory.h"

namespace audio_visualizer {

// Background and foreground are static; a spectrum pixel is either transparent
// or white with alpha 200. Precompose both states using the original arithmetic.
class FrameRenderer {
public:
    using Image = cimg_library::CImg<float>;
    using Buffer = std::vector<unsigned char>;

    FrameRenderer(const Image &background, const Image &foreground, int band_count);

    Image RenderImage(const bakuage::AlignedPodVector<float> &spectrum) const;
    Buffer RenderPpm(const bakuage::AlignedPodVector<float> &spectrum) const;

private:
    static constexpr int kBlockCount = 24;
    struct Rectangle { int x0, y0, x1, y1; };

    template <class CopyRectangle>
    void ForEachRectangle(const bakuage::AlignedPodVector<float> &spectrum,
                          const CopyRectangle &copy) const;

    Image background_, lit_;
    std::vector<std::array<Rectangle, kBlockCount>> rectangles_;
    Buffer ppm_background_, ppm_lit_;
    size_t header_size_ = 0;
};

}  // namespace audio_visualizer

#endif
