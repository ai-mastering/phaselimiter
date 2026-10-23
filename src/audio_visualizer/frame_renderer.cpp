#include "frame_renderer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#include "bakuage/vector_math.h"

namespace audio_visualizer {
namespace {

// Keep the original sequence of vector operations, including alpha-zero pixels,
// so precomposition preserves floating-point rounding before integer conversion.
void Overlay(const FrameRenderer::Image &src, FrameRenderer::Image *dest) {
    bakuage::AlignedPodVector<float> one_minus_alpha(src.width());
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < src.height(); ++y) {
            const float *color = src.data(0, y, 0, c);
            const float *alpha = src.data(0, y, 0, 3);
            float *out = dest->data(0, y, 0, c);
            bakuage::VectorSubConstantRev(alpha, 255.0f, one_minus_alpha.data(), src.width());
            bakuage::VectorMulInplace(one_minus_alpha.data(), out, src.width());
            bakuage::VectorMadInplace(alpha, color, out, src.width());
            bakuage::VectorMulConstantInplace(1.0 / 255.0, out, src.width());
        }
    }
}

FrameRenderer::Buffer EncodePpm(const FrameRenderer::Image &image) {
    // Match CImg's automatic PNG bit depth. P6 stores 16-bit samples big endian.
    const bool wide = image.max() >= 256;
    const std::string header = "P6\n" + std::to_string(image.width()) + " " +
        std::to_string(image.height()) + (wide ? "\n65535\n" : "\n255\n");
    const size_t pixels = static_cast<size_t>(image.width()) * image.height();
    FrameRenderer::Buffer buffer(header.size() + pixels * 3 * (wide ? 2 : 1));
    std::memcpy(buffer.data(), header.data(), header.size());
    auto *dest = buffer.data() + header.size();
    const float *red = image.data(0, 0, 0, 0);
    const float *green = image.data(0, 0, 0, 1);
    const float *blue = image.data(0, 0, 0, 2);
    if (wide) {
        for (size_t i = 0; i < pixels; ++i) {
            for (float value : {red[i], green[i], blue[i]}) {
                const auto pixel = static_cast<unsigned short>(value);
                *dest++ = pixel >> 8;
                *dest++ = pixel & 255;
            }
        }
    } else {
        for (size_t i = 0; i < pixels; ++i) {
            *dest++ = static_cast<unsigned char>(red[i]);
            *dest++ = static_cast<unsigned char>(green[i]);
            *dest++ = static_cast<unsigned char>(blue[i]);
        }
    }
    return buffer;
}

}  // namespace

FrameRenderer::FrameRenderer(const Image &background, const Image &foreground, int band_count)
    : background_(background), lit_(background) {
    if (band_count <= 0 || background.is_empty() || background.depth() != 1 ||
        background.spectrum() != 3 || foreground.width() != background.width() ||
        foreground.height() != background.height() || foreground.depth() != 1 ||
        foreground.spectrum() != 4) {
        throw std::invalid_argument("Invalid visualizer dimensions or band count");
    }

    Image mask(background.width(), background.height(), 1, 4, 0);
    Overlay(mask, &background_);
    Overlay(foreground, &background_);
    for (int c = 0; c < 4; ++c) mask.get_shared_channel(c).fill(c == 3 ? 200 : 255);
    Overlay(mask, &lit_);
    Overlay(foreground, &lit_);

    rectangles_.resize(band_count);
    for (int i = 0; i < band_count; ++i) {
        // Preserve the original geometry, integer truncation and inclusive edges.
        const double center_x = background.width() / 2;
        const double center_y = background.height() * 0.7;
        const double spectrum_width = background.width() * 0.5;
        const double spectrum_height = background.height() * 0.3;
        const double left = center_x - spectrum_width / 2;
        const double bottom = center_y + spectrum_height / 2;
        const double x1 = left + spectrum_width * i / band_count;
        const double x2 = left + spectrum_width * (i + 1) / band_count;
        const double space_x = (x2 - x1) * 0.5;
        for (int j = 0; j < kBlockCount; ++j) {
            const double y1 = bottom - spectrum_height * (j + 1) / kBlockCount;
            const double y2 = bottom - spectrum_height * j / kBlockCount;
            const double space_y = (y2 - y1) * 0.5;
            int xa = x1 + space_x / 2, xb = x2 - space_x / 2;
            int ya = y1 + space_y / 2, yb = y2 - space_y / 2;
            if (xa > xb) std::swap(xa, xb);
            if (ya > yb) std::swap(ya, yb);
            rectangles_[i][j] = {std::max(0, xa), std::max(0, ya),
                std::min(background.width() - 1, xb), std::min(background.height() - 1, yb)};
        }
    }

    // A single cached depth is safe only when both states fit in 8 bits.
    // Otherwise determine the bit depth from the actual frame, as PNG did.
    if (background_.max() < 256 && lit_.max() < 256) {
        ppm_background_ = EncodePpm(background_);
        ppm_lit_ = EncodePpm(lit_);
        header_size_ = ppm_background_.size() - static_cast<size_t>(background.width()) * background.height() * 3;
    }
}

template <class CopyRectangle>
void FrameRenderer::ForEachRectangle(const bakuage::AlignedPodVector<float> &spectrum,
                                    const CopyRectangle &copy) const {
    if (spectrum.size() != rectangles_.size()) {
        throw std::invalid_argument("Visualizer spectrum band count differs from renderer");
    }
    for (size_t i = 0; i < spectrum.size(); ++i) {
        // Silence can produce a degenerate normalization range; draw no blocks
        // for non-finite levels rather than converting NaN to an integer.
        if (!std::isfinite(spectrum[i])) continue;
        const float level = std::max(0.0f, std::min(1.0f, spectrum[i]));
        const int count = std::floor(level * kBlockCount + 0.5);
        for (int j = 0; j < count; ++j) {
            const auto &r = rectangles_[i][j];
            if (r.x0 <= r.x1 && r.y0 <= r.y1) copy(r);
        }
    }
}

FrameRenderer::Image FrameRenderer::RenderImage(const bakuage::AlignedPodVector<float> &spectrum) const {
    Image image(background_);
    ForEachRectangle(spectrum, [&](const Rectangle &r) {
        for (int c = 0; c < 3; ++c) {
            for (int y = r.y0; y <= r.y1; ++y) {
                std::memcpy(image.data(r.x0, y, 0, c), lit_.data(r.x0, y, 0, c),
                            static_cast<size_t>(r.x1 - r.x0 + 1) * sizeof(float));
            }
        }
    });
    return image;
}

FrameRenderer::Buffer FrameRenderer::RenderPpm(const bakuage::AlignedPodVector<float> &spectrum) const {
    if (ppm_background_.empty()) return EncodePpm(RenderImage(spectrum));

    Buffer buffer(ppm_background_);
    ForEachRectangle(spectrum, [&](const Rectangle &r) {
        for (int y = r.y0; y <= r.y1; ++y) {
            const size_t offset = header_size_ +
                (static_cast<size_t>(y) * background_.width() + r.x0) * 3;
            std::memcpy(buffer.data() + offset, ppm_lit_.data() + offset,
                        static_cast<size_t>(r.x1 - r.x0 + 1) * 3);
        }
    });
    return buffer;
}

}  // namespace audio_visualizer
