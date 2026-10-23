#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <string>

#include "gtest/gtest.h"
#include "audio_visualizer/frame_renderer.h"
#include "bakuage/vector_math.h"

namespace {
using Renderer = audio_visualizer::FrameRenderer;
using Image = Renderer::Image;
using Spectrum = bakuage::AlignedPodVector<float>;

// Reference the previous per-frame renderer, including its rounding and CImg's
// rectangle clipping, independently of the precomposed implementation.
void ReferenceOverlay(const Image &src, Image *dest) {
    bakuage::AlignedPodVector<float> alpha(src.width());
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < src.height(); ++y) {
            const auto *a = src.data(0, y, 0, 3);
            auto *out = dest->data(0, y, 0, c);
            bakuage::VectorSubConstantRev(a, 255.0f, alpha.data(), src.width());
            bakuage::VectorMulInplace(alpha.data(), out, src.width());
            bakuage::VectorMadInplace(a, src.data(0, y, 0, c), out, src.width());
            bakuage::VectorMulConstantInplace(1.0 / 255.0, out, src.width());
        }
    }
}

Image ReferenceFrame(const Image &background, const Image &foreground, const Spectrum &spectrum) {
    Image image(background), mask(background.width(), background.height(), 1, 4, 0);
    for (size_t i = 0; i < spectrum.size(); ++i) {
        const double center_x = image.width() / 2;
        const double center_y = image.height() * 0.7;
        const double spectrum_width = image.width() * 0.5;
        const double spectrum_height = image.height() * 0.3;
        const double spectrum_left = center_x - spectrum_width / 2;
        const double spectrum_bottom = center_y + spectrum_height / 2;
        const double x1 = spectrum_left + spectrum_width * i / spectrum.size();
        const double x2 = spectrum_left + spectrum_width * (i + 1) / spectrum.size();
        const double space_x = (x2 - x1) * 0.5;
        const int count = std::floor(spectrum[i] * 24 + 0.5);
        for (int j = 0; j < count; ++j) {
            const double y1 = spectrum_bottom - spectrum_height * (j + 1) / 24;
            const double y2 = spectrum_bottom - spectrum_height * j / 24;
            const double space_y = (y2 - y1) * 0.5;
            const float color[] = {255, 255, 255, 200};
            mask.draw_rectangle(x1 + space_x / 2, y1 + space_y / 2, 0,
                                x2 - space_x / 2, y2 - space_y / 2, 1, color);
        }
    }
    ReferenceOverlay(mask, &image);
    ReferenceOverlay(foreground, &image);
    return image;
}

void ExpectPpmPixels(const Renderer::Buffer &ppm, const Image &expected) {
    const bool wide = expected.max() >= 256;
    const std::string header = "P6\n" + std::to_string(expected.width()) + " " +
        std::to_string(expected.height()) + (wide ? "\n65535\n" : "\n255\n");
    ASSERT_EQ(header.size() + expected.size() * (wide ? 2 : 1), ppm.size());
    ASSERT_EQ(header, std::string(ppm.begin(), ppm.begin() + header.size()));
    const auto *p = ppm.data() + header.size();
    for (int y = 0; y < expected.height(); ++y) {
        for (int x = 0; x < expected.width(); ++x) {
            for (int c = 0; c < 3; ++c) {
                const unsigned int value = wide ? static_cast<unsigned short>(expected(x, y, 0, c))
                                                : static_cast<unsigned char>(expected(x, y, 0, c));
                const unsigned int actual = wide ? 256u * p[0] + p[1] : p[0];
                p += wide ? 2 : 1;
                ASSERT_EQ(value, actual) << x << "," << y << "," << c;
            }
        }
    }
}

TEST(AudioVisualizerRenderer, MatchesPreviousRendererAtNormalOddAndClippedDimensions) {
    const int dimensions[][2] = {{640, 360}, {641, 361}, {17, 13}, {1, 1}};
    std::mt19937 rng(537);
    for (const auto &size : dimensions) {
        SCOPED_TRACE(std::to_string(size[0]) + "x" + std::to_string(size[1]));
        Image background(size[0], size[1], 1, 3), foreground(size[0], size[1], 1, 4);
        // Fractional colors and alpha exercise rounding after image resizing.
        for (auto &x : background) x = (rng() % 65281) / 256.0f;
        for (auto &x : foreground) x = (rng() % 65281) / 256.0f;
        for (int y = 0; y < foreground.height(); ++y) {
            for (int x = 0; x < foreground.width(); ++x) {
                if (x % 3 == 0) foreground(x, y, 0, 3) = 0;
                if (x % 3 == 1) foreground(x, y, 0, 3) = 255;
            }
        }
        Renderer renderer(background, foreground, 30);
        for (int frame = 0; frame < 6; ++frame) {
            Spectrum levels(30);
            for (size_t i = 0; i < levels.size(); ++i) {
                levels[i] = frame == 0 ? 0 : frame == 1 ? 1 : (rng() % 10001) / 10000.0f;
                if (frame == 2) levels[i] = (i % 24 + 0.4999f) / 24;
                if (frame == 3) levels[i] = (i % 24 + 0.5001f) / 24;
            }
            const auto expected = ReferenceFrame(background, foreground, levels);
            const auto actual = renderer.RenderImage(levels);
            ASSERT_EQ(0, std::memcmp(expected.data(), actual.data(), expected.size() * sizeof(float)));
            ExpectPpmPixels(renderer.RenderPpm(levels), expected);
        }
    }
}

TEST(AudioVisualizerRenderer, Preserves16BitPixelsAndChoosesDepthPerFrame) {
    Image foreground(1, 1, 1, 4, 0);
    for (float value : {257.0f, 4096.0f, 65535.0f}) {
        Image background(1, 1, 1, 3, value);
        Renderer renderer(background, foreground, 1);
        for (float level : {0.0f, 1.0f}) {
            const Spectrum levels(1, level);
            const auto expected = ReferenceFrame(background, foreground, levels);
            ExpectPpmPixels(renderer.RenderPpm(levels), expected);
            if (value == 257) {
                EXPECT_EQ(level == 0, expected.max() >= 256);
            }
        }
    }
}

TEST(AudioVisualizerRenderer, EmptyAndNonfiniteLevelsDrawNoBlocks) {
    Image background(23, 19, 1, 3, 0), foreground(23, 19, 1, 4, 0);
    Renderer renderer(background, foreground, 3);
    Spectrum empty(3, 0.0f);
    const auto expected = renderer.RenderPpm(empty);
    empty[0] = std::numeric_limits<float>::quiet_NaN();
    empty[1] = std::numeric_limits<float>::infinity();
    empty[2] = -std::numeric_limits<float>::infinity();
    EXPECT_EQ(expected, renderer.RenderPpm(empty));
}

TEST(AudioVisualizerRenderer, RejectsMismatchedBandCount) {
    Image background(23, 19, 1, 3, 0), foreground(23, 19, 1, 4, 0);
    Renderer renderer(background, foreground, 3);
    EXPECT_THROW(renderer.RenderPpm(Spectrum(2, 0.0f)), std::invalid_argument);
    EXPECT_THROW(renderer.RenderImage(Spectrum(4, 0.0f)), std::invalid_argument);
}
}  // namespace
