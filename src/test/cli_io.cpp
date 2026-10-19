#include "gtest/gtest.h"

#include <boost/filesystem.hpp>
#include <fstream>
#include <limits>
#include <type_traits>
#include <iterator>
#include <string>
#include <vector>
#include "bakuage/ffmpeg.h"
#include "bakuage/file_utils.h"
#include "phase_limiter/wave_utils.h"
#include "phase_limiter/cli_audio.h"

namespace {
namespace fs = boost::filesystem;
class CliIo : public testing::Test {
protected:
    fs::path directory;
    void SetUp() override {
        directory = fs::temp_directory_path() / fs::unique_path("phaselimiter-%%%%-%%%%");
        fs::create_directory(directory);
    }
    void TearDown() override {
        boost::system::error_code ignored;
        fs::remove_all(directory, ignored);
    }
    static void Write(const fs::path &path, const std::string &value) {
        std::ofstream output(path.string(), std::ios::binary);
        output << value;
    }
    static std::string Read(const fs::path &path) {
        std::ifstream input(path.string(), std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
    }
};

static_assert(!std::is_copy_constructible<bakuage::SndfileWrapper>::value, "file ownership must not be copied");
static_assert(!std::is_copy_assignable<bakuage::SndfileWrapper>::value, "file ownership must not be copied");

TEST(CliAudio, ChecksOversamplingBeforeMultiplication) {
    EXPECT_EQ(8, phase_limiter::CheckedOversamplingFactor(2, 4));
    for (int factor : {0, -1, 3, std::numeric_limits<int>::min()}) {
        EXPECT_THROW(phase_limiter::CheckedOversamplingFactor(factor, 1), std::invalid_argument);
        EXPECT_THROW(phase_limiter::CheckedOversamplingFactor(1, factor), std::invalid_argument);
    }
    EXPECT_THROW(phase_limiter::CheckedOversamplingFactor(1 << 30, 1 << 30), std::length_error);
    EXPECT_THROW(phase_limiter::CheckedOversamplingFactor(1, 1 << 30), std::length_error);
    const size_t limit = std::numeric_limits<int>::max();
    EXPECT_NO_THROW(phase_limiter::CheckDspSampleCount(limit));
    EXPECT_THROW(phase_limiter::CheckDspSampleCount(limit + 1), std::length_error);
    EXPECT_NO_THROW(phase_limiter::CheckDspSampleCount(limit / 4, 4));
    EXPECT_THROW(phase_limiter::CheckDspSampleCount(limit / 4 + 1, 4), std::length_error);
}

TEST(CliAudio, ClampsTrimTimesAndPreservesMargins) {
    auto range = phase_limiter::CalculateAudioFrameRange(441000, 1, 2);
    EXPECT_EQ(44100u, range.start);
    EXPECT_EQ(88200u, range.end);
    EXPECT_EQ(22050u, range.margin_start);
    EXPECT_EQ(110250u, range.margin_end);
    range = phase_limiter::CalculateAudioFrameRange(100, -1, -1);
    EXPECT_EQ(0u, range.start);
    EXPECT_EQ(100u, range.end);
    range = phase_limiter::CalculateAudioFrameRange(100, 1e300, 1e300);
    EXPECT_EQ(100u, range.start);
    EXPECT_EQ(100u, range.end);
    EXPECT_EQ(100u, range.margin_end);
    range = phase_limiter::CalculateAudioFrameRange(441000, 2, 1);
    EXPECT_EQ(range.start, range.end);
    const size_t large = std::numeric_limits<size_t>::max();
    range = phase_limiter::CalculateAudioFrameRange(large, 1e300, -1);
    EXPECT_EQ(large, range.end);
    EXPECT_EQ(large, range.margin_end);
    range = phase_limiter::CalculateAudioFrameRange(0, 0, -1);
    EXPECT_EQ(0u, range.margin_end);
    for (double time : {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        EXPECT_THROW(phase_limiter::CalculateAudioFrameRange(100, time, -1), std::invalid_argument);
        EXPECT_THROW(phase_limiter::CalculateAudioFrameRange(100, 0, time), std::invalid_argument);
    }
}

TEST_F(CliIo, ExplicitCloseConsumesHandle) {
    SF_INFO info = {};
    info.channels = 2;
    info.samplerate = 44100;
    info.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;
    const auto file = (directory / "close.wav").string();
    bakuage::SndfileWrapper output;
    ASSERT_NE(nullptr, output.set(sf_open(file.c_str(), SFM_WRITE, &info)));
    EXPECT_NO_THROW(output.CloseChecked());
    EXPECT_EQ(nullptr, output.get());
    EXPECT_NO_THROW(output.CloseChecked());
}

TEST_F(CliIo, ReadsAndWritesFloatAndDoubleSamples) {
    const auto file = (directory / "typed.wav").string();
    const std::vector<double> doubles{0.25, -0.5, 0.125, 0.75};
    const std::vector<float> floats(doubles.begin(), doubles.end());
    phase_limiter::SaveFloatWave(doubles, file);
    EXPECT_EQ(doubles, phase_limiter::LoadFloatWave<double>(file, 2, 44100));
    EXPECT_EQ(floats, phase_limiter::LoadFloatWave<float>(file, 2, 44100));
    phase_limiter::SaveFloatWave(floats, file);
    EXPECT_EQ(doubles, phase_limiter::LoadFloatWave<double>(file));
    EXPECT_EQ(floats, phase_limiter::LoadFloatWave<float>(file));
}

TEST_F(CliIo, SanitizesDoubleSamplesBeforeSaving) {
    const auto file = (directory / "sanitized.wav").string();
    const std::vector<double> samples{std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(), 0.25};
    phase_limiter::SaveFloatWave(samples, file);
    const std::vector<double> expected{0, 1e7, -1e7, 0.25};
    EXPECT_EQ(expected, phase_limiter::LoadFloatWave<double>(file));
}

TEST_F(CliIo, RejectsUnexpectedAudioFormat) {
    for (int channels : {1, 2}) {
        for (int rate : {44100, 48000}) {
            const auto file = (directory / "input.wav").string();
            const std::vector<float> samples(channels * 16, 0.25f);
            phase_limiter::SaveFloatWave(samples, file, channels, rate);
            if (channels == 2 && rate == 44100) {
                EXPECT_EQ(samples, phase_limiter::LoadFloatWave<float>(file, 2, 44100));
            } else {
                EXPECT_THROW(phase_limiter::LoadFloatWave<float>(file, 2, 44100), std::logic_error);
            }
            // Other callers can still load audio without the limiter's format restriction.
            EXPECT_EQ(samples, phase_limiter::LoadFloatWave<float>(file));
        }
    }
}

TEST_F(CliIo, InvalidWaveDimensionsPreserveExistingOutput) {
    const auto file = directory / "output.wav";
    Write(file, "original");
    const std::vector<float> samples{0.25f, -0.25f, 0.5f};
    for (int channels : {0, -1}) {
        EXPECT_THROW(phase_limiter::SaveFloatWave(samples, file.string(), channels), std::invalid_argument);
    }
    for (int rate : {0, -1}) {
        EXPECT_THROW(phase_limiter::SaveFloatWave(samples, file.string(), 1, rate), std::invalid_argument);
    }
    EXPECT_THROW(phase_limiter::SaveFloatWave(samples, file.string(), 2), std::invalid_argument);
    EXPECT_EQ("original", Read(file));
}

TEST_F(CliIo, PassesLiteralProcessArguments) {
    const fs::path executable = directory / "ffmpeg space $ tool.exe";
    {
        std::ifstream input(PROCESS_FIXTURE_PATH, std::ios::binary);
        std::ofstream output(executable.string(), std::ios::binary);
        output << input.rdbuf();
        ASSERT_TRUE(output.good());
    }
    fs::permissions(executable, fs::owner_all);
    const std::string input = "input space \" $(touch injected) `echo x` &.wav";
    const auto output = directory / "output $ & file.wav";
    bakuage::FFMpeg::Execute(executable.string(), input, output.string(),
                           {"-filter:a", "volume=0.5", "literal \" $ ` &"});
    EXPECT_EQ("-nostdin\n-i\n" + input + "\n-filter:a\nvolume=0.5\nliteral \" $ ` &\n", Read(output));
    EXPECT_THROW(bakuage::FFMpeg::Execute(executable.string(), "fail", output.string(), {}), std::runtime_error);
    EXPECT_THROW(bakuage::FFMpeg::Execute((directory / "missing").string(), input, output.string(), {}), std::exception);
}

TEST_F(CliIo, PublishesAndReplacesOutput) {
    const auto source = directory / "encoded";
    const auto target = directory / "output";
    Write(source, "new audio");
    bakuage::PublishFile(source.string(), target.string());
    EXPECT_EQ("new audio", Read(target));
    Write(source, "replacement");
    bakuage::PublishFile(source.string(), target.string());
    EXPECT_EQ("replacement", Read(target));
    // Publishing does not consume the source used by later processing stages.
    EXPECT_EQ("replacement", Read(source));
}

TEST_F(CliIo, FailedPublicationPreservesExistingOutputAndCleansStaging) {
    const auto target = directory / "output";
    Write(target, "original");
    EXPECT_THROW(bakuage::PublishFile((directory / "missing").string(), target.string()), std::exception);
    EXPECT_EQ("original", Read(target));
    EXPECT_EQ(1, std::distance(fs::directory_iterator(directory), fs::directory_iterator()));
    // Force a failure at rename, after staging has succeeded.
    const auto source = directory / "encoded";
    Write(source, "new");
    const auto target_directory = directory / "existing_directory";
    fs::create_directory(target_directory);
    Write(target_directory / "keep", "original");
    EXPECT_THROW(bakuage::PublishFile(source.string(), target_directory.string()), fs::filesystem_error);
    EXPECT_EQ("original", Read(target_directory / "keep"));
    EXPECT_EQ(3, std::distance(fs::directory_iterator(directory), fs::directory_iterator()));
}
}
