#include "classic_detector/detector.h"
#include "classic_detector/numpy_neighbors.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <boost/filesystem.hpp>

namespace {
classic_detector::Reference Reference(double mean) {
    classic_detector::Reference r;
    r.mean = Eigen::VectorXd::Constant(2, mean);
    r.covariance = Eigen::MatrixXd::Identity(2, 2);
    return r;
}
struct Directory {
    boost::filesystem::path path = boost::filesystem::temp_directory_path() / boost::filesystem::unique_path("classic-%%%%-%%%%");
    Directory() { boost::filesystem::create_directories(path); }
    ~Directory() { boost::system::error_code ec; boost::filesystem::remove_all(path, ec); }
};
}
TEST(ClassicDetector, GaussianDistance) {
    auto a = Reference(0), b = Reference(2);
    EXPECT_DOUBLE_EQ(0, classic_detector::Distance(a, a, 0));
    EXPECT_NEAR(2, classic_detector::Distance(a, b, 0), 1e-14);
    EXPECT_NEAR(classic_detector::Distance(a, b, .001), classic_detector::Distance(b, a, .001), 1e-14);
    b.covariance *= 4;
    EXPECT_NEAR(std::sqrt(3.625), classic_detector::Distance(a, b, 0), 1e-14);
    b.covariance.setZero();
    EXPECT_THROW(classic_detector::Distance(a, b, 0), std::runtime_error);
}
TEST(ClassicDetector, LofTrainingAndNoveltyAreDifferent) {
    classic_detector::Preparation p;
    p.paths = {"a", "b", "c", "outlier"};
    p.distances = {{0, 1, 2, 10}, {1, 0, 1, 9}, {2, 1, 0, 8}, {10, 9, 8, 0}};
    std::vector<classic_detector::Reference> references;
    for (double mean : {0., 1., 2., 10.}) references.push_back(Reference(mean));
    const classic_detector::Detector detector(p, references, .5, .1);
    // Hand-computed sklearn reachability densities: 2/3, 1/2, 2/3, 2/17.
    const auto &lofs = detector.reference_lofs();
    EXPECT_NEAR(.875, lofs[0], 1e-9);
    EXPECT_NEAR(4. / 3, lofs[1], 1e-9);
    EXPECT_NEAR(.875, lofs[2], 1e-9);
    EXPECT_NEAR(119. / 24, lofs[3], 1e-9);
    const auto result = detector.Detect(references[1]);
    EXPECT_EQ(1, result.index);
    EXPECT_NEAR(.875, result.lof, 1e-9); // query includes itself, training excludes it
    EXPECT_NEAR((119. / 24) * .7 + (4. / 3) * .3, result.border_lof, 1e-9);
}
TEST(ClassicDetector, NumpyNeighborTieOrder) {
    EXPECT_EQ((std::vector<size_t>{0, 1, 2}), classic_detector::NumpyNeighbors({0, 0, 0, 1}, 3));
    EXPECT_EQ((std::vector<size_t>{1, 3}), classic_detector::NumpyNeighbors({4, 1, 3, 2}, 2));
    EXPECT_THROW(classic_detector::NumpyNeighbors({1}, 2), std::invalid_argument);
}
TEST(ClassicDetector, RejectsInvalidInputs) {
    classic_detector::Preparation p;
    EXPECT_THROW(classic_detector::Detector(p, {}), std::invalid_argument);
}
TEST(ClassicDetector, MatchesSklearn0191WithTiedNeighbors) {
    // Golden values generated using numpy 1.14.3, scipy 1.0.1, sklearn 0.19.1.
    classic_detector::Preparation p;
    std::vector<classic_detector::Reference> references;
    for (int i = 0; i < 24; ++i) {
        p.paths.push_back(std::to_string(i)); references.push_back(Reference(i / 10.));
        p.distances.emplace_back();
        for (int j = 0; j < 24; ++j) p.distances.back().push_back(std::abs((i * 7) % 24 - (j * 7) % 24));
    }
    const classic_detector::Detector detector(p, references, 5. / 24, .1);
    const std::vector<double> expected = {1.1505555555508167, 1, 1, 1.0794736842079076,
        .9822222222225679, 1, .9416666666683738, 1.0794736842079076, 1, 1,
        1.0794736842079076, .9750000000007812, 1, .9822222222225679,
        1.0794736842079076, 1, 1, 1.1505555555508167, .9875000000003906,
        1, .9822222222225679, .9822222222225679, 1, .9750000000007812};
    for (size_t i = 0; i < expected.size(); ++i) EXPECT_NEAR(expected[i], detector.reference_lofs()[i], 1e-14) << i;
    const auto result = detector.Detect(Reference(.713));
    EXPECT_NEAR(1.0311111111099505, result.lof, 1e-14);
    EXPECT_NEAR(1.0794736842079076, result.border_lof, 1e-14);
}
TEST(ClassicDetector, MatchesNumpy1143PartitionAndUnstableSort) {
    std::vector<double> distances;
    for (int i = 0; i < 32; ++i) distances.push_back((i * 13) % 7);
    EXPECT_EQ((std::vector<size_t>{14,28,7,21}), classic_detector::NumpyNeighbors(distances, 4));
    EXPECT_EQ((std::vector<size_t>{7,21,28,14,0,13,6,27,20}), classic_detector::NumpyNeighbors(distances, 9));
    EXPECT_EQ((std::vector<size_t>{0,28,21,7,14,27,20,6,13,5,19,12,26,25,4,11,18}), classic_detector::NumpyNeighbors(distances, 17));
    EXPECT_EQ((std::vector<size_t>{0,28,7,21,14,27,6,20,13,19,5,26,12,18,11,25,4,24,31,3,10,17,2}), classic_detector::NumpyNeighbors(distances, 23));
}
TEST(ClassicDetector, PreservesRoundedKeyShortcutAndDegenerateScore) {
    classic_detector::Preparation p;
    p.paths = {"a", "b", "c"};
    p.distances = {{0,1,1},{1,0,1},{1,1,0}};
    std::vector<classic_detector::Reference> references = {Reference(0), Reference(1), Reference(2)};
    classic_detector::Detector detector(p, references, .34, .1);
    auto changed = references[0]; changed.mean[1] = 100;
    EXPECT_EQ(detector.Detect(references[0]).lof, detector.Detect(changed).lof);
    const auto result = detector.Detect(references[0]);
    EXPECT_TRUE(std::isnan(result.sound_quality));
    EXPECT_NE(std::string::npos, classic_detector::DetectionJson(result, p).find("NaN"));
}
TEST(ClassicDetector, DuplicateRoundedKeysBypassPreparedDistances) {
    std::vector<classic_detector::Reference> references = {Reference(0), Reference(0), Reference(2)};
    references[1].mean[1] = 3;
    classic_detector::Preparation p;
    p.paths = {"a", "b", "c"};
    p.distances.assign(3, std::vector<double>(3, 100));
    p.distances[2][2] = 0;
    const classic_detector::Detector first(p, references, .34, .1);
    for (size_t i = 0; i < 3; ++i) for (size_t j = 0; j < 3; ++j)
        p.distances[i][j] = classic_detector::Distance(references[i], references[j], .001);
    const classic_detector::Detector second(p, references, .34, .1);
    EXPECT_EQ(first.reference_lofs(), second.reference_lofs());
}
TEST(ClassicDetector, ReadsShortStatisticsAndSubtractsLoudness) {
    Directory dir;
    auto path = (dir.path / "input.json").string();
    std::ofstream(path) << R"({"loudness":-10,"bands_short":[{"mid_mean":-12,"side_mean":-20}],"covariance_short":[[1,0.5],[0.5,2]],"bands":[],"covariance":[]})";
    const auto r = classic_detector::ReadReference(path);
    EXPECT_DOUBLE_EQ(-2, r.mean[0]); EXPECT_DOUBLE_EQ(-10, r.mean[1]);
    EXPECT_DOUBLE_EQ(.5, r.covariance(0, 1)); EXPECT_DOUBLE_EQ(2, r.covariance(1, 1));
    std::ofstream(path) << R"({"loudness":0,"bands_short":[{"mid_mean":1,"side_mean":2}],"covariance_short":[[1]]})";
    EXPECT_THROW(classic_detector::ReadReference(path), std::invalid_argument);
    std::ofstream(path) << "not JSON";
    EXPECT_THROW(classic_detector::ReadReference(path), std::runtime_error);
}
TEST(ClassicDetector, ReadsPreparationWithoutReorderingAndRelocatesBundledPaths) {
    Directory dir;
    boost::filesystem::create_directories(dir.path / "analysis_data");
    std::ofstream((dir.path / "analysis_data/000001.json").string()) << "{}";
    const auto preparation = (dir.path / "preparation.json").string();
    std::ofstream(preparation) << R"({"paths":["b","a"],"distance_matrix":[[0,3],[3,0]],"covariance_regularization":0.02})";
    const auto p = classic_detector::ReadPreparation(preparation);
    EXPECT_EQ((std::vector<std::string>{"b", "a"}), p.paths);
    EXPECT_DOUBLE_EQ(.02, p.regularization); EXPECT_DOUBLE_EQ(3, p.distances[0][1]);
    const auto expected = (dir.path / "analysis_data/000001.json").string();
    EXPECT_EQ(expected, classic_detector::ResolveReferencePath("./resource/analysis_data/000001.json", preparation));
    EXPECT_EQ(expected, classic_detector::ResolveReferencePath("./analysis_data/000001.json", preparation));
    EXPECT_THROW(classic_detector::ResolveReferencePath("missing.json", preparation), std::runtime_error);
    std::ofstream(preparation) << R"({"paths":["a"],"distance_matrix":[[0,1]],"covariance_regularization":0.001})";
    EXPECT_THROW(classic_detector::ReadPreparation(preparation), std::invalid_argument);
}
TEST(ClassicDetector, ClampsNeighborsAndRejectsNonfiniteConfiguration) {
    classic_detector::Preparation p;
    p.paths = {"a", "b", "c"}; p.distances = {{0,1,2},{1,0,1},{2,1,0}};
    std::vector<classic_detector::Reference> references = {Reference(0), Reference(1), Reference(2)};
    const classic_detector::Detector first(p, references, 1, .1), second(p, references, 100, .1);
    EXPECT_EQ(first.reference_lofs(), second.reference_lofs());
    EXPECT_THROW(classic_detector::Detector(p, references, 0, .1), std::invalid_argument);
    EXPECT_THROW(classic_detector::Detector(p, references, .03, 1.1), std::invalid_argument);
}
