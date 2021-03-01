#include "gtest/gtest.h"
#include <limits>
#include "bakuage/utils.h"


TEST(Utils, SignedSqrt) {
    using namespace bakuage;
    EXPECT_EQ(0, SignedSqrt(0.0));
    EXPECT_EQ(1, SignedSqrt(1.0));
    EXPECT_EQ(-1, SignedSqrt(-1.0));
    EXPECT_EQ(2, SignedSqrt(4.0));
    EXPECT_EQ(-3, SignedSqrt(-9.0));
}

TEST(Utils, SanitizeFloat) {
    using namespace bakuage;
    typedef float Float;
    const Float threshold = 1;
    
    EXPECT_EQ(0, SanitizeFloat<Float>(0, threshold));
    EXPECT_EQ(1, SanitizeFloat<Float>(std::numeric_limits<float>::infinity(), threshold));
    EXPECT_EQ(-1, SanitizeFloat<Float>(-std::numeric_limits<float>::infinity(), threshold));
    
    EXPECT_EQ(0, SanitizeFloat<Float>(std::numeric_limits<Float>::quiet_NaN(), threshold));
    EXPECT_EQ(0, SanitizeFloat<Float>(std::numeric_limits<Float>::signaling_NaN(), threshold));
    
    EXPECT_EQ(0.5, SanitizeFloat<Float>(0.5, threshold));
    EXPECT_EQ(1, SanitizeFloat<Float>(2, threshold));
    EXPECT_EQ(-1, SanitizeFloat<Float>(-2, threshold));
}

TEST(Utils, CalcAUC) {
    {
        std::vector<double> xs({ 0, 1 });
        std::vector<double> ys({ 2, 3 });
        EXPECT_NEAR(bakuage::CalcAUC(xs.begin(), xs.end(), ys.begin(), ys.end()), 1, 1e-7);
    }
    {
        std::vector<double> xs({ 2, 3 });
        std::vector<double> ys({ 0, 1 });
        EXPECT_NEAR(bakuage::CalcAUC(xs.begin(), xs.end(), ys.begin(), ys.end()), 0, 1e-7);
    }
    {
        std::vector<double> xs;
        std::vector<double> ys;
        // テストのためにあえて数を変えている
        for (int i = 0; i < 10000; i++) {
            xs.push_back(1.0 * i / 10000);
        }
        for (int i = 0; i < 100000; i++) {
            ys.push_back(1.0 * i / 100000);
        }
        EXPECT_NEAR(bakuage::CalcAUC(xs.begin(), xs.end(), ys.begin(), ys.end()), 0.5, 1e-4);
    }
}

TEST(Utils, IntegrateAlongX) {
    {
        std::vector<double> xs({ 0, 1 });
        std::vector<double> ys({ 0, 1 });
        EXPECT_NEAR(bakuage::IntegrateAlongX(xs.begin(), xs.end(), ys.begin(), ys.end()), 0, 1e-7);
    }
    {
        std::vector<double> xs({ 0, 1 });
        std::vector<double> ys({ 1, 1 });
        EXPECT_NEAR(bakuage::IntegrateAlongX(xs.begin(), xs.end(), ys.begin(), ys.end()), 1, 1e-7);
    }
    {
        std::vector<double> xs({ 0, 0.5, 1 });
        std::vector<double> ys({ 0, 1, 0 });
        EXPECT_NEAR(bakuage::IntegrateAlongX(xs.begin(), xs.end(), ys.begin(), ys.end()), 0.5, 1e-7);
    }
}

