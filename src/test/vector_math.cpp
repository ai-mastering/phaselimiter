#include "gtest/gtest.h"
#include <limits>
#include <vector>
#include "bakuage/vector_math.h"

TEST(VectorMath, VectorLInfFloat) {
    EXPECT_EQ(0, bakuage::VectorLInf<float>(nullptr, 0));
    for (int size : {1, 7, 16, 17, 8192}) {
        SCOPED_TRACE(size);
        // Exclude the first element to exercise an unaligned input.
        std::vector<float> storage(size + 1, 0);
        float *input = storage.data() + 1;
        EXPECT_EQ(0, bakuage::VectorLInf(input, size));
        input[0] = 0.5f;
        input[size - 1] = -1.25f;
        EXPECT_EQ(1.25f, bakuage::VectorLInf(input, size));
        input[size - 1] = 2.0f;
        EXPECT_EQ(2.0f, bakuage::VectorLInf(input, size));
    }
}

TEST(VectorMath, VectorSanitizeInplace) {
    using namespace bakuage;
    typedef float Float;
    const Float threshold = 1;
    
    Float zero = 0;
    VectorSanitizeInplace<Float>(threshold, &zero, 1);
    EXPECT_EQ(0, zero);
    
    Float inf = std::numeric_limits<float>::infinity();
    VectorSanitizeInplace<Float>(threshold, &inf, 1);
    EXPECT_EQ(1, inf);
    
    Float neg_inf = -std::numeric_limits<float>::infinity();
    VectorSanitizeInplace<Float>(threshold, &neg_inf, 1);
    EXPECT_EQ(-1, neg_inf);
    
    Float quiet_nan = std::numeric_limits<Float>::quiet_NaN();
    VectorSanitizeInplace<Float>(threshold, &quiet_nan, 1);
    EXPECT_EQ(0, quiet_nan);
    
    Float sig_nan = std::numeric_limits<Float>::signaling_NaN();
    VectorSanitizeInplace<Float>(threshold, &sig_nan, 1);
    EXPECT_EQ(0, sig_nan);
    
    Float inside = 0.5;
    VectorSanitizeInplace<Float>(threshold, &inside, 1);
    EXPECT_EQ(0.5, inside);
    
    Float neg_inside = -0.5;
    VectorSanitizeInplace<Float>(threshold, &neg_inside, 1);
    EXPECT_EQ(-0.5, neg_inside);
    
    Float outside = 2;
    VectorSanitizeInplace<Float>(threshold, &outside, 1);
    EXPECT_EQ(1, outside);
    
    Float neg_outside = -2;
    VectorSanitizeInplace<Float>(threshold, &neg_outside, 1);
    EXPECT_EQ(-1, neg_outside);
}

