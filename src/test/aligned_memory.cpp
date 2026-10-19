#include "gtest/gtest.h"
#include <cstdint>
#include <limits>
#include "bakuage/memory.h"

TEST(AlignedMemory, AlignmentAndOverflow) {
    for (size_t alignment : {size_t(16), size_t(64), size_t(128)}) {
        for (size_t size : {size_t(0), size_t(1), size_t(129)}) {
            void *memory = bakuage::AlignedMalloc(size, alignment);
            ASSERT_NE(nullptr, memory);
            EXPECT_EQ(0u, reinterpret_cast<uintptr_t>(memory) % alignment);
            if (size) std::memset(memory, 0x5a, size);
            bakuage::AlignedFree(memory);
        }
    }
    EXPECT_THROW(bakuage::AlignedMalloc(std::numeric_limits<size_t>::max()), std::bad_alloc);
    EXPECT_THROW(bakuage::AlignedMalloc(16, 0), std::invalid_argument);
    EXPECT_THROW(bakuage::AlignedMalloc(16, 3), std::invalid_argument);
}

TEST(AlignedMemory, FailedGrowthPreservesVectorAndRemainsUsable) {
    bakuage::AlignedPodVector<float> values{1.0f, 2.0f};
    float *original = values.data();
    EXPECT_THROW(values.resize(std::numeric_limits<size_t>::max()), std::length_error);
    EXPECT_EQ(original, values.data());
    ASSERT_EQ(2u, values.size());
    EXPECT_FLOAT_EQ(1.0f, values[0]);
    EXPECT_FLOAT_EQ(2.0f, values[1]);
    values.resize(8);
    EXPECT_FLOAT_EQ(1.0f, values[0]);
    EXPECT_FLOAT_EQ(2.0f, values[1]);
    for (size_t i = 2; i < values.size(); ++i) EXPECT_FLOAT_EQ(0, values[i]);
    values.resize(1);
    values.shrink_to_fit();
    ASSERT_EQ(1u, values.size());
    EXPECT_FLOAT_EQ(1.0f, values[0]);
    values.resize(0);
    values.shrink_to_fit();
    EXPECT_EQ(nullptr, values.data());
    values.push_back(3.0f);
    EXPECT_FLOAT_EQ(3.0f, values[0]);
}
