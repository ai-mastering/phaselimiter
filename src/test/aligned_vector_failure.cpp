#include "gtest/gtest.h"
#include <cstdlib>
#include <new>
#include "bakuage/memory.h"

namespace {
bool fail_allocation = false;
}

// This executable supplies its own allocator to exercise allocation failure
// deterministically, without exhausting memory or changing the production API.
namespace bakuage {
void *AlignedMalloc(size_t bytes, size_t) {
    if (fail_allocation) throw std::bad_alloc();
    void *memory = std::malloc(bytes);
    if (!memory) throw std::bad_alloc();
    return memory;
}
void AlignedFree(void *memory) { std::free(memory); }
}

TEST(AlignedVectorFailure, FailedAllocationPreservesDataAndCapacity) {
    bakuage::AlignedPodVector<float> values{1.0f, 2.0f};
    float *original = values.data();
    fail_allocation = true;
    EXPECT_THROW(values.resize(8), std::bad_alloc);
    EXPECT_EQ(original, values.data());
    EXPECT_EQ(2u, values.size());
    EXPECT_FLOAT_EQ(1.0f, values[0]);
    EXPECT_FLOAT_EQ(2.0f, values[1]);
    // A failed growth must not leave an inflated capacity that skips allocation.
    EXPECT_THROW(values.resize(8), std::bad_alloc);
    fail_allocation = false;
    values.resize(8);
    EXPECT_FLOAT_EQ(1.0f, values[0]);
    EXPECT_FLOAT_EQ(2.0f, values[1]);
    for (size_t i = 2; i < values.size(); ++i) EXPECT_FLOAT_EQ(0, values[i]);
}
