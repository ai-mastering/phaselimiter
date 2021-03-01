#include "gtest/gtest.h"

#if 0
#include "bakuage/memory.h"

TEST(TemporaryMemory, Test) {
    using namespace bakuage;
    for (int i = 0; i < 16; i++) {
        TemporaryMemoryReservation reservation(32);
        EXPECT_EQ(32, reservation.size());
        EXPECT_FALSE(reservation.allocated());
        for (int j = 0; j < 16; j++)
        {
            auto memory = reservation.Alloc();
            EXPECT_EQ(32, memory.size());
            std::memset(memory.get(), 0, memory.size());
            EXPECT_TRUE(reservation.allocated());
        }
        EXPECT_FALSE(reservation.allocated());
    }
}

TEST(TemporaryAlignedPodVector, Test) {
    using namespace bakuage;
    TemporaryAlignedPodVectorReservation<int> reservation(32);
    EXPECT_EQ(32, reservation.size());
    for (int i = 0; i < 16; i++)
    {
        auto memory = reservation.Alloc();
        EXPECT_EQ(32, memory.size());
        std::memset(memory.data(), 0, sizeof(int) * memory.size());
    }
}
#endif
