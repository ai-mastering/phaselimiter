#include "gtest/gtest.h"
#include "bakuage/memory.h"
#include "bakuage/delay_filter2.h"


namespace {
    struct DelayFilter2TestParam {
        int total_size;
        int delay;
        int process_size;
        int process_size2;
        
        friend std::ostream& operator<<(std::ostream& os, const DelayFilter2TestParam& param) {
            os << "DelayFilter2TestParam"
            << " total_size " << param.total_size
            << " delay " << param.delay
            << " process_size " << param.process_size
            << " process_size2 " << param.process_size2
            ;
            return os;
        }
    };
    const DelayFilter2TestParam test_params[] = {
        { 100, 0, 1, 1 },
        { 100, 0, 7, 1 },
        { 100, 0, 1, 7 },
        { 100, 1, 1, 1 },
        { 100, 1, 1, 7 },
        { 100, 1, 7, 1 },
        { 100, 1, 7, 7 },
        { 100, 7, 1, 1 },
        { 100, 7, 1, 16 },
        { 100, 7, 16, 1 },
        { 100, 7, 16, 16 },
        { 1000, 100, 1, 1 },
        { 1000, 100, 1, 32 },
        { 1000, 100, 1, 256 },
        { 1000, 100, 32, 1 },
        { 1000, 100, 32, 32 },
        { 1000, 100, 32, 256 },
        { 1000, 100, 256, 1 },
        { 1000, 100, 256, 32 },
        { 1000, 100, 256, 256 },
    };
    
    class DelayFilter2Test : public ::testing::TestWithParam<DelayFilter2TestParam> {};
}

TEST_P(DelayFilter2Test, Inplace) {
    using namespace bakuage;
    const auto param = GetParam();
    AlignedPodVector<float> output(param.total_size);
    for (int i = 0; i < param.total_size; i++) {
        output[i] = i;
    }
    
    DelayFilter2<float> filter(param.delay);
    int i = 0;
    int j = 0;
    while (i < param.total_size) {
        const auto size = std::min<int>(param.total_size - i, j % 2 == 0 ? param.process_size : param.process_size2);
        filter.Clock(output.data() + i, output.data() + i + size, output.data() + i);
        i += size;
        j++;
    }
    
    for (int i = param.delay; i < param.total_size; i++) {
        EXPECT_EQ(i - param.delay, output[i]);
    }
}


TEST_P(DelayFilter2Test, Outofplace) {
    using namespace bakuage;
    const auto param = GetParam();
    AlignedPodVector<float> input(param.total_size);
    AlignedPodVector<float> output(param.total_size);
    for (int i = 0; i < param.total_size; i++) {
        input[i] = i;
    }
    
    DelayFilter2<float> filter(param.delay);
    int i = 0;
    int j = 0;
    while (i < param.total_size) {
        const auto size = std::min<int>(param.total_size - i, j % 2 == 0 ? param.process_size : param.process_size2);
        filter.Clock(input.data() + i, input.data() + i + size, output.data() + i);
        i += size;
        j++;
    }
    
    for (int i = param.delay; i < param.total_size; i++) {
        EXPECT_EQ(i - param.delay, output[i]);
    }
}

INSTANTIATE_TEST_CASE_P(DelayFilter2TestInstance,
                        DelayFilter2Test,
                        ::testing::ValuesIn(test_params));
