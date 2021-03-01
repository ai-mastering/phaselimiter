#include "gtest/gtest.h"
#include "gflags/gflags.h"

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);

    gflags::SetVersionString("1.0.0-oss");
    gflags::SetUsageMessage("bin/test [gfalgs options] [test options]");
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    return RUN_ALL_TESTS();
}
