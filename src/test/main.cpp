#include "gtest/gtest.h"
#include "gflags/gflags.h"
#include <oneapi/tbb/global_control.h>

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);

    gflags::SetVersionString("1.0.0-oss");
    gflags::SetUsageMessage("bin/test [gfalgs options] [test options]");
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    tbb::task_scheduler_handle scheduler{tbb::attach{}};
    const int result = RUN_ALL_TESTS();
    tbb::finalize(scheduler);
    return result;
}
