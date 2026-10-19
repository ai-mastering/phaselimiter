#include "gtest/gtest.h"
#include "gflags/gflags.h"
#include <tbb/task_scheduler_init.h>

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);

    gflags::SetVersionString("1.0.0-oss");
    gflags::SetUsageMessage("bin/test [gfalgs options] [test options]");
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    tbb::task_scheduler_init scheduler;
    const int result = RUN_ALL_TESTS();
    scheduler.blocking_terminate();
    return result;
}
