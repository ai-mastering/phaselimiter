#include "gtest/gtest.h"
#include "bakuage/sndfile_wrapper.h"

struct sf_private_tag {
    int close_calls = 0;
};

// Supply the C API in this isolated executable, without linking libsndfile.
// Invalidating a live descriptor can terminate the Windows C runtime.
extern "C" int sf_close(SNDFILE *file) {
    ++file->close_calls;
    return SF_ERR_SYSTEM;
}

extern "C" const char *sf_error_number(int) {
    return "injected close failure";
}

TEST(SndfileCloseFailure, CheckedCloseConsumesHandleAndReportsFailure) {
    sf_private_tag file;
    {
        bakuage::SndfileWrapper output;
        output.set(&file);
        EXPECT_THROW(output.CloseChecked(), std::runtime_error);
        EXPECT_EQ(nullptr, output.get());
        EXPECT_EQ(1, file.close_calls);
        EXPECT_NO_THROW(output.CloseChecked());
        EXPECT_EQ(1, file.close_calls);
    }
    EXPECT_EQ(1, file.close_calls);
}

TEST(SndfileCloseFailure, DestructorDoesNotThrowOnFailure) {
    sf_private_tag file;
    {
        bakuage::SndfileWrapper output;
        output.set(&file);
    }
    EXPECT_EQ(1, file.close_calls);
}
