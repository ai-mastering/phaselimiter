#include "bakuage/stacktrace.h"

#include <stdio.h>
#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>

namespace {
    // https://stackoverflow.com/questions/77005/how-to-automatically-generate-a-stacktrace-when-my-program-crashes
    void SignalHandler(int sig) {
        // get void*'s for all entries on the stack
        void *array[10];
        const size_t size = backtrace(array, 10);
        
        // print out all the frames to stderr
        fprintf(stderr, "Error: signal %d:\n", sig);
        backtrace_symbols_fd(array, size, STDERR_FILENO);
        exit(1);
    }
}

namespace bakuage {
    void RegisterStacktracePrinter() {
        // 予行演習で呼ぶ。クラッシュ後にDLLのロードができない場合があるので
        {
            void *array[10];
            const size_t size = backtrace(array, 10);
            fprintf(stderr, "%s", "backtrace test");
            backtrace_symbols_fd(array, size, STDERR_FILENO);
        }
        
        signal(SIGABRT, SignalHandler);
        signal(SIGSEGV, SignalHandler);
    }
}
