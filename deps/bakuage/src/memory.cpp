#include "bakuage/memory.h"

#ifndef BAKUAGE_DISABLE_TBB
#define BAKUAGE_MEMORY_USE_TBB
#endif

#ifdef BAKUAGE_MEMORY_USE_TBB
#include <tbb/scalable_allocator.h>
#endif

namespace bakuage {
    // 長さもアラインするので、 + 2 * alignment
    void *AlignedMalloc(size_t size, size_t alignment) {
#ifdef BAKUAGE_MEMORY_USE_TBB
        return scalable_aligned_malloc(CeilInt(size, alignment), alignment);
#else
        assert(alignment <= 128);
        uintptr_t mem = (uintptr_t)calloc(size + 2 * alignment, 1);
        uintptr_t aligned = CeilInt(mem + 1, alignment);
        uint8_t *res = (uint8_t *)aligned;
        res[-1] = aligned - mem;
        return (void *)res;
#endif
    }

    void AlignedFree(void *ptr) {
        if (!ptr) return;
#ifdef BAKUAGE_MEMORY_USE_TBB
        return scalable_aligned_free(ptr);
#else
        uint8_t *p = (uint8_t *)ptr;
        p -= p[-1];
        free(p);
#endif
    }
}

size_t getPeakRSS();
size_t getCurrentRSS();

namespace bakuage {

size_t GetPeakRss( )
{
    return getPeakRSS();
}

size_t GetCurrentRss( )
{
    return getCurrentRSS();
}

}
