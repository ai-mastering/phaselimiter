#include "bakuage/memory.h"

#ifndef BAKUAGE_DISABLE_TBB
#define BAKUAGE_MEMORY_USE_TBB
#endif

#ifdef BAKUAGE_MEMORY_USE_TBB
#include <oneapi/tbb/scalable_allocator.h>
#endif

namespace bakuage {
    void *AlignedMalloc(size_t size, size_t alignment) {
        if (alignment < sizeof(void *) || (alignment & (alignment - 1)) != 0) {
            throw std::invalid_argument("alignment must be a power of two and at least pointer-sized");
        }
#ifndef BAKUAGE_MEMORY_USE_TBB
        if (alignment > 128) {
            throw std::invalid_argument("fallback allocator supports alignment up to 128 bytes");
        }
#endif
        const size_t limit = static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max());
        if (alignment > limit / 2 || size > limit - 2 * alignment) {
            throw std::bad_alloc();
        }
#ifdef BAKUAGE_MEMORY_USE_TBB
        const size_t rounded_size = (std::max<size_t>(size, 1) + alignment - 1) & ~(alignment - 1);
        void *result = scalable_aligned_malloc(rounded_size, alignment);
        if (!result) throw std::bad_alloc();
        return result;
#else
        void *allocation = calloc(size + 2 * alignment, 1);
        if (!allocation) throw std::bad_alloc();
        const uintptr_t address = reinterpret_cast<uintptr_t>(allocation);
        const size_t offset = alignment - address % alignment;
        uint8_t *result = static_cast<uint8_t *>(allocation) + offset;
        result[-1] = static_cast<uint8_t>(offset);
        return result;
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
