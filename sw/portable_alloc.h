/* Host/testbench-only helper (not used by the kernel): aligned allocation that works with MinGW/MSVC,
 * where C11 aligned_alloc is missing. Memory from llm_aligned_alloc is never freed by the drivers
 * (process exit); on Windows it would have to go back through _aligned_free, not free().
 */
#ifndef PORTABLE_ALLOC_H
#define PORTABLE_ALLOC_H

#include <stdlib.h>
#ifdef _WIN32
#include <malloc.h>
#endif

static inline void *llm_aligned_alloc(size_t align, size_t bytes) {
    bytes = (bytes + align - 1) / align * align;
#ifdef _WIN32
    return _aligned_malloc(bytes, align);
#else
    return aligned_alloc(align, bytes);
#endif
}

#endif
