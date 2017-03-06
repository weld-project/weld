#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "parlib.h"

#define CACHE_BITS 6
#define CACHE_LINE (1 << CACHE_BITS)
#define MASK (~(CACHE_LINE - 1))

static inline int64_t num_cache_blocks(int64_t size) {
  // ceil of number of blocks
  return (size + (CACHE_LINE - 1)) >> CACHE_BITS;
}

extern "C" void *get_merger_at_index(void *m, int64_t size, int32_t i) {
  intptr_t ptr = reinterpret_cast<intptr_t>(m);
  ptr = (ptr + (CACHE_LINE - 1)) & MASK;
  return reinterpret_cast<void *>(ptr + num_cache_blocks(size) * i * CACHE_LINE);
}

// zero-initializes storage
extern "C" void *new_merger(int64_t size, int32_t nworkers) {
  int64_t total_blocks = num_cache_blocks(size) * nworkers;
  // extra space to ensure first block is aligned to boundary
  void *space = malloc(total_blocks * CACHE_LINE + (CACHE_LINE - 1));
  for (int32_t i = 0; i < nworkers; i++) {
    memset(get_merger_at_index(space, size, i), 0, size);
  }
  return space;
}

extern "C" void free_merger(void *m) {
  free(m);
}
