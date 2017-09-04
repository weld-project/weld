#include "runtime.h"

// zero-initializes storage
extern "C" void *weld_rt_new_merger(int64_t size, int32_t nworkers) {
  int64_t total_blocks = num_cache_blocks(size) * nworkers;
  // extra space to ensure first block is aligned to boundary
  void *space = malloc(total_blocks * CACHE_LINE + (CACHE_LINE - 1));
  for (int32_t i = 0; i < nworkers; i++) {
    memset(weld_rt_get_merger_at_index(space, size, i), 0, size);
  }
  return space;
}

extern "C" void weld_rt_free_merger(void *m) {
  free(m);
}
