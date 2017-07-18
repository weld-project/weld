#include <map>
#include <pthread.h>
#include <stdlib.h>
#include "parlib.h"

using namespace std;

int64_t mem_limit;
map<intptr_t, int64_t> allocs;
int64_t cur_mem;
pthread_mutex_t lock;
int64_t errno;

extern "C" void weld_rt_init(int64_t m_limit) {
  mem_limit = m_limit;
  cur_mem = 0;
  allocs.clear();
  pthread_mutex_init(&lock, NULL);
  set_runid(0);
  weld_rt_set_errno(0, 0);
}

extern "C" void *weld_rt_malloc(int64_t run_id, size_t size) {
  pthread_mutex_lock(&lock);
  if (cur_mem + size > mem_limit) {
    pthread_mutex_unlock(&lock);
    weld_rt_set_errno(run_id, 7);
    weld_abort_thread();
    return NULL;
  }
  cur_mem += size;
  void *mem = malloc(size);
  allocs.emplace(reinterpret_cast<intptr_t>(mem), size);
  pthread_mutex_unlock(&lock);
  return mem;
}

extern "C" void *weld_rt_realloc(int64_t run_id, void *data, size_t size) {
  pthread_mutex_lock(&lock);
  int64_t orig_size = allocs.find(reinterpret_cast<intptr_t>(data))->second;
  if (cur_mem - orig_size + size > mem_limit) {
    pthread_mutex_unlock(&lock);
    weld_rt_set_errno(run_id, 7);
    weld_abort_thread();
    return NULL;
  }
  cur_mem -= orig_size;
  allocs.erase(reinterpret_cast<intptr_t>(data));
  cur_mem += size;
  void *mem = realloc(data, size);
  allocs.emplace(reinterpret_cast<intptr_t>(mem), size);
  pthread_mutex_unlock(&lock);
  return mem;
}

extern "C" void weld_rt_free(int64_t run_id, void *data) {
  pthread_mutex_lock(&lock);
  cur_mem -= allocs.find(reinterpret_cast<intptr_t>(data))->second;
  allocs.erase(reinterpret_cast<intptr_t>(data));
  free(data);
  pthread_mutex_unlock(&lock);
}

extern "C" int64_t weld_rt_get_errno(int64_t run_id) {
  return errno;
}

extern "C" void weld_rt_set_errno(int64_t run_id, int64_t eno) {
  errno = eno;
}

extern "C" void weld_rt_run_free(int64_t run_id) {
  for (auto it = allocs.begin(); it != allocs.end(); it++) {
    free(reinterpret_cast<void *>(it->first));
  }
}

extern "C" int64_t weld_rt_memory_usage(int64_t run_id) {
    return cur_mem;
}
