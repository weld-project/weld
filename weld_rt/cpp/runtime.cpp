#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>
#include <map>
#include <queue>
#include <deque>
#include <algorithm>
#include <exception>
#include "assert.h"
#include "runtime.h"

// These is needed to ensure each grain size is divisible by the SIMD vector size. A value of 64
// should be sufficiently high enough to protect against all the common vector lengths (4, 8,
// 16, 32, 64 - 64 is used for 8-bit values in AVX-512).
#define MAX_SIMD_SIZE   64

#ifdef __APPLE__
#include <sched.h>

typedef int pthread_spinlock_t;

static int pthread_spin_init(pthread_spinlock_t *lock, int pshared) {
    __asm__ __volatile__ ("" ::: "memory");
    *lock = 0;
    return 0;
}

static int pthread_spin_destroy(pthread_spinlock_t *lock) {
    return 0;
}

static int pthread_spin_lock(pthread_spinlock_t *lock) {
    while (1) {
        int i;
        for (i=0; i < 10000; i++) {
            if (__sync_bool_compare_and_swap(lock, 0, 1)) {
                return 0;
            }
        }
        sched_yield();
    }
}

static int pthread_spin_trylock(pthread_spinlock_t *lock) {
    if (__sync_bool_compare_and_swap(lock, 0, 1)) {
        return 0;
    }
    return 1;
}

static int pthread_spin_unlock(pthread_spinlock_t *lock) {
    __asm__ __volatile__ ("" ::: "memory");
    *lock = 0;
    return 0;
}
#endif

/*
The Weld parallel runtime. When the comments refer to a "computation",
this means a single complete execution of a Weld program.
*/
using namespace std;

class weld_abort_exception: public exception {};

typedef deque<work_t *> work_queue;
typedef pthread_spinlock_t work_queue_lock;

int64_t run_id = 0; // the current run ID
pthread_mutex_t global_lock;
// allows each thread to retrieve its run ID
pthread_key_t global_id;

struct run_data {
  pthread_mutex_t lock;
  int32_t n_workers;
  pthread_t *workers;
  work_queue *all_work_queues; // queue per worker
  work_queue_lock *all_work_queue_locks; // lock per queue
  volatile bool done; // if a computation is currently running, have we finished it?
  void *result; // stores the final result of the computation to be passed back
  // to the caller
  int64_t mem_limit;
  map<intptr_t, int64_t> allocs;
  int64_t cur_mem;
  volatile int64_t err; // "errno" is a macro on some systems so we'll call this "err"
};

map<int64_t, run_data*> *runs;

typedef struct {
  int64_t run_id;
  int32_t thread_id;
} thread_data;

extern "C" void weld_runtime_init() {
  pthread_mutex_init(&global_lock, NULL);
  pthread_key_create(&global_id, NULL);
  runs = new map<int64_t, run_data*>;
}

// *** weld_rt functions and helpers ***

extern "C" int32_t weld_rt_thread_id() {
  return reinterpret_cast<thread_data *>(pthread_getspecific(global_id))->thread_id;
}

extern "C" int64_t weld_rt_get_run_id() {
  return reinterpret_cast<thread_data *>(pthread_getspecific(global_id))->run_id;
}

static inline run_data *get_run_data_by_id(int64_t run_id) {
  pthread_mutex_lock(&global_lock);
  run_data *rd = runs->find(run_id)->second;
  pthread_mutex_unlock(&global_lock);
  return rd;
}

static inline run_data *get_run_data() {
  return get_run_data_by_id(weld_rt_get_run_id());
}

// set the result of the computation, called from generated LLVM
extern "C" void weld_rt_set_result(void *res) {
  get_run_data()->result = res;
}

extern "C" int32_t weld_rt_get_nworkers() {
  return get_run_data()->n_workers;
}

extern "C" void weld_rt_abort_thread() {
  throw weld_abort_exception();
}

static inline void set_nest(work_t *task) {
  assert(task->full_task);
  vector<int64_t> idxs;
  vector<int64_t> task_ids;
  idxs.push_back(task->cur_idx);
  task_ids.push_back(task->task_id);
  work_t *cur = task->cont;
  int32_t nest_len = 1;
  while (cur != NULL) {
    idxs.push_back(cur->cur_idx);
    // subtract 1 because the conts give us the continuations and we want the
    // task_id's of the loop bodies before the continuations
    task_ids.push_back(cur->task_id - 1);
    cur = cur->cont;
    nest_len++;
  }
  task->nest_idxs = (int64_t *)malloc(sizeof(int64_t) * nest_len);
  task->nest_task_ids = (int64_t *)malloc(sizeof(int64_t) * nest_len);
  task->nest_len = nest_len;
  // we want the outermost idxs to be the "high-order bits" when we do a comparison of task nests
  reverse_copy(idxs.begin(), idxs.end(), task->nest_idxs);
  reverse_copy(task_ids.begin(), task_ids.end(), task->nest_task_ids);
}

static inline void set_full_task(work_t *task) {
  // possible for task to already be full, e.g. if the head task from a start_loop call is queued
  // but stolen before it can be executed (the thief tries to set_full_task a second time)
  if (task->full_task) {
    return;
  }
  task->full_task = true;
  set_nest(task);
}

// attempt to steal from back of the queue of a random victim
// should be called when own work queue empty
static inline bool try_steal(int32_t my_id, run_data *rd) {
  int32_t victim = rand() % rd->n_workers;
  if (!pthread_spin_trylock(rd->all_work_queue_locks + victim)) {
    work_queue *its_work_queue = rd->all_work_queues + victim;
    if (its_work_queue->empty()) {
      pthread_spin_unlock(rd->all_work_queue_locks + victim);
      return false;
    } else {
      work_t *popped = its_work_queue->back();
      its_work_queue->pop_back();
      pthread_spin_unlock(rd->all_work_queue_locks + victim);
      set_full_task(popped);
      pthread_spin_lock(rd->all_work_queue_locks + my_id);
      (rd->all_work_queues + my_id)->push_front(popped);
      pthread_spin_unlock(rd->all_work_queue_locks + my_id);
      return true;
    }
  } else {
    return false;
  }
}

// called once task function returns
// decrease the dependency count of the continuation, run the continuation
// if necessary, or signal the end of the computation if we are done
static inline void finish_task(work_t *task, int32_t my_id, run_data *rd) {
  if (task->cont == NULL) {
    if (!task->continued) {
      // if this task has no continuation and there was no for loop to end it,
      // the computation is over
      rd->done = true;
    }
    free(task->data);
  } else {
    int32_t previous = __sync_fetch_and_sub(&task->cont->deps, 1);
    if (previous == 1) {
      // run the continuation since we are the last dependency
      set_full_task(task->cont);
      pthread_spin_lock(rd->all_work_queue_locks + my_id);
      (rd->all_work_queues + my_id)->push_front(task->cont);
      pthread_spin_unlock(rd->all_work_queue_locks + my_id);
      // we are the last sibling with this data, so we can free it
      free(task->data);
    }
    if (task->full_task) {
      free(task->nest_idxs);
      free(task->nest_task_ids);
    }
  }
  free(task);
}

// set the continuation of w to cont and increment cont
// dependency count
static inline void set_cont(work_t *w, work_t *cont) {
  w->cont = cont;
  __sync_fetch_and_add(&cont->deps, 1);
}

// called from generated code to schedule a for loop with the given body and continuation
// the data pointers store the closures for the body and continuation
// lower and upper give the iteration range for the loop
// w is the currently executing task
extern "C" void weld_rt_start_loop(work_t *w, void *body_data, void *cont_data, void (*body)(work_t*),
  void (*cont)(work_t*), int64_t lower, int64_t upper, int32_t grain_size) {
  work_t *body_task = (work_t *)malloc(sizeof(work_t));
  memset(body_task, 0, sizeof(work_t));
  body_task->data = body_data;
  body_task->fp = body;
  body_task->lower = lower;
  body_task->upper = upper;
  body_task->cur_idx = lower;
  body_task->task_id = w->task_id + 1;
  body_task->grain_size = grain_size;
  work_t *cont_task = (work_t *)malloc(sizeof(work_t));
  memset(cont_task, 0, sizeof(work_t));
  cont_task->data = cont_data;
  cont_task->fp = cont;
  cont_task->cur_idx = w->cur_idx;
  // ensures continuation and all descendants have greater task ID than body
  cont_task->task_id = w->task_id + 2;
  set_cont(body_task, cont_task);
  if (w != NULL) {
    if (w->cont != NULL) {
      // inherit the current task's continuation
      set_cont(cont_task, w->cont);
    } else {
      // this task has no continuation, but it has been effectively
      // continued by this loop so we don't want to end the computation
      // when this task completes
      w->continued = true;
    }
  }
  set_full_task(body_task);

  int32_t my_id = weld_rt_thread_id();
  run_data *rd = get_run_data();
  pthread_spin_lock(rd->all_work_queue_locks + my_id);
  (rd->all_work_queues + my_id)->push_front(body_task);
  pthread_spin_unlock(rd->all_work_queue_locks + my_id);
}

static inline work_t *clone_task(work_t *task) {
  work_t *clone = (work_t *)malloc(sizeof(work_t));
  memcpy(clone, task, sizeof(work_t));
  clone->full_task = false;
  return clone;
}

// repeatedly break off the second half of the task into a new task
// until the task's size in iterations drops below a certain threshold
static inline void split_task(work_t *task, int32_t my_id, run_data *rd) {
  while (task->upper - task->lower > task->grain_size) {
    work_t *last_half = clone_task(task);
    int64_t mid = (task->lower + task->upper) / 2;

    // The inner loop may be subject to vectorization, so modify the bounds to make the task size
    // divisible by the SIMD vector size.
    if (task->grain_size > 2 * MAX_SIMD_SIZE) {
        mid = (mid / MAX_SIMD_SIZE) * MAX_SIMD_SIZE;
    }

    task->upper = mid;
    last_half->lower = mid;
    last_half->cur_idx = mid;
    // task must have non-NULL cont if it has non-zero number of iterations and therefore
    // is a loop body
    set_cont(last_half, task->cont);
    pthread_spin_lock(rd->all_work_queue_locks + my_id);
    (rd->all_work_queues + my_id)->push_front(last_half);
    pthread_spin_unlock(rd->all_work_queue_locks + my_id);
  }
}

// keeps executing items from the work queue until it is empty
static inline void work_loop(int32_t my_id, run_data *rd) {
  while (true) {
    pthread_spin_lock(rd->all_work_queue_locks + my_id);
    if ((rd->all_work_queues + my_id)->empty()) {
      pthread_spin_unlock(rd->all_work_queue_locks + my_id);
      return;
    } else {
      work_t *popped = (rd->all_work_queues + my_id)->front();
      (rd->all_work_queues + my_id)->pop_front();
      pthread_spin_unlock(rd->all_work_queue_locks + my_id);
      split_task(popped, my_id, rd);
      // Exit the thread if there's an error.
      // We don't need to worry about freeing here; the runtime will
      // free all allocated memory as long as it is allocated with
      // `weld_run_malloc` or `weld_run_realloc`.
      if (rd->err != 0) {
        weld_rt_abort_thread();
      }
      popped->fp(popped);
      finish_task(popped, my_id, rd);
    }
  }
}

static void *thread_func(void *data) {
  pthread_setspecific(global_id, data);
  thread_data *td = reinterpret_cast<thread_data *>(data);
  pthread_mutex_lock(&global_lock);
  run_data *rd = runs->find(td->run_id)->second;
  pthread_mutex_unlock(&global_lock);

  int iters = 0;
  try {
    // this work_loop call is needed to complete any work items that are initially on the queue
    work_loop(td->thread_id, rd);
    while (!rd->done) {
      if (try_steal(td->thread_id, rd)) {
        iters = 0;
        work_loop(td->thread_id, rd);
      } else {
        // If this thread is stalling, periodically check for errors.
        iters++;
        if (iters > 1000000) {
          if (rd->err != 0) {
            break;
          }
          iters = 0;
        }
      }
    }
  } catch (weld_abort_exception &) {
    // swallow aborts here
  }
  free(data);
  return NULL;
}


// *** weld_run functions and helpers ***

// kick off threads running thread_func
// block until the computation is complete
extern "C" int64_t weld_run_begin(void (*run)(work_t*), void *data, int64_t mem_limit, int32_t n_workers) {
  run_data *rd = new run_data;
  pthread_mutex_init(&rd->lock, NULL);
  rd->n_workers = n_workers;
  rd->workers = new pthread_t[n_workers];
  rd->all_work_queue_locks = new work_queue_lock[n_workers];
  rd->all_work_queues = new work_queue[n_workers];
  rd->done = false;
  rd->result = NULL;
  rd->mem_limit = mem_limit;
  rd->cur_mem = 0;
  rd->err = 0;

  work_t *run_task = (work_t *)malloc(sizeof(work_t));
  memset(run_task, 0, sizeof(work_t));
  run_task->data = data;
  run_task->fp = run;
  // this initial task can be thought of as a continuation
  set_full_task(run_task);
  rd->all_work_queues[0].push_front(run_task);

  int64_t my_run_id = __sync_fetch_and_add(&run_id, 1);
  pthread_mutex_lock(&global_lock);
  (*runs)[my_run_id] = rd;
  pthread_mutex_unlock(&global_lock);

  for (int32_t i = 0; i < rd->n_workers; i++) {
    pthread_spin_init(rd->all_work_queue_locks + i, 0);
  }

  for (int32_t i = 0; i < rd->n_workers; i++) {
    thread_data *td = (thread_data *)malloc(sizeof(thread_data));
    td->run_id = my_run_id;
    td->thread_id = i;
    if (rd->n_workers == 1) {
      thread_func(reinterpret_cast<void *>(td));
    } else {
      pthread_create(rd->workers + i, NULL, &thread_func, reinterpret_cast<void *>(td));
    }
  }

  if (rd->n_workers > 1) {
    for (int32_t i = 0; i < rd->n_workers; i++) {
      pthread_join(rd->workers[i], NULL);
    }
  }

  for (int32_t i = 0; i < n_workers; i++) {
    pthread_spin_destroy(rd->all_work_queue_locks + i);
  }
  delete [] rd->all_work_queue_locks;
  delete [] rd->all_work_queues;
  delete [] rd->workers;
  rd->done = true;
  return my_run_id;
}

extern "C" void *weld_run_malloc(int64_t run_id, size_t size) {
  run_data *rd = get_run_data_by_id(run_id);
  pthread_mutex_lock(&rd->lock);
  if (rd->cur_mem + size > rd->mem_limit) {
    pthread_mutex_unlock(&rd->lock);
    weld_run_set_errno(run_id, 7);
    weld_rt_abort_thread();
    return NULL;
  }
  rd->cur_mem += size;
  void *mem = malloc(size);
  rd->allocs[reinterpret_cast<intptr_t>(mem)] = size;
  pthread_mutex_unlock(&rd->lock);
  return mem;
}

extern "C" void *weld_run_realloc(int64_t run_id, void *data, size_t size) {
  run_data *rd = get_run_data_by_id(run_id);
  pthread_mutex_lock(&rd->lock);
  int64_t orig_size = rd->allocs.find(reinterpret_cast<intptr_t>(data))->second;
  if (rd->cur_mem - orig_size + size > rd->mem_limit) {
    pthread_mutex_unlock(&rd->lock);
    weld_run_set_errno(run_id, 7);
    weld_rt_abort_thread();
    return NULL;
  }
  rd->cur_mem -= orig_size;
  rd->allocs.erase(reinterpret_cast<intptr_t>(data));
  rd->cur_mem += size;
  void *mem = realloc(data, size);
  rd->allocs[reinterpret_cast<intptr_t>(mem)] = size;
  pthread_mutex_unlock(&rd->lock);
  return mem;
}

extern "C" void weld_run_free(int64_t run_id, void *data) {
  run_data *rd = get_run_data_by_id(run_id);
  pthread_mutex_lock(&rd->lock);
  rd->cur_mem -= rd->allocs.find(reinterpret_cast<intptr_t>(data))->second;
  rd->allocs.erase(reinterpret_cast<intptr_t>(data));
  free(data);
  pthread_mutex_unlock(&rd->lock);
}

extern "C" void *weld_run_get_result(int64_t run_id) {
  run_data *rd = get_run_data_by_id(run_id);
  return rd->err != 0 ? NULL : rd->result;
}

extern "C" int64_t weld_run_get_errno(int64_t run_id) {
  return get_run_data_by_id(run_id)->err;
}

extern "C" void weld_run_set_errno(int64_t run_id, int64_t err) {
  get_run_data_by_id(run_id)->err = err;
}

extern "C" int64_t weld_run_memory_usage(int64_t run_id) {
  return get_run_data_by_id(run_id)->cur_mem;
}

extern "C" void weld_run_dispose(int64_t run_id) {
  run_data *rd = get_run_data_by_id(run_id);
  assert(rd->done);
  for (map<intptr_t, int64_t>::iterator it = rd->allocs.begin(); it != rd->allocs.end(); it++) {
    free(reinterpret_cast<void *>(it->first));
  }
  pthread_mutex_destroy(&rd->lock);
  delete rd;
  pthread_mutex_lock(&global_lock);
  runs->erase(run_id);
  pthread_mutex_unlock(&global_lock);
}
