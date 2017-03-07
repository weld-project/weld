#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>
#include <queue>
#include <deque>
#include <algorithm>
#include "assert.h"
#include "parlib.h"

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

// allows each thread to retrieve its index
pthread_key_t id;

typedef deque<work_t *> work_queue;
typedef pthread_spinlock_t work_queue_lock;

int32_t W = 1; // number of workers, default 1
int64_t run_id = 1; // the current run ID.
pthread_t *workers = NULL;
work_queue *all_work_queues = NULL; // queue per worker
work_queue_lock *all_work_queue_locks = NULL; // lock per queue

volatile bool done = false; // if a computation is currently running, have we finished it?
volatile void *result = NULL; // stores the final result of the computation to be passed back
// to the caller

// TODO remove this, just for testing
pthread_mutex_t global_lock;

// gives the current thread's index
static inline int32_t my_id() {
  return (int32_t)reinterpret_cast<intptr_t>(pthread_getspecific(id));
}

extern "C" int32_t my_id_public() {
  return (int32_t)reinterpret_cast<intptr_t>(pthread_getspecific(id));
}

extern "C" void take_global_lock() {
  pthread_mutex_lock(&global_lock);
}

extern "C" void release_global_lock() {
  pthread_mutex_unlock(&global_lock);
}

// retrieve the result of the computation
extern "C" void *get_result() {
  if (weld_rt_get_errno(get_runid()) != 0) {
    return NULL;
  }
  return (void *)result;
}

// set the result of the computation, called from generated LLVM
extern "C" void set_result(void *res) {
  result = res;
}

extern "C" int32_t get_nworkers() {
  return W;
}

extern "C" void set_nworkers(int32_t n) {
  W = n;
}

extern "C" int64_t get_runid() {
  return run_id;
}

extern "C" void set_runid(int64_t id) {
  run_id = id;
}

extern "C" void weld_abort_thread() {
    pthread_exit(NULL);
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
static inline bool try_steal() {
  int32_t victim = rand() % W;
  if (!pthread_spin_trylock(all_work_queue_locks + victim)) {
    work_queue *its_work_queue = all_work_queues + victim;
    if (its_work_queue->empty()) {
      pthread_spin_unlock(all_work_queue_locks + victim);
      return false;
    } else {
      work_t *popped = its_work_queue->back();
      its_work_queue->pop_back();
      pthread_spin_unlock(all_work_queue_locks + victim);
      set_full_task(popped);
      pthread_spin_lock((all_work_queue_locks + my_id()));
      (all_work_queues + my_id())->push_front(popped);
      pthread_spin_unlock((all_work_queue_locks + my_id()));
      return true;
    }
  } else {
    return false;
  }
}

// called once task function returns
// decrease the dependency count of the continuation, run the continuation
// if necessary, or signal the end of the computation if we are done
static inline void finish_task(work_t *task) {
  if (task->cont == NULL) {
    if (!task->continued) {
      // if this task has no continuation and there was no for loop to end it,
      // the computation is over
      done = true;
    }
    free(task->data);
  } else {
    int32_t previous = __sync_fetch_and_sub(&task->cont->deps, 1);
    if (previous == 1) {
      // run the continuation since we are the last dependency
      set_full_task(task->cont);
      pthread_spin_lock((all_work_queue_locks + my_id()));
      (all_work_queues + my_id())->push_front(task->cont);
      pthread_spin_unlock((all_work_queue_locks + my_id()));
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
extern "C" void pl_start_loop(work_t *w, void *body_data, void *cont_data, void (*body)(work_t*),
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

  pthread_spin_lock((all_work_queue_locks + my_id()));
  (all_work_queues + my_id())->push_front(body_task);
  pthread_spin_unlock((all_work_queue_locks + my_id()));
}

static inline work_t *clone_task(work_t *task) {
  work_t *clone = (work_t *)malloc(sizeof(work_t));
  memcpy(clone, task, sizeof(work_t));
  clone->full_task = false;
  return clone;
}

// repeatedly break off the second half of the task into a new task
// until the task's size in iterations drops below a certain threshold
static inline void split_task(work_t *task) {
  while (task->upper - task->lower > task->grain_size) {
    work_t *last_half = clone_task(task);
    int64_t mid = (task->lower + task->upper) / 2;
    task->upper = mid;
    last_half->lower = mid;
    last_half->cur_idx = mid;
    // task must have non-NULL cont if it has non-zero number of iterations and therefore
    // is a loop body
    set_cont(last_half, task->cont);
    pthread_spin_lock((all_work_queue_locks + my_id()));
    (all_work_queues + my_id())->push_front(last_half);
    pthread_spin_unlock((all_work_queue_locks + my_id()));
  }
}

// keeps executing items from the work queue until it is empty
static inline void work_loop() {
  while (true) {
    pthread_spin_lock((all_work_queue_locks + my_id()));
    if ((all_work_queues + my_id())->empty()) {
      pthread_spin_unlock((all_work_queue_locks + my_id()));
      return;
    } else {
      work_t *popped = (all_work_queues + my_id())->front();
      (all_work_queues + my_id())->pop_front();
      pthread_spin_unlock((all_work_queue_locks + my_id()));
      split_task(popped);
      // Exit the thread if there's an error.
      // We don't need to worry about freeing here; the runtime will
      // free all allocated memory as long as it is allocated with
      // `weld_rt_malloc` or `weld_rt_realloc`.
      if (weld_rt_get_errno(get_runid()) != 0) {
        weld_abort_thread();
      }
      popped->fp(popped);
      finish_task(popped);
    }
  }
}

static void *thread_func(void *data) {
  intptr_t tid = reinterpret_cast<intptr_t>(data);
  pthread_setspecific(id, (void *)tid);

#ifndef __APPLE__
  // pin thread to CPU
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(tid, &set);
  if (sched_setaffinity(0, sizeof(set), &set) == -1) {
    printf("unable to set affinity for thread %d\n", tid);
  }
#endif

  int iters = 0;

  // this work_loop call is needed to complete any work items that are initially on the queue
  work_loop();
  while (!done) {
    if (try_steal()) {
      iters = 0;
      work_loop();
    } else {
      // If this thread is stalling, periodically check for errors.
      iters++;
      if (iters > 1000000) {
        if (weld_rt_get_errno(get_runid()) != 0) {
          return NULL;
        }
        iters = 0;
      }
    }
  }
  return NULL;
}

// cleanup old computation state and reset to initial values so that
// another computation can be run
static inline void cleanup_computation_state() {
  pthread_key_delete(id);
  pthread_mutex_destroy(&global_lock);

  for (int32_t i = 0; i < W; i++) {
    pthread_spin_destroy(all_work_queue_locks + i);
    all_work_queues[i].clear();
  }

  delete [] all_work_queue_locks;
  delete [] all_work_queues;
  delete [] workers;

  done = false;
}

// start up parallel runtime and kick off threads running thread_func
// block until the computation is complete
// once execute() returns all state has been reset and it is safe to run
// another computation
extern "C" void execute(void (*run)(work_t*), void *data) {
  workers = new pthread_t[W];
  all_work_queue_locks = new work_queue_lock[W];
  all_work_queues = new work_queue[W];

  work_t *run_task = (work_t *)malloc(sizeof(work_t));
  memset(run_task, 0, sizeof(work_t));
  run_task->data = data;
  run_task->fp = run;
  // this initial task can be thought of as a continuation
  set_full_task(run_task);
  all_work_queues[0].push_front(run_task);

  for (int32_t i = 0; i < W; i++) {
    pthread_spin_init(all_work_queue_locks + i, 0);
  }

  pthread_key_create(&id, NULL);
  pthread_mutex_init(&global_lock, NULL);

  for (int32_t i = 0; i < W; i++) {
    pthread_create(workers + i, NULL, &thread_func, reinterpret_cast<void *>(i));
  }

  for (int32_t i = 0; i < W; i++) {
    pthread_join(workers[i], NULL);
  }

  cleanup_computation_state();
}
