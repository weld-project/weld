#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>
#include <queue>
#include <deque>
#include <algorithm>
#include "parlib.h"

// TODO convert all longs to int64_t and ints to int32_t
using namespace std;

#define CACHE_LINE 64

#ifndef W
#define W 1
#endif

pthread_t workers[W];

pthread_key_t id;

typedef deque<work_t *> work_queue __attribute__((aligned(CACHE_LINE)));

typedef pthread_spinlock_t work_queue_lock __attribute__((aligned(CACHE_LINE)));

work_queue all_work_queues[W] __attribute__((aligned(CACHE_LINE)));
work_queue_lock all_work_queue_locks[W] __attribute__((aligned(CACHE_LINE)));

volatile bool started = false;
volatile bool done = false;
volatile void *result = NULL;

pthread_mutex_t global_lock;

// TODO optimize use of my_id throughout this file
static inline int my_id() {
  return (int)reinterpret_cast<intptr_t>(pthread_getspecific(id));
}

extern "C" int my_id_public() {
  return (int)reinterpret_cast<intptr_t>(pthread_getspecific(id));
}

extern "C" void take_global_lock() {
  pthread_mutex_lock(&global_lock);
}

extern "C" void release_global_lock() {
  pthread_mutex_unlock(&global_lock);
}

extern "C" void *get_result() {
  return (void *)result;
}

extern "C" void set_result(void *res) {
  result = res;
}

// call when own work queue empty
static inline int try_steal() {
  // TODO never attempt steal from self
  int victim = rand() % W;
  if (!pthread_spin_trylock(all_work_queue_locks + victim)) {
    work_queue *its_work_queue = all_work_queues + victim;
    if (its_work_queue->empty()) {
      pthread_spin_unlock(all_work_queue_locks + victim);
      return 0;
    } else {
      work_t *popped = its_work_queue->back();
      its_work_queue->pop_back();
      pthread_spin_unlock(all_work_queue_locks + victim);
      pthread_spin_lock((all_work_queue_locks + my_id()));
      (all_work_queues + my_id())->push_front(popped);
      pthread_spin_unlock((all_work_queue_locks + my_id()));
      return 1;
    }
  } else {
    return 0;
  }
}

static inline void finish_frame(work_t *frame) {
  if (frame->cont == NULL) {
    done = true;
    free(frame->data);
  } else {
    /*
    long old, updated;
    do {
      old = frame->cont->task_id;
      updated = std::max(old, frame->task_id + 1);
    } while (!__sync_bool_compare_and_swap(&frame->cont->task_id, old, updated));
    */

    int previous = __sync_fetch_and_sub(&frame->cont->deps, 1);
    if (previous == 1) {
      pthread_spin_lock((all_work_queue_locks + my_id()));
      (all_work_queues + my_id())->push_front(frame->cont);
      pthread_spin_unlock((all_work_queue_locks + my_id()));
      // we are the last sibling with this data, so we can free it
      free(frame->data);
    }
  }
  free(frame);
}

static inline void set_cont(work_t *w, work_t *cont) {
  w->cont = cont;
  __sync_fetch_and_add(&cont->deps, 1);
}

static inline void execute();

extern "C" void pl_start_loop(work_t *w, void *body_data, void *cont_data, void (*body)(work_t*),
  void (*cont)(work_t*), int64_t lower, int64_t upper) {
  work_t *body_task = (work_t *)calloc(sizeof(work_t), 1);
  body_task->data = body_data;
  body_task->fp = body;
  body_task->lower = lower;
  body_task->upper = upper;
  work_t *cont_task = (work_t *)calloc(sizeof(work_t), 1);
  cont_task->data = cont_data; 
  cont_task->fp = cont;
  set_cont(body_task, cont_task);
  if (w != NULL && w->cont != NULL) {
    set_cont(cont_task, w->cont);
  }
  if (started) {
    pthread_spin_lock((all_work_queue_locks + my_id()));
    (all_work_queues + my_id())->push_front(body_task);
    pthread_spin_unlock((all_work_queue_locks + my_id()));
  } else {
    all_work_queues[0].push_front(body_task);
    execute();
  }
}

// call with my_id() queue lock held
static inline void split_task(work_t *task) {
  // TODO make this a constant
  while (task->upper - task->lower >= 1024) {
    work_t *last_half = (work_t *)malloc(sizeof(work_t));
    memcpy(last_half, task, sizeof(work_t));
    int64_t mid = (task->lower + task->upper) / 2;
    task->upper = mid;
    last_half->lower = mid;
    // task must have cont if it has non-zero number of iterations and therefore is loop body
    set_cont(last_half, task->cont);
    (all_work_queues + my_id())->push_front(last_half);
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
      split_task(popped);
      pthread_spin_unlock((all_work_queue_locks + my_id()));
      popped->fp(popped);
      finish_frame(popped);
    }
  }
}

static void *thread_func(void *data) {
  intptr_t tid = reinterpret_cast<intptr_t>(data);
  pthread_setspecific(id, (void *)tid);

#ifndef __APPLE__
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(tid, &set);
  if (sched_setaffinity(0, sizeof(set), &set) == -1) {
    printf("unable to set affinitiy for thread %d\n", tid);
  }
#endif

  // this work_loop call is needed to complete any work items that are initially on the queue
  work_loop();
  while (!done) {
    if (try_steal()) {
      work_loop();
    }
  }
  return NULL;
}

static inline void execute() {
  started = true;

  for (int i = 0; i < W; i++) {
    pthread_spin_init(all_work_queue_locks + i, 0);
  }

  pthread_key_create(&id, NULL);
  pthread_mutex_init(&global_lock, NULL);

  for (int i = 0; i < W; i++) {
    pthread_create(workers + i, NULL, &thread_func, reinterpret_cast<void *>(i));
  }

  for (int i = 0; i < W; i++) {
    pthread_join(workers[i], NULL);
  }

  pthread_key_delete(id);
  pthread_mutex_destroy(&global_lock);

  for (int i = 0; i < W; i++) {
    pthread_spin_destroy(all_work_queue_locks + i);
    all_work_queues[i].clear();
  }

  started = false;
  done = false;
}
