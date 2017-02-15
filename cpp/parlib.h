#ifndef _PARLIB_H_
#define _PARLIB_H_

// Memory allocation functions for Weld.
extern "C" void *weld_rt_malloc(int64_t run_id, size_t size);
extern "C" void *weld_rt_realloc(int64_t run_id, void *data, size_t size);
extern "C" void weld_rt_free(int64_t run_id, void *data);

// work item
struct work_t {
  // parameters for the task function
  void *data;
  // [lower, upper) gives the range of iteration indices for this task
  // it is [0, 0) if the task is not a loop body
  int64_t lower;
  int64_t upper;
  // set in the user program -- the current iteration index this task is on,
  // 0 if not a loop body task
  int64_t cur_idx;
  // true if task was directly stolen from another queue, is the head (earliest in serial order)
  // task of a loop instance, or is a continuation.
  // If true, we need to set the nest_* fields so that we can create a new piece for
  // this task and its siblings in any associative builders.
  // Tasks that are not full tasks (non-stolen, non-head tasks of loop instances) do not need
  // separate nest_* fields because they will be executed in serial order (and by the same thread)
  // after their associated full tasks and can share these full tasks' pieces (and nest_* fields)
  // in any associative builders.
  int32_t full_task; // boolean
  // The list of loop indices of all containing loops for this task.
  // The whole program is assumed to be run in an outer loop with a single iteration.
  // This list is constructed by walking up the list of continuations of this task
  // and taking their cur_idx's. Indices for more outer loops will be earlier in the list.
  int64_t *nest_idxs;
  // For each entry of nest_idx, the task_id corresponding to the task from which it was obtained.
  // An algorithm for determining whether task A follows task B in the program's serial order
  // is as follows: compare A.nest_idxs to B.nest_idxs, starting from the low indices. If
  // two elements at the same index are different, the task with the lower element comes earlier
  // in the serial order. If the two elements are the same, compare the two elements at the same
  // position in nest_task_ids. If these two elements are different, the task with the lower
  // element comes earlier in the serial order, by the definition of task_id below. If the
  // two nest_task_ids elements are equal, proceed to the next nest_idxs index. (If the
  // nest_len's of the tasks are different, compare the arrays only up to the length of the
  // shorter one.) No two distinct tasks should be determined to be equal by this algorithm.
  int64_t *nest_task_ids;
  // length of nest_idxs and nest_task_ids
  int32_t nest_len;
  // If task B must execute after task A, and tasks A and B have identical nest_idxs, B.task_id
  // is guaranteed to be larger than A.task_id.
  int64_t task_id;
  // task function
  void (*fp)(work_t*);
  // the continuation, NULL if no continuation
  work_t *cont;
  // if this task is a continuation, the number of remaining dependencies (0 otherwise)
  int32_t deps;
  // if this task has no continuation of its own, indicates whether program execution
  // was continued by the start of another loop at the end of this task
  int32_t continued; // boolean
  // largest task size (in # of iterations) that we should not split further (defaults to 0
  // for non-loop body tasks)
  int32_t grain_size;
};

typedef struct work_t work_t;

struct vec_piece {
  void *data;
  int64_t size;
  int64_t capacity;
  int64_t *nest_idxs;
  int64_t *nest_task_ids;
  int32_t nest_len;
};

typedef struct {
  void *data;
  int64_t size;
} vec_output;

extern "C" {
  int32_t my_id_public();
  void set_result(void *res);
  void *get_result();
  int32_t get_nworkers();
  void set_nworkers(int32_t n);
  int64_t get_runid();
  void set_runid(int64_t rid);
  void pl_start_loop(work_t *w, void *body_data, void *cont_data, void (*body)(work_t*),
    void (*cont)(work_t*), int64_t lower, int64_t upper, int32_t grain_size);
  void execute(void (*run)(work_t*), void* data);

  void *new_vb(int64_t elem_size, int64_t starting_cap);
  void new_piece(void *v, work_t *w);
  vec_piece *cur_piece(void *v, int32_t my_id);
  
}

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

#endif
