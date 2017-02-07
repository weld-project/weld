#ifndef _PARLIB_H_
#define _PARLIB_H_

// work item
struct work_t {
  void *data;
  int64_t lower;
  int64_t upper;
  int64_t cur_idx;
  int32_t stolen; // boolean
  int64_t *nest_pos;
  int32_t nest_pos_len;
  // to disambiguate the relative order of tasks with identical loop bounds
  int64_t task_id;
  // work function
  void (*fp)(work_t*);
  // cont is dependent on the completion of this work item (and possibly others as well)
  work_t *cont;
  // number of remaining parent dependencies
  int32_t deps;
  int32_t continued; // boolean
};

typedef struct work_t work_t;

extern "C" {
  int32_t my_id_public();
  void set_result(void *res);
  void *get_result();
  int32_t get_nworkers();
  void set_nworkers(int32_t n);
  int64_t get_runid();
  void set_runid(int64_t rid);
  void pl_start_loop(work_t *w, void *body_data, void *cont_data, void (*body)(work_t*),
    void (*cont)(work_t*), int64_t lower, int64_t upper);
  void execute(void (*run)(work_t*), void* data);
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
