#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <assert.h>
#include "parlib.h"

#include <vector>
#include <algorithm>

/*
The parallel VecBuilder backend. A new piece is created for each full
task. When result is called, the pieces are sorted according to the serial
task ordering described in parlib.h. The data arrays of the sorted pieces are
concatenated to produce the correct output vector.
*/

using namespace std;

struct vec_builder {
  vector<vec_piece> pieces;
  vec_piece *thread_curs;
  int64_t elem_size;
  int64_t starting_cap;
  pthread_mutex_t lock;
};

static inline void print_piece(vec_piece vp) {
  for (int i = 0; i < vp.nest_len; i++) {
    printf("(%lld, %lld) ", vp.nest_idxs[i], vp.nest_task_ids[i]);
  }
}

static inline void print_pieces(vec_builder *vb) {
  for (int i = 0; i < vb->pieces.size(); i++) {
    print_piece(vb->pieces[i]);
    printf("\n");
  }
}

bool vp_compare(vec_piece one, vec_piece two) {
  for (int32_t i = 0; i < min(one.nest_len, two.nest_len); i++) {
    if (one.nest_idxs[i] != two.nest_idxs[i]) {
      return one.nest_idxs[i] < two.nest_idxs[i];
    }
    // the nest_idxs are equal, but task_ids can break ties
    if (one.nest_task_ids[i] != two.nest_task_ids[i]) {
      return one.nest_task_ids[i] < two.nest_task_ids[i];
    }
  }
  // Distinct tasks should never have equal nest_*'s by this
  // comparison algorithm, but sometimes we can reach here because either
  // one == two or there were multiple aliases for a single VecBuilder, so
  // multiple pieces were created for the same task. (Creating multiple
  // pieces for the same task is ok, since all but one of them will be empty,
  // so their relative ordering doesn't matter.)
  return false;
}

extern "C" void *new_vb(int64_t elem_size, int64_t starting_cap) {
  vec_builder *vb = new vec_builder();
  vb->thread_curs = (vec_piece *)malloc(sizeof(vec_piece) * get_nworkers());
  memset(vb->thread_curs, 0, sizeof(vec_piece) * get_nworkers());
  pthread_mutex_init(&vb->lock, NULL);
  vb->elem_size = elem_size;
  vb->starting_cap = starting_cap;
  return vb;
}

extern "C" void new_piece(void *v, work_t *w) {
  vec_builder *vb = (vec_builder *)v;
  int64_t *nest_idxs = (int64_t *)malloc(sizeof(int64_t) * w->nest_len);
  memcpy(nest_idxs, w->nest_idxs, sizeof(int64_t) * w->nest_len);
  int64_t *nest_task_ids = (int64_t *)malloc(sizeof(int64_t) * w->nest_len);
  memcpy(nest_task_ids, w->nest_task_ids, sizeof(int64_t) * w->nest_len);
  int32_t my_id = my_id_public();
  if (vb->thread_curs[my_id].data != NULL) {
    pthread_mutex_lock(&vb->lock);
    vb->pieces.push_back(vb->thread_curs[my_id]);
    pthread_mutex_unlock(&vb->lock);
  }
  vb->thread_curs[my_id].nest_idxs = nest_idxs;
  vb->thread_curs[my_id].nest_len = w->nest_len;
  vb->thread_curs[my_id].nest_task_ids = nest_task_ids;
  // we need weld_rt_malloc here because this data is realloc'ed by the user program and
  // can become large
  vb->thread_curs[my_id].data = weld_rt_malloc(get_runid(), vb->elem_size * vb->starting_cap);
  vb->thread_curs[my_id].size = 0;
  vb->thread_curs[my_id].capacity = vb->starting_cap;
}

// a non-full task executes in correct serial order (and by the same thread)
// after its associated full task (and potentially other non-full tasks also
// assicated with the same full task), so it can simply write into its full
// task's piece (the cur_piece) without creating a new one
extern "C" vec_piece *cur_piece(void *v, int32_t my_id) {
  vec_builder *vb = (vec_builder *)v;
  return &vb->thread_curs[my_id];
}

extern "C" vec_output result_vb(void *v) {
  vec_builder *vb = (vec_builder *)v;
  for (int32_t i = 0; i < get_nworkers(); i++) {
    if (vb->thread_curs[i].data != NULL) {
      vb->pieces.push_back(vb->thread_curs[i]);
    }
  }
  std::sort(vb->pieces.begin(), vb->pieces.end(), vp_compare);
  int64_t output_size = 0;
  for (int64_t i = 0; i < vb->pieces.size(); i++) {
    output_size += vb->pieces[i].size;
  }

  // also needs weld_rt_malloc because it's the final result and not freed by the runtime
  uint8_t *output = (uint8_t *)weld_rt_malloc(get_runid(), vb->elem_size * output_size);
  int64_t cur_start = 0;
  for (int64_t i = 0; i < vb->pieces.size(); i++) {
    memcpy(output + cur_start, vb->pieces[i].data, vb->elem_size * vb->pieces[i].size);
    cur_start += vb->elem_size * vb->pieces[i].size;
    free(vb->pieces[i].nest_idxs);
    free(vb->pieces[i].nest_task_ids);
    weld_rt_free(get_runid(), vb->pieces[i].data);
  }
  pthread_mutex_destroy(&vb->lock);
  free(vb->thread_curs);
  delete vb;
  vec_output vo = (vec_output) {output, output_size};
  return vo;
}
