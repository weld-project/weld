#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <assert.h>
#include "runtime.h"

#include <vector>
#include <algorithm>

/*
The parallel VecBuilder backend. A new piece is created for each full
task. When result is called, the pieces are sorted according to the serial
task ordering described in parlib.h. The data arrays of the sorted pieces are
concatenated to produce the correct output vector.
*/

using namespace std;

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

extern "C" void *weld_rt_new_vb(int64_t elem_size, int64_t starting_cap, int32_t fixed_size) {
  vec_builder *vb = new vec_builder();
  // TODO rename merger to be more generic and cover this use case
  vb->thread_curs = weld_rt_new_merger(sizeof(vec_piece), weld_rt_get_nworkers());
  pthread_mutex_init(&vb->lock, NULL);
  vb->elem_size = elem_size;
  vb->starting_cap = starting_cap;
  vb->fixed_size = fixed_size != 0;
  if (vb->fixed_size) {
    vb->fixed_vector = weld_run_malloc(weld_rt_get_run_id(), vb->elem_size * vb->starting_cap);
  }
  return vb;
}

extern "C" void weld_rt_new_vb_piece(void *v, work_t *w, int32_t is_init_piece) {
  vec_builder *vb = (vec_builder *)v;
  vec_piece *cur_piece = (vec_piece *)weld_rt_get_merger_at_index(vb->thread_curs, sizeof(vec_piece),
    weld_rt_thread_id());
  if (!vb->fixed_size) {
    int64_t *nest_idxs = (int64_t *)malloc(sizeof(int64_t) * w->nest_len);
    memcpy(nest_idxs, w->nest_idxs, sizeof(int64_t) * w->nest_len);
    int64_t *nest_task_ids = (int64_t *)malloc(sizeof(int64_t) * w->nest_len);
    memcpy(nest_task_ids, w->nest_task_ids, sizeof(int64_t) * w->nest_len);
    if (cur_piece->data != NULL) {
      pthread_mutex_lock(&vb->lock);
      vb->pieces.push_back(*cur_piece);
      pthread_mutex_unlock(&vb->lock);
    }
    cur_piece->nest_idxs = nest_idxs;
    cur_piece->nest_len = w->nest_len;
    cur_piece->nest_task_ids = nest_task_ids;
    // we need weld_run_malloc here because this data is realloc'ed by the user program and
    // can become large
    cur_piece->data = weld_run_malloc(weld_rt_get_run_id(), vb->elem_size * vb->starting_cap);
    cur_piece->size = 0;
    cur_piece->capacity = vb->starting_cap;
  } else {
    // If this is the initial piece (i.e. the one created right after the vb is created),
    // we want to use an offset of 0. Only for pieces created for new tasks should an
    // offset of w->cur_idx be used.
    cur_piece->data = (uint8_t *)vb->fixed_vector +
      (is_init_piece ? 0 : vb->elem_size * w->cur_idx);
    cur_piece->size = 0;
    cur_piece->capacity = vb->starting_cap; // larger than the real capacity for this task,
    // but it doesn't matter because the real capacity won't be reached in the fixed case
  }
}

extern "C" vec_output weld_rt_result_vb(void *v) {
  vec_builder *vb = (vec_builder *)v;
  vec_output vo;
  if (!vb->fixed_size) {
    for (int32_t i = 0; i < weld_rt_get_nworkers(); i++) {
      vec_piece *cur_piece = (vec_piece *)weld_rt_get_merger_at_index(vb->thread_curs, sizeof(vec_piece), i);
      if (cur_piece->data != NULL) {
        vb->pieces.push_back(*cur_piece);
      }
    }
    std::sort(vb->pieces.begin(), vb->pieces.end(), vp_compare);
    int64_t output_size = 0;
    for (int64_t i = 0; i < vb->pieces.size(); i++) {
      output_size += vb->pieces[i].size;
    }

    // also needs weld_run_malloc because it's the final result and not freed by the runtime
    uint8_t *output = (uint8_t *)weld_run_malloc(weld_rt_get_run_id(), vb->elem_size * output_size);
    int64_t cur_start = 0;
    // TODO probably want to parallelize this
    for (int64_t i = 0; i < vb->pieces.size(); i++) {
      memcpy(output + cur_start, vb->pieces[i].data, vb->elem_size * vb->pieces[i].size);
      cur_start += vb->elem_size * vb->pieces[i].size;
      free(vb->pieces[i].nest_idxs);
      free(vb->pieces[i].nest_task_ids);
      weld_run_free(weld_rt_get_run_id(), vb->pieces[i].data);
    }
    vo = (vec_output) {output, output_size};
  } else {
    vo = (vec_output) {vb->fixed_vector, vb->starting_cap};
  }
  pthread_mutex_destroy(&vb->lock);
  weld_rt_free_merger(vb->thread_curs);
  delete vb;
  return vo;
}
