#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>
#include "parlib.h"

#include <vector>
#include <algorithm>

// Memory allocation functions for Weld.
extern "C" void *weld_rt_malloc(int64_t run_id, size_t size);
extern "C" void *weld_rt_realloc(int64_t run_id, void *data, size_t size);
extern "C" void weld_rt_free(int64_t run_id, void *data);

using namespace std;

struct vec_piece {
  int64_t *key;
  int32_t key_length;
  int64_t task_id;
  void *data;
  int64_t size;
  int64_t capacity;
};

struct vec_builder {
  vector<vec_piece> pieces;
  vec_piece *thread_curs;
  int64_t elem_size;
  int64_t starting_cap;
  pthread_mutex_t lock;
};

typedef struct {
  void *data;
  int64_t size;
} vec_output;

bool vp_compare(vec_piece one, vec_piece two) {
  if (one.task_id < two.task_id) {
    return true;
  } else if (one.task_id > two.task_id) {
    return false;
  }
  for (int i = 0; i < min(one.key_length, two.key_length); i++) {
    if (one.key[i] < two.key[i]) {
      return true;
    } else if (one.key[i] > two.key[i]) {
      return false;
    }
  }
  // case where you have a loop body that does a merge and then more merges inside an inner loop
  if (one.key_length < two.key_length) {
    return true;
  } else if (one.key_length > two.key_length) {
    return false;
  }
  return false;
}

extern "C" void *new_vb(int64_t elem_size, int64_t starting_cap) {
  // TODO set default allocator to be weld_rt_malloc
  vec_builder *vb = new vec_builder();
  vb->thread_curs = (vec_piece *)weld_rt_malloc(get_runid(), sizeof(vec_piece) * get_nworkers());
  memset(vb->thread_curs, 0, sizeof(vec_piece) * get_nworkers());
  pthread_mutex_init(&vb->lock, NULL);
  vb->elem_size = elem_size;
  vb->starting_cap = starting_cap;
  return vb;
}

extern "C" void new_piece(void *v, work_t *w) {
  vec_builder *vb = (vec_builder *)v;
  int64_t *key = (int64_t *)weld_rt_malloc(get_runid(), sizeof(int64_t) * w->nest_pos_len);
  memcpy(key, w->nest_pos, sizeof(int64_t) * w->nest_pos_len);
  int32_t my_id = my_id_public();
  if (vb->thread_curs[my_id].data != NULL) {
    pthread_mutex_lock(&vb->lock);
    vb->pieces.push_back(vb->thread_curs[my_id]);
    pthread_mutex_unlock(&vb->lock);
  }
  vb->thread_curs[my_id].key = key;
  vb->thread_curs[my_id].key_length = w->nest_pos_len;
  vb->thread_curs[my_id].task_id = w->task_id;
  vb->thread_curs[my_id].data = weld_rt_malloc(get_runid(), vb->elem_size * vb->starting_cap);
  vb->thread_curs[my_id].size = 0;
  vb->thread_curs[my_id].capacity = vb->starting_cap;
}

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

  uint8_t *output = (uint8_t *)weld_rt_malloc(get_runid(), vb->elem_size * output_size);
  int64_t cur_start = 0;
  for (int64_t i = 0; i < vb->pieces.size(); i++) {
    memcpy(output + cur_start, vb->pieces[i].data, vb->elem_size * vb->pieces[i].size);
    cur_start += vb->elem_size * vb->pieces[i].size;
    weld_rt_free(get_runid(), vb->pieces[i].key);
    weld_rt_free(get_runid(), vb->pieces[i].data);
  }
  pthread_mutex_destroy(&vb->lock);
  weld_rt_free(get_runid(), vb->thread_curs);
  delete vb;
  vec_output vo = (vec_output) {output, output_size};
  return vo;
}
