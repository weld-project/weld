#ifndef _VEC_H_
#define _VEC_H_

#include "strt.h"

#include <assert.h>

//r A Weld vector, which is always defined as a pointer and capacity pair.

template<typename T>
struct Vec {

public:
  T *data;
  int64_t capacity;

public:
  // Extend the vector to hold `num_elements` elements.
  void extend(WeldRunHandleRef run, int64_t num_elements) {
    if (capacity >= num_elements) {
      return;
    } 
    int64_t new_size = num_elements * sizeof(T);
    data = (uint8_t *)weld_runst_realloc(run, data, new_size);
    assert(data);
    capacity = num_elements;
  }
};

#endif
