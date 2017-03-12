#ifndef __COMMON_H__
#define __COMMON_H__

#include <immintrin.h>

#include "memory.h"

// This is how the vec type is represented in LLVM

namespace weld {

// Typedefs matching LLVM types.
typedef bool       i1;
typedef int8_t     i8;
typedef int16_t    i16;
typedef int32_t    i32;
typedef int64_t    i64;

template<typename T>
struct vec {
  T *ptr;
  i64 size;
};

template<typename T>
vec<T> make_vec(i64 size) {
  vec<T> t;
  t.ptr = (T *)malloc(size * sizeof(T));
  t.size = size;

  return t;
}

// Some common fixed vector types. Other fixed vector types should be used
// carefully because it's likely unpredictable how they'll be expressed in memory
typedef __m128i     i1x4;
typedef __m128i     i1x8;
typedef __m128i     i32x4;

// Some convinience macros for converting vectors into arrays

inline uint16_t *i1x8_to_ptr(i1x8 *v) {
  return (uint16_t *)v;
}

inline uint32_t *i1x4_to_ptr(i1x4 *v) {
  return (uint32_t *)v;
}

inline uint32_t *i32x4_to_ptr(i32x4 *v) {
  return (uint32_t *)v;
}

}

#endif /**  __COMMON_H__ **/
