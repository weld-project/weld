// THIS IS A GENERATED C++ FILE.

#include <stdint.h> // For explicitly sized integer types.
#include <stdlib.h> // For malloc

// Defines Weld's primitive numeric types.
typedef bool        i1;
typedef uint8_t     u8;
typedef int8_t      i8;
typedef uint16_t   u16;
typedef int16_t    i16;
typedef uint32_t   u32;
typedef int32_t    i32;
typedef uint64_t   u64;
typedef int64_t    i64;
typedef float      f32;
typedef double     f64;

// Defines the basic Vector struture using C++ templates.
template<typename T>
struct vec {
  T *ptr;
  i64 size;
};

template<typename T>
vec<T> new_vec(i64 size) {
  vec<T> t;
  t.ptr = (T *)malloc(size * sizeof(T));
  t.size = size;
  return t;
}
