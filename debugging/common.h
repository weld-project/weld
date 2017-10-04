
#include <stdint.h>

typedef float f32;
typedef double f64;

typedef int8_t i8;
typedef int32_t i32;
typedef int64_t i64;

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

/** Reads the entire file filename into memory. */
long read_all(const char *filename, char **buf) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "%s: ", filename);
        perror("read_all");
        exit(1);
    }

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);  // same as rewind(f);

    char *string = (char *)malloc(fsize + 1);
    fread(string, fsize, 1, f);
    fclose(f);

    string[fsize] = 0;

    *buf = string;
    return fsize + 1;
}
