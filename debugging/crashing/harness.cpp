// clang++-3.8 -O0 -g -Iinclude -I../weld_rt/cpp harness.cpp oom.ll ../weld_rt/cpp/runtime.cpp ../weld_rt/cpp/vb.cpp ../weld_rt/cpp/merger.cpp -o run
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>

#include "common.h"
#include "weld.h"

#define BIG 1

#ifdef BIG
#define FILENAME "../programs/crashing2-big.weld"
#else
#define FILENAME "../programs/crashing2.weld"
#endif

#define SIZE 10000

struct s0 {
  bool _1;
  vec<i8> _2;
};

struct s1 {
  f64 _f1;
  i64 _i1;
#ifdef BIG
  f64 _f2;
  i64 _i2;
  f64 _f3;
  i64 _i3;
  f64 _f4;
  i64 _i4;
  f64 _f5;

  i64 _i5;
  f64 _f6;
  i64 _i6;
  f64 _f7;
  i64 _i7;
  f64 _f8;
  i64 _i8;
  f64 _f9;
  i64 _i9;
  f64 _f10;
  i64 _i10;
  f64 _f11;
  i64 _i11;
  f64 _f12;
  i64 _i12;
  f64 _f13;
  i64 _i13;
  f64 _f14;
  i64 _i14;
  f64 _f15;
  i64 _i15;
  f64 _f16;
  i64 _i16;
  f64 _f17;
  i64 _i17;
  f64 _f18;
  i64 _i18;
  f64 _f19;
  i64 _i19;
  f64 _f20;
  i64 _i20;
  f64 _f21;
  i64 _i21;
  f64 _f22;
  i64 _i22;
  f64 _f23;
  i64 _i23;
  f64 _f24;
  i64 _i24;
  f64 _f25;
  i64 _i25;
  f64 _f26;
  i64 _i26;
  f64 _f27;
  i64 _i27;
  f64 _f28;
  i64 _i28;
  f64 _f29;
  i64 _i29;
  f64 _f30;
  i64 _i30;
  f64 _f31;
  i64 _i31;
  f64 _f32;
  i64 _i32;
  f64 _f33;
  i64 _i33;
  f64 _f34;
  i64 _i34;
  f64 _f35;
  i64 _i35;
  f64 _f36;
  i64 _i36;
  f64 _f37;
  i64 _i37;
  f64 _f38;
  i64 _i38;
  f64 _f39;
  i64 _i39;
  f64 _f40;
  i64 _i40;
  f64 _f41;
  i64 _i41;
  f64 _f42;
  i64 _i42;
#endif
};

struct both {
  struct s0 _1;
  struct s1 _2;
};

struct arguments {
  int32_t partitionIndex;
  vec<both> kvs;
};

// Setup and call run() here.

int main(int argc, char **argv) {

  struct arguments a;
  memset(&a, 0, sizeof(a));
  a.kvs = make_vec<both>(SIZE);
  for (int i = 0; i < SIZE; i++) {
    a.kvs.ptr[i]._1._2 = make_vec<i8>(100);
  }

  // This is the Weld program.
  char *program;
  long bytes = read_all(FILENAME, &program);

  weld_error_t e = weld_error_new();
  weld_conf_t conf = weld_conf_new();

  weld_conf_set(conf, "weld.threads", "1");

  weld_module_t m = weld_module_compile(program, conf, e);
  weld_conf_free(conf);

  if (weld_error_code(e)) {
    const char *err = weld_error_message(e);
    printf("Error message: %s\n", err);
    exit(1);
  }

  struct timeval start, end, diff;
  gettimeofday(&start, 0);
  printf("running...\n");

  // Run the module and get the result.
  weld_value_t arg = weld_value_new(&a);

  conf = weld_conf_new();
  weld_value_t result = weld_module_run(m, conf, arg, e);
  void *result_data = weld_value_data(result);

  printf("%d\n", *((uint32_t *)result_data));
  if (weld_error_code(e)) {
    const char *err = weld_error_message(e);
    printf("Error message: %s\n", err);
    exit(1);
  }

  gettimeofday(&end, 0);
  timersub(&end, &start, &diff);

  printf("Weld: %ld.%06ld\n", (long)diff.tv_sec, (long)diff.tv_usec);
  return 0;
}
