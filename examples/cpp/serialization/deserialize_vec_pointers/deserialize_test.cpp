#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include <string.h>

#include <assert.h>

// Include the Weld API.
#include "../../../../weld-capi/weld.h"


template <class T>
struct weld_vec {
    T *data;
    int64_t length;
};

struct pair {
  int32_t a;
  int32_t b;
};

struct vpair {
  int32_t a;
  weld_vec<int32_t> b;
};

struct args {
    weld_vec<vpair> vector;
};

// A program that serializes a vector with pointers, and then immediately deserializes it.
const char *program = "|v:vec[{i32,vec[i32]}]| deserialize[vec[{i32,vec[i32]}]](serialize(v))";

int main() {
    // Compile Weld module.
    weld_error_t e = weld_error_new();
    weld_conf_t conf = weld_conf_new();
    weld_module_t m = weld_module_compile(program, conf, e);
    weld_context_t context = weld_context_new(conf);

    weld_conf_free(conf);

    if (weld_error_code(e)) {
        const char *err = weld_error_message(e);
        printf("Error message: %s\n", err);
        exit(1);
    }

    weld_vec<vpair> v;
    const uint64_t length = 25;
    vpair *data = (vpair *)malloc(sizeof(vpair) * length);
    for (int i = 0; i < length; i++) {
      data[i].a = i % 10;

      data[i].b.data = (int32_t *)malloc(sizeof(int32_t) * length);
      data[i].b.length = length;
      for (int j = 0; j < length; j++) {
        data[i].b.data[j] = j;
      }
    }

    v.data = data;
    v.length = length;

    struct args a;
    a.vector = v;

    weld_value_t arg = weld_value_new(&a);

    printf("Running...\n");
    fflush(stdout);

    // Run the module and get the result.
    weld_value_t result = weld_module_run(m, context, arg, e);
    if (weld_error_code(e)) {
        const char *err = weld_error_message(e);
        printf("Error message: %s\n", err);
        exit(1);
    }

    weld_vec<vpair> *res_vec = (weld_vec<vpair> *)weld_value_data(result);

    printf("Output length %lld\n", res_vec->length);
    for (int i = 0; i < length; i++) {
      printf("(%d %d)\n", res_vec->data[i].a, res_vec->data[i].b.length);
      for (int j = 0; j < res_vec->data[i].b.length; j++) {
        printf("%d ", res_vec->data[i].b.data[j]);
      }
      printf("\n");
    }

    free(data);


    // Free the values.
    weld_value_free(result);
    weld_value_free(arg);
    weld_context_free(context);

    weld_error_free(e);
    weld_module_free(m);
    return 0;
}
