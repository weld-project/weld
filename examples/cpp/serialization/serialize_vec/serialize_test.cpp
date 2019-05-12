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

struct triple {
  int a;
  int b;
  int c;
};

struct args {
    struct weld_vec<weld_vec<triple> > vector;
};

const char *program = "|x:vec[vec[{i32,i32,i32}]]| serialize(x)";

// Parses a single serialized vec<i32>
char *parse_buffer(char *data) {
  int64_t length = *((int64_t *)data);

  data += sizeof(int64_t);
  for (int i = 0; i < length; i++) {
    triple value = *((triple *)data);
    assert(value.a == i);
    assert(value.b == i);
    assert(value.b == i);
    printf("(%d %d %d) ", value.a, value.b, value.c);
    data += sizeof(triple);
  }
  printf("\n");
  return data;
}

int main() {
    // Compile Weld module.
    weld_error_t e = weld_error_new();
    weld_conf_t conf = weld_conf_new();

    weld_module_t m = weld_module_compile(program, conf, e);
    weld_conf_free(conf);

    if (weld_error_code(e)) {
        const char *err = weld_error_message(e);
        printf("Error message: %s\n", err);
        exit(1);
    }

    weld_vec<weld_vec<triple> > v;
    const uint64_t length = 8;

    weld_vec<triple> *data = (weld_vec<triple> *)malloc(sizeof(weld_vec<triple>) * length);
    for (int i = 0; i < length; i++) {
      data[i].length = length;
      data[i].data = (triple *)malloc(sizeof(triple) * length);
      for (int j = 0; j < length; j++) {
        data[i].data[j].a = j;
        data[i].data[j].b = j;
        data[i].data[j].c = j;
      }
    }

    v.data = data;
    v.length = length;

    struct args a;
    a.vector = v;

    weld_value_t arg = weld_value_new(&a);

    // Run the module and get the result.
    conf = weld_conf_new();
    weld_context_t context = weld_context_new(conf);
    weld_value_t result = weld_module_run(m, context, arg, e);
    if (weld_error_code(e)) {
        const char *err = weld_error_message(e);
        printf("Error message: %s\n", err);
        exit(1);
    }

    weld_vec<uint8_t> *res_vec = (weld_vec<uint8_t> *)weld_value_data(result);
    uint8_t *result_data = res_vec->data;

    printf("Output data buffer length %ld\n", res_vec->length);

    int64_t *length_ptr = (int64_t *)result_data;
    int64_t serialized_length = *length_ptr;

    printf("%lld\n", serialized_length);

    assert(serialized_length == length);
    length_ptr++;

    char *serialized_data = (char *)length_ptr;
    for (int i = 0; i < serialized_length; i++) {
      serialized_data = parse_buffer(serialized_data);
    }

    free(data);

    // Free the values.
    weld_value_free(result);
    weld_value_free(arg);
    weld_conf_free(conf);
    weld_context_free(context);

    weld_error_free(e);
    weld_module_free(m);
    return 0;
}
