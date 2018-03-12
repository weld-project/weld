#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include <string.h>

#include <assert.h>

// Include the Weld API.
#include "../../../c/weld.h"


template <class T>
struct weld_vec {
    T *data;
    int64_t length;
};

struct pair {
  int32_t a;
  int32_t b;
};

struct args {
    weld_vec<pair> vector;
};

const char *program_no_pointers = "|v:vec[{i32,i32}]| serialize(result(for(v, dictmerger[i32,i32,+], |b,i,e| merge(b, e))))";

// Serialized as Length:i64, Key:i32, Value:i32, Key:i32, Value:i32, ...
void parse_nopointers(int8_t *data) {
  int64_t length = *((int64_t *)data);
  data += sizeof(int64_t);

  printf("KV pairs: %lld\n", length);
  for (int i = 0; i < length; i++) {
    pair value = *((pair *)data);
    printf("(%d->%d) ", value.a, value.b);
    data += sizeof(pair);
  }
  printf("\n");
}

int test_dictionary_nopointers() {
    // Compile Weld module.
    weld_error_t e = weld_error_new();
    weld_conf_t conf = weld_conf_new();

    //weld_conf_set(conf, "weld.compile.dumpCode", "true");

    weld_module_t m = weld_module_compile(program_no_pointers, conf, e);
    weld_conf_free(conf);

    if (weld_error_code(e)) {
        const char *err = weld_error_message(e);
        printf("Error message: %s\n", err);
        exit(1);
    }

    weld_vec<pair> v;
    const uint64_t length = 100;
    pair *data = (pair *)malloc(sizeof(pair) * length);
    for (int i = 0; i < length; i++) {
        data[i].a = i;
        data[i].b = i;
    }

    v.data = data;
    v.length = length;

    struct args a;
    a.vector = v;

    weld_value_t arg = weld_value_new(&a);

    // Run the module and get the result.
    conf = weld_conf_new();
    weld_value_t result = weld_module_run(m, conf, arg, e);
    if (weld_error_code(e)) {
        const char *err = weld_error_message(e);
        printf("Error message: %s\n", err);
        exit(1);
    }

    weld_vec<int8_t> *res_vec = (weld_vec<int8_t> *)weld_value_data(result);
    int8_t *result_data = res_vec->data;

    printf("Output data buffer length %lld\n", res_vec->length);
    parse_nopointers(result_data);

    free(data);

    // Free the values.
    weld_value_free(result);
    weld_value_free(arg);
    weld_conf_free(conf);

    weld_error_free(e);
    weld_module_free(m);
    return 0;
}

const char *program_pointers = "|v:vec[{i32,i32}]| serialize(result(for(v, groupmerger[i32,i32], |b,i,e| merge(b, e))))";

int8_t *parse_veci32(int8_t *data) {
  int64_t length = *((int64_t *)data);
  data += sizeof(int64_t);

  printf("Length: %lld [", length);

  int32_t *elements = (int32_t *)data;
  for (int i = 0; i < length; i++) {
    printf("%d ", elements[i]);
  }
  printf("]\n");
  return data + length * sizeof(int32_t);
}

// Serialized as Length:i64, Key:i32, ValueLength:i64, Val1:i32, ..., Key:i32, ValLength:i64, Val1: i32, ...
void parse_pointers(int8_t *data) {
  int64_t length = *((int64_t *)data);
  data += sizeof(int64_t);
  printf("KV pairs: %lld\n", length);
  for (int i = 0; i < length; i++) {
    int32_t *key_ptr = (int32_t *)data;
    printf("Key(%d) -> ", *key_ptr);
    data += sizeof(int32_t);
    data = parse_veci32(data);
  }
}

int test_dictionary_pointers() {
    // Compile Weld module.
    weld_error_t e = weld_error_new();
    weld_conf_t conf = weld_conf_new();

    weld_conf_set(conf, "weld.compile.dumpCode", "true");

    weld_module_t m = weld_module_compile(program_pointers, conf, e);
    weld_conf_free(conf);

    if (weld_error_code(e)) {
        const char *err = weld_error_message(e);
        printf("Error message: %s\n", err);
        exit(1);
    }

    weld_vec<pair> v;
    const uint64_t length = 100;
    pair *data = (pair *)malloc(sizeof(pair) * length);
    for (int i = 0; i < length; i++) {
        data[i].a = i % 10;
        data[i].b = i;
    }

    v.data = data;
    v.length = length;

    struct args a;
    a.vector = v;

    weld_value_t arg = weld_value_new(&a);

    // Run the module and get the result.
    conf = weld_conf_new();
    weld_value_t result = weld_module_run(m, conf, arg, e);
    if (weld_error_code(e)) {
        const char *err = weld_error_message(e);
        printf("Error message: %s\n", err);
        exit(1);
    }

    weld_vec<int8_t> *res_vec = (weld_vec<int8_t> *)weld_value_data(result);
    int8_t *result_data = res_vec->data;

    printf("Output data buffer length %lld\n", res_vec->length);
    parse_pointers(result_data);

    free(data);

    // Free the values.
    weld_value_free(result);
    weld_value_free(arg);
    weld_conf_free(conf);

    weld_error_free(e);
    weld_module_free(m);
    return 0;
}

int main() {
  test_dictionary_nopointers();
  test_dictionary_pointers();
}
