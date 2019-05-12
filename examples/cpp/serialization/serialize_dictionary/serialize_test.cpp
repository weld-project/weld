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
    weld_vec<pair> vector;
};

typedef void (*SerializeCheckerFn)(void *);

// Programs that deserialize.
const char *program_no_pointers = "|v:vec[{i32,i32}]| serialize(result(for(v, dictmerger[i32,i32,+], |b,i,e| merge(b, e))))";
const char *program_pointers = "|v:vec[{i32,i32}]| serialize(result(for(v, groupmerger[i32,i32], |b,i,e| merge(b, e))))";

// Programs that deserialize outputs of the above programs.
const char *deser_no_pointers = "|v:vec[u8]| tovec(deserialize[dict[i32,i32]](v))";
const char *deser_pointers = "|v:vec[u8]| tovec(deserialize[dict[i32,vec[i32]]](v))";

/** Runs a serialize test on a program that takes a vec[{i32,i32}] has input. */
weld_value_t serialize_test(const char *program, SerializeCheckerFn checker) {
    weld_error_t e = weld_error_new();
    weld_conf_t conf = weld_conf_new();

    weld_module_t m = weld_module_compile(program, conf, e);

    weld_conf_free(conf);

    if (weld_error_code(e)) {
        const char *err = weld_error_message(e);
        fprintf(stderr, "Error message: %s\n", err);
        exit(1);
    }

    weld_vec<pair> v;
    const int64_t length = 100;
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

    //  XXX Leaks We should return/free this context!
    weld_context_t context = weld_context_new(conf);
    weld_value_t result = weld_module_run(m, context, arg, e);
    if (weld_error_code(e)) {
        const char *err = weld_error_message(e);
        fprintf(stderr, "Error message: %s\n", err);
        exit(1);
    }

    void *result_data = weld_value_data(result);
    checker(result_data);

    free(data);

    // Free the values.
    weld_value_free(arg);
    weld_conf_free(conf);

    weld_error_free(e);
    weld_module_free(m);

    return result;
}

/** Runs a serialize test on a program that takes a vec[u8] has input. */
void deserialize_test(const char *program,
    weld_vec<uint8_t> *a,
    SerializeCheckerFn checker) {

    weld_error_t e = weld_error_new();
    weld_conf_t conf = weld_conf_new();

    weld_module_t m = weld_module_compile(program, conf, e);

    weld_conf_free(conf);

    if (weld_error_code(e)) {
        const char *err = weld_error_message(e);
        fprintf(stderr, "Error message: %s\n", err);
        exit(1);
    }

    weld_value_t arg = weld_value_new(a);

    // Run the module and get the result.
    conf = weld_conf_new();
    weld_context_t context = weld_context_new(conf);
    weld_value_t result = weld_module_run(m, context, arg, e);
    if (weld_error_code(e)) {
        const char *err = weld_error_message(e);
        fprintf(stderr, "Error message: %s\n", err);
        exit(1);
    }

    void *result_data = weld_value_data(result);
    checker(result_data);

    // Free the values.
    weld_value_free(result);
    weld_value_free(arg);
    weld_conf_free(conf);
    weld_context_free(context);

    weld_error_free(e);
    weld_module_free(m);
}

// Serialized as Length:i64, Key:i32, Value:i32, Key:i32, Value:i32, ...
void check_nopointers(void *inp) {
  weld_vec<uint8_t> *res_vec = (weld_vec<uint8_t> *)inp;
  int64_t length = res_vec->length;
  uint8_t *data = res_vec->data;

  // Print the number of bytes in the serialized buffer.
  printf("Output data buffer length %lld\n", length);

  int64_t size = *((int64_t *)data);
  data += sizeof(int64_t);

  printf("KV pairs: %lld\n", size);
  for (int i = 0; i < size; i++) {
    pair value = *((pair *)data);
    printf("(%d %d)\n", value.a, value.b);
    data += sizeof(pair);
  }
}

uint8_t *parse_veci32(uint8_t *data) {
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
void check_pointers(void *inp) {
  weld_vec<uint8_t> *res_vec = (weld_vec<uint8_t> *)inp;
  int64_t length = res_vec->length;
  uint8_t *data = res_vec->data;

  int64_t size = *((int64_t *)data);
  data += sizeof(int64_t);

  printf("KV pairs: %lld\n", size);
  for (int i = 0; i < size; i++) {
    int32_t *key_ptr = (int32_t *)data;
    printf("Key(%d) -> ", *key_ptr);
    data += sizeof(int32_t);
    data = parse_veci32(data);
  }
}

void check_deserialize_nopointers(void *inp) {
  weld_vec<pair> *res_vec = (weld_vec<pair> *)inp;
  int64_t length = res_vec->length;
  pair *data = res_vec->data;

  printf("Length %lld\n", length);
  for (int i = 0; i < length; i++) {
    printf("(%d %d)\n", data[i].a, data[i].b);
  }
}

void check_deserialize_pointers(void *inp) {
  weld_vec<vpair> *res_vec = (weld_vec<vpair> *)inp;
  int64_t length = res_vec->length;
  vpair *data = res_vec->data;

  printf("Length %lld\n", length);
  for (int i = 0; i < length; i++) {
    printf("%d -> [", data[i].a);
    weld_vec<int32_t> v = data[i].b;
    for (int j = 0; j < v.length; j++) {
      printf("%d ", v.data[j]);
    }
    printf("]\n"); 
  }
}

int main() {
  /*
  printf("Serializing Dictionary of type 'dict[i32,i32]':\n");
  weld_value_t no_ptrs = serialize_test(program_no_pointers, check_nopointers);
  printf("Deserializing Dictionary of type 'dict[i32,i32]':\n");
  deserialize_test(deser_no_pointers, (weld_vec<uint8_t> *)weld_value_data(no_ptrs), check_deserialize_nopointers);
  weld_value_free(no_ptrs);
  */

  // set to INFO level
  weld_set_log_level(3);

  weld_value_t ptrs = serialize_test(program_pointers, check_pointers);
  deserialize_test(deser_pointers,
      (weld_vec<uint8_t> *)weld_value_data(ptrs),
      check_deserialize_pointers);
  // weld_value_free(ptrs);
  printf("Success!\n");
  return 0;
}
