#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include <sys/time.h>
#include <string.h>

// Include the Weld API.
#include "../../../weld-capi/weld.h"

struct weld_vector {
    int32_t *data;
    int64_t length;
};

struct args {
    struct weld_vector vector;
};

const char *program = "|x:vec[i32]| result(for(x, merger[i64,+],\
                       |b,i,e| merge(b, i64(e))))";

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

    weld_vector v;
    const uint64_t length = 10000;
    int32_t *data = (int32_t *)malloc(sizeof(int32_t) * length);
    int32_t expect = 0;
    for (int i = 0; i < length; i++) {
        data[i] = 1;
        expect += data[i];
    }

    v.data = data;
    v.length = length;

    struct args a;
    a.vector = v;

    weld_value_t arg = weld_value_new(&a);

    // Run the module and get the result.
    conf = weld_conf_new();

    // Create a new context.
    weld_context_t context = weld_context_new(conf);

    for (int i = 0; i < 10; i++) {
      struct timeval timecheck;
      gettimeofday(&timecheck, NULL);
      long start = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

      weld_value_t result = weld_module_run(m, context, arg, e);

      gettimeofday(&timecheck, NULL);
      long end = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

      long ms = end - start;
      printf("Weld: %ld ms\n", ms);

      gettimeofday(&timecheck, NULL);
      start = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

      long baseline_result = 0;
      for (int i = 0; i < length; i++) {
        baseline_result += data[i];
      }

      gettimeofday(&timecheck, NULL);
      end = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;
      ms = end - start;
      printf("Baseline: %ld ms\n", ms);

      if (weld_error_code(e)) {
        const char *err = weld_error_message(e);
        printf("Error message: %s\n", err);
        exit(1);
      }
      void *result_data = weld_value_data(result);
      printf("Answer: %lld\n", *(int64_t *)result_data);
      printf("Expect: %d\n", expect);
      printf("Baseline: %lld\n", baseline_result);

      weld_context_free(context);

      // Even though we free the context, the WeldValue holds a reference to it,
      // so the memory is still live. We can reobtain a reference to the context here.
      context = weld_value_context(result);

      printf("Context Memory Usage Before Free: %lld\n",
          weld_context_memory_usage(context));

      // Free the result.
      weld_value_free(result);

      printf("Context Memory Usage Before Free: %lld\n",
          weld_context_memory_usage(context));
    }

    // Free the input data.
    free(data);

    // Free the values.
    weld_value_free(arg);
    weld_conf_free(conf);

    // Now free the context - this makes the reference count of the context 0, thus freeing it.
    weld_context_free(context);

    weld_error_free(e);
    weld_module_free(m);
    return 0;
}
