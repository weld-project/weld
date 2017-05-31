#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include <string.h>

// Include the Weld API.
#include "../../../c/weld.h"

struct weld_vector {
    int32_t *data;
    int64_t length;
};

int main() {
    // Compile Weld module.
    weld_error_t e = weld_error_new();
    weld_conf_t conf = weld_conf_new();
    weld_module_t m = weld_module_compile("|x:vec[i32]| result(for(x, merger[i32,+], |b,i,e| merge(b,e)))", conf, e);
    weld_conf_free(conf);

    if (weld_error_code(e)) {
        const char *err = weld_error_message(e);
        printf("Error message: %s\n", err);
        exit(1);
    }

    weld_vector v;
    const uint64_t length = 1000;
    int32_t *data = (int32_t *)malloc(sizeof(int32_t) * length);
    for (int i = 0; i < length; i++) {
        data[i] = 1;
    }

    v.data = data;
    v.length = length;

    weld_value_t arg = weld_value_new(&v);

    // Run the module and get the result.
    conf = weld_conf_new();
    weld_value_t result = weld_module_run(m, conf, arg, e);
    void *result_data = weld_value_data(result);
    printf("Answer: %d\n", *(int32_t *)result_data);

    free(data);

    // Free the values.
    weld_value_free(result);
    weld_value_free(arg);
    weld_conf_free(conf);

    weld_error_free(e);
    weld_module_free(m);
    return 0;
}
