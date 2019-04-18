#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include <string.h>

// Include the Weld API.
#include "../../../weld-capi/weld.h"

const char *program = "|x:i32, ys:vec[i64]|\
let a = result(for(ys, appender[i32], |b, i, y| merge(b, x)));\
let b = result(for(ys, appender[i64], |b, i, y| merge(b, y)));\
{a, b}";

template <typename T>
struct vec {
    T *data;
    int64_t length;
};

struct args {
    int32_t x;
    struct vec<int64_t> ys;
};

struct retval {
    struct vec<int32_t> a;
    struct vec<int64_t> b;
};

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

    {
        struct vec<int64_t> ys;
        ys.data = (int64_t *)malloc(sizeof(int64_t) * 4);
        ys.data[0] = ys.data[1] = ys.data[2] = ys.data[3] = 2;
        ys.length = 4;

        struct args input;
        input.x = 5;
        input.ys = ys;

        weld_value_t arg = weld_value_new(&input);

        weld_conf_t conf = weld_conf_new();
        weld_context_t context = weld_context_new(conf);

        weld_value_t result = weld_module_run(m, context, arg, e);
        if (weld_error_code(e)) {
            const char *err = weld_error_message(e);
            printf("Error message: %s\n", err);
            exit(1);
        }
        struct retval *result_data = (struct retval *)weld_value_data(result);

        printf("Answer lengths: %lld %lld\n", result_data->a.length, result_data->b.length);
        printf("Answer a values: %d %d %d %d\n", result_data->a.data[0],
                result_data->a.data[1],
                result_data->a.data[2],
                result_data->a.data[3]);

        printf("Answer b values: %lld %lld %lld %lld\n", result_data->b.data[0],
                result_data->b.data[1],
                result_data->b.data[2],
                result_data->b.data[3]);

        // Free the values.
        weld_value_free(result);
        weld_value_free(arg);
        weld_conf_free(conf);
    }

    weld_error_free(e);
    weld_module_free(m);
    printf("Freeing data and quiting!\n");


    return 0;
}
