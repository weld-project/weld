#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include <string.h>

// Include the Weld API.
#include "../../../c/weld.h"

extern "C" void add_five(int64_t *x, int64_t *result) {
    *result = *x + 5;
    printf("called add_five: %lld + 5 = %lld\n", *x, *result);
}

int main() {
    // Compile Weld module.
    weld_error_t e = weld_error_new();
    weld_conf_t conf = weld_conf_new();
    weld_module_t m = weld_module_compile("|x:i64| cudf[add_five,i64](x)", conf, e);
    weld_conf_free(conf);

    if (weld_error_code(e)) {
        const char *err = weld_error_message(e);
        printf("Error message: %s\n", err);
        exit(1);
    }

    while(true) {
        char buf[4096];
        char *c;
        printf(">>> ");
        fgets(buf, sizeof(buf), stdin);

        if (strncmp(buf, "quit", sizeof(buf)) == 0) {
            break;
        }

        unsigned long x = strtoul(buf, &c, 10);
        if (c == buf) {
            printf("nope, try again.\n");
            continue;
        }

        int64_t input = (int64_t)x;
        weld_value_t arg = weld_value_new(&input);

        // Run the module and get the result.
        weld_conf_t conf = weld_conf_new();
        weld_value_t result = weld_module_run(m, conf, arg, e);
        void *result_data = weld_value_data(result);
        printf("Answer: %lld\n", *(int64_t *)result_data);

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
