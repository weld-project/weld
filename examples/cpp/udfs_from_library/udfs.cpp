// This program calls a UDF from a shared library loaded using weld_load_library.

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include <string.h>

// Include the Weld API.
#include "../../../weld-capi/weld.h"

int main() {
    // Load our shared library; its name will be passed by make as the macro UDFLIB
    weld_error_t e = weld_error_new();
    weld_load_library(UDFLIB, e);
    if (weld_error_code(e)) {
        const char *err = weld_error_message(e);
        printf("Error loading library: %s\n", err);
        exit(1);
    }

    // Compile Weld module.
    weld_conf_t conf = weld_conf_new();
    weld_module_t m = weld_module_compile("|x:i64| cudf[add_five,i64](x)", conf, e);

    if (weld_error_code(e)) {
        const char *err = weld_error_message(e);
        printf("Error during compilation: %s\n", err);
        exit(1);
    }

    // Run the module and get the result.
    weld_context_t context = weld_context_new(conf);
    weld_conf_free(conf);

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


        weld_value_t result = weld_module_run(m, context, arg, e);
        void *result_data = weld_value_data(result);
        printf("Answer: %lld\n", *(int64_t *)result_data);

        // Free the values.
        weld_value_free(result);
        weld_value_free(arg);
    }

    weld_context_free(context);
    weld_error_free(e);
    weld_module_free(m);
    printf("Freeing data and quiting!\n");

    return 0;
}
