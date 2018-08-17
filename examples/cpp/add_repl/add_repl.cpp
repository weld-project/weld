#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include <string.h>

// Include the Weld API.
#include "../../../c/weld.h"

int main() {
    // Compile Weld module.
    weld_error_t e = weld_error_new();
    weld_conf_t conf = weld_conf_new();

    // Print out the SIR while we execute. This is useful for debugging a crashing
    // program.
    weld_conf_set(conf, "weld.compile.traceExecution", "true");

    // Set a small 1KB memory memory limit for the runtime. We don't need more than
    // that for this simple program!
    weld_conf_set(conf, "weld.memory.limit", "1024");

    weld_module_t m = weld_module_compile("|x:i64| x+5L", conf, e);

    weld_context_t context = weld_context_new(conf);
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
        weld_value_t result = weld_module_run(m, context, arg, e);
        void *result_data = weld_value_data(result);
        printf("Answer: %lldB\n", *(int64_t *)result_data);

        // Show the memory usage.
        printf("memory usage: %lld\n",
            weld_context_memory_usage(context));

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
