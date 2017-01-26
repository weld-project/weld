#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include <string.h>

// Include the Weld API.
#include "../../../weld.h"

int main() {
    // Compile Weld module.
    weld_error_t e = NULL;
    weld_module_t m = weld_module_compile("|x:i64| x+5L", "configuration", &e);

    if (weld_error_success(e)) {
        const char *err = weld_error_message(e);
        printf("Error message: %s\n", err);
    }

    // Create a Weld Object for the argument.
    
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
        weld_object_t arg = weld_object_new(&input);

        // Run the module and get the result.
        weld_object_t result = weld_module_run(m, arg, &e);
        void *result_data = weld_object_data(result);
        printf("Answer: %lld\n", *(int64_t *)result_data);

        // Free the objects.
        weld_object_free(result);
        weld_object_free(arg);
    }

    weld_error_free(e);
    weld_module_free(m);
    printf("Freeing data and quiting!\n");

    return 0;
}
