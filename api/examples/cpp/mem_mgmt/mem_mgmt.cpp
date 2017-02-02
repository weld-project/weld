#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include <unistd.h>
#include <string.h>

// Include the Weld API.
#include "../../../weld.h"

struct vec {
    int32_t *data;
    int64_t length;
};

extern "C" void weld_rt_free(void *data);
extern "C" void *weld_rt_malloc(int64_t _, size_t size);

int main() {
    const size_t LEN = 10000000;

    printf("sleeping before input allocation so memory can be observed..\n");
    sleep(10);

    struct vec inp;
    inp.data = (int32_t *)malloc(sizeof(int32_t) * LEN);
    inp.length = LEN;

    printf("sleeping before module compilation so memory can be observed..\n");
    sleep(10);

    // Compile Weld module.
    weld_error_t e = NULL;
    weld_module_t m = weld_module_compile("|x:vec[i32]| map(x, |e| e+1)", "configuration", &e);

    if (weld_error_code(e)) {
        printf("Weld compile returned error: %s\n", weld_error_message(e));
        exit(1);
    }

    weld_value_t arg = weld_value_new(&inp);

    printf("starting run loop...\n");
    sleep(1);

    for (int i = 0; i < 5010; i++) {
        weld_error_t e = NULL;
        // This allocates some data.
        weld_value_t res = weld_module_run(m, arg, &e);

        // This frees the value wrapper object.
        weld_value_free(res);
        weld_error_free(e);

        // This frees it. Without this line, we should see a large memory blowup.
        weld_run_free(0);

        if (i > 5000) {
            sleep(1);
        }
    }

    // Clean up other resources.
    weld_error_free(e);
    weld_value_free(arg);
    weld_module_free(m);

    printf("sleeping before quit so memory can be observed..\n");
    sleep(10);

    return 0;
}
