#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include <unistd.h>
#include <string.h>

// Include the Weld API.
#include "../../../weld.h"

int main() {
    const size_t LEN = 1000000000;

    for (int i = 0; i < 1000; i++) {
        sleep(1);
        int *x = (int *)weld_rt_malloc(0, sizeof(int) * LEN);

        x[0] = 1;
        x[1] = 2;
        x[2] = 3;
        x[3] = 4;
        x[4] = 5;

        // This should prevent a huge memory blow up (comment it out and run top to see memory blowup)
        //weld_rt_free((void *)x);
    }

    return 0;
}
