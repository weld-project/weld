
void example() {
    const char *prog = "|a,b| a+b";
    const char *conf = "";

    weld_module_t m = weld_module_compile(prog, conf);

    int *args = (int *)malloc(sizeof(int) * 2);
    args[0] = 2;
    args[1] = 2;

    weld_object_t obj = weld_object_new((void *)args);
    weld_object_t ret = weld_module_run(m, obj);

    int *result = (int *)weld_object_data(ret);
    printf("%d\n", *result);

    // Clean up.
    weld_object_free(obj);
    weld_object_free(ret);
}
