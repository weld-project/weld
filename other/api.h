#ifndef _WELD_H_
#define _WELD_H_

/** 
 * The C API for Weld.
 */

/**
 * The top level type of a Weld value.
 */
typedef enum {
    I32 = 0,
    I64,
    F32,
    F64,
    VEC,
    DICT,
    STRUCT,
} weld_type_t;

/**
 * A Weld value represented as a raw data pointer.
 */
typedef struct {
    weld_type_t ty;     /** The type of the top level object. */
    void *data;         /** A pointer to the data for this object. */

    // Private Fields
    bool _owned;         /** If true, this value is allocated and owned by the Weld Runtime. */
} weld_value_t;

/**
 * A Weld configuration.
 *
 * The configuration is used by the Weld compiler and runtime to determine how
 * code should be generated and executed. The configuration controls both
 * compile-time parameters and runtime parameters (the latter which can be changed
 * across runs of the same compiled module).
 */
typedef struct {
    unsigned cores; /** The number of cores to run */
     /** 
     * The maximum amount of memory the module is allowed to allocate on one
     * run, in kilobytes. 
     */
    size_t mem_limit;
} weld_config_t; // TODO(shoumik) Make compile and runtime config different?

/**
 * A compiled Weld module.
 *
 * A Weld module represents a compiled runnable Weld program.
 */
typedef struct {
    weld_config_t conf; /** The compile time configuration for this module */
    void *binary;       /** A pointer to the binary for this module */
} weld_module_t;

/** Takes a Weld program and produces a module.
 *
 * The Weld module can be executed using the `weld_run_module` call.
 */
weld_module_t weld_compile_module(const char *program, weld_config_t *conf);

/** Runs a Weld module against a list of arguments and returns a result.
 *
 * This function does not perform any type checking; the caller is responsible
 * for ensuring that the passed argument is compatible with the data
 * expected by the Weld module. 
 *
 * The args argument is a contiguous block of memory. The value in the buffer
 * can be viewed as a struct where the field at index i corresponds to the ith
 * argument of the Weld function encapsulated by module.
 *
 * @param module the module to run.
 * @param args A formatted buffer of arguments for the module.
 *
 * @return a Weld value capturing the return value of the function. The return
 * value is *owned* by the Weld runtime and must be freed using `weld_free`.
 */
weld_value_t weld_run_module(weld_module_t *module, weld_value_t *args);

/* Frees a Weld value owned by the Weld runtime. If a value not owned by the
 * runtime is passed as an argument, this function does nothing.
 *
 * @param value the value to free
 */
void weld_free(weld_value_t *value);

#endif
