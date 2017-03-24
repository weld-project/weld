/**
 *
 * Weld Runtime C Interface.
 * Link the Weld library with -lweld.
 *
 */
#ifndef _WELD_H_
#define _WELD_H_

// Types

/** A type encapsulating a Weld value. */
typedef void* weld_value_t;

/** A runnable Weld module. */
typedef void* weld_module_t;

/** A handle to a Weld error. */
typedef void* weld_error_t;

/** A hanlde to a Weld configuration. */
typedef void* weld_conf_t;

// ************* Values ****************

/** Returns a Weld-readable value with the given input buffer.
 *
 * A Weld value created using this method is owned by the caller.
 * The caller must ensure that the data buffer remains a valid pointer
 * to memory until after the value is used by the runtime. The value
 * must be freed with `weld_value_free`.
 *
 * @param data the data this struct captures.
 * @return a new Weld value.
 */
extern "C" weld_value_t 
weld_value_new(void *data);

/** Returns 1 if the value's data is owned by the Weld runtime, or
 * 0 otherwise.
 *
 * A value owned by the Weld runtime is freed using the `weld_value_free`
 * call. Non-owned values must have their *data buffers* (retrieved using
 * `weld_value_data`) freed by the caller; this Weld value must still
 * be garbage collected using `weld_value_free` however.
 *
 * @param obj the value to check
 * @return 1 if owned, 0 otherwise.
 */
extern "C" int 
weld_value_run(weld_value_t obj);

/** Returns this value's data buffer.
 *
 * @param obj the value whose data buffer should be retrieved.
 * @return a void * data buffer. The caller is responsible for knowing
 * the type of the buffer and casting it appropriately.
 */
extern "C" void* 
weld_value_data(weld_value_t obj);

/* Frees a Weld value.
 *
 * Each Weld value must be freed using this call. Owned values also
 * free their data buffers; non-owned values require the caller to free
 * the buffer explicitly. This Weld value is invalid after this call.
 *
 * @param obj the value to free.
 */
extern "C" void 
weld_value_free(weld_value_t);

// ************* Modules ****************

/** Compiles a Weld module.
 *
 * Takes a string and configuration and returns a runnable module.
 *
 * @param code a Weld program to compile
 * @param conf a configuration for the module.
 * @param err a Weld erro for this compilation.
 * @return a runnable module.
 */
extern "C" weld_module_t 
weld_module_compile(const char *code, weld_conf_t, weld_error_t);

/** Runs a module using the given argument.
 *
 * Multi-argument Weld functions take a Weld value encapsulating
 * a single struct as an argument. The field at index i in the struct
 * represents the ith argument of the Weld function.
 *
 * @param module the module to run.
 * @param conf a configuration for this run.
 * @param arg the argument for the module's function.
 * @param err a Weld error for this run.
 * @return an owned Weld value representing the return value. The caller
 * is responsible for knowing what the type of the return value is based on
 * the module she runs.
 */
extern "C" weld_value_t 
weld_module_run(weld_module_t, weld_conf_t, weld_value_t, weld_error_t);

/** Garbage collects a module.
 *
 * @param module the module to garbage collect.
 */
extern "C" void 
weld_module_free(weld_module_t);

// ************* Errors ****************

/** Return a new Weld error.
 *
 */
extern "C" weld_error_t
weld_error_new();

/** Returns an error code, or 0 if there was no error.
 *
 * @param err the error to check
 * @param 0 if the error was a success, or a nonzero error code otherwise.
 */
extern "C" int
weld_error_code(weld_error_t);

/** Returns an error message for a given error.
 *
 * @param err the error
 * @return a string error message.
 */
extern "C" const char *
weld_error_message(weld_error_t);

/** Free a Weld error.
 *
 * @param err the error
 */
extern "C" void 
weld_error_free(weld_error_t);

// ************* Configuration ****************

/** Return a new Weld configuraiton.
 *
 */
extern "C" weld_conf_t
weld_conf_new();

/** Returns a value for a Weld configuration key.
 *
 * @param key the key to look up.
 * @return the string value for the key, or NULL if the key does not exist.
 */
extern "C" const char*
weld_conf_get(weld_conf_t, const char *key);

/** Set a value for a Weld configuration key.
 *
 * @param key the key
 * @param key the value
 */
extern "C" void
weld_conf_set(weld_conf_t, const char *key, const char *value);

/** Free a Weld configuration.
 *
 */
extern "C" weld_conf_t
weld_conf_free(weld_conf_t);

#endif

