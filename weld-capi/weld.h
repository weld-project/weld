#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

/*
 * An opauqe handle to a Weld configuration.
 */
typedef WeldConf *WeldConfRef;

/*
 * An opauqe handle to a Weld context.
 */
typedef WeldContext *WeldContextRef;

/*
 * An opauqe handle to a Weld error.
 */
typedef WeldError *WeldErrorRef;

/*
 * An opauqe handle to a Weld module.
 */
typedef WeldModule *WeldModuleRef;

/*
 * An opauqe handle to a Weld data value.
 */
typedef WeldValue *WeldValueRef;

/*
 * Free a configuration.
 *
 * The passsed configuration should have been allocated using `weld_conf_new`.
 */
void weld_conf_free(WeldConfRef ptr);

/*
 * Get a value associated with a key for a configuration.
 *
 * This function is a wrapper for `WeldConf::get`.
 */
const char *weld_conf_get(WeldConfRef ptr, const char *key);

/*
 * Creates a new configuration.
 *
 * This function is a wrapper for `WeldConf::new`.
 */
WeldConfRef weld_conf_new(void);

/*
 * Set the value of `key` to `value`.
 *
 * This function is a wrapper for `WeldConf::set`.
 */
void weld_conf_set(WeldConfRef ptr, const char *key, const char *value);

/*
 * Frees a context.
 *
 * All contexts created by the FFI should be freed. This includes contexts obtained from
 * `weld_context_new` and `weld_value_context`.
 */
void weld_context_free(WeldContextRef ptr);

/*
 * Gets the memory allocated by a Weld context.
 *
 * This includes all live memory allocated in the given context.
 */
int64_t weld_context_memory_usage(WeldContextRef context);

/*
 * Creates a new context.
 *
 * This function is a wrapper for `WeldContext::new`.
 */
WeldContextRef weld_context_new(WeldConfRef conf);

/*
 * Returns the error code for a Weld error.
 *
 * This function is a wrapper for `WeldError::code`.
 */
WeldRuntimeErrno weld_error_code(WeldErrorRef err);

/*
 * Frees a Weld error object.
 */
void weld_error_free(WeldErrorRef err);

/*
 * Returns a Weld error message.
 *
 * This function is a wrapper for `WeldError::message`.
 */
const char *weld_error_message(WeldErrorRef err);

/*
 * Creates a new Weld error object.
 */
WeldErrorRef weld_error_new(void);

/*
 * Load a dynamic library that a Weld program can access.
 *
 * The dynamic library is a C dynamic library identified by its filename.
 * This function is a wrapper for `load_linked_library`.
 */
void weld_load_library(const char *filename, WeldErrorRef err);

/*
 * Compiles a Weld program into a runnable module.
 *
 * The compilation can be configured using the passed configuration pointer. This function stores
 * an error (which indicates a compilation error or success) in `err`.
 *
 * This function is a wrapper for `WeldModule::compile`.
 */
WeldModuleRef weld_module_compile(const char *code, WeldConfRef conf, WeldErrorRef err);

/*
 * Frees a module.
 *
 * Freeing a module does not free the memory it may have allocated. Values returned by the module
 * must be freed explicitly using `weld_value_free`.
 */
void weld_module_free(WeldModuleRef ptr);

/*
 * Runs a compiled Weld module.
 *
 * The module is run with a given configuration and argument list, and returns the result wrapped
 * as a `WeldValue`. If the run raised a runtime error, the method writes the erorr into `err`,
 * and a the method returns `null`. Otherwise, `err` indicates success.
 *
 * This function is a wrapper for `WeldModule::run`.
 */
WeldValueRef weld_module_run(WeldModuleRef module,
                             WeldContextRef context,
                             WeldValueRef arg,
                             WeldErrorRef err);

/*
 * Enables logging to stderr in Weld with the given log level.
 *
 * This function is ignored if it has already been called once, or if some other code in the
 * process has initialized logging using Rust's `log` crate.
 */
void weld_set_log_level(WeldLogLevel level);

/*
 * Returns the context of a value.
 *
 * Since contexts are internally reference-counted, this function increases the reference count of
 * the context. The context must be freed with `weld_context_free` to decrement the internal
 * reference count. Since Weld values owned by a context internally hold a reference to the
 * context, the value is guaranteed to be live until `weld_value_free` is called.
 */
WeldContextRef weld_value_context(WeldValueRef value);

/*
 * Returns the data pointer for a Weld value.
 *
 * This function is a wrapper for `WeldValue::data`. If this value is owned by the runtime, the
 * returned pointer should never be freed -- instead, use `weld_value_free` to free the data.
 */
const void *weld_value_data(WeldValueRef value);

/*
 * Frees a Weld value.
 *
 * All Weld values must be freed using this call. Weld values which are owned by the runtime also
 * free the data they contain.  Weld values which are not owned by the runtime only free the
 * structure used to wrap the data; the actual data itself is owned by the caller.
 */
void weld_value_free(WeldValueRef value);

/*
 * Returns a new Weld value.
 *
 * The value returned by this method is *not* owned by the runtime, so any data this value refers
 * to must be managed by the caller. The created value will always have a NULL context.
 *
 * This function is a wrapper for `WeldValue::new_from_data`.
 */
WeldValueRef weld_value_new(const void *data);

/*
 * Returns the Run ID of the value.
 *
 * If this value was not returned by a Weld program, this function returns `-1`.
 */
int64_t weld_value_run(WeldValueRef value);
