/**
 *
 * Weld Runtime C Interface.
 * Link the Weld library with -lweld.
 *
 */
#ifndef _WELD_H_
#define _WELD_H_

#include <stdint.h>

// Log level constants -- should match WeldLogLevel.

/** A Weld log level. */
typedef uint64_t weld_log_level_t;

static const weld_log_level_t WELD_LOG_LEVEL_OFF = 0;
static const weld_log_level_t WELD_LOG_LEVEL_ERROR = 1;
static const weld_log_level_t WELD_LOG_LEVEL_WARN = 2;
static const weld_log_level_t WELD_LOG_LEVEL_INFO = 3;
static const weld_log_level_t WELD_LOG_LEVEL_DEBUG = 4;
static const weld_log_level_t WELD_LOG_LEVEL_TRACE = 5;

// Types

/** A type encapsulating a Weld value. */
typedef struct weld_value_* weld_value_t;

/** A runnable Weld module. */
typedef struct weld_module_* weld_module_t;

/** A handle to a Weld error. */
typedef struct weld_error_* weld_error_t;

/** A hanlde to a Weld configuration. */
typedef struct weld_conf_* weld_conf_t;

/** A hanlde to a Weld context. */
typedef struct weld_context_* weld_context_t;

// ************* Contexts **************

/**
 * Returns a new Weld context configured with the given configuration.
 *
 * If the context failed to initialize properly, this function returns NULL.
 * A context  can fail to initialize if the provided configuration is malformed.
 *
 * @param conf the configuration for the context.
 * @return a new context.
 */
extern "C" weld_context_t
weld_context_new(weld_conf_t conf);

/* Gets the memory usage of the Weld context in bytes.
 *
 * @param context the context whose memory usage to report.
 * @return the memory usage in bytes.
 */
extern "C" int64_t
weld_context_memory_usage(weld_context_t context);

/**
 * Frees a weld context.
 *
 * @param context the context to free.
 */
extern "C" void
weld_context_free(weld_context_t);

// ************* Values ****************

/**
 * Returns a Weld-readable value with the given input buffer.
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

/**
 * Returns the context that owns this value.
 *
 * @param value the value whose context to return.
 * @return The context of this value, or NULL if the context was freed already.
 */
extern "C" weld_context_t
weld_value_context(weld_value_t obj);

/**
 * Returns this value's data buffer.
 *
 * @param obj the value whose data buffer should be retrieved.
 * @return a void * data buffer. The caller is responsible for knowing
 * the type of the buffer and casting it appropriately.
 */
extern "C" void*
weld_value_data(weld_value_t obj);

/**
 * Frees a Weld value.
 *
 * Each Weld value must be freed using this call. Owned values also
 * free their data buffers; non-owned values require the caller to free
 * the buffer explicitly. This Weld value is invalid after this call.
 *
 * @param obj the value to free.
 */
extern "C" void
weld_value_free(weld_value_t obj);


// ************* Modules ****************

/**
 * Compiles a Weld module.
 *
 * Takes a string and configuration and returns a runnable module.
 *
 * @param code a Weld program to compile.
 * @param conf a configuration for the module.
 * @param err will hold any error raised during compilation.
 * @return a runnable module.
 */
extern "C" weld_module_t
weld_module_compile(const char *code, weld_conf_t, weld_error_t);

/**
 * Runs a module using the given argument.
 *
 * Multi-argument Weld functions take a Weld value encapsulating
 * a single struct as an argument. The field at index i in the struct
 * represents the ith argument of the Weld function.
 *
 * @param module the module to run.
 * @param conf a configuration for this run.
 * @param arg the argument for the module's function.
 * @param err will hold any error raised during execution.
 * @return an owned Weld value representing the return value. The caller
 * is responsible for knowing what the type of the return value is based on
 * the module she runs.
 */
extern "C" weld_value_t
weld_module_run(weld_module_t, weld_context_t, weld_value_t, weld_error_t);

/**
 * Garbage collects a module.
 *
 * @param module the module to garbage collect.
 */
extern "C" void
weld_module_free(weld_module_t);

// ************* Errors ****************

/**
 * Return a new Weld error holder.
 */
extern "C" weld_error_t
weld_error_new();

/**
 * Returns an error code, or 0 if there was no error.
 *
 * @param err the error to check
 * @param 0 if the error was a success, or a nonzero error code otherwise.
 */
extern "C" int
weld_error_code(weld_error_t err);

/**
 * Returns an error message for a given error.
 *
 * @param err the error
 * @return a string error message.
 */
extern "C" const char *
weld_error_message(weld_error_t err);

/**
 * Free a Weld error.
 *
 * @param err the error to free
 */
extern "C" void
weld_error_free(weld_error_t err);

// ************* Configuration ****************

/**
 * Returns a new Weld configuration.
 */
extern "C" weld_conf_t
weld_conf_new();

/**
 * Returns a value for a Weld configuration key.
 *
 * @param key the key to look up.
 * @return the string value for the key, or NULL if the key does not exist.
 */
extern "C" const char*
weld_conf_get(weld_conf_t conf, const char *key);

/**
 * Set a value for a Weld configuration key.
 *
 * @param key the key
 * @param key the value
 */
extern "C" void
weld_conf_set(weld_conf_t conf, const char *key, const char *value);

/**
 * Free a Weld configuration.
 *
 * @param conf the configuration to free
 */
extern "C" weld_conf_t
weld_conf_free(weld_conf_t conf);

// ************* Other Functions ****************

/**
 * Loads a dynamically linked library and makes it available to the LLVM compiler and JIT.
 * This function is safe to call multiple times on the same library file.
 *
 * @param filename path of the library to load, including .so extension on Linux
 * @param err will hold any errors raised during execution
 */
extern "C" void
weld_load_library(const char *filename, weld_error_t err);

/**
 * Enables logging to stderr in Weld with the given level (one of the WELD_LOG_* constants).
 * This function is ignored if it has already been called once, or if some other code in the
 * process has initialized logging using Rust's `log` crate.
 *
 * @param level the log level to use.
 */
extern "C" void
weld_set_log_level(weld_log_level_t level);

#endif

