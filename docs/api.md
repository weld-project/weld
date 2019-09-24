# Weld C API

This document describes the C API for interfacing with Weld.

- [Weld C API](#weld-c-api)
  * [Values](#values)
    + [API](#api)
  * [Module](#module)
    + [API](#api-1)
  * [Errors](#errors)
    + [API](#api-2)
  * [Configurations](#configurations)
    + [API](#api-3)

## Values

A value is a piece of data which can be consumed by Weld. Data wrapped using the API's value type must be marshalled
into a format which Weld understands. The table below describes how each type in Weld is represented in C. `(...)` represents a C struct with fields in the specified order (e.g., `(uint8_t, uint16_t)` is a struct where the first byte is a `uint8_t` and the second two bytes is a `uint16_t`.


  Type | C Representation | Explanation
  ------------- | ------------- | -------------
  `i8` | `int8_t` |
  `i16` | `int16_t` |
  `i32` | `int32_t` |
  `i64` | `int64_t` |
  `f32` | `float` |
  `f64` | `double` |
  `{T1, T2 ... }` | `(T1,T2,...)`|
  `vec[T]` | `(*T,int64_t)` | pointer to data, size of vector

Values can be initialized using user data using the `weld_value_new` call. After its use, a value must be freed using the `weld_value_free` call. Note that
a value initialized by `weld_value_new` must have its data freed by the caller as well (e.g., by calling C's `free` function explicitly).
Values are also returned by Weld; these values are _owned_ by the runtime, so their data buffers are freed automatically after a `weld_value_free` call.

### API

```C
/** A type encapsulating a Weld value. */
typedef void* weld_value_t;

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

```


## Module

A module represents a runnable Weld program.

### API

```C

/** A runnable Weld module. */
typedef void* weld_module_t;

/** Compiles a Weld module.
 *
 * Takes a string and configuration and returns a runnable module.
 *
 * @param code a Weld program to compile
 * @param conf a configuration for the module.
 * @param err a Weld error for this compilation.
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

```

## Errors

Errors are returned by the Weld runtime, and may signify either runtime errors (e.g., array out of bounds)
or compilation errors.

### API

```C

/** A handle to a Weld error. */
typedef void* weld_error_t;

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

```

## Configurations

A configuration specifies tunable parameters when generating and running code. The table below
specifies some example that can be tuned by using configurations (where both keys and values
are encoded as C strings.

  Configuration | Value
  ------------- | -------------
  `weld.threads` | A string value, e.g., `"1"`
  `weld.memory.limit` | A memory limit for Weld in bytes


### API

```C

/** A hanlde to a Weld configuration. */
typedef void* weld_conf_t;

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
extern "C" const char *
weld_conf_get(weld_conf_t, const char *key);

/** Set a value for a Weld configuration key.
 *
 * @param key the key
 * @param key the value
 */
extern "C" void
weld_conf_set(weld_conf_t, const char *key, const char *value);

/** Return a new Weld configuraiton.
 *
 */
extern "C" void
weld_conf_free(weld_conf_t);

```

