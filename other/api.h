/**
 *
 * Weld Runtime Interface.
 *
 */
#ifndef _WELD_H_
#define _WELD_H_

// Types

/** An object encapsulating a Weld value. */
typedef void* weld_object_t;

/** A runnable Weld module. */
typedef void* weld_module_t;

// Objects

/** Returns a Weld-readable object with the given input buffer.
 *
 * A Weld object created using this method is owned by the caller.
 * The caller must ensure that the data buffer remains a valid pointer
 * to memory until after the object is used by the runtime. The object
 * must be freed with `weld_object_free`.
 *
 * @param data the data this object captures.
 * @return a new Weld object.
 */
weld_object_t 
weld_object_new(void *data);

/** Returns 1 if the object's data is owned by the Weld runtime, or
 * 0 otherwise.
 *
 * An object owned by the Weld runtime is freed using the `weld_object_free`
 * call. Non-owned objects must have their *data buffers* (retrieved using
 * `weld_object_data`) freed by the caller; this Weld object must still
 * be garbage collected using `weld_object_free` however.
 *
 * @param obj the object to check
 * @return 1 if owned, 0 otherwise.
 */
int 
weld_object_owned(weld_object_t obj);

/** Returns this object's data buffer.
 *
 * @param obj the object whose data buffer should be retrieved.
 * @return a void * data buffer. The caller is responsible for knowing
 * the type of the buffer and casting it appropriately.
 */
void* 
weld_object_data(weld_object_t);

/* Frees a Weld object.
 *
 * Each Weld object must be freed using this call. Owned objects also
 * free their data buffers; non-owned objects require the caller to free
 * the buffer explicitly. This Weld object is invalid after this call.
 *
 * @param obj the object to free.
 */
void 
weld_object_free(weld_object_t);

// Modules

/** Compiles a Weld module.
 *
 * Takes a string and configuration and returns a runnable module.
 *
 * @param program a Weld program to compile
 * @param conf a configuration for the module.
 * @return a runnable module.
 */
weld_module_t 
weld_module_compile(const char *program, const char *conf);

/** Runs a module using the given argument.
 *
 * Multi-argument Weld functions take a Weld object encapsulating
 * a single struct as an argument. The field at index i in the struct
 * represents the ith argument of the Weld function.
 *
 * @param module the module to run.
 * @param arg the argument for the module's function.
 * @return an owned Weld object representing the return value. The caller
 * is responsible for knowing what the type of the return value is based on
 * the module she runs.
 */
weld_object_t 
weld_module_run(weld_module_t, weld_object_t arg);

/** Garbage collects a module.
 *
 * @param module the module to garbage collect.
 */
void 
weld_module_free(weld_module_t);

#endif

