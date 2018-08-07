#ifndef _STRT_H_
#define _STRT_H_

#include <stdint.h>

struct WeldOpaqueRunHandle {};

// Reference to the run handle.
typedef struct WeldOpaqueRunHandle *WeldRunHandleRef;

// Weld errors. This must match the definition in weld::runtime.
typedef enum {
    /// Indicates success.
    ///
    /// This will always be 0.  
    Success = 0,
    /// Invalid configuration.
    ConfigurationError,
    /// Dynamic library load error.
    LoadLibraryError,
    /// Weld compilation error.
    CompileError,
    /// Array out-of-bounds error.
    ArrayOutOfBounds,
    /// A Weld iterator was invalid.
    BadIteratorLength,
    /// Mismatched Zip error.
    ///
    /// This error is thrown if the vectors in a Zip have different lengths.
    MismatchedZipSize,
    /// Out of memory error.
    ///
    /// This error is thrown if the amount of memory allocated by the runtime exceeds the limit set
    /// by the configuration.
    OutOfMemory,
    RunNotFound,
    /// An unknown error.
    Unknown,
    /// A deserialization error.
    ///
    /// This error occurs if a buffer being deserialized has an invalid length.
    DeserializationError,
    /// A key was not found in a dictionary.
    KeyNotFoundError,
    /// Maximum errno value.
    ///
    /// All errors will have a value less than this value and greater than 0.
    ErrnoMax,
} WeldRuntimeErrno;

extern "C" {
  void *weld_runst_malloc(WeldRunHandleRef run, int64_t bytes);
  void *weld_runst_realloc(WeldRunHandleRef run, void *pointer, int64_t new_size);
  void *weld_runst_free(WeldRunHandleRef run, void *pointer);
  void *weld_runst_set_errno(WeldRunHandleRef run, WeldRuntimeErrno errno);
}

#endif
