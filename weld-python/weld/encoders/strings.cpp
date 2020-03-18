
#include "Python.h"
#include "numpy/arrayobject.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <iostream>

namespace weld {

/// A Weld vector.
template <typename T>
class Vector {
 public:
  /// Default constructor.
  Vector() : data(nullptr), length(0) {}

  /// Creates a new empty vector with space for `length` elements.
  explicit Vector(int64_t capacity) : length(capacity) {
    data = new T[length];
  }

  T* data;
  int64_t length;

};

template class Vector<int8_t>;
template class Vector<Vector<int8_t>>;

// API for Python
extern "C" {
  /// Converts a Numpy array of Python strings to an Weld vector of strings.
  ///
  /// We only support strings with a dtype of 'S' for now, which indicates a bytearray.
  Vector<Vector<int8_t>> NumpyArrayOfStringsToWeld(PyObject* in) {
    PyArrayObject* inp = reinterpret_cast<PyArrayObject*>(in);
    if (PyArray_DESCR(inp)->kind != 'S') {
      // Returns an empty vector.
      return Vector<Vector<int8_t>>();
    }

    int64_t nd = static_cast<int64_t>(PyArray_NDIM(inp));
    if (nd != 1) {
      // String vectors should alays have a dimension of 1.
      return Vector<Vector<int8_t>>();
    }

    int64_t num_strings = static_cast<int64_t>(PyArray_DIMS(inp)[0]);
    // Maximum length of any string.
    int64_t stride = static_cast<int64_t>(PyArray_STRIDES(inp)[0]);
    int8_t* data = reinterpret_cast<int8_t*>(PyArray_DATA(inp));
    int8_t* base = data;

    // TODO(shoumik.palkar): How to free this?
    Vector<Vector<int8_t>> output(num_strings);

    for (long i = 0; i < num_strings; ++i) {
      output.data[i].length = strnlen(reinterpret_cast<char*>(data), stride);
      // TODO(shoumik.palkar): This array should hold a reference to the Python object.
      output.data[i].data = reinterpret_cast<int8_t*>(base + i * stride);
      data += stride;
    }
    return output;
  }

   /// Converts a Weld vector of strings into a NumPy vector-of-strings. The dtype of the
   /// NumPY vector is "STRING", i.e., a bytearray.
   ///
   /// We allocate a single array contiguously
   /// and `memcpy` each into the buffer instead of converting eveything back into a Python
   /// string object, which is more expensive (due to conversion to UCS4) and makes using this
   /// array easier in other Weld computations down the line.
  PyObject* WeldArrayOfStringsToNumPy(Vector<Vector<int8_t>> inp) {
    Py_Initialize();

    // Stride is the size of the largest string. The minimum stride is 1 (for an empty string):
    // in this case, each string is just a single 0-valued byte.
    npy_intp stride = 1;
    for (long i = 0; i < inp.length; ++i) {
      if (inp.data[i].length > stride) {
        stride = inp.data[i].length;
      }
    }

    _import_array();

    // TODO(shoumik.palkar): How do we free this?
    int8_t* buffer = new int8_t[stride * inp.length];
    // Strings with lengths less than stride must be null-terminated.
    memset(buffer, 0, stride * inp.length);
    // Make sure the allocation succeeded.
    if (buffer == nullptr) {
      return nullptr;
    }

    for (long i = 0; i < inp.length; i++) {
      // memcpy each of the strings.
      memcpy(buffer + i * stride, inp.data[i].data, inp.data[i].length);
    }

    // Construct the NumPy array.
    npy_intp dims[] = { static_cast<npy_intp>(inp.length) };
    npy_intp strides[] = { stride };
    PyObject* out = PyArray_New(&PyArray_Type, /*nd=*/1, dims, NPY_STRING,
                                /*strides=*/strides, reinterpret_cast<void*>(buffer),
                                /*itemsize=*/stride, /*flags=*/0, nullptr);
    return out;
  }

  // Define this to be a module with no methods -- we'll load it as a dynamic library and call it with ctypes directly.
  static PyMethodDef methods[] = { {NULL, NULL, 0, NULL} };

  static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_strings",
    "",
    -1,
    methods
  };

  PyMODINIT_FUNC PyInit__strings(void) {
    return PyModule_Create(&module);
  }

} // extern "C"

}; // namespace weld
