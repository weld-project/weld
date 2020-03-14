
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
  Vector() : length(0), data(nullptr) {}

  /// Creates a new empty vector with space for `length` elements.
  explicit Vector(int64_t capacity) : length(capacity) {
    data = new T[length];
  }

  int64_t length;
  T* data;

};

template class Vector<int8_t>;
template class Vector<Vector<int8_t>>;

// API for Python
extern "C" {
  /// Converts a Numpy array of Python strings to an Weld vector of strings.
  ///
  /// We only support strings with a dtype of 'S' for now, which indicates a bytearray.
  PyObject* NumpyArrayOfStringsToWeld(PyObject* self, PyObject *args) {
    PyObject *in;
    if (!PyArg_ParseTuple(args, "O", &in)) {
        return nullptr;
    }

    PyArrayObject* inp = reinterpret_cast<PyArrayObject*>(in);
    if (PyArray_DESCR(inp)->kind != 'S') {
      return nullptr;
    }

    int64_t num_strings = static_cast<int64_t>(PyArray_DIMS(inp)[0]);
    int8_t* data = reinterpret_cast<int8_t*>(PyArray_DATA(inp));
    int8_t* base = data;

    Vector<Vector<int8_t>> output(num_strings);

    for (long i = 0; i < num_strings; ++i) {
      output.data[i].length = strlen(reinterpret_cast<char*>(data));
      output.data[i].data = reinterpret_cast<int8_t*>(base + i * inp->strides[0]);
      data += (PyArray_STRIDES(inp)[0]);
    }

    // return output;
    return nullptr;
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

    // Stride is the size of the largest string.
    npy_intp stride = std::numeric_limits<npy_intp>::min();
    for (long i = 0; i < inp.length; ++i) {
      if (inp.data[i].length > stride) {
        stride = inp.data[i].length;
      }
    }

    _import_array();

    // TODO(shoumik.palkar): How do we free this?
    int8_t* buffer = new int8_t[stride * inp.length];
    // Make sure the allocation succeeded.
    if (buffer == nullptr) {
      return nullptr;
    }

    for (long i = 0; i < inp.length; i++) {
      // memcpy each of the strings.
      memcpy(buffer + i * stride, inp.data[i].data, inp.data[i].length);
      // This format expects a null-terminated byte.
      buffer[i * stride + inp.data[i].length] = 0;
    }

    // Construct the NumPy array.
    npy_intp dims = static_cast<npy_intp>(inp.length);
    PyObject* out = PyArray_New(&PyArray_Type, 1, &dims, NPY_STRING,  &stride, reinterpret_cast<void*>(buffer), stride, 0, nullptr);
    return out;
  }

  /// Python module definition
  static PyMethodDef methods[] = {
    {"NumpyArrayOfStringsToWeld", NumpyArrayOfStringsToWeld, METH_VARARGS, "some nonsense"},
    //{"WeldArrayOfStringsToNumPy", WeldArrayOfStringsToNumPy, METH_VARARGS, "some nonsense"},
    {NULL, NULL, 0, NULL}
  };

  static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "strings",
    "",
    -1,
    methods
  };

  PyMODINIT_FUNC PyInit_strings(void) {
    return PyModule_Create(&module);
  }

} // extern "C"

}; // namespace weld
