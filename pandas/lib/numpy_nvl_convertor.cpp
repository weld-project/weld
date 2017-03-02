#include "Python.h"
#include "numpy/arrayobject.h"
#include "common.h"
#include <cstdlib>
#include <cstdio>
using namespace std;

/**
 * Converts numpy array to NVL vector.
 */
extern "C"
nvl::vec<int32_t> numpy_to_nvl_int_arr(PyObject* in) {
    PyArrayObject* inp = (PyArrayObject*) in;
    int64_t dimension = (int64_t) PyArray_DIMS(inp)[0];
    nvl::vec<int32_t> t;
    t.size = dimension;
    t.ptr = (int32_t*) PyArray_DATA(inp);
    return t;
}

/**
 * Converts numpy array to NVL vector.
 */
extern "C"
nvl::vec<long> numpy_to_nvl_long_arr(PyObject* in) {
    PyArrayObject* inp = (PyArrayObject*) in;
    int64_t dimension = (int64_t) PyArray_DIMS(inp)[0];
    nvl::vec<long> t;
    t.size = dimension;
    t.ptr = (long*) PyArray_DATA(inp);
    return t;
}

/**
 * Converts numpy array to NVL vector.
 */
extern "C"
nvl::vec<double> numpy_to_nvl_double_arr(PyObject* in) {
    PyArrayObject* inp = (PyArrayObject*) in;
    int64_t dimension = (int64_t) PyArray_DIMS(inp);
    nvl::vec<double> t;
    t.size = dimension;
    t.ptr = (double*) PyArray_DATA(inp);
    return t;
}

/**
 * Converts numpy array to NVL char vector.
 */
extern "C"
nvl::vec<uint8_t> numpy_to_nvl_char_arr(PyObject* in) {
    int64_t dimension = (int64_t) PyString_Size(in);
    nvl::vec<uint8_t> t;
    t.size = dimension;
    t.ptr = (uint8_t*) PyString_AsString(in);
    return t;
}

/**
 * Converts numpy array to NVL bit vector.
  */
extern "C"
nvl::vec<bool> numpy_to_nvl_bool_arr(PyObject* in) {
    PyArrayObject* inp = (PyArrayObject*) in;
    int64_t dimension = (int64_t) PyArray_DIMS(inp)[0];
    nvl::vec<bool> t;
    t.size = dimension;
    t.ptr = (bool*) PyArray_DATA(inp);
    return t;
}

/**
 * Converts numpy array to NVL vector, with ndim = 2.
 */
extern "C"
nvl::vec<nvl::vec<int> > numpy_to_nvl_int_arr_arr(PyObject* in) {
    PyArrayObject* inp = (PyArrayObject*) in;
    int64_t dimension = (int64_t) PyArray_DIMS(inp)[0];
    nvl::vec<nvl::vec<int> > t;
    t = nvl::make_vec<nvl::vec<int> >(dimension);
    for (int i = 0; i < t.size; i++) {
        t.ptr[i].size = (int64_t) PyArray_DIMS(inp)[1];
        t.ptr[i].ptr = (int *)(inp->data + i * inp->strides[0]);
    }

    return t;
}

/**
 * Converts numpy array to NVL vector, with ndim = 2.
 */
extern "C"
nvl::vec<nvl::vec<long> > numpy_to_nvl_long_arr_arr(PyObject* in) {
    PyArrayObject* inp = (PyArrayObject*) in;
    int64_t dimension1 = (int64_t) PyArray_DIMS(inp)[0];
    int64_t dimension2 = (int64_t) PyArray_DIMS(inp)[1];
    nvl::vec<nvl::vec<long> > t = nvl::make_vec<nvl::vec<long> >(dimension1);
    if ((dimension1 * 8) == PyArray_STRIDES(inp)[1]) {
        // Matrix is transposed.
        long *new_buffer = (long *) malloc(sizeof(long) * dimension1 * dimension2);
        long *old_buffer = (long *) inp->data;
        for (int i = 0; i < t.size; i++) {
            t.ptr[i].size = dimension2;
            for (int j = 0; j < dimension2; j++) {
                *(new_buffer + j) = old_buffer[(j * dimension1) + i];
            }
            t.ptr[i].ptr = new_buffer;
            new_buffer += dimension2;
        }
    } else {
        for (int i = 0; i < t.size; i++) {
            t.ptr[i].size = dimension2;
            t.ptr[i].ptr = (long *)(inp->data + i * PyArray_STRIDES(inp)[0]);
        }
    }

    return t;
}

/**
 * Converts numpy array to NVL vector, with ndim = 2.
 */
extern "C"
nvl::vec<nvl::vec<double> > numpy_to_nvl_double_arr_arr(PyObject* in) {
    PyArrayObject* inp = (PyArrayObject*) in;
    int64_t dimension1 = (int64_t) PyArray_DIMS(inp)[0];
    int64_t dimension2 = (int64_t) PyArray_DIMS(inp)[1];
    nvl::vec<nvl::vec<double> > t = nvl::make_vec<nvl::vec<double> >(dimension1);
    if ((dimension1 * 8) == PyArray_STRIDES(inp)[1]) {
        // Matrix is transposed.
        double *new_buffer = (double *) malloc(sizeof(long) * dimension1 * dimension2);
        double *old_buffer = (double *) inp->data;
        for (int i = 0; i < t.size; i++) {
            t.ptr[i].size = dimension2;
            for (int j = 0; j < dimension2; j++) {
                *(new_buffer + j) = old_buffer[(j * dimension1) + i];
            }
            t.ptr[i].ptr = new_buffer;
            new_buffer += dimension2;
        }
    } else {
        for (int i = 0; i < t.size; i++) {
            t.ptr[i].size = dimension2;
            t.ptr[i].ptr = (double *)(inp->data + i * PyArray_STRIDES(inp)[0]);
        }
    }

    return t;
}

/**
 * Converts numpy array of strings to NVL vector, with ndim = 2.
 */
extern "C"
nvl::vec<nvl::vec<uint8_t> > numpy_to_nvl_char_arr_arr(PyObject* in) {
    PyArrayObject* inp = (PyArrayObject*) in;
    int64_t dimension = (int64_t) PyArray_DIMS(inp)[0];
    nvl::vec<nvl::vec<uint8_t> > t;
    t = nvl::make_vec<nvl::vec<uint8_t> >(dimension);
    uint8_t* ptr = (uint8_t *) PyArray_DATA(inp);
    uint8_t* data = ptr;
    for (int i = 0; i < t.size; i++) {
        t.ptr[i].size = strlen((char *) ptr);
        if ((int) inp->dimensions[1] < t.ptr[i].size) {
            t.ptr[i].size = (int) inp->dimensions[1];
        }
        t.ptr[i].ptr = (uint8_t *)(data + i * inp->strides[0]);
        ptr += (PyArray_STRIDES(inp)[0]);
    }

    return t;
}

/**
 * Converts NVL vector to numpy array.
 */
extern "C"
PyObject* nvl_to_numpy_int_arr(nvl::vec<int> inp) {
    Py_Initialize();
    npy_intp size = {inp.size};
    _import_array();
    PyObject* out = PyArray_SimpleNewFromData(1, &size, NPY_INT32, (char*)inp.ptr);
    return out;
}

/**
 * Converts NVL vector to numpy array.
 */
extern "C"
PyObject* nvl_to_numpy_long_arr(nvl::vec<long> inp) {
    Py_Initialize();
    npy_intp size = {inp.size};
    _import_array();
    PyObject* out = PyArray_SimpleNewFromData(1, &size, NPY_INT64, (char*)inp.ptr);
    return out;
}

/**
 * Converts NVL vector to numpy float array.
 */
extern "C"
PyObject* nvl_to_numpy_double_arr(nvl::vec<double> inp) {
    Py_Initialize();
    npy_intp size = {inp.size};
    _import_array();
    PyObject* out = PyArray_SimpleNewFromData(1, &size, NPY_DOUBLE, (char*)inp.ptr);
    return out;
}

/**
 * Converts NVL vector to (bool) numpy array.
 */
extern "C"
PyObject* nvl_to_numpy_bool_arr(nvl::vec<bool> inp) {
    Py_Initialize();
    npy_intp size = {inp.size};
    _import_array();
    PyObject* out = PyArray_SimpleNewFromData(1, &size, NPY_BOOL, (char*)inp.ptr);
    return out;
}

/**
 * Converts NVL vector-of-int-vectors to two-dimensional numpy array.
 */
extern "C"
PyObject* nvl_to_numpy_int_arr_arr(nvl::vec< nvl::vec<int> > inp) {
    Py_Initialize();

    int num_rows = inp.size;
    int num_cols = inp.ptr[0].size;

    npy_intp size[2] = {num_rows, num_cols};
    int *ptr_array = (int *) malloc(sizeof(int) * num_rows * num_cols);

    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            ptr_array[i * num_cols + j] = *((int *) inp.ptr[i].ptr + j);
        }
    }

    _import_array();
    PyObject* out = PyArray_SimpleNewFromData(2, size, NPY_INT32, (char*)ptr_array);
    return out;
}

/**
 * Converts NVL vector-of-long-vectors to two-dimensional numpy array.
 */
extern "C"
PyObject* nvl_to_numpy_long_arr_arr(nvl::vec< nvl::vec<long> > inp) {
    Py_Initialize();

    int num_rows = inp.size;
    int num_cols = inp.ptr[0].size;

    npy_intp size[2] = {num_rows, num_cols};
    long *ptr_array = (long *) malloc(sizeof(long) * num_rows * num_cols);

    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            ptr_array[i * num_cols + j] = *((long *) inp.ptr[i].ptr + j);
        }
    }

    _import_array();
    PyObject* out = PyArray_SimpleNewFromData(2, size, NPY_INT64, (char*)ptr_array);
    return out;
}

/**
 * Converts NVL vector-of-char-vectors to NumPy vector-of-strings.
 */
extern "C"
PyObject* nvl_to_numpy_char_arr_arr(nvl::vec< nvl::vec<uint8_t> > inp) {
    Py_Initialize();

    int num_rows = inp.size;

    PyObject** ptr_array = (PyObject**) malloc(sizeof(PyObject*) * num_rows);
    _import_array();

    for (int i = 0; i < num_rows; i++) {
        int size = inp.ptr[i].size;
        PyObject* buffer = PyString_FromStringAndSize((const char*) inp.ptr[i].ptr, size);
        ptr_array[i] = buffer;
    }
    npy_intp size = num_rows;

    PyObject* out = PyArray_SimpleNewFromData(1, &size, NPY_OBJECT, (void*) ptr_array);
    return out;
}
