#include "Python.h"
#include "numpy/arrayobject.h"
#include "common.h"
#include <cstdlib>
#include <cstdio>
#include <iostream>
//#include <omp.h> //uncomment to enable parallel encode

using namespace std;

/**
 * Converts numpy array to Weld vector.
 */
extern "C"
weld::vec<int32_t> numpy_to_weld_int_arr(PyObject* in) {
  PyArrayObject* inp = (PyArrayObject*) in;
  int64_t dimension = (int64_t) PyArray_DIMS(inp)[0];
  weld::vec<int32_t> t;
  t.size = dimension;
  t.ptr = (int32_t*) PyArray_DATA(inp);
  return t;
}

/**
 * Converts numpy array to Weld vector.
 */
extern "C"
weld::vec<long> numpy_to_weld_long_arr(PyObject* in) {
  PyArrayObject* inp = (PyArrayObject*) in;
  int64_t dimension = (int64_t) PyArray_DIMS(inp)[0];
  weld::vec<long> t;
  t.size = dimension;
  t.ptr = (long*) PyArray_DATA(inp);
  return t;
}

/**
 * Converts numpy array to Weld vector.
 */
extern "C"
weld::vec<double> numpy_to_weld_double_arr(PyObject* in) {
  PyArrayObject* inp = (PyArrayObject*) in;
  int64_t dimension = (int64_t) PyArray_DIMS(inp)[0];
  weld::vec<double> t;
  t.size = dimension;
  t.ptr = (double*) PyArray_DATA(inp);
  return t;
}

/**
 * Converts numpy array to Weld char vector.
 */
extern "C"
weld::vec<uint8_t> numpy_to_weld_char_arr(PyObject* in) {
  int64_t dimension = (int64_t) PyString_Size(in);
  weld::vec<uint8_t> t;
  t.size = dimension;
  t.ptr = (uint8_t*) PyString_AsString(in);
  return t;
}

/**
 * Converts numpy array to Weld bit vector.
 */
extern "C"
weld::vec<bool> numpy_to_weld_bool_arr(PyObject* in) {
  PyArrayObject* inp = (PyArrayObject*) in;
  int64_t dimension = (int64_t) PyArray_DIMS(inp)[0];
  weld::vec<bool> t;
  t.size = dimension;
  t.ptr = (bool*) PyArray_DATA(inp);
  return t;
}

/**
 * Converts numpy array to Weld vector, with ndim = 2.
 */
extern "C"
weld::vec<weld::vec<int> > numpy_to_weld_int_arr_arr(PyObject* in) {
  PyArrayObject* inp = (PyArrayObject*) in;
  int64_t dimension = (int64_t) PyArray_DIMS(inp)[0];
  weld::vec<weld::vec<int> > t;
  t = weld::make_vec<weld::vec<int> >(dimension);
  for (int i = 0; i < t.size; i++) {
    t.ptr[i].size = (int64_t) PyArray_DIMS(inp)[1];
    t.ptr[i].ptr = (int *)(inp->data + i * inp->strides[0]);
  }

  return t;
}

/**
 * Converts numpy array to Weld vector, with ndim = 2.
 */
extern "C"
weld::vec<weld::vec<long> > numpy_to_weld_long_arr_arr(PyObject* in) {
  PyArrayObject* inp = (PyArrayObject*) in;
  int64_t dimension1 = (int64_t) PyArray_DIMS(inp)[0];
  int64_t dimension2 = (int64_t) PyArray_DIMS(inp)[1];
  weld::vec<weld::vec<long> > t = weld::make_vec<weld::vec<long> >(dimension1);
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

struct numpy_to_weld_double_arr_arr_args {
  const PyArrayObject* inp;
  double* ptr;
  weld::vec<weld::vec<double> >* t;
  int start;
  int end;
};

/* Helper method used for parallelized encoder. */
void *numpy_to_weld_double_arr_arr_helper(void* args_) {
  struct numpy_to_weld_double_arr_arr_args* args = (struct numpy_to_weld_double_arr_arr_args*) args_;
  double* ptr = args->ptr;
  const PyArrayObject* inp = args->inp;
  weld::vec<weld::vec<double> >* t = args->t;
  for (int i = args->start; i < args->end; i++) {
    t->ptr[i].size = (int64_t) PyArray_DIMS(inp)[1];
    t->ptr[i].ptr = (double *)(inp->data + i * PyArray_STRIDES(inp)[0]);
    ptr += (inp->strides[0]);
  }
}

/**
 * Converts numpy array to Weld vector, with ndim = 2.
 */
extern "C"
weld::vec<weld::vec<double> > numpy_to_weld_double_arr_arr(PyObject* in, int num_threads) {
  PyArrayObject* inp = (PyArrayObject*) in;
  int64_t dimension1 = (int64_t) PyArray_DIMS(inp)[0];
  int64_t dimension2 = (int64_t) PyArray_DIMS(inp)[1];
  weld::vec<weld::vec<double> > t = weld::make_vec<weld::vec<double> >(dimension1);
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
    struct numpy_to_weld_double_arr_arr_args args[num_threads];
    int fringe_length = t.size % num_threads;

    for (int i = 0; i < num_threads; i++) {
      args[i].inp = inp;
      args[i].ptr = (double *) inp->data;
      args[i].t = &t;
      args[i].start = (t.size / num_threads) * i;
      if (i < fringe_length) {
	args[i].start += i;
      } else {
	args[i].start += fringe_length;
      }
      args[i].ptr += (args[i].start * inp->strides[0]);
      args[i].end = (t.size / num_threads) * (i + 1);
      if ((i+1) < fringe_length) {
	args[i].end += (i+1);
      } else {
	args[i].end += fringe_length;
      }
    }

    // #pragma omp parallel for // uncomment to enable parallel encode
    for (int i = 0; i < num_threads; i++) {
      numpy_to_weld_double_arr_arr_helper(&args[i]);
    }
  }

  return t;
}

struct numpy_to_weld_char_arr_arr_args {
  const PyArrayObject* inp;
  uint8_t* ptr;
  weld::vec<weld::vec<uint8_t> >* t;
  int start;
  int end;
};

// /* Helper method used for parallelized encoder. */
// void *numpy_to_weld_char_arr_arr_helper(void* args_) {
//   struct numpy_to_weld_char_arr_arr_args* args = (struct numpy_to_weld_char_arr_arr_args*) args_;
//   uint8_t* ptr = args->ptr;
//   const PyArrayObject* inp = args->inp;
//   uint8_t* data = (uint8_t *) PyArray_DATA(inp);
//   weld::vec<weld::vec<uint8_t> >* t = args->t;
//   for (int i = args->start; i < args->end; i++) {
//     t->ptr[i].size = (int64_t) strlen((char *) ptr);
//     if ((int) inp->dimensions[1] < t->ptr[i].size) {
//       t->ptr[i].size = (int) inp->dimensions[1];
//     }
//     t->ptr[i].ptr = (uint8_t *)(data + i * inp->strides[0]);
//     ptr += (PyArray_STRIDES(inp)[0]);
//   }
// }

// /**
//  * Converts numpy array of strings to Weld vector, with ndim = 2.
//  */
// extern "C"
// weld::vec<weld::vec<uint8_t> > numpy_to_weld_char_arr_arr(PyObject* in, int num_threads) {
  
//   PyArrayObject* inp = (PyArrayObject*) in;
//   int64_t dimension = (int64_t) PyArray_DIMS(inp)[0];
//   weld::vec<weld::vec<uint8_t> > t;
//   t = weld::make_vec<weld::vec<uint8_t> >(dimension);
//   uint8_t* ptr = (uint8_t *) PyArray_DATA(inp);
//   struct numpy_to_weld_char_arr_arr_args args[num_threads];
//   int fringe_length = t.size % num_threads;
//   for (int i = 0; i < num_threads; i++) {
    
//     args[i].inp = inp;
//     args[i].ptr = (uint8_t *) PyArray_DATA(inp);
//     args[i].t = &t;
//     args[i].start = (t.size / num_threads) * i;
//     if (i < fringe_length) {
//       args[i].start += i;
//     } else {
//       args[i].start += fringe_length;
//     }
//     args[i].ptr += (args[i].start * inp->strides[0]);
//     args[i].end = (t.size / num_threads) * (i + 1);
//     if ((i+1) < fringe_length) {
//       args[i].end += (i+1);
//     } else {
//       args[i].end += fringe_length;
//     }
//     return t;
//   }

//   // #pragma omp parallel for // uncomment to enable parallel encode
//   for (int i = 0; i < num_threads; i++) {
//     numpy_to_weld_char_arr_arr_helper(&args[i]);
//   }

//   return t;
// }

/**
 * Converts numpy array of strings to Weld vector, with ndim = 2.
 */
extern "C"
weld::vec<weld::vec<uint8_t> > numpy_to_weld_char_arr_arr(PyObject* in) {
  PyArrayObject* inp = (PyArrayObject*) in;
  int64_t dimension = (int64_t) PyArray_DIMS(inp)[0];
  weld::vec<weld::vec<uint8_t> > t;
  t = weld::make_vec<weld::vec<uint8_t> >(dimension);
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
 * Converts Weld vector to numpy array.
 */
extern "C"
PyObject* weld_to_numpy_int_arr(weld::vec<int> inp) {
  Py_Initialize();
  npy_intp size = {inp.size};
  _import_array();
  PyObject* out = PyArray_SimpleNewFromData(1, &size, NPY_INT32, (char*)inp.ptr);
  return out;
}

/**
 * Converts Weld vector to numpy array.
 */
extern "C"
PyObject* weld_to_numpy_long_arr(weld::vec<long> inp) {
  Py_Initialize();
  npy_intp size = {inp.size};
  _import_array();
  PyObject* out = PyArray_SimpleNewFromData(1, &size, NPY_INT64, (char*)inp.ptr);
  return out;
}

/**
 * Converts Weld vector to numpy float array.
 */
extern "C"
PyObject* weld_to_numpy_double_arr(weld::vec<double> inp) {
  Py_Initialize();
  npy_intp size = {inp.size};
  _import_array();
  PyObject* out = PyArray_SimpleNewFromData(1, &size, NPY_DOUBLE, (char*)inp.ptr);
  return out;
}

/**
 * Converts Weld vector to (bool) numpy array.
 */
extern "C"
PyObject* weld_to_numpy_bool_arr(weld::vec<bool> inp) {
  Py_Initialize();
  npy_intp size = {inp.size};
  _import_array();
  PyObject* out = PyArray_SimpleNewFromData(1, &size, NPY_BOOL, (char*)inp.ptr);
  return out;
}

/**
 * Converts Weld vector-of-int-vectors to two-dimensional numpy array.
 */
extern "C"
PyObject* weld_to_numpy_int_arr_arr(weld::vec< weld::vec<int> > inp) {
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
 * Converts Weld vector-of-long-vectors to two-dimensional numpy array.
 */
extern "C"
PyObject* weld_to_numpy_long_arr_arr(weld::vec< weld::vec<long> > inp) {
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
 * Converts Weld vector-of-double-vectors to two-dimensional numpy array.
 */
extern "C"
PyObject* weld_to_numpy_double_arr_arr(weld::vec< weld::vec<double> > inp) {
  Py_Initialize();

  int num_rows = inp.size;
  int num_cols = inp.ptr[0].size;

  npy_intp size[2] = {num_rows, num_cols};
  double *ptr_array = (double *) malloc(sizeof(double) * num_rows * num_cols);

  for (int i = 0; i < num_rows; i++) {
    for (int j = 0; j < num_cols; j++) {
      ptr_array[i * num_cols + j] = *((double *) inp.ptr[i].ptr + j);
    }
  }

  _import_array();
  PyObject* out = PyArray_SimpleNewFromData(2, size, NPY_FLOAT64, (char*)ptr_array);
  return out;
}

/**
 * Converts Weld vector-of-char-vectors to NumPy vector-of-strings.
 */
extern "C"
PyObject* weld_to_numpy_char_arr_arr(weld::vec< weld::vec<uint8_t> > inp) {
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
