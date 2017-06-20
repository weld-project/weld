from pyarrow.lib cimport Array
from libcpp.memory cimport shared_ptr, make_shared
from libc.stdint cimport uint8_t, int64_t, uintptr_t
from ctypes import cast, addressof

from pyarrow.lib cimport pyarrow_wrap_array, DataType
from pyarrow.includes.libarrow cimport *

cdef extern from "arrow/array.h" namespace "arrow":
    cdef cppclass PrimitiveArray:
        int64_t length()
        shared_ptr[CBuffer] data()
        int64_t null_count()

cdef extern from "arrow/loader.h" namespace "arrow":
    cdef CStatus MakePrimitiveArray(const shared_ptr[CDataType]& type, 
    int64_t length, const shared_ptr[CBuffer]& data,
    const shared_ptr[CBuffer]& null_bitmap, int64_t null_count, int64_t offset,
    shared_ptr[CArray] *out)

def arrow_to_weld_arr(Array array, weld_vec_ctype):
    
    cdef PrimitiveArray *pa = <PrimitiveArray*> array.sp_array.get()
    size = pa[0].length()
    null_count = pa[0].null_count()
    if null_count != 0:
        raise Exception("Cannot serialize arrow data with nulls to weld")

    ptr = <uintptr_t> pa[0].data().get()[0].data()
    ptr = cast(ptr, weld_vec_ctype._fields_[0][1])
    # TODO(jhscott): evaluate whether this causes a use-after-free bug
    return weld_vec_ctype(ptr, size)

def weld_arr_to_arrow(weld_vec, DataType arrow_type):
    cdef:
        uintptr_t ptr_as_int = addressof(weld_vec.ptr.contents)
        uint8_t* ptr = <uint8_t*> ptr_as_int
        int64_t length = weld_vec.size
        shared_ptr[CBuffer] null_buffer = make_shared[CBuffer](<uint8_t*> NULL, 0)
        shared_ptr[CBuffer] data_buffer = make_shared[CBuffer](ptr, length)
        shared_ptr[CArray] out

    MakePrimitiveArray(arrow_type.sp_type, length, data_buffer, null_buffer, 0, 0, &out)
    return pyarrow_wrap_array(out)