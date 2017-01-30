
from ctypes import *

# Load the Weld Dynamic Library.
weld = CDLL("../../../../target/debug/libweld.dylib")

# Set up the function pointers for the other functions we'll call.
# TODO(shoumik) Could move this into another file or class.

weld_module_compile = weld.weld_module_compile
weld_module_compile.argtypes = [c_char_p, c_char_p, POINTER(c_void_p)]
weld_module_compile.restype = c_void_p

weld_module_run = weld.weld_module_run
weld_module_run.argtypes = [c_void_p, c_void_p, POINTER(c_void_p)]
weld_module_run.restype = c_void_p

weld_value_new = weld.weld_value_new
weld_value_new.argtypes = [c_void_p]
weld_value_new.restype = c_void_p

weld_value_data = weld.weld_value_data
weld_value_data.argtypes = [c_void_p]
weld_value_data.restype = c_void_p

weld_error_free = weld.weld_error_free
weld_error_free.argtypes = [c_void_p]
weld_error_free.restype = None

weld_value_free = weld.weld_value_free
weld_value_free.argtypes = [c_void_p]
weld_value_free.restype = None

weld_module_free = weld.weld_module_free
weld_module_free.argtypes = [c_void_p]
weld_module_free.restype = None
