#
# Implements a wrapper around the Weld API.
#

from ctypes import *

import platform
import os
import copy

system = platform.system()
if system == 'Linux':
    path = "libweld.so"
elif system == 'Windows':
    path = "libweld.dll"
elif system == 'Darwin':
    path = "libweld.dylib"
else:
    raise OSError("Unsupported platform {}", system)

# Load the Weld Dynamic Library.
weld = CDLL(os.environ["WELD_HOME"] + "/target/release/" + path)

# Used for some type checking carried out by ctypes
class c_weld_module(c_void_p): pass
class c_weld_conf(c_void_p): pass
class c_weld_value(c_void_p): pass

class WeldModule(c_void_p):
    def __init__(self, code, conf, err):
        weld_module_compile = weld.weld_module_compile
        weld_module_compile.argtypes = [c_char_p, c_weld_conf, POINTER(WeldError)]
        weld_module_compile.restype = c_weld_module

        code = c_char_p(code)
        self.module = weld_module_compile(code, conf.conf, byref(err))

    def run(self, conf, arg, err):
        weld_module_run = weld.weld_module_run
        # module, conf, arg, &err
        weld_module_run.argtypes = [c_weld_module, c_weld_conf, c_weld_value, POINTER(WeldError)]
        weld_module_run.restype = c_weld_value
        ret = weld_module_run(self.module, conf.conf, arg.val, byref(err))
        return WeldValue(ret, assign=True)

    def __del__(self):
        weld_module_free = weld.weld_module_free
        weld_module_free.argtypes = [c_weld_module]
        weld_module_free.restype = None
        weld_module_free(self.module)


class WeldValue(c_void_p):
    def __init__(self, value, assign=False):
        if assign is False:
            weld_value_new = weld.weld_value_new
            weld_value_new.argtypes = [c_void_p]
            weld_value_new.restype = c_weld_value
            self.val = weld_value_new(value)
        else:
            self.val = value
        self.freed = False

    def _check(self):
        if self.freed:
            raise ValueError("Attempted to use freed WeldValue")

    def data(self):
        self._check()
        weld_value_data = weld.weld_value_data
        weld_value_data.argtypes = [c_weld_value]
        weld_value_data.restype = c_void_p
        return weld_value_data(self.val)

    def free(self):
        self._check()
        weld_value_free = weld.weld_value_free
        weld_value_free.argtypes = [c_weld_value]
        weld_value_free.restype = None
        self.freed = True
        return weld_value_free(self.val)


class WeldConf(c_void_p):
    def __init__(self):
        weld_conf_new = weld.weld_conf_new
        weld_conf_new.argtypes = []
        weld_conf_new.restype = c_weld_conf
        self.conf = weld_conf_new()

    def get(self, key):
        key = c_char_p(key)
        weld_conf_get = weld.weld_conf_get
        weld_conf_get.argtypes = [c_weld_conf, c_char_p]
        weld_conf_get.restype = c_char_p
        val = weld_conf_get(self.conf, key)
        return copy.copy(val.value)

    def set(self, key, value):
        key = c_char_p(key)
        value = c_char_p(value)
        weld_conf_set = weld.weld_conf_set
        weld_conf_set.argtypes = [c_weld_conf, c_char_p, c_char_p]
        weld_conf_set.restype = None
        weld_conf_set(self.conf, key, value)

    def __del__(self):
        weld_conf_free = weld.weld_conf_free
        weld_conf_free.argtypes = [c_weld_conf]
        weld_conf_free.restype = None
        weld_conf_free(self.conf)


class WeldError(c_void_p):
    def code(self):
        weld_error_code = weld.weld_error_code
        weld_error_code.argtypes = [WeldError]
        weld_error_code.restype = c_uint64
        return weld_error_code(self)

    def message(self):
        weld_error_message = weld.weld_error_message
        weld_error_message.argtypes = [WeldError]
        weld_error_message.restype = c_char_p
        val = weld_error_message(self)
        return copy.copy(val)

    def __del__(self):
        weld_error_free = weld.weld_error_free
        weld_error_free.argtypes = [WeldError]
        weld_error_free.restype = None
        weld_error_free(self)
