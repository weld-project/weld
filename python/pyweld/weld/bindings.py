#
# Implements a wrapper around the Weld API.
#

from ctypes import *

import os
import platform
import copy

import pkg_resources

system = platform.system()
if system == 'Linux':
    lib_file = "libweld.so"
elif system == 'Windows':
    lib_file = "libweld.dll"
elif system == 'Darwin':
    lib_file = "libweld.dylib"
else:
    raise OSError("Unsupported platform {}", system)

lib_file = pkg_resources.resource_filename(__name__, lib_file)

weld = CDLL(lib_file, mode=RTLD_GLOBAL)

# Used for some type checking carried out by ctypes

class c_weld_module(c_void_p):
    pass

class c_weld_conf(c_void_p):
    pass

class c_weld_err(c_void_p):
    pass

class c_weld_value(c_void_p):
    pass

class WeldModule(c_void_p):

    def __init__(self, code, conf, err):
        weld_module_compile = weld.weld_module_compile
        weld_module_compile.argtypes = [
            c_char_p, c_weld_conf, c_weld_err]
        weld_module_compile.restype = c_weld_module

        code = c_char_p(code)
        self.module = weld_module_compile(code, conf.conf, err.error)

    def run(self, conf, arg, err):
        weld_module_run = weld.weld_module_run
        # module, conf, arg, &err
        weld_module_run.argtypes = [
            c_weld_module, c_weld_conf, c_weld_value, c_weld_err]
        weld_module_run.restype = c_weld_value
        ret = weld_module_run(self.module, conf.conf, arg.val, err.error)
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

    def memory_usage(self):
        self._check()
        weld_value_memory_usage = weld.weld_value_memory_usage
        weld_value_memory_usage.argtypes = [c_weld_value]
        weld_value_memory_usage.restype = c_int64
        return weld_value_memory_usage(self.val)

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
        return copy.copy(val)

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

    def __init__(self):
        weld_error_new = weld.weld_error_new
        weld_error_new.argtypes = []
        weld_error_new.restype = c_weld_err
        self.error = weld_error_new()

    def code(self):
        weld_error_code = weld.weld_error_code
        weld_error_code.argtypes = [c_weld_err]
        weld_error_code.restype = c_uint64
        return weld_error_code(self.error)

    def message(self):
        weld_error_message = weld.weld_error_message
        weld_error_message.argtypes = [c_weld_err]
        weld_error_message.restype = c_char_p
        val = weld_error_message(self.error)
        return copy.copy(val)

    def __del__(self):
        weld_error_free = weld.weld_error_free
        weld_error_free.argtypes = [c_weld_err]
        weld_error_free.restype = None
        weld_error_free(self.error)

WeldLogLevelOff = 0
WeldLogLevelError = 1
WeldLogLevelWarn = 2
WeldLogLevelInfo = 3
WeldLogLevelDebug = 4
WeldLogLevelTrace = 5

def weld_set_log_level(log_level):
     """
     Sets the log_level for Weld:
        0 = No Logs,
        1 = Error,
        2 = Warn,
        3 = Info,
        4 = Debug,
        5 = Trace.
     """
     weld.weld_set_log_level(log_level)
