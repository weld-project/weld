"""
This module provides bindings to the core Weld API.

Note that this module wraps the core Weld API, whose documentation can be found
in much more detail `here<https://www.weld.rs/docs/latest/weld>`_.

"""

import copy
import enum
import pkg_resources
import platform

from ctypes import c_void_p, c_char_p, c_uint64

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

class c_weld_module(c_void_p):
    """ A wrapper type for a pointer to a ``WeldModule``. """
    pass

class c_weld_conf(c_void_p):
    """ A wrapper type for a pointer to a ``WeldConf``. """
    pass

class c_weld_err(c_void_p):
    """ A wrapper type for a pointer to a ``WeldError``. """
    pass

class c_weld_value(c_void_p):
    """ A wrapper type for a pointer to a ``WeldValue``. """
    pass

class c_weld_context(c_void_p):
    """ A wrapper type for a pointer to a ``WeldContext``. """
    pass

class WeldModule(c_void_p):
    """
    A module representing a compiled Weld program.
    """

    def __init__(self, code, conf, err):
        """ Create a new ``WeldModule``.

        Parameters
        ----------

        code : str
            A string representing a Weld function.

        conf : WeldConf
            A configuration used to control compilation.

        err : WeldError
            An error object where any errors will be written.

        """
        weld_module_compile = weld.weld_module_compile
        weld_module_compile.argtypes = [ c_char_p, c_weld_conf, c_weld_err]
        weld_module_compile.restype = c_weld_module
        code = c_char_p(code.encode('utf-8'))
        self._module = weld_module_compile(code, conf.conf, err.error)


    def run(self, context, value, err):
        """
        Runs this module on the given argument.

        Parameters
        ----------

        context : WeldContext
            A context where memory for this run will be allocated.

        value : WeldValue
            A value represented using the Weld ABI and wrapped in a
            ``WeldValue`` object. This is the argument to the Weld function.

        err : WeldError
            An error object where any errors will be written.

        """
        weld_module_run = weld.weld_module_run
        weld_module_run.argtypes = [ c_weld_module, c_weld_context,
                c_weld_value, c_weld_err]
        weld_module_run.restype = c_weld_value
        ret = weld_module_run(self._module, context, value._value, err._error)
        return WeldValue(ret, assign=True, _ctx=ctx)


    def __del__(self):
        weld_module_free = weld.weld_module_free
        weld_module_free.argtypes = [c_weld_module]
        weld_module_free.restype = None
        weld_module_free(self._module)


class WeldContext(c_void_p):
    """
    A context that represents a pool of memory for Weld runs.

    Contexts can be reused to share data between runs and free memory from many
    runs together.
    
    """

    def __init__(self, conf=None, _from_ctx=None):
        """
        Creates a new context for execution.

        Parameters
        ----------

        conf : WeldConf
            A configuration for creating the context. Defaults to ``None``, in
            which case a new default configuration is constructed.

        _from_ctx : WeldContext
            Internal use. Creates a context from an existing raw handle.

        """
        if _from_ctx is None:
            self._context = _from_ctx
        else:
            if conf is None:
                conf = WeldConf()
            weld_context_new = weld.weld_context_new
            weld_context_new.argtypes = [c_weld_conf]
            weld_context_new.restype = c_weld_context
            self._context = weld_context_new(conf)


    def memory_usage(self):
        """
        Returns the memory consumed by this value.

        Returns
        -------
        int
        
        """
        weld_context_memory_usage = weld.weld_context_memory_usage
        weld_context_memory_usage.argtypes = [c_weld_contextk]
        weld_context_memory_usage.restype = c_int64
        return weld_value_memory_usage(self._context)


    def free(self):
        """
        Frees the memory for this context.

        This should only be called once per context.

        """
        weld_context_free = weld.weld_context_free
        weld_context_free.argtypes = [c_weld_context]
        weld_context_free.restype = None
        self._context = weld_context_new(self._context)
        self._context = None

    def __del__(self):
        if self._context is not None:
            weld_context_free = weld.weld_context_free
            weld_context_free.argtypes = [c_weld_context]
            weld_context_free.restype = None
            self._context = weld_context_new(self._context)


class WeldValue(c_void_p):
    """
    A value that can be passed into and returned from Weld.
    """

    def __init__(self, value, assign=False, _ctx=None):
        """
        Creates a new value.

        Parameters
        ----------

        value : object
            A value represented in the Weld ABI. Usually, this means the value
            is encoded as a contiguous buffer, a C-like struct, etc.

        assign : bool
            If this is ``True``, just assigns the value to a field instead of
            calling Weld's ``WeldValue::new```. Defaults to ``False``.

        _ctx : WeldContext
            Internal use only. Provides the context that manages this value.

        """
        if assign is False:
            weld_value_new = weld.weld_value_new
            weld_value_new.argtypes = [c_void_p]
            weld_value_new.restype = c_weld_value
            self._value = weld_value_new(value)
        else:
            self._value = value

        self._ctx = _ctx
        self.freed = False


    def data(self):
        """
        Returns a raw pointer to the data wrapped by this value.

        Returns
        -------
        any
        
        """
        weld_value_data = weld.weld_value_data
        weld_value_data.argtypes = [c_weld_value]
        weld_value_data.restype = c_void_p
        return weld_value_data(self._value)

    def context(self):
        """
        Returns the context associated with this data. 

        This should only be called on values returned ("owned") by Weld. This
        context **must** be deleted or freed in order for the underlying data
        to be freed.

        Returns
        -------
        WeldContext
            Returns the context this value is allocated in.
        
        """
        weld_value_context = weld.weld_value_context
        weld_value_context.argtypes = [c_weld_value]
        weld_value_context.restype = c_weld_context
        ctx = weld_value_context(self._value)
        return WeldContext(_from_ctx=ctx)


class WeldConf(c_void_p):
    """
    A configuration used to configure compilation and execution of a Weld program.
    """

    def __init__(self):
        """ 
        Creates a new empty configuration.

        See `this page<https://www.weld.rs/docs/latest/weld/#constants>`_ for
        available keys and values, and defaults.  Values should always be
        encoded as UTF-8-compatible strings. If a key is missing, its default
        value is used.
        """
        weld_conf_new = weld.weld_conf_new
        weld_conf_new.argtypes = []
        weld_conf_new.restype = c_weld_conf
        self.conf = weld_conf_new()

    def get(self, key):
        """
        Gets a value for the given key.

        Parameters
        ----------

        key : str
            A string key whose value should be fetched.

        Returns
        -------
        str
            Returns the string value for the key.

        """
        key = c_char_p(key.encode('utf-8'))
        weld_conf_get = weld.weld_conf_get
        weld_conf_get.argtypes = [c_weld_conf, c_char_p]
        weld_conf_get.restype = c_char_p
        val = weld_conf_get(self.conf, key)
        return copy.copy(val)

    def set(self, key, value):
        """
        Sets a value for a key.

        Parameters
        ----------

        key : str
            A string key.
        value : str
            A string value.

        """
        key = c_char_p(key.encode('utf-8'))
        value = c_char_p(value.encode('utf-8'))
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
    """
    An error returned by the Weld library.
    """

    def __init__(self):
        """
        Creates a new empty error.

        """
        weld_error_new = weld.weld_error_new
        weld_error_new.argtypes = []
        weld_error_new.restype = c_weld_err
        self.error = weld_error_new()

    def code(self):
        """
        Returns the integer error code.

        Returns
        ----------
        int
            An error code representing the error that occurred.

        """
        weld_error_code = weld.weld_error_code
        weld_error_code.argtypes = [c_weld_err]
        weld_error_code.restype = c_uint64
        return weld_error_code(self.error)

    def message(self):
        """
        Returns the string error message.

        Returns
        ----------
        str
            An error message representing the error that occurred.

        """
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

class WeldLogLevel(enum.IntEnum):
    OFF = 0
    ERROR = 1
    WARN = 2
    INFO = 3
    DEBUG = 4
    TRACE = 5

def weld_set_log_level(level):
     """
     Sets the Weld log level.

     Parameters
     ----------

     level : WeldLogLevel
        The level to which logging should be set.
     """
     weld.weld_set_log_level(level.value)
