"""
This module provides an interface for compiling Weld functions and calling them
as Python functions.

"""

from weld.core import *
from weld.encoders import WeldEncoder, WeldDecoder
from weld.encoders.primitives import PrimitiveWeldEncoder, PrimitiveWeldDecoder
from weld.types import WeldType

import ctypes
import logging

def _args_factory(arg_names, arg_types):
    """
    Constructs a ctypes Structure with the given field names and field types.

    This is used to construct the argument struct for a Weld program.

    Parameters
    ----------

    arg_names : list of str
        Field names for the arguments.
    arg_types : list of ctypes
        Types for each field.

    Returns
    -------
    Args class
        Returns a class representing a ctypes Structure.

    """
    class Args(ctypes.Structure):
        _fields_ = list(zip(arg_names, arg_types))
    return Args

def compile(program, arg_types, encoders, restype, decoder, conf=None):
    """Compiles a program and returns a function for calling it.

    Parameters
    ----------
    program : str
        A string representing a Weld program.
    arg_types : list of WeldType
        A list describing the types of each argument. This argument will
        eventually be removed since this information can be retrieved from the
        provided program, but the mechanisms for passing that information to
        the Python bindings has not been built yet.
    encoders: list of (WeldEncoders and/or None)
        A list that provides an encoder instance for each argument. One encoder
        must be provided for each argument (the ``i``th encoder corresponds to
        the ``i``th argument). Users can optionally pass ``None`` in place of
        an encoder, in which case the ``PrimitiveWeldEncoder`` will be used.
    restype : WeldType
        The WeldType of the result. This argument will eventually be removed
        since this information can be retrieved from the provided program, but
        the mechanisms for passing that information to the Python bindings has
        not been built yet.
    decoder : WeldDecoder
        A decoder for the return value.
    conf : WeldConf
        Configuration for compilation and runtime.

    Returns
    -------
    function(args) -> (any, WeldContext)
        Returns a function that takes argument of the specified types and
        returns a value that is decoded using the provided decoder. The
        function also returns the context in which the value is allocated, in
        case the decoder does not copy data.

    Examples
    --------

    The following example compiles a simple program that adds one to an
    integer. We use the default primitive encoders/decoders by passing
    ``None``:

    >>> from weld.types import *
    >>> func = compile("|x: i32| x + 1",
    ...        [I32()],  [None],
    ...        I32(), None)
    ...
    >>> func(100)[0]
    101
    >>> func(200)[0]
    201

    The next example adds two numbers together, following the same principle as above:

    >>> func = compile("|x: i32, y: i32| x + y",
    ...        [I32(), I32()],  [None, None],
    ...        I32(), None)
    ...
    >>> func(5, 6)[0]
    11

    We can also pass encoders explicitly:

    >>> func = compile("|x: i32| x + 1",
    ...        [I32()],  [PrimitiveWeldEncoder()],
    ...        I32(), PrimitiveWeldDecoder())
    ...
    >>> func(100)[0]
    101

    """

    # Checks
    assert len(arg_types) == len(encoders)
    assert False not in [isinstance(ty, WeldType) for ty in arg_types]
    assert False not in [encoder is None or isinstance(encoder, WeldEncoder) for encoder in encoders]
    assert isinstance(restype, WeldType)
    assert decoder is None or isinstance(decoder, WeldDecoder)

    if conf is None:
        conf = WeldConf()

    module = WeldModule(program, conf)

    # TODO(shoumik): Add assertion checking module.return_type() vs. decoder's supported types.

    def func(*args, context=None):
        # Field names.
        names = []
        # C type of each argument.
        arg_c_types = []
        # Encoded version of each argument.
        encoded = []

        primitive_encoder = PrimitiveWeldEncoder()
        primitive_decoder = PrimitiveWeldDecoder()

        for (i, (arg, arg_type, encoder)) in enumerate(zip(args, arg_types, encoders)):

            names.append("_{}".format(i))
            arg_c_types.append(arg_type.ctype_class)

            logging.debug(i, arg, arg_type)

            if encoder is not None:
                encoded.append(encoder.encode(arg, arg_type))
            else:
                encoded.append(primitive_encoder.encode(arg, arg_type))

        logging.debug(names)
        logging.debug(arg_c_types)
        logging.debug(encoded)

        Args = _args_factory(names, arg_c_types)
        raw_args = Args()

        for name, value in zip(names, encoded):
            setattr(raw_args, name, value)

        raw_args_pointer = ctypes.addressof(raw_args)
        value = WeldValue(raw_args_pointer)

        if context is None:
            context = WeldContext(conf)

        result = module.run(context, value)

        pointer_type = ctypes.POINTER(restype.ctype_class)
        data = ctypes.cast(result.data(), pointer_type)

        if decoder is not None:
            result = decoder.decode(data, restype, context)
        else:
            result = primitive_decoder.decode(data, restype, context)
        return (result, context)

    return func
