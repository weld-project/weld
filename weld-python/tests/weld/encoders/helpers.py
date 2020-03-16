
import ctypes

def encdec_factory(encoder, decoder, eq=None):
    """ Returns a function that encodes and decodes a value.

    Parameters
    ----------
    encoder : WeldEncoder
        the encoder class to use.
    decoder : WeldDecoder
        the decoder class to use.
    eq : function (T, T) => bool, optional (default=None)
        the equality function to use. If this is `None`, the `==` operator is
        used.

    Returns
    -------
    function

    """
    def encdec(value, ty, assert_equal=True, err=False):
        """ Helper function that encodes a value and decodes it.

        The function asserts that the original value and the decoded value are
        equal.

        Parameters
        ----------
        value : any
            The value to encode and decode
        ty : WeldType
            the WeldType of the value
        assert_equal : bool (default True)
            Checks whether the original value and decoded value are equal.
        err :  bool (default False)
            If True, expects an error.

        """
        enc = encoder()
        dec = decoder()

        try:
            result = dec.decode(ctypes.pointer(enc.encode(value, ty)), ty)
        except Exception as e:
            if err:
                return
            else:
                raise e

        if err:
            raise RuntimeError("Expected error during encode/decode")

        if assert_equal:
            if eq is not None:
                assert eq(value, result)
            else:
                assert value == result

    return encdec
