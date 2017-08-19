from weldarray import *
from weldnumpy import *

# FIXME: Should this be in weldnump? pytest gives errors when trying to import
# weldarray in weldnumpy...so keeping it here for now.

def array(arr, *args, **kwargs):
    '''
    Wrapper around weldarray - first create np.array and then convert to
    weldarray.
    '''
    return weldarray(np.array(arr, *args, **kwargs))

