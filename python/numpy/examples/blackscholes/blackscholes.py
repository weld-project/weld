#!/usr/bin/env python
import argparse
import time
from os import sys, path
sys.path.append("/lfs/1/pari/weld/python/numpy")
from weldnumpy import weldarray
import numpy as np
import weldnumpy as wn
import scipy.special as ss
import grizzly.grizzly as gr
from grizzly.lazy_op import LazyOpResult

# invsqrt2 = 1.0 #0.707
invsqrt2 = 0.707

def get_data(num_els):
    np.random.seed(2592)
    # random prices between 1 and 101
    price = np.float64(np.random.rand(num_els) * np.float64(100.0))
    # random prices between 0 and 101
    strike = np.float64(np.random.rand(num_els) * np.float64(100.0))
    # random maturity between 0 and 4
    t = np.float64(np.float64(1.0) + np.random.rand(num_els) * np.float64(6.0))
    # random rate between 0 and 1
    rate = np.float64(np.float64(0.01) + np.random.rand(num_els))
    # random volatility between 0 and 1
    vol = np.float64(np.float64(0.01) + np.random.rand(num_els))
    print('***********Generated Data*************')
    return price, strike, t, rate, vol


def blackscholes(price, strike, t, rate, vol, intermediate_eval, use_group):
    '''
    Implements the Black Scholes pricing model using NumPy and SciPy.
    Based on the code given by Intel, but cleaned up.

    The following links were used to define the constants c05 and c10:

    http://codereview.stackexchange.com/questions/108533/fastest-possible-cython-for-black-scholes-algorithm
    http://gosmej1977.blogspot.com/2013/02/black-and-scholes-formula.html
    '''
    c05 = np.float64(3.0)
    c10 = np.float64(1.5)
    rsig = rate + (vol*vol) * c05
    vol_sqrt = vol * np.sqrt(t)

    d1 = (np.log(price / strike) + rsig * t) / vol_sqrt
    d2 = d1 - vol_sqrt

    # these are numpy arrays, so use scipy's erf function. scipy's ufuncs also
    # get routed through the common ufunc routing mechanism, so these work just
    # fine on weld arrays.
    d1 = c05 + c05 * ss.erf(d1 * invsqrt2)
    d2 = c05 + c05 * ss.erf(d2 * invsqrt2)

    e_rt = np.exp((0.0-rate) * t)

    # An alternative to using the group operator is to manually evaluate all
    # the intermediate arrays whose values will be reused later in
    # computations. This is harder to do precisely, and we fail to apply
    # certain optimizations that weld could when using the group operator.
    if isinstance(price, weldarray) and intermediate_eval:
        price = price.evaluate()
        d1 = d1.evaluate()
        d2 = d2.evaluate()
        e_rt = e_rt.evaluate()
        strike = strike.evaluate()

    call = (price * d1) - (e_rt * strike * d2)
    put = e_rt * strike * (c10 - d2) - price * (c10 - d1)

    # the group operator! This essentially does the same thing as the
    # intermediate evaluation option used above - it avoids recomputions.
    if isinstance(call, weldarray) and use_group:
        lazy_ops = generate_lazy_op_list([call, put])
        outputs = gr.group(lazy_ops).evaluate(True, passes=wn.CUR_PASSES)

        call = weldarray(outputs[0])
        put = weldarray(outputs[1])

    # if we were not using the group operator.
    if isinstance(call, weldarray) and not use_group:
        call = call.evaluate()
        put = put.evaluate()

    return call, put

def generate_lazy_op_list(arrays):
    '''
    Slightly hacky way to match the group operator syntax.
    '''
    ret = []
    for a in arrays:
        lazy_arr = LazyOpResult(a.weldobj, a._weld_type, 1)
        ret.append(lazy_arr)
    return ret

def run_blackscholes(args, use_weld):
    p, s, t, r, v = get_data(args.num_els)
    if use_weld:
        p = weldarray(p)
        s = weldarray(s)
        t = weldarray(t)
        r = weldarray(r)
        v = weldarray(v)

    start = time.time()
    call, put = blackscholes(p, s, t, r, v, args.intermediate_eval, args.use_group)

    if isinstance(call, weldarray):
        call = call.evaluate()
        put = put.evaluate()
	print("**************************************************")
	print "weld took: %.4f (result=%.4f)" % (time.time() - start, call[0])
	print("**************************************************")
    else:
	print("**************************************************")
        print "numpy took: %.4f (result=%.4f)" % (time.time() - start, call[0])
	print("**************************************************")

    return call, put

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="give num_els of arrays used for blackscholes"
    )
    parser.add_argument('-n', "--num_els", type=int, required=False, default=1000,
                        help="num_els of 1d arrays")
    parser.add_argument('-ie', "--intermediate_eval", type=int, required=False,
                        default=1, help="num_els of 1d arrays")
    parser.add_argument('-g', "--use_group", type=int, required=False,
                        default=0, help="num_els of 1d arrays")
    parser.add_argument('-numpy', "--use_numpy", type=int, required=False, default=0,
                        help="use numpy or not in this run")
    parser.add_argument('-weld', "--use_weld", type=int, required=False, default=1,
                        help="use weld or not in this run")

    args = parser.parse_args()

    if args.use_numpy:
        call, put = run_blackscholes(args, False)
        print("*********Finished Numpy**********")
    else:
        print('Not running numpy')

    if args.use_weld:
        call2, put2 = run_blackscholes(args, True)
    else:
        print('Not running weld')

    # Correctness check.
    if args.use_numpy and args.use_weld:
        print("Using np.allclose to compare results of NumPy and Weld for call: ",
                np.allclose(call, call2.view(np.ndarray)))
        print("Using np.allclose to compare results of NumPy and Weld for put: ",
                np.allclose(put, put2.view(np.ndarray)))
        print(np.linalg.norm(put - put2.view(np.ndarray)))
        print(np.linalg.norm(call - call2.view(np.ndarray)))

        # mistakes = 0
        # for i in range(len(call)):
            # if (call[i] - call2[i] > 5):
                # mistakes += 1
                # print(i)
            # if (call[i] != call2[i]):
                # print("call: ", call[i])
                # print("call2: ", call2[i])
        # print("mistakes: ", mistakes)

