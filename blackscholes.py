#!/usr/bin/env python

import argparse
import time
from weldnumpy import weldarray
import scipy.special as ss
import grizzly.grizzly as gr
from grizzly.lazy_op import LazyOpResult

invsqrt2 = 0.707

def get_data(num_els):
    np.random.seed(0)
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

    # TODO: would be nice if this could be avoided...
    if isinstance(vol, weldarray):
        price = price.evaluate()
        strike = strike.evaluate()
        t = t.evaluate()
        rate = rate.evaluate()
        vol = vol.evaluate()
    
    return price, strike, t, rate, vol


def blackscholes(price, strike, t, rate, vol, int_eval, use_group):
    '''
    Implements the Black Scholes pricing model using NumPy and SciPy.
    Based on the code given by Intel, but cleaned up.

    The following links were used to define the constants c05 and c10:

    http://codereview.stackexchange.com/questions/108533/fastest-possible-cython-for-black-scholes-algorithm
    http://gosmej1977.blogspot.com/2013/02/black-and-scholes-formula.html
    '''
    c05 = np.float64(3.0)
    c10 = np.float64(1.5)

    rsig = rate + (vol * vol) * c05
    vol_sqrt = vol * np.sqrt(t)

    d1 = (np.log(price / strike) + rsig * t) / vol_sqrt
    d2 = d1 - vol_sqrt

    if isinstance(price, weldarray):
        # weldarrays. weldnumpy supports erf directly
        d1 = c05 + c05 * wn.erf(d1 * invsqrt2)
        d2 = c05 + c05 * wn.erf(d2 * invsqrt2)
    else:
        # these are numpy arrays, so use scipy's erf function
        d1 = c05 + c05 * ss.erf(d1 * invsqrt2)
        d2 = c05 + c05 * ss.erf(d2 * invsqrt2)
    
    e_rt = np.exp((-rate) * t)

    if isinstance(price, weldarray) and int_eval:
        price = price.evaluate()
        d1 = d1.evaluate()
        d2 = d2.evaluate()
        e_rt = e_rt.evaluate()
        strike = strike.evaluate()

    call = price * d1 - e_rt * strike * d2
    put = e_rt * strike * (c10 - d2) - price * (c10 - d1)

    if isinstance(call, weldarray) and use_group:
        print('going to use group!!!!')
        lazy_ops = generate_lazy_op_list([call, put])
        call, put = gr.group(lazy_ops).evaluate(True)

    if isinstance(call, weldarray) and not use_group:
        call = call.evaluate()
        put = put.evaluate()
    
    return call, put

def generate_lazy_op_list(arrays):
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
    call, put = blackscholes(p, s, t, r, v, args.int_eval, args.use_group)

    if use_weld:
        print "-------------> Weld: %.4f (result=%.4f)" % (time.time() - start, call[0])
    else:
        print "-------------> Numpy: %.4f (result=%.4f)" % (time.time() - start, call[0])

    return call, put

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="give num_els of arrays used for blackscholes"
    )
    parser.add_argument('-n', "--num_els", type=int, required=True,
                        help="num_els of 1d arrays")
    parser.add_argument('-ie', "--int_eval", type=int, required=True,
                        help="num_els of 1d arrays")
    parser.add_argument('-g', "--use_group", type=int, 
                        default=1, help="num_els of 1d arrays")
    parser.add_argument('-p', "--remove_pass", type=str, 
                        default="whatever_string", help="will remove the pass containing this str")

    args = parser.parse_args()

    import numpy as np
    import weldnumpy as wn
    call2, put2 = run_blackscholes(args, True)
    call, put = run_blackscholes(args, False)

    print("close put, put2: ", np.allclose(put, put2))
    print("close call1, call2: ", np.allclose(call, call2))

    print call
    print call2

