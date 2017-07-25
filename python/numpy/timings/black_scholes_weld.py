# Implements the Black Scholes pricing model using Weld.
#
# The following links were used to define the constants c05 and c10:
#
# http://codereview.stackexchange.com/questions/108533/fastest-possible-cython-for-black-scholes-algorithm
# http://gosmej1977.blogspot.com/2013/02/black-and-scholes-formula.html

import numpy as np
import sys
sys.path.append('..')
from weldarray import WeldArray
import os
import time
import random

invsqrt2 = 1.0 #0.707
verbose = False

os.environ["NUM_WORKERS"] = "12"

import black_scholes_config as bsc

def black_scholes_weld(price, strike, t, rate, vol):
    """
    All inputs are float32 NumPy arrays.

    price: stock price at time 0.
    strike: strike price
    t: time to maturity in trading years.
    vol: volatility of the stock.
    rate: continuously compounded risk-free rate

    Returns: call and put of each stock as a vector

    Order of operations in terms of usage:

    +, *, sqrt, log, /, -, erf, exp,
    """
    # Should be easy to replicate in weld.
    price = WeldArray(price)
    strike = WeldArray(strike)
    t = WeldArray(t)
    rate = WeldArray(rate)
    vol = WeldArray(vol)

    c05 = 0.5
    c10 = 0.0

    rsig = rate + vol * vol * c05
    # Need sqrt operator in weld.
    vol_sqrt = vol * np.sqrt(t)

    # Have log operator.
    d1 = (np.log(price / strike) + rsig * t) / vol_sqrt
    d2 = d1 - vol_sqrt

    # FIXME: need to implement erf in weldarray numpy
    # d1 = c05 + c05 * nnp.erf(d1 * invsqrt2)
    # d2 = c05 + c05 * nnp.erf(d2 * invsqrt2)
    d1 = c05 + c05 * np.exp(d1 * invsqrt2)
    d2 = c05 + c05 * np.exp(d2 * invsqrt2)

    e_rt = np.exp((-rate) * t)

    call = price * d1 - e_rt * strike * d2
    put = e_rt * strike * (c10 - d2) - price * (c10 - d1)

    # Find equivalent weld version.
    call = call.eval()
    put = put.eval()

    # FIXME: Return total time / or just time spent doing computations as
    # it was done before?
    return call, put, None

# Sorted by frequency of the operator
ordered_ops =  ["erf", "*", "-", "/", "+", "exp", "sqrt", "log"]

def run(data, num_ops, trial):
    """
    Runs a single trial of black scholes with the given trial number.
    """
    # Remove all the operators
    price, strike, t, rate, vol = data

    # print nnp.implemented_operators()
    call_weld, put_weld, time = black_scholes_weld(price, strike, t, rate, vol)
    print trial, "|", num_ops, "|", "Weld", "|", time
    return call_weld, put_weld, time

if __name__ == '__main__':
    data = bsc.get_data()
    num_ops = int(sys.argv[1]) if len(sys.argv) > 1 else len(ordered_ops)
    trials = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    run(data, num_ops, trials)


