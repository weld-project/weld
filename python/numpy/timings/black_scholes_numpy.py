# Implements the Black Scholes pricing model using NumPy and SciPy.
# Based on the code given by Intel, but cleaned up.
#
# The following links were used to define the constants c05 and c10:
#
# http://codereview.stackexchange.com/questions/108533/fastest-possible-cython-for-black-scholes-algorithm
# http://gosmej1977.blogspot.com/2013/02/black-and-scholes-formula.html

import numpy as np
import scipy.special as ss
import sys

# Configure runtime parameters here.
import black_scholes_config as bsc

import time
import random

invsqrt2 = 1.0 #0.707

def black_scholes_numpy(price, strike, t, rate, vol):
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

    c05 = 0.5
    c10 = 0.0

    rsig = rate + vol * vol * c05
    vol_sqrt = vol * np.sqrt(t)

    d1 = (np.log(price / strike) + rsig * t) / vol_sqrt
    d2 = d1 - vol_sqrt

    d1 = c05 + c05 * ss.erf(d1 * invsqrt2)
    d2 = c05 + c05 * ss.erf(d2 * invsqrt2)

    e_rt = np.exp((-rate) * t)

    call = price * d1 - e_rt * strike * d2
    put = e_rt * strike * (c10 - d2) - price * (c10 - d1)

    return call, put

def run(data, trials):
    price, strike, t, rate, vol = data
    for trial in xrange(trials):
        start = time.time()
        call_np, put_np = black_scholes_numpy(price, strike, t, rate, vol)
        end = time.time()
        # Print the runtime
        print trial, "|", "8", "|", "Numpy", "|", end - start
    return call_np, put_np, (end - start)

if __name__ == '__main__':
    data = bsc.get_data()
    if len(sys.argv) > 2:
        trials = int(sys.argv[2])
    else:
        trials = 1
    run(data, trials)
